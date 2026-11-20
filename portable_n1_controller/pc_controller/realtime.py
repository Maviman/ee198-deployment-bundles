"""The real-deployment control loop (``run_controller.py --source udp``).

Latency design
--------------
Perception sends a pose frame for EVERY camera frame (30 fps on the fast
path). The policy was trained at a 10 Hz tick, so the controller acts on one
frame per 100 ms. It chooses the frame that arrives nearest each tick's due
time and acts on it the instant it arrives, so the command is computed from a
pose that is only perception-latency old. A timer-driven loop would add up to
a frame interval of staleness. The tick timeline advances by exactly the
period (not "now + period"), so the average rate stays at the trained 10 Hz
whatever the camera rate. Use a camera rate that is a multiple of the tick
rate (10, 20, 30 fps) and ticks are evenly spaced too.

Safety (unchanged semantics, plus two additions)
------------------------------------------------
- No frame for ``stall_s`` (0.3 s): E-stop every car, reset velocity history,
  re-arm automatically when frames resume. As before.
- NEW: frames whose perception-side latency exceeds ``max_pose_age_s`` are
  "late" and never acted on. If only late frames arrive for ``stall_s``,
  that is a stall too. The policy collapses beyond ~0.2 s of latency
  (Phase 2 robustness grid), so acting on stale poses is worse than stopping.
- A LOST car is not a stall (2026-11-19). Perception keeps sending frames with that car
  marked lost; pc_controller/deadzone.py brings it back into view (blind recovery) or stops
  it, and stands every car still while the evader is lost. A stall is now only what it says:
  no frames at all (camera or process down, network gone), or only late ones.
- NEW: operator arm/disarm. With ``start_disarmed`` the policy runs and its
  output is shown on the dashboard, but every car receives E-stop packets
  until an operator arms (`arena go`). Disarm (`arena halt`, the dashboard
  STOP button, or Ctrl-C) takes effect on the next packet. E-stop packets are
  ACKed, so link quality is visible before anything moves.

The control port listens on localhost only: arming is done from the control
Orin itself (the `arena` CLI SSHes there). Nothing on the LAN can arm the cars.
"""

from __future__ import annotations

import json
import math
import select
import socket
import time

import numpy as np

from pc_controller.deadzone import CarSpec, DeadZone, fresh_seen
from pc_controller.pose_stream import UdpPoseSource


class _Window:
    def __init__(self, n: int = 100) -> None:
        self.n, self.v = n, []

    def add(self, x: float) -> None:
        self.v.append(float(x))
        if len(self.v) > self.n:
            del self.v[: len(self.v) - self.n]

    def stats(self) -> dict:
        if not self.v:
            return {}
        s = sorted(self.v)
        at = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]  # noqa: E731
        return {"p50": round(at(0.5), 2), "p95": round(at(0.95), 2), "max": round(s[-1], 2)}


class HoldZone:
    """Stand-still zone around the evader (``run_controller.py --hold-radius``).

    While a car is within ``radius`` metres of the evader its command is [0, 0]: every
    button released. This is NOT an E-stop and NOT a capture: the car stays ARMED, the
    link stays live (the firmware's dead-man keeps being fed), the poses keep being
    tracked and the policy keeps being asked on every tick. The car goes back to the
    policy as soon as the evader is more than ``radius + resume_margin`` away, so it
    follows an evader that moves off. The margin is hysteresis: pose noise on the
    boundary must not chatter the buttons. Each car is judged on its own distance.

    Only the command that is SENT is overridden. A model with an action history (k > 0)
    still records its own un-held output there; the deployed n1_catch has no history.
    """

    def __init__(self, radius: float, resume_margin: float = 0.10) -> None:
        if not radius > 0.0:
            raise ValueError("hold radius must be > 0 metres")
        self.radius = float(radius)
        self.resume_radius = self.radius + max(0.0, float(resume_margin))
        self.held: list[bool] = []

    def reset(self) -> None:
        """After a stall or re-arm: nobody is held until the next pose says so."""
        self.held = []

    def apply(self, action, pursuers, evader, log=None):
        """``action`` (throttle, steer per pursuer) with every held car zeroed.

        Returns ``action`` itself when nobody is held, otherwise a copy; the policy's own
        output is never modified in place."""
        if len(self.held) != len(pursuers):
            self.held = [False] * len(pursuers)
        out = action
        for i, p in enumerate(pursuers):
            d = math.hypot(p.x - evader.x, p.y - evader.y)
            was = self.held[i]
            now = d <= self.radius or (was and d <= self.resume_radius)
            if now != was:
                self.held[i] = now
                if log is not None:
                    log(f"car {i} HOLD: {d:.2f} m from the evader (<= {self.radius:g} m): buttons "
                        f"released, still armed and tracking" if now else
                        f"car {i} RESUME: evader {d:.2f} m away (> {self.resume_radius:g} m): driving again")
            if now:
                if out is action:
                    out = action.copy()
                out[2 * i] = 0.0
                out[2 * i + 1] = 0.0
        return out


class Telemetry:
    def __init__(self, target: str | None) -> None:
        self.addr = None
        self.sock = None
        if target:
            host, _, port = target.rpartition(":")
            try:
                host = socket.gethostbyname(host or "127.0.0.1")   # once, not per packet
            except OSError:
                return                                             # telemetry is optional
            self.addr = (host, int(port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)

    def send(self, payload: dict) -> None:
        if self.sock is not None:
            try:
                self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.addr)
            except OSError:
                pass


class ControlPort:
    """Localhost-only UDP command port: {"cmd": "arm"|"disarm"|"status"|"manual"|"manual_off"}.

    "manual" ({"cmd": "manual", "thr": t, "steer": s, "car": i}) drives car i by hand (the
    dashboard's manual drive): refused while the policy is armed, and it lasts
    MANUAL_DEADMAN_S unless refreshed, so a closed page or a dropped network stops the car."""

    def __init__(self, port: int | None) -> None:
        self.sock = None
        if port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", int(port)))
            self.sock.setblocking(False)

    def poll(self):
        """Yield (cmd, message, reply_addr) for every pending command."""
        if self.sock is None:
            return
        while True:
            try:
                raw, addr = self.sock.recvfrom(1024)
            except (BlockingIOError, ConnectionResetError, OSError):
                return
            try:
                msg = json.loads(raw.decode("utf-8"))
                yield str(msg.get("cmd", "")), msg, addr
            except (ValueError, AttributeError):
                continue

    def reply(self, addr, payload: dict) -> None:
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), addr)
        except OSError:
            pass


MANUAL_DEADMAN_S = 0.3        # manual drive stops this long after the last "manual" command
MANUAL_PERIOD_S = 0.05        # manual commands go to the car at 20 Hz, poses or not


def _clamp1(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else max(-1.0, min(1.0, v))


def run_realtime(loop, link, *, pose_port: int, tick_hz: float = 10.0, stall_s: float = 0.3,
                 max_pose_age_s: float = 0.15, telemetry: str | None = None,
                 control_port: int | None = None, start_disarmed: bool = False,
                 model_name: str = "", log=print, stop=None,
                 hold_radius: float | None = None, recovery: bool = True,
                 recovery_min_throttle: float = 0.0, recovery_binary_drive: bool = False,
                 edge_brake: bool = True) -> None:
    """Run the loop until Ctrl-C/SIGTERM, or until ``stop`` (an optional threading.Event,
    for tests and embedders) is set. The pose socket is closed on the way out either way;
    the caller owns the link. ``hold_radius`` (metres) turns on the HoldZone around the
    evader; None or 0 leaves the policy in charge at every distance. ``recovery`` False:
    a car the cameras lose is only stopped, never driven blind (pc_controller/deadzone.py);
    the other two say how this car's drive takes a slow recovery throttle (same module)."""
    n = loop.policy.num_pursuers
    handles_unseen = bool(getattr(loop, "handles_unseen", False)) and recovery
    dead = DeadZone(
        n, spec=CarSpec.from_manifest(getattr(loop.policy, "manifest", None) or {}) if recovery else None,
        runtime_handles_unseen=handles_unseen,
        runtime_status=getattr(loop, "recovery_status", None) if handles_unseen else None,
        dt=1.0 / tick_hz, min_throttle=recovery_min_throttle, binary_drive=recovery_binary_drive,
        guard=edge_brake)
    last_seen = None
    source = UdpPoseSource(pose_port, expected_pursuers=n)
    tel = Telemetry(telemetry)
    ctl = ControlPort(control_port)
    capture_radius = float(getattr(loop, "capture_radius", 0.0))
    hold = HoldZone(hold_radius) if hold_radius else None

    period = 1.0 / tick_hz
    armed = not start_disarmed
    state = "WAITING"
    due = None
    last_good = time.perf_counter()
    frame_dt = period / 3.0              # EMA of frame inter-arrival, seeded for 30 fps
    last_arrival = None
    ticks = stalls = estops = late = 0
    w_period, w_vis, w_ctl, w_total, w_infer = _Window(), _Window(), _Window(), _Window(), _Window()
    last_tick_t = None
    last_action = None
    last_poses = None
    last_tel = 0.0
    last_keepalive = 0.0
    stall_reason = ""
    manual = None                        # {"car", "thr", "steer", "until"} while driven by hand
    last_manual_send = 0.0
    log(f"listening for pose frames on UDP :{pose_port}; tick {tick_hz:g} Hz; "
        f"{'DISARMED until `arena go`' if not armed else 'ARMED'} (ctrl-C to stop)")
    log({"runtime": "DEAD ZONE: a car the cameras lose is brought back by the model's own recovery "
                    "(brake, then back along its track); past the tape it drives back in",
         "controller": "DEAD ZONE: a car the cameras lose is brought back by the controller (brake, then "
                       "back along its own track); past the tape it drives back in"
                       + ("; EDGE BRAKE on (too fast to stop before the tape: brake)" if edge_brake else ""),
         "stop-only": "DEAD ZONE: a car the cameras lose is STOPPED until it is seen again (no blind "
                      "recovery" + (" requested" if not recovery else ": the model's manifest lacks the car's numbers") + ")",
         }[dead.mode]
        + (f"; recovery throttle as on/off pulses (drive binary)" if recovery_binary_drive else
           f"; recovery throttle at least {recovery_min_throttle:g}" if recovery_min_throttle else ""))
    if hold is not None:
        log(f"HOLD ZONE on: a car within {hold.radius:g} m of the evader stands still (buttons released, "
            f"still armed); it drives again once the evader is {hold.resume_radius:g} m away")

    def send_state_telemetry(now: float) -> None:
        cars = []
        if link is not None:
            for i in range(len(link.targets)):
                st = link.car_stats.get(i, {})
                age = None if st.get("last_ack") is None else round(now - st["last_ack"], 2)
                cars.append({"idx": i, "addr": link.label(i), "rtt_ms": st.get("rtt_ms"),
                             "rssi": st.get("rssi"), "acks": st.get("acks", 0),
                             "applied": st.get("applied"), "slot": st.get("slot"),
                             "ack_age_s": age})
        dist = None
        if last_poses is not None:
            ps, ev = last_poses
            dist = [round(math.hypot(p.x - ev.x, p.y - ev.y), 3) for p in ps]
        tel.send({
            "src": "control", "wall": round(time.time(), 3), "state": state, "armed": armed,
            "model": model_name, "n": n, "ticks": ticks, "tick_hz": tick_hz,
            "period_ms": w_period.stats(),
            "lat_ms": {"vision": w_vis.stats(), "control": w_ctl.stats(),
                       "total": w_total.stats(), "infer": w_infer.stats()},
            "frames_hz": round(1.0 / frame_dt, 1) if frame_dt > 0 else None,
            "drained": source.drained, "bad_frames": source.bad_frames, "late_frames": late,
            "stalls": stalls, "estops": estops, "stall_reason": stall_reason,
            "cmd": None if last_action is None else [round(float(v), 3) for v in last_action],
            "poses": None if last_poses is None else {
                "pursuers": [[round(p.x, 4), round(p.y, 4), round(p.heading, 4)] for p in last_poses[0]],
                "evader": [round(last_poses[1].x, 4), round(last_poses[1].y, 4), round(last_poses[1].heading, 4)]},
            # which poses the cameras see right now (pursuers in order, then the evader); a
            # False one is drawn where it was LAST seen. deadzone: what is being done about it.
            "seen": last_seen, "deadzone": dead.status(),
            "dist": dist, "capture_radius": capture_radius, "cars": cars,
            "hold_radius": None if hold is None else hold.radius,
            "hold": None if hold is None else list(hold.held),
            "roles": list(getattr(loop, "roles", None) or []) or None,   # role commanders only
            "manual": None if manual is None else {"car": manual["car"],
                                                   "cmd": [round(manual["thr"], 3), round(manual["steer"], 3)]},
        })

    def estop(reason: str) -> None:
        nonlocal estops
        if link is not None:
            link.send_stop()
        estops += 1

    try:
        while stop is None or not stop.is_set():
            now = time.perf_counter()
            for cmd, msg, addr in ctl.poll():
                error = None
                if cmd == "arm":
                    if manual is not None:
                        manual = None
                        log("MANUAL DRIVE ended: armed by operator")
                    armed = True
                    # A run starts clean: speeds, the model's memory and above all the recovery's
                    # memory of "commands sent" (while disarmed none were) are from before.
                    loop.reset()
                    dead.reset()
                    if hold is not None:
                        hold.reset()
                    log("ARMED by operator")
                elif cmd == "disarm":
                    if armed:
                        log("DISARMED by operator -> E-stop")
                    if manual is not None:
                        log("MANUAL DRIVE ended by STOP")
                    armed = False
                    manual = None
                    estop("operator")
                elif cmd == "manual":
                    car = int(msg.get("car", 0) or 0)
                    if armed:
                        error = "the policy is driving (ARMED): press STOP first"
                    elif link is None:
                        error = "no car link (dry run)"
                    elif not 0 <= car < n:
                        error = f"car {car} is not in this fleet (0..{n - 1})"
                    else:
                        if manual is None:
                            log(f"MANUAL DRIVE on: car {car} driven from the dashboard")
                        manual = {"car": car, "thr": _clamp1(msg.get("thr")), "steer": _clamp1(msg.get("steer")),
                                  "until": now + MANUAL_DEADMAN_S}
                elif cmd == "manual_off":
                    if manual is not None:
                        log("MANUAL DRIVE off -> E-stop")
                        estop("manual off")
                    manual = None
                reply = {"armed": armed, "state": state, "ticks": ticks, "manual": manual is not None}
                if error:
                    reply["error"] = error
                ctl.reply(addr, reply)

            # Manual drive: the operator's command at 20 Hz whatever the poses do, until
            # the dashboard stops refreshing it (MANUAL_DEADMAN_S), then an E-stop.
            if manual is not None and link is not None:
                if now > manual["until"]:
                    log("MANUAL DRIVE stopped: no command from the dashboard for "
                        f"{MANUAL_DEADMAN_S:g} s -> E-stop")
                    manual = None
                    estop("manual deadman")
                elif now - last_manual_send >= MANUAL_PERIOD_S:
                    act = np.zeros(2 * n)
                    act[2 * manual["car"]] = manual["thr"]
                    act[2 * manual["car"] + 1] = manual["steer"]
                    link.send_commands(act)
                    last_manual_send = now
                    last_action = act

            # Not driving (waiting for the first pose, or stalled): keep telling
            # every car to hold E-stop at 2 Hz. The cars would sit in failsafe
            # anyway; this keeps their ACKs, and so the dashboard's radio health,
            # alive while you are checking the setup.
            if state != "RUNNING" and manual is None and link is not None and now - last_keepalive > 0.5:
                link.send_stop()
                last_keepalive = now

            # Wait on poses, ACKs and operator commands together, so each is
            # handled (and each ACK timestamped) the moment it lands.
            timeout = max(0.005, min(stall_s - (now - last_good), 0.05))
            # Rebuilt every pass: a serial port that fails and reopens is a new fd.
            watch = [source.sock] + (link.selectables() if link is not None else []) + (
                [ctl.sock] if ctl.sock is not None else [])
            ready, _, _ = select.select(watch, [], [], timeout)
            if link is not None:
                link.poll_acks()          # ACKs from every transport; also services serial ports
            if ctl.sock is not None and ctl.sock in ready:
                continue                  # handled at the top of the loop
            got = source.drain_latest() if source.sock in ready else None
            now = time.perf_counter()

            if got is None:
                if now - last_good > stall_s and state != "STALLED":
                    frames_arriving = last_arrival is not None and now - last_arrival < stall_s
                    stall_reason = ("pose frames arriving but too old (perception latency over budget)"
                                    if frames_arriving else "no pose frames")
                    log(f"POSE FEED STALLED >{stall_s:.1f}s ({stall_reason}) -> E-stop, waiting")
                    if manual is None:            # manual drive needs no poses
                        estop("stall")
                    loop.reset()          # stale histories would corrupt velocity estimates
                    dead.reset()
                    if hold is not None:
                        hold.reset()
                    state, due = "STALLED", None
                    stalls += 1
                if now - last_tel > 0.25:
                    last_tel = now
                    send_state_telemetry(now)
                continue

            pursuers, evader, meta = got
            arrival = meta["arrival"]
            if last_arrival is not None:
                gap = arrival - last_arrival
                if 0.0 < gap < 0.25:
                    frame_dt = 0.9 * frame_dt + 0.1 * gap
            last_arrival = arrival
            vis_lat = meta.get("lat")
            if vis_lat is not None and vis_lat > max_pose_age_s:
                late += 1
                continue                  # never act on a stale pose; the stall timer runs on
            last_good = now
            if state == "STALLED":
                log("pose feed recovered, re-armed" if armed else "pose feed recovered (disarmed)")
            if state in ("STALLED", "WAITING"):
                state = "RUNNING"

            if due is not None and arrival < due - frame_dt / 2.0:
                continue                  # not this tick's frame; a nearer one is coming
            # --- tick ---------------------------------------------------------
            t_inf = time.perf_counter()
            seen, lost_s = fresh_seen(pursuers, evader, meta)
            arena = meta.get("arena")
            if arena:
                try:
                    model_half = dead.spec.half_m if dead.spec is not None else None
                    if dead.set_arena_half(min(float(arena[0]), float(arena[1]))):
                        log(f"DEAD ZONE: the cameras are calibrated on a {2 * dead.spec.half_m:.2f} m square, the "
                            f"model was trained on {2 * model_half:.2f} m: the tape is taken from the cameras")
                except (TypeError, ValueError, IndexError):
                    pass
            if handles_unseen and seen is not None:
                action = loop.tick(pursuer_poses=pursuers, evader_pose=evader, seen=seen)
            else:
                action = loop.tick(pursuer_poses=pursuers, evader_pose=evader)
            model_action = action
            if hold is not None:
                action = hold.apply(action, pursuers, evader, log)
            # Last: a recovering car's command replaces whatever the model or the hold zone said.
            action = dead.apply(action, pursuers, evader, seen, lost_s, log, raw=model_action,
                                sending=armed and manual is None)
            last_seen = None if seen is None else [bool(s) for s in seen]
            t_done = time.perf_counter()
            if link is not None and manual is None:
                if armed:
                    link.send_commands(action)
                else:
                    link.send_stop()
            t_sent = time.perf_counter()
            if due is None or arrival - due > period / 2.0:
                due = arrival + period            # first tick, or fell badly behind: restart the timeline
            else:
                due += period                     # fixed timeline: the average stays at tick_hz
            if last_tick_t is not None:
                w_period.add((t_sent - last_tick_t) * 1000.0)
            last_tick_t = t_sent
            ticks += 1
            ctl_ms = (t_sent - arrival) * 1000.0
            w_ctl.add(ctl_ms)
            w_infer.add((t_done - t_inf) * 1000.0)
            if vis_lat is not None:
                w_vis.add(vis_lat * 1000.0)
                w_total.add(vis_lat * 1000.0 + ctl_ms)
            if manual is None:
                last_action = action
            last_poses = (pursuers, evader)
            if now - last_tel > 0.09:
                last_tel = now
                send_state_telemetry(now)
            if ticks % 100 == 0:
                tot = w_total.stats()
                log(f"tick {ticks} {'ARMED' if armed else 'disarmed'}  cmd "
                    + " ".join(f"[{action[i]:+.2f},{action[i + 1]:+.2f}]" for i in range(0, len(action), 2))
                    + (f"  camera->command p50 {tot['p50']:.1f} ms p95 {tot['p95']:.1f} ms" if tot else "")
                    + f"  period p95 {w_period.stats().get('p95', 0):.0f} ms")
    except KeyboardInterrupt:
        log("stopping")
    finally:
        source.close()
