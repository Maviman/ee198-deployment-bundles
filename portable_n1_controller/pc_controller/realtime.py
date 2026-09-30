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
    """Localhost-only UDP command port: {"cmd": "arm"|"disarm"|"status"}."""

    def __init__(self, port: int | None) -> None:
        self.sock = None
        if port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", int(port)))
            self.sock.setblocking(False)

    def poll(self):
        """Yield (cmd, reply_addr) for every pending command."""
        if self.sock is None:
            return
        while True:
            try:
                raw, addr = self.sock.recvfrom(1024)
            except (BlockingIOError, ConnectionResetError, OSError):
                return
            try:
                yield str(json.loads(raw.decode("utf-8")).get("cmd", "")), addr
            except (ValueError, AttributeError):
                continue

    def reply(self, addr, payload: dict) -> None:
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), addr)
        except OSError:
            pass


def run_realtime(loop, link, *, pose_port: int, tick_hz: float = 10.0, stall_s: float = 0.3,
                 max_pose_age_s: float = 0.15, telemetry: str | None = None,
                 control_port: int | None = None, start_disarmed: bool = False,
                 model_name: str = "", log=print) -> None:
    n = loop.policy.num_pursuers
    source = UdpPoseSource(pose_port, expected_pursuers=n)
    tel = Telemetry(telemetry)
    ctl = ControlPort(control_port)
    env = loop.adapter.env
    capture_radius = float(getattr(env, "capture_radius", 0.0))

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
    log(f"listening for pose frames on UDP :{pose_port}; tick {tick_hz:g} Hz; "
        f"{'DISARMED until `arena go`' if not armed else 'ARMED'} (ctrl-C to stop)")

    def send_state_telemetry(now: float) -> None:
        cars = []
        if link is not None:
            for i, (host, port) in enumerate(link.targets):
                st = link.car_stats.get(i, {})
                age = None if st.get("last_ack") is None else round(now - st["last_ack"], 2)
                cars.append({"idx": i, "addr": f"{host}:{port}", "rtt_ms": st.get("rtt_ms"),
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
            "dist": dist, "capture_radius": capture_radius, "cars": cars,
        })

    def estop(reason: str) -> None:
        nonlocal estops
        if link is not None:
            link.send_stop()
        estops += 1

    watch = [source.sock] + ([link.sock] if link is not None else []) + (
        [ctl.sock] if ctl.sock is not None else [])
    try:
        while True:
            now = time.perf_counter()
            for cmd, addr in ctl.poll():
                if cmd == "arm":
                    armed = True
                    log("ARMED by operator")
                elif cmd == "disarm":
                    if armed:
                        log("DISARMED by operator -> E-stop")
                    armed = False
                    estop("operator")
                ctl.reply(addr, {"armed": armed, "state": state, "ticks": ticks})

            # Not driving (waiting for the first pose, or stalled): keep telling
            # every car to hold E-stop at 2 Hz. The cars would sit in failsafe
            # anyway; this keeps their ACKs, and so the dashboard's radio health,
            # alive while you are checking the setup.
            if state != "RUNNING" and link is not None and now - last_keepalive > 0.5:
                link.send_stop()
                last_keepalive = now

            # Wait on poses, ACKs and operator commands together, so each is
            # handled (and each ACK timestamped) the moment it lands.
            timeout = max(0.005, min(stall_s - (now - last_good), 0.05))
            ready, _, _ = select.select(watch, [], [], timeout)
            if link is not None and link.sock in ready:
                link.poll_acks()
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
                    estop("stall")
                    loop.reset()          # stale histories would corrupt velocity estimates
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
            action = loop.tick(pursuer_poses=pursuers, evader_pose=evader)
            t_done = time.perf_counter()
            if link is not None:
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
            last_action, last_poses = action, (pursuers, evader)
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
