"""System identification of the real car, from the overhead camera, for an
Isaac Sim model.

The car is on/off on both axes (V3 button firmware over a DRV8833: FWD/BACK
and LEFT/RIGHT, no proportional control), so every test is a sequence of
full presses and releases, and every result is read from the vision Orin's
pose frames (30 Hz, arena-centred metres, heading from the roof tag).

    run      drive the battery, record, analyse, write the Isaac Sim file
    analyze  re-run the analysis (and plots) on a recorded run directory

    python tools/physics_id.py run --esp ble:hive-car0 --pose-port 9870 --out ~/physics_runs
    python tools/physics_id.py run --esp 127.0.0.1:18888 --pose-port 19870 --arena 1.83 --sim
    python tools/physics_id.py analyze ~/physics_runs/<stamp> --plots

Battery (each test is planned from wherever the car is, repositioning with a
guarded bang-bang "go to" when the room is not there):

    static    car still: pose noise floor, frame rate, perception latency
    straight  FWD (and BACK) from rest until the guard, then release and coast
              -> command latency, accel time constant, top speed, coast-down
                 decel (rolling + drivetrain drag), run-on after release
    brake     FWD to speed, then BACK until stopped
              -> braking decel, longitudinal friction estimate
    circle    FWD + full LEFT (and RIGHT) -> turning radius, yaw rate, speed
              in the turn, lateral acceleration, steady sideslip at the tag,
              steering response time constant
    skid      FWD to speed, then full steer with throttle held, then release
              -> transient yaw / sideslip at speed, lateral grip

SAFETY. The car has no brakes but the BACK button, and it rolls on for
~0.4 s after a release. Every command passes a guard that predicts where the
car will be after the run-on and releases the buttons if that point is
outside the arena (minus the body size and a margin) or inside a keep-out
around the evader tag. A pose older than --max-pose-age also releases
everything: no pose, no drive. Ctrl-C sends an E-stop. The guard's run-on
horizon is re-estimated from the first coast-down.

Run the controller down first: it holds the car's radio link and the pose
port (`svc_stop controller` on the control Orin, or `./arena stop`).
Everything here is numpy-only; plots need matplotlib (system python3).
"""

from __future__ import annotations

import argparse
import csv
import warnings
import json
import math
import os
import platform
import signal
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOOL_VERSION = "1.0"
G = 9.80665

# ----------------------------------------------------------------------------- math helpers


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def unwrap(h: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(h, dtype=float))


def sg_derivative(t: np.ndarray, y: np.ndarray, half: int = 3, order: int = 2) -> np.ndarray:
    """Derivative of y(t) by a local polynomial fit (Savitzky-Golay without scipy).
    Robust to the uneven 30 Hz sampling of UDP frames."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if hi - lo < order + 1:
            lo, hi = max(0, min(lo, n - order - 1)), min(n, max(hi, order + 1))
        tt = t[lo:hi] - t[i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = np.polyfit(tt, y[lo:hi], min(order, hi - lo - 1))
        except (np.linalg.LinAlgError, ValueError):
            continue
        out[i] = np.polyval(np.polyder(c), 0.0)
    return out


def fit_circle(x: np.ndarray, y: np.ndarray):
    """Algebraic (Kasa) least-squares circle: (cx, cy, r, rms_residual) or None."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 6:
        return None
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx * cx + cy * cy - F
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.hypot(x - cx, y - cy) - r
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(res ** 2)))


def fit_first_order(t: np.ndarray, v: np.ndarray, t0: float):
    """v(t) = vmax * (1 - exp(-(t - t0) / tau)) for t >= t0. Grid over tau, closed
    form for vmax. Returns (vmax, tau, rms) or None."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    m = np.isfinite(v) & (t >= t0)
    if m.sum() < 5:
        return None
    tt, vv = t[m] - t0, v[m]
    best = None
    for tau in np.geomspace(0.03, 4.0, 160):
        basis = 1.0 - np.exp(-tt / tau)
        denom = float(np.dot(basis, basis))
        if denom <= 0:
            continue
        vmax = float(np.dot(basis, vv) / denom)
        rms = float(np.sqrt(np.mean((vmax * basis - vv) ** 2)))
        if best is None or rms < best[2]:
            best = (vmax, float(tau), rms)
    return best


def linfit(t: np.ndarray, y: np.ndarray):
    """slope, intercept, rms or None."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(t)
    if m.sum() < 3:
        return None
    c = np.polyfit(t[m], y[m], 1)
    rms = float(np.sqrt(np.mean((np.polyval(c, t[m]) - y[m]) ** 2)))
    return float(c[0]), float(c[1]), rms


def nanmedian(a) -> float | None:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else float(np.median(a))


def rnd(x, nd=4):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: rnd(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ----------------------------------------------------------------------------- pose intake


class PoseTap:
    """Listens on the controller's pose port and keeps every frame, with the
    label of the test phase that was running when it arrived.

    Time base: frames carry the vision host's capture time ``t`` and its own
    capture->send latency ``lat``. The two Orins are not clock-synced, so the
    analysis uses ``t_local = arrival - lat`` (this host's perf_counter), the
    same clock the command log uses. Within a run, differences of ``t`` are
    still the best measure of inter-frame time (no network jitter)."""

    def __init__(self, port: int, pursuer_slot: int = 0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.slot = pursuer_slot
        self.frames: list[dict] = []
        self.recent: deque = deque(maxlen=12)
        self.lock = threading.Lock()
        self.label = ("idle", 0)          # (phase, segment id), set by the driver
        self.evader = None
        self.bad = 0
        self.n_received = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pose-tap", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            arrival = time.perf_counter()
            try:
                fr = json.loads(raw.decode("utf-8"))
                p = fr["pursuers"][self.slot]
                ev = fr.get("evader")
                lat = float(fr.get("lat") or 0.0)
                rec = {"t_vision": float(fr["t"]), "arrival": arrival, "lat": lat,
                       "t_local": arrival - lat, "seq": fr.get("seq"),
                       "x": float(p["x"]), "y": float(p["y"]), "h": float(p["heading"])}
            except (ValueError, KeyError, IndexError, TypeError):
                self.bad += 1
                continue
            with self.lock:
                self.n_received += 1
                rec["phase"], rec["seg"] = self.label
                self.frames.append(rec)
                self.recent.append(rec)
                if ev is not None:
                    try:
                        self.evader = (float(ev["x"]), float(ev["y"]))
                    except (KeyError, TypeError, ValueError):
                        pass

    def set_label(self, phase: str, seg: int) -> None:
        with self.lock:
            self.label = (phase, seg)

    def state(self):
        """Latest pose with a short-window velocity estimate, or None if no frame.
        Returns dict(x, y, h, vx, vy, speed, v_long, yaw_rate, age, t_local, n)."""
        with self.lock:
            rec = list(self.recent)
        if not rec:
            return None
        last = rec[-1]
        now = time.perf_counter()
        st = {"x": last["x"], "y": last["y"], "h": last["h"], "age": now - last["arrival"],
              "t_local": last["t_local"], "vx": 0.0, "vy": 0.0, "speed": 0.0, "v_long": 0.0,
              "yaw_rate": 0.0, "n": len(rec), "evader": self.evader}
        win = [r for r in rec if last["t_vision"] - r["t_vision"] <= 0.2]
        if len(win) >= 3:
            t = np.array([r["t_vision"] for r in win])
            if t[-1] - t[0] > 0.04:
                cx = np.polyfit(t - t[-1], [r["x"] for r in win], 1)[0]
                cy = np.polyfit(t - t[-1], [r["y"] for r in win], 1)[0]
                ch = np.polyfit(t - t[-1], unwrap([r["h"] for r in win]), 1)[0]
                st.update(vx=float(cx), vy=float(cy), speed=float(math.hypot(cx, cy)),
                          yaw_rate=float(ch))
                st["v_long"] = float(cx * math.cos(last["h"]) + cy * math.sin(last["h"]))
        return st

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close()


# ----------------------------------------------------------------------------- the guard


class Guard:
    """Decides whether a command may go out, from where the car will be after
    its run-on. ``half`` is the arena half-size (m). ``body`` is how far the
    body reaches from the tag; ``margin`` is extra clearance to the tape."""

    def __init__(self, half: float, *, body: float = 0.12, margin: float = 0.12,
                 horizon_s: float = 0.6, evader_keepout: float = 0.30,
                 max_pose_age: float = 0.25) -> None:
        self.half = half
        self.body = body
        self.margin = margin
        self.horizon_s = horizon_s
        self.evader_keepout = evader_keepout
        self.max_pose_age = max_pose_age
        self.trips = 0
        self.last_reason = ""

    @property
    def limit(self) -> float:
        return self.half - self.body - self.margin

    def inside(self, x: float, y: float, extra: float = 0.0) -> bool:
        lim = self.limit - extra
        return abs(x) <= lim and abs(y) <= lim

    def room(self, st, direction: float) -> float:
        """Free distance from the tag along +heading (direction=+1) or -heading
        (-1) before the body reaches the margin, less the evader keep-out."""
        d = np.array([math.cos(st["h"]), math.sin(st["h"])]) * direction
        p = np.array([st["x"], st["y"]])
        lim = self.limit
        ts = []
        for i in range(2):
            if abs(d[i]) > 1e-6:
                ts.append(((lim if d[i] > 0 else -lim) - p[i]) / d[i])
        room = max(0.0, min(ts)) if ts else 0.0
        ev = st.get("evader")
        if ev is not None and room > 0:
            rel = np.array(ev) - p
            along = float(np.dot(rel, d))
            perp = abs(float(rel[0] * d[1] - rel[1] * d[0]))
            if along > 0 and perp < self.evader_keepout + self.body:
                room = min(room, max(0.0, along - self.evader_keepout - self.body))
        return room

    def check(self, st, thr: float) -> str | None:
        """Reason to release the buttons, or None if ``thr`` may be sent."""
        if st is None:
            return "no pose"
        if st["age"] > self.max_pose_age:
            return f"pose {st['age']*1000:.0f} ms old"
        if st["speed"] > 3.5:
            return f"implausible speed {st['speed']:.1f} m/s"
        if thr == 0.0:
            return None
        p0 = (st["x"], st["y"])
        if not self.inside(p0[0], p0[1]):
            # Past the limit: only a command that points back in may be sent, and
            # never while the car is still moving outward (whatever the button
            # claims to do, releasing is the one thing guaranteed to help).
            out_vel = any(abs(p0[i]) > self.limit and (st["vx"], st["vy"])[i] * p0[i] > 0.05 for i in range(2))
            if out_vel:
                return "outside and moving out"
        # Braking (throttle against the motion) is allowed while inside.
        if st["v_long"] * thr < -0.05 and self.inside(p0[0], p0[1]):
            return None
        d = np.array([math.cos(st["h"]), math.sin(st["h"])]) * (1.0 if thr > 0 else -1.0)
        vel = np.array([st["vx"], st["vy"]])
        # Prediction: coast on for the horizon along the current ARC (yaw rate
        # held, capped at a quarter turn), plus a little reach in the commanded
        # direction. A straight extrapolation would forbid any circle that
        # passes near the tape even when the circle itself fits.
        p = np.array([st["x"], st["y"]])
        w = st.get("yaw_rate", 0.0)
        if abs(w) > 0.3 and st["speed"] > 0.15:
            phi = max(-math.pi / 2, min(math.pi / 2, w * self.horizon_s))
            vdir = math.atan2(vel[1], vel[0])
            chord = st["speed"] * self.horizon_s * (math.sin(phi / 2) / (phi / 2) if abs(phi) > 1e-3 else 1.0)
            pred = p + chord * np.array([math.cos(vdir + phi / 2), math.sin(vdir + phi / 2)]) + d * 0.05
        else:
            pred = p + vel * self.horizon_s + d * 0.05
        if not self.inside(pred[0], pred[1]):
            # Already past the limit (the car was placed there, or rolled on):
            # a command that reduces the overshoot on every violated axis is
            # the way back in, so let it through; anything else is held.
            lim = self.limit
            viol = [i for i in range(2) if abs(p[i]) > lim]
            inward = bool(viol) and all(d[i] * p[i] < 0 for i in viol) and \
                all(abs(pred[i]) <= abs(p[i]) + 1e-6 or abs(pred[i]) <= lim for i in range(2))
            if not inward:
                return f"predicted ({pred[0]:+.2f},{pred[1]:+.2f}) outside"
        if self.room(st, 1.0 if thr > 0 else -1.0) < 0.05:
            return "no room ahead"
        ev = st.get("evader")
        if ev is not None and math.hypot(pred[0] - ev[0], pred[1] - ev[1]) < self.evader_keepout:
            return "evader keep-out"
        return None


# ----------------------------------------------------------------------------- axis map

WIRE_BUTTONS = ("FWD", "BACK", "LEFT", "RIGHT")          # the firmware's order; GPIO 1, 2, 42, 41 by default
DEFAULT_GPIO = {"FWD": 1, "BACK": 2, "LEFT": 42, "RIGHT": 41}


def parse_axis_map(spec: str) -> dict:
    """'fwd=RIGHT,back=LEFT,left=BACK,right=FWD' -> {desired button: wire button}.
    'identity' maps every button to itself. The wire button is what the
    firmware calls the pin it will drive; the desired button is what the car
    should physically do."""
    if spec.strip().lower() in ("identity", "", "none"):
        return {b: b for b in WIRE_BUTTONS}
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k, v = k.strip().upper(), v.strip().upper()
        if k not in WIRE_BUTTONS or v not in WIRE_BUTTONS:
            raise ValueError(f"bad axis map entry {part!r}")
        out[k] = v
    if sorted(out) != sorted(WIRE_BUTTONS) or sorted(out.values()) != sorted(WIRE_BUTTONS):
        raise ValueError("axis map must assign all four of FWD, BACK, LEFT, RIGHT exactly once")
    return out


def apply_axis_map(amap: dict, thr: float, steer: float) -> tuple[float, float]:
    """Desired [throttle, steer] -> the [throttle, steer] to put on the wire so
    the firmware presses the pins that physically do what we want."""
    want = []
    if thr > 0:
        want.append("FWD")
    elif thr < 0:
        want.append("BACK")
    if steer > 0:
        want.append("LEFT")
    elif steer < 0:
        want.append("RIGHT")
    w_thr = w_steer = 0.0
    for b in want:
        wire = amap[b]
        if wire == "FWD":
            w_thr = 1.0
        elif wire == "BACK":
            w_thr = -1.0
        elif wire == "LEFT":
            w_steer = 1.0
        else:
            w_steer = -1.0
    return w_thr, w_steer


def axis_map_as_pins(amap: dict) -> str:
    """The firmware console command that would make the identity map true."""
    gpio = {b: DEFAULT_GPIO[amap[b]] for b in WIRE_BUTTONS}
    return "pins " + " ".join(str(gpio[b]) for b in WIRE_BUTTONS)


# ----------------------------------------------------------------------------- the driver


class Driver:
    """Sends one [throttle, steer] per tick over the car link, through the guard,
    logging every command. Also owns the pose tap and the segment labels."""

    RATE_HZ = 10.0

    def __init__(self, link, tap: PoseTap, guard: Guard, log, *, sim: bool = False) -> None:
        self.link = link
        self.tap = tap
        self.guard = guard
        self.log = log
        self.sim = sim
        self.commands: list[dict] = []
        self.segments: list[dict] = []
        self.seg_id = 0
        self.t0 = time.perf_counter()
        self.acks = 0
        self.lost = 0
        self.rtts: list[float] = []
        self.axis_map = {b: b for b in WIRE_BUTTONS}   # desired button -> wire button
        self.wrong_dir_ticks = 0   # consecutive ticks the car accelerated against the throttle
        self.abort_reason = None
        self.radius = 0.6          # full-lock turning radius used for planning (updated by the circle test)
        self.move_m = 0.5          # distance one planning move is expected to cover (press + run-on)
        self.move_s = 0.7          # how long a planning move holds the buttons

    # -- segments -----------------------------------------------------------
    def begin(self, test: str, variant: str, **info) -> int:
        self.seg_id += 1
        self.segments.append({"seg": self.seg_id, "test": test, "variant": variant,
                              "t_start": time.perf_counter(), "phases": [], **info})
        self.log(f"== seg {self.seg_id}: {test} {variant}")
        return self.seg_id

    def end(self, **info) -> None:
        s = self.segments[-1]
        s["t_end"] = time.perf_counter()
        s.update(info)
        self.tap.set_label("idle", 0)

    def phase(self, name: str) -> None:
        s = self.segments[-1]
        s["phases"].append({"name": name, "t": time.perf_counter()})
        self.tap.set_label(name, s["seg"])

    # -- one tick -----------------------------------------------------------
    def tick(self, thr: float, steer: float, *, phase: str, force: bool = False) -> dict:
        """Send one command (after the guard) and sleep out the tick period.
        Returns dict(state, sent_thr, sent_steer, blocked)."""
        tick_start = time.perf_counter()
        st = self.tap.state()
        reason = None if force else self.guard.check(st, thr)
        s_thr, s_steer = (0.0, 0.0) if reason else (thr, steer)
        if reason and (thr != 0.0 or steer != 0.0):
            self.guard.trips += 1
            if reason != self.guard.last_reason:
                self.log(f"   guard: {reason} -> released")
            self.guard.last_reason = reason
        elif not reason:
            self.guard.last_reason = ""
        # Does the car move the way the throttle says? Moving against it while
        # not already carrying that motion is a wrong pin map: stop everything.
        if st is not None and st["age"] < 0.3 and s_thr != 0.0 and st["v_long"] * s_thr < -0.15 \
                and not (st["v_long"] * s_thr < 0 and phase in ("brake",)):
            self.wrong_dir_ticks += 1
            if self.wrong_dir_ticks >= 3 and self.abort_reason is None:
                self.abort_reason = (f"car moves against its throttle (v_long {st['v_long']:+.2f} m/s on thr {s_thr:+.0f}): "
                                     "wrong axis map / pins")
                self.log("   ABORT: " + self.abort_reason)
                s_thr, s_steer = 0.0, 0.0
        else:
            self.wrong_dir_ticks = 0
        if self.abort_reason is not None:
            s_thr, s_steer = 0.0, 0.0
        w_thr, w_steer = apply_axis_map(self.axis_map, s_thr, s_steer)
        pkt = self.link.send_commands(np.array([w_thr, w_steer]))
        ack = self.link.wait_ack(pkt["seq"], timeout_s=0.08)
        rtt = (time.perf_counter() - tick_start) * 1000.0
        if ack is None:
            self.lost += 1
        else:
            self.acks += 1
            self.rtts.append(rtt)
        self.commands.append({"t": tick_start, "seq": pkt["seq"], "thr": s_thr, "steer": s_steer,
                              "wire_thr": w_thr, "wire_steer": w_steer,
                              "want_thr": thr, "want_steer": steer, "blocked": bool(reason),
                              "phase": phase, "seg": self.seg_id, "acked": ack is not None,
                              "applied": None if ack is None else ack.get("applied")})
        time.sleep(max(0.0, 1.0 / self.RATE_HZ - (time.perf_counter() - tick_start)))
        return {"state": st, "thr": s_thr, "steer": s_steer, "blocked": bool(reason)}

    def neutral(self, seconds: float, phase: str = "neutral") -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.tick(0.0, 0.0, phase=phase)

    def wait_pose(self, seconds: float = 10.0):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            st = self.tap.state()
            if st is not None and st["age"] < self.guard.max_pose_age:
                return st
            self.tick(0.0, 0.0, phase="wait-pose")
        return None

    def wait_still(self, timeout: float = 4.0, phase: str = "settle") -> None:
        """Neutral until the car has been below 0.04 m/s for 0.4 s."""
        end = time.perf_counter() + timeout
        still_since = None
        while time.perf_counter() < end:
            r = self.tick(0.0, 0.0, phase=phase)
            st = r["state"]
            if st is not None and st["age"] < 0.3 and st["speed"] < 0.04:
                still_since = still_since or time.perf_counter()
                if time.perf_counter() - still_since > 0.4:
                    return
            else:
                still_since = None

    # -- maneuvers ------------------------------------------------------------
    # -- planning (one-step lookahead with the car's own kinematics) ---------
    def predict(self, st, gear: float, steer: float, dist: float) -> dict:
        """Pose after driving ``dist`` metres in ``gear`` (+1/-1) with ``steer``
        (-1/0/+1), as a kinematic bicycle whose full-lock radius is self.radius."""
        x, y, h = st["x"], st["y"], st["h"]
        if steer == 0.0:
            nx, ny, nh = x + gear * dist * math.cos(h), y + gear * dist * math.sin(h), h
        else:
            R = self.radius
            cx, cy = x - steer * R * math.sin(h), y + steer * R * math.cos(h)
            phi = gear * steer * dist / R
            dx, dy = x - cx, y - cy
            nx = cx + dx * math.cos(phi) - dy * math.sin(phi)
            ny = cy + dx * math.sin(phi) + dy * math.cos(phi)
            nh = h + phi
        out = dict(st)
        out.update(x=nx, y=ny, h=wrap(nh), vx=0.0, vy=0.0, speed=0.0, v_long=0.0)
        return out

    def plan_moves(self, objective, tol: float, *, max_moves: int = 8, label: str = "plan") -> bool:
        """Greedy: try the six short moves (gear x steer), execute the one whose
        predicted end pose scores best on ``objective`` (lower is better), stop,
        look again. Each move is guarded like everything else."""
        self.phase(label)
        refused: set = set()          # (gear, steer) the guard would not let move from here
        last_pose = None
        for move in range(max_moves):
            st = self.wait_pose()
            if st is None:
                return False
            f = objective(st)
            if f <= tol:
                return True
            if last_pose is not None and math.hypot(st["x"] - last_pose[0], st["y"] - last_pose[1]) > 0.08:
                refused.clear()       # we moved: the refusals were about the old pose
            last_pose = (st["x"], st["y"])
            best = None
            for gear in (1.0, -1.0):
                for steer in (0.0, 1.0, -1.0):
                    if (gear, steer) in refused:
                        continue
                    end = self.predict(st, gear, steer, self.move_m)
                    mid = self.predict(st, gear, steer, self.move_m * 0.5)
                    pen = objective(end)
                    for q in (mid, end):
                        if not self.guard.inside(q["x"], q["y"]):
                            pen += 5.0 + max(abs(q["x"]), abs(q["y"])) - self.guard.limit
                        ev = st.get("evader")
                        if ev is not None and math.hypot(q["x"] - ev[0], q["y"] - ev[1]) < self.guard.evader_keepout + self.guard.body:
                            pen += 5.0
                    if best is None or pen < best[0]:
                        best = (pen, gear, steer)
            if best is None:
                self.log(f"   {label}: every move refused from here; giving up")
                return False
            pen, gear, steer = best
            self.log(f"   {label}: score {f:.2f} -> move {'FWD' if gear > 0 else 'BACK'} "
                     f"{'LEFT' if steer > 0 else 'RIGHT' if steer < 0 else 'straight'} (predicted {pen:.2f})")
            blocked = sent = 0
            t_end = time.perf_counter() + self.move_s
            while time.perf_counter() < t_end:
                r = self.tick(gear, steer, phase=label)
                blocked = blocked + 1 if r["blocked"] else 0
                sent += 0 if r["blocked"] else 1
                if blocked >= 3:
                    break
            if sent == 0:
                refused.add((gear, steer))
            self.wait_still(2.5, phase=label)
            # learn how far one move really carries the car (press + run-on)
            st2 = self.tap.state()
            if sent >= 3 and st2 is not None and st2["age"] < 0.3:
                moved = math.hypot(st2["x"] - st["x"], st2["y"] - st["y"])
                if 0.05 < moved < 2.0:
                    self.move_m = 0.6 * self.move_m + 0.4 * moved
        st = self.wait_pose()
        return st is not None and objective(st) <= tol

    def make_room(self, need: float, prefer: float = 1.0, *, allow_opposite: bool = True) -> float | None:
        """Get ``need`` metres of straight room ahead (prefer=+1) or behind (-1).
        Returns the direction that has it, or None."""
        st = self.wait_pose()
        if st is None:
            return None
        for d in ((prefer, -prefer) if allow_opposite else (prefer,)):
            if self.guard.room(st, d) >= need:
                return d
        self.log(f"   make_room: need {need:.2f} m {'ahead' if prefer > 0 else 'behind'}, have "
                 f"{self.guard.room(st, prefer):.2f}; repositioning")
        self.begin("reposition", "make-room")
        try:
            self.plan_moves(lambda s: max(0.0, need - self.guard.room(s, prefer)), 0.0, label="make-room")
        finally:
            self.end()
        st = self.wait_pose()
        if st is None:
            return None
        rooms = {d: self.guard.room(st, d) for d in ((prefer, -prefer) if allow_opposite else (prefer,))}
        d_best = max(rooms, key=rooms.get)
        if rooms[d_best] >= need:
            return d_best
        if rooms[d_best] >= 0.6 * need:
            self.log(f"   make_room: settling for {rooms[d_best]:.2f} m of {need:.2f}")
            return d_best
        return None

    def test_axes(self, press_s: float = 0.5) -> dict | None:
        """Which wire button does what. From a settled car near the centre, press
        each of FWD, BACK, LEFT, RIGHT alone (as the firmware names them) and
        read the tag's displacement along its heading; then press the one that
        drove forward together with each button of the other pair and read the
        heading change. Returns the inferred axis map (desired -> wire), or None.
        The axis map in force is bypassed here: wire buttons are pressed raw."""
        self.begin("axes", "discovery")
        saved = self.axis_map
        self.axis_map = {b: b for b in WIRE_BUTTONS}
        results = {}
        try:
            st = self.wait_pose()
            if st is None:
                return None
            if max(abs(st["x"]), abs(st["y"])) > 0.65:
                self.log(f"   axes: car at ({st['x']:+.2f},{st['y']:+.2f}) is too close to the tape; "
                         "place it within 0.6 m of the centre")
                return None
            wire_cmd = {"FWD": (1.0, 0.0), "BACK": (-1.0, 0.0), "LEFT": (0.0, 1.0), "RIGHT": (0.0, -1.0)}

            def press(label, thr, steer):
                self.wait_still(3.0, phase="axes")
                s0 = self.tap.state()
                self.phase(f"axes-{label}")
                t_end = time.perf_counter() + press_s
                while time.perf_counter() < t_end:
                    st_now = self.tap.state()
                    # raw guard: never past 0.85 m from the centre, never faster than 1.8 m/s
                    if st_now is None or st_now["age"] > 0.3 or max(abs(st_now["x"]), abs(st_now["y"])) > 0.85 \
                            or st_now["speed"] > 1.8:
                        self.tick(0.0, 0.0, phase=f"axes-{label}", force=True)
                        break
                    self.tick(thr, steer, phase=f"axes-{label}", force=True)
                self.wait_still(3.0, phase="axes")
                s1 = self.tap.state()
                if s0 is None or s1 is None:
                    return None
                h0 = s0["h"]
                dx, dy = s1["x"] - s0["x"], s1["y"] - s0["y"]
                along = dx * math.cos(h0) + dy * math.sin(h0)
                turned = math.degrees(wrap(s1["h"] - h0))
                r = {"along_m": along, "dist_m": math.hypot(dx, dy), "turned_deg": turned}
                self.log(f"   axes: {label:10s} -> moved {r['dist_m']:.2f} m ({along:+.2f} along heading), "
                         f"turned {turned:+.0f} deg")
                return r

            for b in WIRE_BUTTONS:
                results[b] = press(b, *wire_cmd[b])
            moving = {b: r["along_m"] for b, r in results.items() if r and r["dist_m"] > 0.08}
            if not moving:
                self.log("   axes: nothing moved the car on any single button")
                return None
            drive_fwd = max(moving, key=moving.get)
            if moving[drive_fwd] <= 0.05:
                # every mover went backwards along the tag heading: the tag is on backwards
                self.log("   axes: every mover drove AGAINST the tag heading: tag mounted backwards?")
            pair = ("FWD", "BACK") if drive_fwd in ("FWD", "BACK") else ("LEFT", "RIGHT")
            drive_back = pair[0] if pair[1] == drive_fwd else pair[1]
            other = ("LEFT", "RIGHT") if pair == ("FWD", "BACK") else ("FWD", "BACK")
            turns = {}
            for b in other:
                thr = wire_cmd[drive_fwd][0] + wire_cmd[b][0]
                steer = wire_cmd[drive_fwd][1] + wire_cmd[b][1]
                r = press(f"{drive_fwd}+{b}", thr, steer)
                turns[b] = r["turned_deg"] if r else 0.0
            left = max(turns, key=turns.get)          # CCW while driving forward = left
            right = other[0] if other[1] == left else other[1]
            amap = {"FWD": drive_fwd, "BACK": drive_back, "LEFT": left, "RIGHT": right}
            self.log(f"   axes: inferred desired->wire map {amap}; firmware equivalent: `{axis_map_as_pins(amap)}`"
                     + ("  (identity: pins are right)" if all(k == v for k, v in amap.items()) else ""))
            self.segments[-1]["axis_map"] = amap
            self.segments[-1]["raw"] = {"single": results, "turns": turns}
            return amap
        finally:
            self.axis_map = saved
            self.end()

    def test_static(self, seconds: float) -> None:
        self.begin("static", "still")
        self.phase("static")
        self.neutral(seconds, phase="static")
        self.end()

    def test_straight(self, direction: float, *, need: float, max_drive_s: float = 3.5,
                      coast_s: float = 3.0, brake: bool = False) -> bool:
        """direction +1: FWD, -1: BACK. With brake=True the run ends with the
        opposite button held until the car stops (the brake test)."""
        name = "brake" if brake else "straight"
        variant = "fwd" if direction > 0 else "rev"
        d = self.make_room(need, prefer=direction)
        if d is None:
            self.log(f"   {name} {variant}: could not find {need:.2f} m of room; skipped")
            return 0.0
        direction = d
        variant = "fwd" if direction > 0 else "rev"
        self.begin(name, variant)
        self.wait_still(3.0)
        st0 = self.tap.state()
        self.phase("drive")
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < max_drive_s:
            r = self.tick(direction, 0.0, phase="drive")
            if r["blocked"]:
                break
        if brake:
            self.phase("brake")
            t_b = time.perf_counter()
            while time.perf_counter() - t_b < 2.0:
                r = self.tick(-direction, 0.0, phase="brake", force=True)
                st = r["state"]
                if st is not None and st["v_long"] * direction < 0.08:
                    break
        self.phase("coast")
        end = time.perf_counter() + coast_s
        still_since = None
        while time.perf_counter() < end:
            r = self.tick(0.0, 0.0, phase="coast")
            st = r["state"]
            if st is not None and st["speed"] < 0.04:
                still_since = still_since or time.perf_counter()
                if time.perf_counter() - still_since > 0.6:
                    break
            else:
                still_since = None
        self.end(start_pose=None if st0 is None else [st0["x"], st0["y"], st0["h"]])
        return direction

    def circle_fit_score(self, st, side: float, radius: float) -> float:
        """How far (m) the predicted circle pokes outside the limits (0 = fits)."""
        n = (-math.sin(st["h"]), math.cos(st["h"]))
        cx, cy = st["x"] + side * radius * n[0], st["y"] + side * radius * n[1]
        over = max(abs(cx) + radius - self.guard.limit, abs(cy) + radius - self.guard.limit, 0.0)
        ev = st.get("evader")
        if ev is not None:
            dmin = abs(math.hypot(cx - ev[0], cy - ev[1]) - radius)
            over = max(over, self.guard.evader_keepout + self.guard.body - dmin)
        return over

    def test_circle(self, side: float, *, radius_guess: float, seconds: float = 6.0) -> bool:
        variant = "left" if side > 0 else "right"
        self.radius = radius_guess
        st = self.wait_pose()
        if st is None:
            return False
        if self.circle_fit_score(st, side, radius_guess) > 0.08:
            self.begin("reposition", f"circle-{variant}")
            self.plan_moves(lambda s: self.circle_fit_score(s, side, radius_guess), 0.08, label="circle-setup")
            self.end()
        self.begin("circle", variant)
        self.wait_still(3.0)
        self.phase("drive")
        t_start = time.perf_counter()
        blocked = 0
        while time.perf_counter() - t_start < seconds:
            r = self.tick(1.0, side, phase="drive")
            blocked = blocked + 1 if r["blocked"] else 0
            if blocked >= 4:
                break
        self.phase("coast")
        self.wait_still(3.0, phase="coast")
        self.end()
        return True

    def test_skid(self, side: float, *, need: float, steer_s: float = 1.2) -> bool:
        variant = "left" if side > 0 else "right"
        d = self.make_room(need, prefer=1.0, allow_opposite=False)
        if d is None:
            self.log(f"   skid {variant}: no forward room; skipped")
            return False
        self.begin("skid", variant)
        self.wait_still(3.0)
        self.phase("drive")
        t_start = time.perf_counter()
        # Straight until top speed (or 60 % of the room is used), then the step.
        st0 = self.tap.state()
        while time.perf_counter() - t_start < 3.0:
            r = self.tick(d, 0.0, phase="drive")
            st = r["state"]
            if r["blocked"]:
                break
            if st is not None and st0 is not None:
                gone = math.hypot(st["x"] - st0["x"], st["y"] - st0["y"])
                if gone > 0.55 * need and st["speed"] > 0.5:
                    break
        self.phase("steer")
        t_s = time.perf_counter()
        blocked = 0
        while time.perf_counter() - t_s < steer_s:
            r = self.tick(d, side, phase="steer")
            blocked = blocked + 1 if r["blocked"] else 0
            if blocked >= 3:
                break
        self.phase("coast")
        self.wait_still(3.0, phase="coast")
        self.end()
        return True


# ----------------------------------------------------------------------------- recording


def save_run(out: Path, driver: Driver, tap: PoseTap, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trajectory.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_local", "t_vision", "arrival", "lat", "seq", "x", "y", "heading", "phase", "seg"])
        for r in tap.frames:
            w.writerow([f"{r['t_local']:.4f}", f"{r['t_vision']:.4f}", f"{r['arrival']:.4f}",
                        f"{r['lat']:.4f}", r["seq"], f"{r['x']:.4f}", f"{r['y']:.4f}",
                        f"{r['h']:.4f}", r["phase"], r["seg"]])
    with (out / "commands.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "seq", "thr", "steer", "want_thr", "want_steer", "blocked", "phase", "seg", "acked", "applied",
                    "wire_thr", "wire_steer"])
        for c in driver.commands:
            w.writerow([f"{c['t']:.4f}", c["seq"], c["thr"], c["steer"], c["want_thr"], c["want_steer"],
                        int(c["blocked"]), c["phase"], c["seg"], int(c["acked"]),
                        "" if c.get("applied") is None else int(bool(c["applied"])),
                        c.get("wire_thr", c["thr"]), c.get("wire_steer", c["steer"])])
    meta = dict(meta)
    meta["segments"] = driver.segments
    meta["link"] = {"acks": driver.acks, "lost": driver.lost,
                    "rtt_ms_median": nanmedian(driver.rtts) if driver.rtts else None,
                    "car_stats": {str(k): v for k, v in getattr(driver.link, "car_stats", {}).items()}}
    meta["guard_trips"] = driver.guard.trips
    meta["abort_reason"] = driver.abort_reason
    meta["axis_map"] = driver.axis_map
    meta["frames"] = len(tap.frames)
    meta["bad_frames"] = tap.bad
    (out / "run_meta.json").write_text(json.dumps(rnd(meta, 5), indent=2))


def load_run(d: Path):
    traj = []
    with (d / "trajectory.csv").open() as fh:
        for row in csv.DictReader(fh):
            traj.append({k: (float(v) if k in ("t_local", "t_vision", "arrival", "lat", "x", "y", "heading")
                             else v) for k, v in row.items()})
            traj[-1]["seg"] = int(traj[-1]["seg"])
    cmds = []
    with (d / "commands.csv").open() as fh:
        for row in csv.DictReader(fh):
            cmds.append({"t": float(row["t"]), "seq": int(row["seq"]), "thr": float(row["thr"]),
                         "steer": float(row["steer"]), "blocked": row["blocked"] == "1",
                         "phase": row["phase"], "seg": int(row["seg"]), "acked": row["acked"] == "1"})
    meta = json.loads((d / "run_meta.json").read_text())
    return traj, cmds, meta


# ----------------------------------------------------------------------------- analysis


class Track:
    """One segment's frames as arrays with derived kinematics."""

    def __init__(self, frames: list[dict]) -> None:
        frames = sorted(frames, key=lambda r: r["t_vision"])
        self.n = len(frames)
        self.t = np.array([r["t_vision"] for r in frames])          # even inter-frame time
        self.t_local = np.array([r["t_local"] for r in frames])      # the command clock
        self.x = np.array([r["x"] for r in frames])
        self.y = np.array([r["y"] for r in frames])
        self.h = unwrap([r.get("heading", r.get("h")) for r in frames]) if self.n else np.array([])
        self.phase = np.array([r["phase"] for r in frames])
        if self.n >= 5:
            self.vx = sg_derivative(self.t, self.x)
            self.vy = sg_derivative(self.t, self.y)
            self.speed = np.hypot(self.vx, self.vy)
            self.yaw_rate = sg_derivative(self.t, self.h)
            self.v_long = self.vx * np.cos(self.h) + self.vy * np.sin(self.h)
            self.v_lat = -self.vx * np.sin(self.h) + self.vy * np.cos(self.h)
            # sideslip at the tag: velocity direction relative to heading (signed, CCW+)
            sgn = np.where(self.v_long < 0, -1.0, 1.0)      # sideslip relative to the way the car travels
            self.beta = np.where(self.speed > 0.15, np.arctan2(self.v_lat * sgn, self.v_long * sgn), np.nan)
            self.accel = sg_derivative(self.t, self.v_long)
        else:
            z = np.full(self.n, np.nan)
            self.vx = self.vy = self.speed = self.yaw_rate = self.v_long = self.v_lat = self.beta = self.accel = z

    def mask(self, phase: str) -> np.ndarray:
        return self.phase == phase

    TEST_PHASES = ("static", "drive", "brake", "coast", "steer")

    def gaps(self) -> dict:
        m = np.isin(self.phase, self.TEST_PHASES) if self.n else np.zeros(0, bool)
        t = self.t[m]
        if len(t) < 2:
            return {"frames": int(len(t))}
        dt = np.diff(t)
        return {"frames": int(len(t)), "rate_hz": float((len(t) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else None,
                "max_gap_s": float(dt.max()), "gaps_over_0p2s": int((dt > 0.2).sum())}


def first_command_time(cmds: list[dict], seg: int, phase: str, pred) -> float | None:
    for c in cmds:
        if c["seg"] == seg and c["phase"] == phase and not c["blocked"] and pred(c):
            return c["t"]
    return None


def analyze_static(tr: Track) -> dict:
    m = tr.mask("static")
    if m.sum() < 10:
        return {"ok": False, "reason": "too few frames", **tr.gaps()}
    x, y, h = tr.x[m], tr.y[m], tr.h[m]
    return {"ok": True, **tr.gaps(),
            "pose_std_m": [float(np.std(x)), float(np.std(y))],
            "pose_p2p_m": [float(np.ptp(x)), float(np.ptp(y))],
            "heading_std_deg": float(math.degrees(np.std(h))),
            "speed_noise_rms_mps": float(np.sqrt(np.nanmean(tr.speed[m] ** 2))),
            "yaw_rate_noise_rms_dps": float(math.degrees(np.sqrt(np.nanmean(tr.yaw_rate[m] ** 2)))),
            "mean_pose": [float(x.mean()), float(y.mean()), float(wrap(h.mean()))]}


def analyze_straight(tr: Track, cmds: list[dict], seg: int, direction: float, brake: bool) -> dict:
    out: dict = {"direction": "fwd" if direction > 0 else "rev", **tr.gaps()}
    drive = tr.mask("drive")
    if drive.sum() < 5 or tr.n < 10:
        out.update(ok=False, reason="too few frames in drive phase")
        return out
    t_cmd = first_command_time(cmds, seg, "drive", lambda c: c["thr"] * direction > 0)
    t_rel = first_command_time(cmds, seg, "brake" if brake else "coast", lambda c: True)
    s = tr.v_long * direction                           # speed in the commanded direction
    sign_ok = np.nanmean(s[drive]) > 0
    # motion onset: first frame after the command where s exceeds 0.08 m/s and stays
    onset_i = None
    if t_cmd is not None:
        idx = np.where((tr.t_local > t_cmd) & (s > 0.08))[0]
        for i in idx:
            if i + 2 < tr.n and np.all(s[i:i + 3] > 0.08):
                onset_i = int(i)
                break
    out["drive_cmd_t"] = t_cmd
    if onset_i is None:
        out.update(ok=False, reason="no motion detected after the command", heading_matches_motion=bool(sign_ok))
        return out
    t_on = float(tr.t_local[onset_i])
    out["command_to_motion_latency_s"] = t_on - t_cmd
    # the drive phase after onset
    drive_idx = np.where(drive & (tr.t_local >= t_on))[0]
    tt = tr.t[drive_idx]
    ss = s[drive_idx]
    if len(tt) >= 5:
        t0 = tr.t[onset_i] - (t_on - t_cmd) * 0.0
        fit = fit_first_order(tt, ss, float(tr.t[onset_i]) - 0.03)
        if fit:
            vmax, tau, rms = fit
            out["accel_fit"] = {"v_inf_mps": vmax, "tau_s": tau, "rms_mps": rms,
                                "a0_mps2": vmax / tau if tau > 0 else None}
        # plateau: last 0.3 s of the drive if the slope is small
        tail = drive_idx[tr.t[drive_idx] > tr.t[drive_idx][-1] - 0.3]
        if len(tail) >= 4:
            lf = linfit(tr.t[tail], s[tail])
            out["plateau"] = {"speed_mps": float(np.nanmean(s[tail])), "slope_mps2": lf[0] if lf else None,
                              "reached": bool(lf and abs(lf[0]) < 0.6)}
        out["peak_speed_mps"] = float(np.nanmax(ss))
        out["drive_distance_m"] = float(math.hypot(tr.x[drive_idx[-1]] - tr.x[drive_idx[0]],
                                                   tr.y[drive_idx[-1]] - tr.y[drive_idx[0]]))
        out["drive_duration_s"] = float(tt[-1] - tt[0])
        # heading drift while driving straight
        out["heading_drift_deg_per_m"] = (float(math.degrees(tr.h[drive_idx[-1]] - tr.h[drive_idx[0]])) / out["drive_distance_m"]
                                          if out["drive_distance_m"] > 0.2 else None)
        # early acceleration: slope over the first 0.4 s after onset
        early = drive_idx[tr.t[drive_idx] <= tr.t[onset_i] + 0.4]
        lf = linfit(tr.t[early], s[early]) if len(early) >= 4 else None
        if lf:
            out["early_accel_mps2"] = lf[0]
    out["heading_matches_motion"] = bool(sign_ok)
    # release -> run-on
    if t_rel is not None:
        i_rel = int(np.searchsorted(tr.t_local, t_rel))
        if i_rel < tr.n:
            out["speed_at_release_mps"] = float(s[min(i_rel, tr.n - 1)])
            after = np.where(tr.t_local >= t_rel)[0]
            stop = [i for i in after if i + 2 < tr.n and np.all(tr.speed[i:i + 3] < 0.06)]
            if stop:
                i_stop = stop[0]
                out["release_to_stop_s"] = float(tr.t_local[i_stop] - t_rel)
                out["runon_distance_m"] = float(math.hypot(tr.x[i_stop] - tr.x[i_rel], tr.y[i_stop] - tr.y[i_rel]))
            if brake:
                bm = tr.mask("brake")
                bi = np.where(bm & (s > 0.1))[0]
                if len(bi) >= 4:
                    lf = linfit(tr.t[bi], s[bi])
                    if lf:
                        out["brake_decel_mps2"] = -lf[0]
                        out["brake_fit_rms_mps"] = lf[2]
                        out["mu_long_est"] = -lf[0] / G
            else:
                cm = tr.mask("coast")
                ci = np.where(cm & (s > 0.08) & (tr.t_local > t_rel + 0.05))[0]
                if len(ci) >= 4:
                    lf = linfit(tr.t[ci], s[ci])
                    if lf:
                        out["coast_decel_mps2"] = -lf[0]
                        out["coast_fit_rms_mps"] = lf[2]
                        out["rolling_resistance_coeff"] = -lf[0] / G
                    # a v^2 drag check: regress decel on speed^2
                    a = -tr.accel[ci] * direction
                    v2 = s[ci] ** 2
                    lf2 = linfit(v2, a)
                    if lf2:
                        out["coast_decel_vs_v2"] = {"c0_mps2": lf2[1], "k_per_m": lf2[0], "rms": lf2[2]}
    out["ok"] = True
    return out


def analyze_circle(tr: Track, cmds: list[dict], seg: int, side: float, phase: str = "drive") -> dict:
    out: dict = {"side": "left" if side > 0 else "right", **tr.gaps()}
    m = tr.mask(phase)
    if m.sum() < 8:
        out.update(ok=False, reason="too few frames")
        return out
    t_cmd = first_command_time(cmds, seg, phase, lambda c: c["steer"] * side > 0 and c["thr"] != 0)
    out["steer_cmd_t"] = t_cmd
    idx = np.where(m)[0]
    w = tr.yaw_rate * side                           # positive = turning the commanded way
    # steady state: the last 60 % of the phase, or once |w| is within 20 % of its median
    t_rel = tr.t[idx] - tr.t[idx][0]
    dur = float(t_rel[-1])
    steady = idx[t_rel > max(0.8, 0.4 * dur)]
    if len(steady) < 6:
        steady = idx[len(idx) // 2:]
    cf = fit_circle(tr.x[steady], tr.y[steady])
    if cf:
        cx, cy, r, rms = cf
        out["radius_tag_m"] = r
        out["circle_center"] = [cx, cy]
        out["circle_fit_rms_m"] = rms
        ang = np.unwrap(np.arctan2(tr.y[steady] - cy, tr.x[steady] - cx))
        out["arc_swept_deg"] = float(math.degrees(abs(ang[-1] - ang[0])))
    out["yaw_rate_steady_radps"] = nanmedian(w[steady])
    out["speed_steady_mps"] = nanmedian(tr.speed[steady])
    out["sideslip_tag_steady_deg"] = None if nanmedian(tr.beta[steady]) is None else math.degrees(nanmedian(tr.beta[steady]) * side)
    if out["yaw_rate_steady_radps"] and out["speed_steady_mps"]:
        wv = out["yaw_rate_steady_radps"]
        v = out["speed_steady_mps"]
        if abs(wv) > 1e-3:
            out["radius_kinematic_m"] = v / abs(wv)
        out["lateral_accel_mps2"] = v * abs(wv)
        out["lateral_accel_g"] = v * abs(wv) / G
        if out.get("radius_tag_m") and out["sideslip_tag_steady_deg"] is not None:
            # kinematic bicycle with no rear slip: tan(beta_tag) = a_tag / R
            out["tag_offset_from_rear_axle_m_est"] = out["radius_tag_m"] * math.tan(math.radians(out["sideslip_tag_steady_deg"]))
    # transient: yaw-rate rise after the steer command
    if t_cmd is not None and out["yaw_rate_steady_radps"]:
        after = idx[tr.t_local[idx] > t_cmd]
        if len(after) >= 5:
            wss = out["yaw_rate_steady_radps"]
            onset = [i for i in after if np.isfinite(w[i]) and w[i] > 0.15 * abs(wss)]
            if onset:
                out["steer_to_yaw_latency_s"] = float(tr.t_local[onset[0]] - t_cmd)
                fit = fit_first_order(tr.t[after], w[after], float(tr.t[onset[0]]) - 0.02)
                if fit:
                    out["yaw_rise_fit"] = {"w_inf_radps": fit[0], "tau_s": fit[1], "rms": fit[2]}
                i90 = [i for i in after if np.isfinite(w[i]) and w[i] > 0.9 * abs(wss)]
                if i90:
                    out["yaw_rise_time_90pct_s"] = float(tr.t_local[i90[0]] - t_cmd)
    out["reversing"] = bool(nanmedian(tr.v_long[steady]) is not None and nanmedian(tr.v_long[steady]) < 0)
    if phase == "steer":
        if dur < 0.3 or (out.get("speed_steady_mps") or 0.0) < 0.3:
            out.update(ok=False, reason=f"steer phase too short ({dur:.1f} s at {out.get('speed_steady_mps') or 0:.2f} m/s)")
            return out
    elif (out.get("speed_steady_mps") or 0.0) < 0.2 or (out.get("arc_swept_deg") or 0.0) < 40.0:
        out.update(ok=False, reason=f"arc too short ({out.get('arc_swept_deg') or 0:.0f} deg, "
                                    f"{out.get('speed_steady_mps') or 0:.2f} m/s): guard cut it")
        return out
    out["peak_yaw_rate_radps"] = float(np.nanmax(w[idx])) if np.isfinite(w[idx]).any() else None
    out["peak_sideslip_tag_deg"] = (float(math.degrees(np.nanmax(np.abs(tr.beta[idx]))))
                                    if np.isfinite(tr.beta[idx]).any() else None)
    out["ok"] = True
    return out


def analyze_skid(tr: Track, cmds: list[dict], seg: int, side: float) -> dict:
    out = analyze_circle(tr, cmds, seg, side, phase="steer")
    drive = tr.mask("drive")
    if drive.sum() >= 4:
        out["entry_speed_mps"] = float(np.nanmax(tr.speed[drive]))
    st = tr.mask("steer")
    if st.sum() >= 4:
        out["speed_drop_in_turn_mps"] = float(np.nanmax(tr.speed[st]) - np.nanmin(tr.speed[st]))
        b = tr.beta[st]
        if np.isfinite(b).any():
            out["sideslip_growth_deg"] = float(math.degrees(np.nanmax(np.abs(b)) - np.nanmin(np.abs(b))))
            out["skidding"] = bool(np.nanmax(np.abs(b)) > math.radians(15))
    return out


def summarize(results: dict, hand: dict) -> dict:
    """Roll the per-run results into one set of numbers, then map them onto
    Isaac Sim quantities. Anything that needs a hand measurement is flagged."""
    def med(items, key):
        vals = [it[key] for it in items if it.get("ok") and it.get(key) is not None]
        return (None if not vals else float(np.median(vals))), len(vals)

    S: dict = {}
    fwd = results.get("straight_fwd", [])
    rev = results.get("straight_rev", [])
    brk = results.get("brake", [])
    circ_l = results.get("circle_left", [])
    circ_r = results.get("circle_right", [])
    skid = results.get("skid", [])

    def plateau(items):
        vals = [it["plateau"]["speed_mps"] for it in items if it.get("ok") and it.get("plateau", {}).get("reached")]
        if vals:
            return float(np.median(vals)), len(vals), "plateau"
        vals = [it["accel_fit"]["v_inf_mps"] for it in items if it.get("ok") and it.get("accel_fit")]
        if vals:
            return float(np.median(vals)), len(vals), "first-order fit (plateau not reached)"
        vals = [it["peak_speed_mps"] for it in items if it.get("ok") and it.get("peak_speed_mps")]
        return (float(np.median(vals)) if vals else None), len(vals), "peak (lower bound)"

    v_f, n_f, how_f = plateau(fwd + brk + skid)
    v_r, n_r, how_r = plateau(rev)
    S["top_speed_fwd_mps"] = {"value": v_f, "n": n_f, "how": how_f}
    S["top_speed_rev_mps"] = {"value": v_r, "n": n_r, "how": how_r}
    taus = [it["accel_fit"]["tau_s"] for it in fwd + brk + skid if it.get("ok") and it.get("accel_fit")]
    S["accel_time_constant_s"] = {"value": float(np.median(taus)) if taus else None, "n": len(taus)}
    a0 = [it["early_accel_mps2"] for it in fwd + brk + skid if it.get("ok") and it.get("early_accel_mps2")]
    S["accel_initial_mps2"] = {"value": float(np.median(a0)) if a0 else None, "n": len(a0)}
    a0r = [it["early_accel_mps2"] for it in rev if it.get("ok") and it.get("early_accel_mps2")]
    S["accel_initial_rev_mps2"] = {"value": float(np.median(a0r)) if a0r else None, "n": len(a0r)}
    S["command_to_motion_latency_s"] = dict(zip(("value", "n"), med(fwd + rev + brk + skid, "command_to_motion_latency_s")))
    S["coast_decel_mps2"] = dict(zip(("value", "n"), med(fwd + rev, "coast_decel_mps2")))
    S["rolling_resistance_coeff"] = dict(zip(("value", "n"), med(fwd + rev, "rolling_resistance_coeff")))
    S["runon_distance_m"] = dict(zip(("value", "n"), med(fwd + rev, "runon_distance_m")))
    S["release_to_stop_s"] = dict(zip(("value", "n"), med(fwd + rev, "release_to_stop_s")))
    S["brake_decel_mps2"] = dict(zip(("value", "n"), med(brk, "brake_decel_mps2")))
    S["mu_long_est"] = dict(zip(("value", "n"), med(brk, "mu_long_est")))
    S["turn_radius_left_m"] = dict(zip(("value", "n"), med(circ_l, "radius_tag_m")))
    S["turn_radius_right_m"] = dict(zip(("value", "n"), med(circ_r, "radius_tag_m")))
    S["yaw_rate_steady_radps"] = dict(zip(("value", "n"), med(circ_l + circ_r, "yaw_rate_steady_radps")))
    S["speed_in_turn_mps"] = dict(zip(("value", "n"), med(circ_l + circ_r, "speed_steady_mps")))
    S["lateral_accel_steady_g"] = dict(zip(("value", "n"), med(circ_l + circ_r, "lateral_accel_g")))
    S["sideslip_tag_steady_deg"] = dict(zip(("value", "n"), med(circ_l + circ_r, "sideslip_tag_steady_deg")))
    S["tag_offset_from_rear_axle_m_est"] = dict(zip(("value", "n"), med(circ_l + circ_r, "tag_offset_from_rear_axle_m_est")))
    S["steer_to_yaw_latency_s"] = dict(zip(("value", "n"), med(circ_l + circ_r + skid, "steer_to_yaw_latency_s")))
    yt = [it["yaw_rise_fit"]["tau_s"] for it in circ_l + circ_r + skid if it.get("ok") and it.get("yaw_rise_fit")]
    S["yaw_rise_time_constant_s"] = {"value": float(np.median(yt)) if yt else None, "n": len(yt)}
    S["peak_lateral_accel_g"] = {"value": max([it["lateral_accel_g"] for it in circ_l + circ_r + skid
                                               if it.get("ok") and it.get("lateral_accel_g")] or [None]) if any(
        it.get("ok") and it.get("lateral_accel_g") for it in circ_l + circ_r + skid) else None}
    S["peak_sideslip_tag_deg"] = dict(zip(("value", "n"), med(skid + circ_l + circ_r, "peak_sideslip_tag_deg")))
    S["skidding_observed"] = any(it.get("skidding") for it in skid)
    st = results.get("static", [])
    S["pose_noise"] = st[-1] if st else None
    hd = [it["heading_matches_motion"] for it in fwd + rev + brk if it.get("ok") and "heading_matches_motion" in it]
    S["tag_heading_matches_drive_direction"] = (all(hd) if hd else None)

    # ------------------------------------------------------------ Isaac Sim mapping
    L = hand.get("wheelbase_m")
    r_w = hand.get("wheel_radius_m")
    mass = hand.get("mass_kg")
    track = hand.get("track_m")
    R_l, R_r = S["turn_radius_left_m"]["value"], S["turn_radius_right_m"]["value"]
    R = float(np.mean([v for v in (R_l, R_r) if v])) if (R_l or R_r) else None
    v_top = v_f
    tau = S["accel_time_constant_s"]["value"]
    a_coast = S["coast_decel_mps2"]["value"]
    a_brake = S["brake_decel_mps2"]["value"]
    placeholders = []

    def need(name, val, note):
        if val is None:
            placeholders.append({"parameter": name, "note": note})
        return val

    isaac: dict = {
        "notes": [
            "Positions are of the roof tag, not the rear axle; tag_offset_from_rear_axle_m_est comes from the",
            "steady-turn sideslip assuming no rear-wheel slip. Hand-measure it to confirm.",
            "Friction values from coast/brake/turn are effective decelerations divided by g: use them as PhysX",
            "material friction only after checking the wheel model (a locked-wheel skid gives mu_k directly;",
            "a driven turn gives a lower bound unless skidding_observed is true).",
            "The real car is binary: throttle in {-1, 0, +1} (BACK/none/FWD) and steer in {-1, 0, +1}.",
        ],
        "command_model": {
            "throttle_levels": [-1, 0, 1],
            "steer_levels": [-1, 0, 1],
            "binary_threshold": 1.0 / 3.0,
            "command_rate_hz": Driver.RATE_HZ,
            "command_to_motion_latency_s": S["command_to_motion_latency_s"]["value"],
            "steer_to_yaw_latency_s": S["steer_to_yaw_latency_s"]["value"],
            "firmware_failsafe_s": 0.3,
        },
        "longitudinal": {
            "top_speed_fwd_mps": v_top,
            "top_speed_rev_mps": v_r,
            "first_order_time_constant_s": tau,
            "initial_accel_mps2": S["accel_initial_mps2"]["value"],
            "coast_decel_mps2": a_coast,
            "brake_decel_mps2": a_brake,
            "runon_after_release_m": S["runon_distance_m"]["value"],
            "runon_after_release_s": S["release_to_stop_s"]["value"],
        },
        "lateral": {
            "turn_radius_tag_left_m": R_l,
            "turn_radius_tag_right_m": R_r,
            "yaw_rate_steady_radps": S["yaw_rate_steady_radps"]["value"],
            "speed_in_turn_mps": S["speed_in_turn_mps"]["value"],
            "lateral_accel_steady_g": S["lateral_accel_steady_g"]["value"],
            "sideslip_tag_steady_deg": S["sideslip_tag_steady_deg"]["value"],
            "yaw_rise_time_constant_s": S["yaw_rise_time_constant_s"]["value"],
            "peak_sideslip_tag_deg": S["peak_sideslip_tag_deg"]["value"],
            "skidding_observed": S["skidding_observed"],
        },
        "chassis": {
            "mass_kg": need("mass_kg", mass, "weigh the car with battery and ESP"),
            "wheelbase_m": need("wheelbase_m", L, "axle centre to axle centre"),
            "track_m": need("track_m", track, "tyre centre to tyre centre"),
            "wheel_radius_m": need("wheel_radius_m", r_w, "loaded tyre radius"),
            "body_length_m": hand.get("body_length_m"),
            "body_width_m": hand.get("body_width_m"),
            "tag_offset_from_rear_axle_m": hand.get("tag_offset_m"),
            "tag_offset_from_rear_axle_m_est": S["tag_offset_from_rear_axle_m_est"]["value"],
            "source": {k: ("hand" if hand.get(k) is not None else "missing") for k in
                       ("mass_kg", "wheelbase_m", "track_m", "wheel_radius_m", "body_length_m", "body_width_m", "tag_offset_m")},
        },
        "physx_material": {
            "static_friction": None,
            "dynamic_friction": None,
            "restitution": 0.1,
            "how": "dynamic from the brake test (mu_long_est) when the wheels lock; otherwise the larger of "
                   "lateral_accel_steady_g and mu_long_est is a LOWER bound. static = dynamic * 1.1 as a start.",
        },
        "wheel_drive": {},
        "steering": {},
        "bicycle_model_for_training_configs": {},
    }
    mu = None
    cands = [v for v in (S["mu_long_est"]["value"], S["peak_lateral_accel_g"]["value"]) if v]
    if cands:
        mu = max(cands)
        isaac["physx_material"]["dynamic_friction"] = mu
        isaac["physx_material"]["static_friction"] = mu * 1.1
        isaac["physx_material"]["is_lower_bound"] = not (S["skidding_observed"] or (S["mu_long_est"]["value"] or 0) >= (S["peak_lateral_accel_g"]["value"] or 0))
    isaac["physx_material"]["rolling_resistance_coeff"] = S["rolling_resistance_coeff"]["value"]

    wd = isaac["wheel_drive"]
    wd["type"] = "velocity drive, on/off"
    if v_top and r_w:
        wd["target_angular_velocity_radps"] = v_top / r_w
        wd["target_rpm"] = v_top / r_w * 60.0 / (2 * math.pi)
    else:
        wd["target_angular_velocity_radps"] = None
        placeholders.append({"parameter": "wheel_drive.target_angular_velocity_radps", "note": "needs wheel_radius_m"})
    if v_top and tau:
        wd["speed_time_constant_s"] = tau
    if v_top and tau and mass and r_w:
        a_peak = v_top / tau
        n_driven = int(hand.get("driven_wheels", 2))
        wd["driven_wheels"] = n_driven
        wd["peak_wheel_torque_Nm_total"] = mass * (a_peak + (a_coast or 0.0)) * r_w
        wd["peak_wheel_torque_Nm_per_wheel"] = wd["peak_wheel_torque_Nm_total"] / n_driven
        wd["joint_drive_damping_hint"] = wd["peak_wheel_torque_Nm_per_wheel"] / max(wd["target_angular_velocity_radps"], 1e-6)
    else:
        placeholders.append({"parameter": "wheel_drive.peak_wheel_torque_Nm", "note": "needs mass_kg and wheel_radius_m"})
    if v_top and tau and a_coast is not None:
        wd["rolling_drag_torque_fraction"] = a_coast / (v_top / tau)

    stg = isaac["steering"]
    stg["type"] = "bang-bang: full lock left / centre / full lock right"
    stg["turn_radius_tag_m"] = R
    if R and L:
        a_tag = hand.get("tag_offset_m")
        if a_tag is None:
            a_tag = S["tag_offset_from_rear_axle_m_est"]["value"] or 0.0
        R_rear = math.sqrt(max(R * R - a_tag * a_tag, 1e-6))
        stg["turn_radius_rear_axle_m"] = R_rear
        stg["max_steer_angle_deg"] = math.degrees(math.atan2(L, R_rear))
        if track:
            stg["ackermann_inner_deg"] = math.degrees(math.atan2(L, R_rear - track / 2))
            stg["ackermann_outer_deg"] = math.degrees(math.atan2(L, R_rear + track / 2))
    else:
        stg["max_steer_angle_deg"] = None
        placeholders.append({"parameter": "steering.max_steer_angle_deg", "note": "needs wheelbase_m (atan(L / R))"})
    stg["slew_time_constant_s"] = S["yaw_rise_time_constant_s"]["value"]
    stg["latency_s"] = S["steer_to_yaw_latency_s"]["value"]
    if S["yaw_rise_time_constant_s"]["value"]:
        stg["joint_drive_hint"] = ("position drive; pick stiffness/damping so the lock-to-lock step settles in ~"
                                   f"{3 * S['yaw_rise_time_constant_s']['value']:.2f} s (3 tau). Part of that tau is tyre relaxation, not the servo.")

    bm = isaac["bicycle_model_for_training_configs"]
    bm["max_speed_mps"] = v_top
    bm["max_reverse_speed_mps"] = v_r
    bm["max_accel_mps2"] = S["accel_initial_mps2"]["value"]
    bm["max_reverse_accel_mps2"] = S["accel_initial_rev_mps2"]["value"]
    bm["max_brake_mps2"] = a_brake
    bm["coast_decel_mps2"] = a_coast
    bm["wheelbase_m"] = L
    bm["max_steering_angle_deg"] = stg.get("max_steer_angle_deg")
    bm["command_latency_s"] = S["command_to_motion_latency_s"]["value"]
    bm["steering_time_constant_s"] = S["yaw_rise_time_constant_s"]["value"]
    bm["note"] = "keys match configs/*.json vehicles.attackers; speed is a FIRST-ORDER response, not a rate limit"

    return {"summary": S, "isaac_sim": isaac, "placeholders_to_measure_by_hand": placeholders}


def analyze_run(d: Path, hand: dict | None = None, *, plots: bool = False, log=print) -> dict:
    traj, cmds, meta = load_run(d)
    hand = dict(meta.get("hand_measurements") or {})
    hand.update({k: v for k, v in (hand or {}).items() if v is not None})
    by_seg: dict[int, list] = {}
    for r in traj:
        by_seg.setdefault(r["seg"], []).append(r)
    results: dict[str, list] = {}
    per_segment = []
    for seg in meta["segments"]:
        sid = seg["seg"]
        tr = Track(by_seg.get(sid, []))
        test, variant = seg["test"], seg["variant"]
        if test == "static":
            res = analyze_static(tr)
            key = "static"
        elif test in ("straight", "brake"):
            res = analyze_straight(tr, cmds, sid, 1.0 if variant == "fwd" else -1.0, brake=(test == "brake"))
            key = "brake" if test == "brake" else f"straight_{variant}"
        elif test == "circle":
            res = analyze_circle(tr, cmds, sid, 1.0 if variant == "left" else -1.0)
            key = f"circle_{variant}"
        elif test == "skid":
            res = analyze_skid(tr, cmds, sid, 1.0 if variant == "left" else -1.0)
            key = "skid"
        elif test == "axes":
            res = {"ok": True, "axis_map": seg.get("axis_map"), "raw": seg.get("raw")}
            key = "axes"
        else:
            continue
        res["seg"] = sid
        results.setdefault(key, []).append(res)
        per_segment.append({"seg": sid, "test": test, "variant": variant, "result": res})
        log(f"  seg {sid:2d} {test:8s} {variant:5s} -> " + (describe(res) if res.get("ok") else f"NOT USABLE: {res.get('reason')}"))
    roll = summarize(results, hand)
    out = {
        "meta": {
            "tool": f"physics_id.py {TOOL_VERSION}", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_dir": str(d), "recorded_utc": meta.get("recorded_utc"), "car": meta.get("esp"),
            "arena_half_m": meta.get("arena_half_m"), "frames": meta.get("frames"),
            "pose_source": meta.get("pose_source"), "sim": meta.get("sim", False), "host": meta.get("host"),
            "link": meta.get("link"), "guard_trips": meta.get("guard_trips"), "hand_measurements": hand,
            "coordinate_frame": "arena-centred metres, +x right, +y up, heading 0 = +x, CCW positive; poses are the roof tag centre",
        },
        **roll,
        "per_run": results,
    }
    (d / "car_physics.json").write_text(json.dumps(rnd(out, 4), indent=2))
    write_report(d, out)
    if plots:
        try:
            make_plots(d, traj, cmds, meta, out, log)
        except ImportError:
            import subprocess
            r = subprocess.run(["python3", str(Path(__file__).resolve()), "analyze", str(d), "--plots"],
                               capture_output=True, text=True)
            log("  plots drawn with python3" if r.returncode == 0 else f"  plots skipped: {r.stderr.strip()[-300:]}")
        except Exception as exc:   # plots are a convenience, never a failure
            log(f"  plots skipped: {exc}")
    return out


def describe(res: dict) -> str:
    bits = []
    for k, fmt in (("command_to_motion_latency_s", "lat {:.2f}s"), ("peak_speed_mps", "peak {:.2f}m/s"),
                   ("coast_decel_mps2", "coast {:.2f}m/s2"), ("brake_decel_mps2", "brake {:.2f}m/s2"),
                   ("runon_distance_m", "run-on {:.2f}m"), ("radius_tag_m", "R {:.2f}m"),
                   ("yaw_rate_steady_radps", "yaw {:.2f}rad/s"), ("speed_steady_mps", "v {:.2f}m/s"),
                   ("sideslip_tag_steady_deg", "beta {:.1f}deg"), ("heading_std_deg", "h-noise {:.2f}deg")):
        v = res.get(k)
        if v is not None:
            bits.append(fmt.format(v))
    if res.get("accel_fit"):
        bits.append(f"tau {res['accel_fit']['tau_s']:.2f}s v_inf {res['accel_fit']['v_inf_mps']:.2f}")
    if res.get("pose_std_m"):
        bits.append(f"xy-noise {1000*res['pose_std_m'][0]:.1f}/{1000*res['pose_std_m'][1]:.1f}mm")
    return "  ".join(bits) or "ok"


def write_report(d: Path, out: dict) -> None:
    S = out["summary"]
    isaac = out["isaac_sim"]
    lines = [f"# Car physics identification -- {out['meta'].get('recorded_utc', '')}", ""]
    lines.append(f"Run directory: `{d}`  ·  car `{out['meta'].get('car')}`  ·  {out['meta'].get('frames')} pose frames"
                 + ("  ·  **SIMULATED**" if out["meta"].get("sim") else ""))
    lines += ["", "## Headline numbers", "", "| quantity | value | n | how |", "|---|---|---|---|"]
    for k, unit in (("top_speed_fwd_mps", "m/s"), ("top_speed_rev_mps", "m/s"), ("accel_time_constant_s", "s"),
                    ("accel_initial_mps2", "m/s²"), ("command_to_motion_latency_s", "s"), ("coast_decel_mps2", "m/s²"),
                    ("runon_distance_m", "m"), ("release_to_stop_s", "s"), ("brake_decel_mps2", "m/s²"),
                    ("mu_long_est", ""), ("turn_radius_left_m", "m"), ("turn_radius_right_m", "m"),
                    ("yaw_rate_steady_radps", "rad/s"), ("speed_in_turn_mps", "m/s"), ("lateral_accel_steady_g", "g"),
                    ("sideslip_tag_steady_deg", "deg"), ("tag_offset_from_rear_axle_m_est", "m"),
                    ("steer_to_yaw_latency_s", "s"), ("yaw_rise_time_constant_s", "s"), ("peak_sideslip_tag_deg", "deg")):
        e = S.get(k) or {}
        v = e.get("value")
        lines.append(f"| {k} | {'—' if v is None else f'{v:.3f} {unit}'} | {e.get('n', '')} | {e.get('how', '')} |")
    lines.append(f"| skidding_observed | {S.get('skidding_observed')} | | |")
    lines.append(f"| tag heading matches drive direction | {S.get('tag_heading_matches_drive_direction')} | | |")
    pn = S.get("pose_noise") or {}
    if pn.get("ok"):
        lines += ["", f"Pose noise (car still): σx/σy {1000*pn['pose_std_m'][0]:.1f}/{1000*pn['pose_std_m'][1]:.1f} mm, "
                      f"heading σ {pn['heading_std_deg']:.2f}°, {pn.get('rate_hz', 0):.1f} frames/s, speed noise "
                      f"{pn['speed_noise_rms_mps']:.3f} m/s rms."]
    lines += ["", "## Isaac Sim parameters", "", "```json", json.dumps(rnd(isaac, 4), indent=2), "```"]
    if out["placeholders_to_measure_by_hand"]:
        lines += ["", "## Still needed by hand", ""]
        for p in out["placeholders_to_measure_by_hand"]:
            lines.append(f"- **{p['parameter']}** -- {p['note']}")
        lines += ["", "Pass them as `--wheelbase --track --wheel-radius --mass --tag-offset` to `analyze` and the derived "
                      "values fill in without re-driving."]
    lines += ["", "## Every run", ""]
    for key, items in out["per_run"].items():
        for it in items:
            lines.append(f"- seg {it.get('seg')} `{key}`: " + (describe(it) if it.get("ok") else f"not usable ({it.get('reason')})"))
    (d / "report.md").write_text("\n".join(lines) + "\n")


def make_plots(d: Path, traj, cmds, meta, out, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd = d / "plots"
    pd.mkdir(exist_ok=True)
    by_seg: dict[int, list] = {}
    for r in traj:
        by_seg.setdefault(r["seg"], []).append(r)
    half = meta.get("arena_half_m") or 1.2192
    # top-down of everything
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], "k--", lw=1)
    for seg in meta["segments"]:
        fr = by_seg.get(seg["seg"], [])
        if len(fr) < 2:
            continue
        xs = [r["x"] for r in fr]
        ys = [r["y"] for r in fr]
        ax.plot(xs, ys, lw=1.2, label=f"{seg['seg']} {seg['test']} {seg['variant']}")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("tag trajectories")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(pd / "trajectories.png", dpi=130)
    plt.close(fig)
    # per segment: speed / yaw rate vs time with the command trace
    for seg in meta["segments"]:
        fr = by_seg.get(seg["seg"], [])
        if len(fr) < 8 or seg["test"] in ("reposition",):
            continue
        tr = Track(fr)
        t0 = tr.t_local[0]
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(tr.t_local - t0, tr.v_long, label="v_long (heading axis)")
        axes[0].plot(tr.t_local - t0, tr.speed, alpha=0.5, label="|v|")
        axes[0].set_ylabel("m/s")
        axes[0].legend(fontsize=7)
        axes[1].plot(tr.t_local - t0, np.degrees(tr.yaw_rate), label="yaw rate")
        axes[1].plot(tr.t_local - t0, np.degrees(tr.beta), alpha=0.6, label="sideslip at tag")
        axes[1].set_ylabel("deg/s, deg")
        axes[1].legend(fontsize=7)
        cs = [c for c in cmds if c["seg"] == seg["seg"]]
        if cs:
            ct = np.array([c["t"] for c in cs]) - t0
            axes[2].step(ct, [c["thr"] for c in cs], where="post", label="throttle sent")
            axes[2].step(ct, [c["steer"] for c in cs], where="post", label="steer sent")
            axes[2].set_ylim(-1.2, 1.2)
            axes[2].legend(fontsize=7)
        axes[2].set_xlabel("s")
        for ph in seg.get("phases", []):
            for a in axes:
                a.axvline(ph["t"] - t0, color="gray", lw=0.6, ls=":")
            axes[0].text(ph["t"] - t0, axes[0].get_ylim()[1] * 0.9, ph["name"], fontsize=6, rotation=90)
        fig.suptitle(f"seg {seg['seg']}: {seg['test']} {seg['variant']}")
        fig.tight_layout()
        fig.savefig(pd / f"seg{seg['seg']:02d}_{seg['test']}_{seg['variant']}.png", dpi=120)
        plt.close(fig)
    log(f"  plots in {pd}")


# ----------------------------------------------------------------------------- run


def hand_from_args(args) -> dict:
    return {"wheelbase_m": args.wheelbase, "track_m": args.track, "wheel_radius_m": args.wheel_radius,
            "mass_kg": args.mass, "tag_offset_m": args.tag_offset, "body_length_m": args.body_length,
            "body_width_m": args.body_width, "driven_wheels": args.driven_wheels}


def add_hand_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("hand measurements (optional; fill in the Isaac Sim derived values)")
    g.add_argument("--wheelbase", type=float, help="axle to axle, m")
    g.add_argument("--track", type=float, help="tyre centre to tyre centre, m")
    g.add_argument("--wheel-radius", type=float, help="loaded tyre radius, m")
    g.add_argument("--mass", type=float, help="kg, with battery and ESP")
    g.add_argument("--tag-offset", type=float, help="tag centre forward of the rear axle, m")
    g.add_argument("--body-length", type=float)
    g.add_argument("--body-width", type=float)
    g.add_argument("--driven-wheels", type=int, default=2)


def ble_preclean(spec: str, log) -> None:
    """BlueZ keeps a GATT connection open after the client process dies, and a
    connected car stops advertising, so a connect-by-name then fails with
    'not seen'. Drop any such connection before opening the link."""
    import subprocess
    if not spec.lower().startswith("ble:"):
        return
    peer = spec[4:].strip()
    try:
        if ":" not in peer:                     # a name: look its address up
            out = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True, timeout=10).stdout
            addrs = [ln.split()[1] for ln in out.splitlines() if ln.strip().endswith(" " + peer)]
            if not addrs:
                return
            peer = addrs[0]
        info = subprocess.run(["bluetoothctl", "info", peer], capture_output=True, text=True, timeout=10).stdout
        if "Connected: yes" in info:
            log(f"BlueZ still holds a connection to {peer}: dropping it first")
            subprocess.run(["bluetoothctl", "disconnect", peer], capture_output=True, text=True, timeout=10)
            time.sleep(1.5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def cmd_run(args) -> int:
    from pc_controller.esp_link import EspLink, parse_targets, target_uri
    from pc_controller.serial_link import SerialLinkError

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out).expanduser() / stamp
    out.mkdir(parents=True, exist_ok=True)
    logfh = (out / "physics_id.log").open("w")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logfh.write(line + "\n")
        logfh.flush()

    half = args.arena / 2.0
    guard = Guard(half, body=args.body_reach, margin=args.margin, horizon_s=args.horizon,
                  evader_keepout=args.evader_keepout, max_pose_age=args.max_pose_age)
    log(f"physics_id {TOOL_VERSION}: arena {args.arena:.3f} m (half {half:.3f}), drive limit ±{guard.limit:.2f} m, "
        f"run-on horizon {guard.horizon_s:.2f} s, evader keep-out {guard.evader_keepout:.2f} m")
    try:
        tap = PoseTap(args.pose_port, pursuer_slot=args.slot)
    except OSError as exc:
        log(f"cannot listen on udp/{args.pose_port}: {exc}. Is the controller still running? "
            "(svc_stop controller / ./arena stop)")
        return 2
    log(f"listening for pose frames on udp/{args.pose_port} (pursuer slot {args.slot})")
    targets = parse_targets(args.esp)
    if len(targets) != 1:
        log("physics_id drives one car: give a single --esp")
        return 2
    link = None
    for attempt in range(1, args.connect_tries + 1):
        ble_preclean(args.esp, log)
        try:
            link = EspLink(targets, log=lambda m: log(f"link: {m}"))
            log(f"opening {target_uri(link.addr)} (try {attempt}/{args.connect_tries}) ...")
            link.wait_ready()
            break
        except SerialLinkError as exc:
            log(f"car link: {exc}")
            if link is not None:
                link._release()
                link = None
            time.sleep(2.0)
    if link is None:
        log("could not bring the car link up")
        tap.close()
        return 2
    log("car link up")
    driver = Driver(link, tap, guard, log, sim=args.sim)
    driver.radius = args.radius_guess
    driver.move_m = args.move_m
    if args.axis_map != "auto":
        driver.axis_map = parse_axis_map(args.axis_map)
        meta["axis_map"] = driver.axis_map
        if any(k != v for k, v in driver.axis_map.items()):
            log(f"axis map in force (desired -> wire): {driver.axis_map}")
    stop_flag = {"stop": False}

    def _term(_sig, _frm):
        stop_flag["stop"] = True
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    meta = {"recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "esp": args.esp,
            "pose_port": args.pose_port, "arena_half_m": half, "guard": {"body": guard.body, "margin": guard.margin,
            "horizon_s_initial": guard.horizon_s, "evader_keepout": guard.evader_keepout}, "sim": args.sim,
            "host": platform.node(), "hand_measurements": hand_from_args(args), "tests": args.tests,
            "pose_source": "udp pose frames (vision service)"}
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    radius_guess = args.radius_guess
    try:
        st = driver.wait_pose(15.0)
        if st is None:
            log("no pose frames: is the car in the arena with its tag visible, and the evader tag too? "
                "(the vision service sends nothing while either is missing)")
            return 3
        log(f"car at ({st['x']:+.2f},{st['y']:+.2f}) heading {math.degrees(st['h']):.0f} deg; "
            f"evader {st['evader']}; pose age {st['age']*1000:.0f} ms")
        driver.neutral(1.0, phase="arm")
        if args.axis_map == "auto":
            amap = driver.test_axes()
            if amap is None:
                log("axis discovery failed; stopping before any guarded driving")
                stop_flag["stop"] = True
            else:
                driver.axis_map = amap
                meta["axis_map"] = amap
        for rep in range(args.repeats):
            for test in tests:
                if stop_flag["stop"] or driver.abort_reason:
                    break
                if test == "axes":
                    if rep == 0:
                        driver.test_axes()
                elif test == "static":
                    if rep == 0:
                        driver.test_static(args.static_s)
                elif test == "straight":
                    drove = driver.test_straight(1.0, need=args.need)
                    if drove and rep == 0:
                        # the first coast-down sets the guard's run-on horizon
                        traj = [r for r in tap.frames if r["seg"] == driver.seg_id]
                        try:
                            res = analyze_straight(Track(traj), driver.commands, driver.seg_id, drove, brake=False)
                        except Exception as exc:       # analysis bug: keep driving, analyse offline
                            log(f"   (in-run analysis failed: {exc})")
                            res = {}
                        if res.get("ok") and res.get("release_to_stop_s"):
                            new_h = min(0.8, max(0.35, 1.3 * res["release_to_stop_s"]))
                            log(f"   measured release->stop {res['release_to_stop_s']:.2f} s, run-on {res.get('runon_distance_m', float('nan')):.2f} m; "
                                f"guard horizon {guard.horizon_s:.2f} -> {new_h:.2f} s")
                            guard.horizon_s = new_h
                        if res.get("ok") and res.get("heading_matches_motion") is False:
                            log("   WARNING: the car moved against its tag heading: tag mounted backwards, "
                                "or FWD/BACK swapped. Guard predictions use the heading; stopping the battery.")
                            stop_flag["stop"] = True
                            break
                    driver.test_straight(-1.0, need=args.need_rev)
                elif test == "brake":
                    driver.test_straight(1.0, need=args.need, brake=True)
                elif test == "circle":
                    ok = driver.test_circle(1.0, radius_guess=radius_guess, seconds=args.circle_s)
                    if ok:
                        traj = [r for r in tap.frames if r["seg"] == driver.seg_id]
                        try:
                            res = analyze_circle(Track(traj), driver.commands, driver.seg_id, 1.0)
                        except Exception as exc:
                            log(f"   (in-run analysis failed: {exc})")
                            res = {}
                        if res.get("ok") and res.get("radius_tag_m") and 0.15 < res["radius_tag_m"] < 2.0 \
                                and (res.get("arc_swept_deg") or 0) >= 60:
                            radius_guess = float(res["radius_tag_m"])
                            log(f"   left circle radius {radius_guess:.2f} m (arc {res.get('arc_swept_deg', 0):.0f} deg); planning with it")
                    driver.test_circle(-1.0, radius_guess=radius_guess, seconds=args.circle_s)
                elif test == "skid":
                    driver.test_skid(1.0, need=args.need)
                    driver.test_skid(-1.0, need=args.need)
                else:
                    log(f"unknown test {test!r}")
        driver.neutral(0.5, phase="done")
    except KeyboardInterrupt:
        log("interrupted -> E-stop")
    finally:
        try:
            link.close()
        finally:
            tap.close()
            ble_preclean(args.esp, log)
            log(f"link: {driver.acks} acks, {driver.lost} unacked, rtt median "
                f"{nanmedian(driver.rtts) or float('nan'):.0f} ms; guard trips {guard.trips}; "
                f"{len(tap.frames)} frames recorded")
            save_run(out, driver, tap, meta)
            log(f"saved {out}")
    log("analysis:")
    analyze_run(out, hand_from_args(args), plots=args.plots, log=log)
    log(f"wrote {out / 'car_physics.json'} and {out / 'report.md'}")
    logfh.close()
    return 0


def cmd_analyze(args) -> int:
    d = Path(args.run_dir).expanduser()
    if not (d / "trajectory.csv").exists():
        print(f"{d}: no trajectory.csv")
        return 2
    hand = hand_from_args(args)
    meta = json.loads((d / "run_meta.json").read_text())
    merged = dict(meta.get("hand_measurements") or {})
    merged.update({k: v for k, v in hand.items() if v is not None})
    meta["hand_measurements"] = merged
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2))
    analyze_run(d, merged, plots=args.plots)
    print(f"wrote {d / 'car_physics.json'} and {d / 'report.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="drive the battery and record")
    p.add_argument("--esp", required=True, help="the car: ble:hive-car0, serial:/dev/..., or ip:port")
    p.add_argument("--pose-port", type=int, default=9870)
    p.add_argument("--slot", type=int, default=0, help="pursuer slot of the car in the pose frame")
    p.add_argument("--arena", type=float, default=2.4384, help="arena side, m (8 ft = 2.4384)")
    p.add_argument("--out", default="~/physics_runs")
    p.add_argument("--tests", default="static,straight,brake,circle,skid")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--static-s", type=float, default=4.0)
    p.add_argument("--circle-s", type=float, default=6.0)
    p.add_argument("--need", type=float, default=1.4, help="straight room needed for a forward run, m")
    p.add_argument("--need-rev", type=float, default=1.1, help="straight room needed for a reverse run, m")
    p.add_argument("--radius-guess", type=float, default=0.6, help="turning radius to plan the first circle, m")
    p.add_argument("--move-m", type=float, default=0.5, help="distance one repositioning move covers, m")
    p.add_argument("--horizon", type=float, default=0.6, help="guard run-on horizon, s (re-estimated after the first coast)")
    p.add_argument("--margin", type=float, default=0.12, help="clearance to the tape, m")
    p.add_argument("--body-reach", type=float, default=0.12, help="how far the body reaches from the tag, m")
    p.add_argument("--evader-keepout", type=float, default=0.30)
    p.add_argument("--max-pose-age", type=float, default=0.25)
    p.add_argument("--connect-tries", type=int, default=6, help="attempts to bring the car link up (BLE can drop while verifying)")
    p.add_argument("--axis-map", default="identity",
                   help="desired->wire button map, e.g. fwd=RIGHT,back=LEFT,left=BACK,right=FWD; 'auto' runs the "
                        "axes discovery first and adopts its result; 'identity' trusts the firmware's pins")
    p.add_argument("--plots", action="store_true", help="also draw plots (needs matplotlib)")
    p.add_argument("--sim", action="store_true", help="mark the run as simulated")
    add_hand_args(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("analyze", help="re-analyse a recorded run")
    p.add_argument("run_dir")
    p.add_argument("--plots", action="store_true")
    add_hand_args(p)
    p.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
