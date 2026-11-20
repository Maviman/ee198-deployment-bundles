"""The dead zone: cars the cameras cannot see, and cars past the tape (2026-11-19).

Everything the cameras do not cover is a dead zone. A car is in it when the pose stream
marks it LOST (pose_stream: meta["seen"] False), which in practice happens at the border:
the car overshoots the tape and leaves the cameras' view. Before this module that ended the
run (vision went silent, the controller stalled, the dashboard disarmed). Now the frames keep
coming with the car marked lost, and this module decides what each car is sent:

  seen, inside the tape           the model's own command, except the EDGE BRAKE (models without
                                  their own limiter): moving too fast to stop before the tape
                                  with a 4 m/s^2 brake after 0.3 s, the throttle becomes a brake;
                                  otherwise a throttle toward the tape is capped at the speed the
                                  car could still stop from (never under GUARD_CREEP, so it can
                                  always reach the tape slowly). The steering stays the model's.
  seen, past the tape             RECOVERY, seen: brake, then drive toward the arena centre
                                  slowly, until REJOIN_INSIDE_M inside the tape
  not seen for BLIND_AFTER_S,     RECOVERY, blind: brake against the last measured motion,
  or dead-reckoned past the tape  then replay the last second of commands that were sent while
                                  the car was SEEN, backwards: newest first, the same steering,
                                  a slow throttle against the direction it was moving. The car
                                  retraces its own track to where the cameras last saw it (and
                                  keeps that direction 1 s more, wheels straight, if it is
                                  still unseen when the memory runs out).
  blind for GIVE_UP_S (4.5 s)     zero command, flagged gave_up (the dashboard ends the run)
  evader not seen                 every car stands still (armed, link alive) until it is back

Why the replay goes backwards along the car's own track, not "toward the centre": a blind car
does not know where it is, only what it did. Ground it was seen on a second ago is ground the
cameras cover, and it is free of obstacles the car has not already passed. The commands are
remembered here, on the controller, which sent every one of them: the car's firmware needs no
change and its 300 ms dead-man still stops it when the radio drops.

"Seen" here is stricter than perception's: perception holds a missed car's pose for its hold
time (0.25 s) before it marks the car lost, and a model fed that frozen pose drives on as if
the car stood still (simulated: a car leaving view at 2 m/s was 0.8 m past the tape before
anything braked). `fresh_seen` therefore counts a pose older than FRESH_S as not seen.

Two kinds of model:
* A runtime that handles unseen cars itself (its module sets HANDLES_UNSEEN = True: the
  rc16_hive_v1 playbook commander, trained with exactly this recovery in the loop,
  rc16/hive/recovery.py). The loop passes `seen` into it and this module only supervises:
  a car unseen for BLIND_STALE_S that the runtime is NOT recovering gets a zero command (the
  runtime would keep driving it on a frozen pose), and it tracks gave_up.
* Any other model (the one-pursuer rc16_n1_v1, the older ones). EdgeRecovery below, a
  one-car numpy port of rc16/hive/recovery.py, overrides the model's command for that car.

Slow throttles on a real car (the recovery asks for 0.19 to 0.30 of full command):
* `min_throttle`: a recovery throttle above the deadband but under this is raised to it (a
  motor that does not turn below some duty: find it with the dashboard's MANUAL DRIVE).
* `binary_drive`: the firmware's `drive binary` (full on at |throttle| >= 1/3, nothing
  below). The recovery throttle becomes pulses: BINARY_PULSE on a share of ticks equal to
  the throttle asked for (first-order sigma-delta), zero on the others.
Both apply to recovery commands only; the model's own commands are never changed.

Terms: metres, seconds, m/s; throttle and steer in [-1, 1], + forward, + left; a tick is one
control step (0.1 s). "Past the tape": the car's centre outside the square |x|, |y| <= half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# rc16/hive/recovery.py's numbers (the 3-car models were trained with them), except the two marked
HISTORY_S = 2.0             # s of sent commands remembered per car (seen ticks only). 1.0 there: this layer
                            # also serves cars with no limiter, which leave faster and further
DRIVE_IN_MPS = 0.50         # m/s: speed limit while driving back in or replaying
DRIVE_IN_TARGET_MPS = 0.40  # m/s: speed asked for while driving back in
STOPPED_MPS = 0.15          # m/s: under this the car counts as stopped
GIVE_UP_S = 4.5             # s blind (while recovering) after which the car is stopped for good. 3.0 there.
                            # A car leaving at 2.7 m/s is about 1.7 m out when it has stopped (0.3 s of
                            # latency + a 4 m/s^2 brake); backing up 1.7 m at 0.5 m/s takes 3.4 s
DEAD_RECKON_MAX_S = 1.0     # s: longest time an unseen car's position is extrapolated
REPLAY_HOLD_MAX = 4         # ticks: longest one remembered tick is held during the replay
REPLAY_EXTRA_S = 1.0        # s the replay's last direction is kept, wheels straight, once the history has run
                            # out and the car is still unseen (the slow replay covers a little less ground
                            # than the car did at speed: simulated, it stopped 1 cm short of the cameras' view)
BRAKE_THROTTLE = 0.75       # blind-brake throttle: the remembered speed in m/s, clamped to [0.25, this]
BRAKE_PLAN_MPS2 = 4.0       # m/s^2: deceleration assumed for the blind brake
LEAD_FLIP_COS = 0.34        # the lead end changes only when the centre is > 110 deg off it

# this layer's own (a one-car model has no shield and may work right at the tape)
ENTER_OUTSIDE_M = 0.05      # m past the tape (seen) at which recovery takes over
REJOIN_INSIDE_M = 0.20      # m inside the tape (both axes, seen) at which the model gets the car back
BLIND_AFTER_S = 0.30        # s unseen, anywhere, after which the blind recovery starts
BLIND_STALE_S = 1.0         # s unseen after which a car its runtime is NOT recovering is stopped
EVADER_BLIND_S = 3.0        # s unseen (perception's own hold included) up to which a runtime that
                            # handles unseen cars may go on without the evader
BINARY_PULSE = 0.40         # throttle sent on an "on" tick in binary-drive mode (>= the 1/3 cut)
FRESH_S = 0.12              # s: a pose captured longer ago than this (against the frame's newest) is not "seen"
GUARD_LATENCY_S = 0.30      # s the car travels on before a brake bites: up to one control tick until the
                            # next command, the radio, the motor (simulated: with 0.15 s a car that had the
                            # room to stop still ended 0.35 m past the tape under a one-tick command delay)
GUARD_PAD_M = 0.02          # m kept in hand by the edge brake
GUARD_MIN_MPS = 0.30        # m/s: the edge brake leaves a car slower than this alone
GUARD_CREEP = 0.12          # the throttle toward the tape is never capped under this (about 0.3 m/s)
COAST_MPS2 = 0.8            # m/s^2 a coasting car loses (dead reckoning only)


def fresh_seen(pursuers, evader, meta: dict, fresh_s: float = FRESH_S) -> tuple[list[bool] | None, list[float] | None]:
    """(seen, lost_s) for DeadZone.apply from a parsed pose frame (pose_stream.parse_pose_frame_meta):
    one entry per pursuer, then the evader. A vehicle counts as seen only if perception says so AND
    its pose was captured within `fresh_s` of the newest capture in the frame. The evader is exempt
    from the freshness rule (it is slow or parked; perception holds it longer on purpose).
    (None, None) for a frame without the fields (an older perception): everything is seen."""
    seen = meta.get("seen")
    if seen is None:
        return None, None
    lost = list(meta.get("lost_s") or [0.0] * len(seen))
    t = meta.get("t")
    out = [bool(s) for s in seen]
    if t is not None:
        for i, pose in enumerate(pursuers):
            age = float(t) - pose.timestamp_s
            if out[i] and age > fresh_s:
                out[i] = False
            if not out[i]:
                lost[i] = max(float(lost[i]), age)
    return out, [float(x) for x in lost]


@dataclass(frozen=True)
class CarSpec:
    """What the recovery needs to know about the arena and the car."""

    half_m: float                   # half the taped square's side
    top_speed_mps: float            # the speed a throttle of 1.0 reaches
    max_steer_rad: float            # full steering lock
    wheelbase_m: float = 0.170
    deadband: float = 0.05          # |throttle| under this does nothing

    @classmethod
    def from_manifest(cls, manifest: dict) -> "CarSpec | None":
        """From a model manifest, or None when it does not say enough (older models)."""
        obs, arena, car = manifest.get("observation") or {}, manifest.get("arena") or {}, manifest.get("car") or {}
        half = arena.get("half_w") or obs.get("arena_half_m")
        top = obs.get("top_speed_mps") or car.get("sim_capped_top_speed_mps") or car.get("max_speed_mps")
        lock = obs.get("max_steer_rad") or (math.radians(car["max_steer_deg"]) if car.get("max_steer_deg") else None)
        if not (half and top and lock):
            return None
        return cls(float(half), float(top), float(lock), float(car.get("wheelbase_m", 0.170)),
                   float(obs.get("deadband", 0.05)))


class EdgeRecovery:
    """Out-of-view / past-the-tape recovery for ONE car (see the module docstring)."""

    def __init__(self, spec: CarSpec, dt: float = 0.1, guard: bool = True) -> None:
        self.spec = spec
        self.dt = float(dt)
        self.guard = bool(guard)
        self.guarding = False           # the edge brake replaced the model's throttle on the last step
        self.steps = max(1, int(round(HISTORY_S / self.dt)))
        self.low_throttle = DRIVE_IN_MPS / spec.top_speed_mps
        self.reset()

    def reset(self) -> None:
        self.recovering = False
        self.gave_up = False
        self.blind = False              # recovering and not seen on the last step
        self.stopped = False            # the seen procedure has braked to a stop once
        self.lead = 0                   # the end that goes in first: +1 nose, -1 tail, 0 not chosen
        self.last = None                # (x, y, yaw, v) when last seen
        self.dr = None                  # dead-reckoned (x, y, v) since then (== last while seen)
        self.t_unseen = 0.0
        self.t_blind = 0.0
        self.brake_left = 0
        self.rp_idx = 0
        self.rp_hold = 0
        self.rp_dir = 0.0               # the direction (+1 / -1 throttle sign) the replay last drove in
        self.rp_extra = 0               # ticks of REPLAY_EXTRA_S left
        self.hist: list[tuple[float, float, float]] = []    # newest first: (throttle, steer, v)

    # -- the two procedures -------------------------------------------------------------------
    def _seen_cmd(self, x: float, y: float, yaw: float, v: float) -> tuple[float, float]:
        sp = self.spec
        if not self.stopped and abs(v) < STOPPED_MPS:
            self.stopped = True
        if not self.stopped:                                   # brake against the motion
            return -math.copysign(min(max(abs(v), 0.25), 0.75), v), 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = c * -x + s * -y, -s * -x + c * -y             # the arena centre in the car's frame
        face = 1 if bx >= 0.0 else -1                          # the end facing the centre now
        if self.lead == 0 or self.lead * bx < -LEAD_FLIP_COS * math.hypot(bx, by):
            self.lead = face
        lead = self.lead
        kappa = 2.0 * by / max(bx * bx + by * by, 1e-6)        # pure pursuit of the centre
        delta = max(-sp.max_steer_rad, min(sp.max_steer_rad, math.atan(sp.wheelbase_m * kappa)))
        if lead * bx < 0.0:                                    # the centre is behind the lead end: full lock
            delta = math.copysign(sp.max_steer_rad, by if by != 0.0 else 1.0)
        along = lead * v                                       # speed along the chosen direction
        if along > DRIVE_IN_MPS:
            thr = -lead * 0.25
        elif along < -0.05:
            thr = lead * 0.25
        else:
            thr = lead * min(max((2.0 * DRIVE_IN_TARGET_MPS - along) / sp.top_speed_mps, 0.0), 0.30)
        # Reversing, the same wheel angle turns the car the other way round the same circle:
        # the pure-pursuit curvature already accounts for it (the circle through the centre).
        return thr, delta / sp.max_steer_rad

    def _blind_cmd(self) -> tuple[float, float]:
        v = self.last[3]
        if self.brake_left > 0:
            # As hard as the speed still remembered: a fixed hard brake held for the planned ticks
            # threw the (simulated) car back the other way at 1 m/s once it had stopped.
            self.brake_left -= 1
            return -math.copysign(min(max(abs(self.dr[2]), 0.25), BRAKE_THROTTLE), v), 0.0
        if self.rp_idx >= len(self.hist):                      # the history ran out
            if self.rp_extra > 0 and self.rp_dir != 0.0:
                self.rp_extra -= 1
                return self.rp_dir * self.low_throttle, 0.0
            return 0.0, 0.0                                    # wait to be seen
        h_thr, h_steer, h_v = self.hist[self.rp_idx]
        if abs(h_v) > 0.05:
            moved = math.copysign(1.0, h_v)
        elif abs(h_thr) >= self.spec.deadband:
            moved = math.copysign(1.0, h_thr)
        else:
            moved = 0.0
        need = min(max(int(math.ceil(abs(h_v) / DRIVE_IN_MPS)), 1), REPLAY_HOLD_MAX)
        self.rp_hold += 1
        if self.rp_hold >= need:
            self.rp_idx += 1
            self.rp_hold = 0
        if moved != 0.0:
            self.rp_dir = -moved
        return -moved * self.low_throttle, h_steer

    def _room(self, x: float, y: float, yaw: float, d: float) -> float:
        """Metres to the tape straight along the nose (d = +1) or the tail (d = -1), less the pad."""
        hx, hy, half = d * math.cos(yaw), d * math.sin(yaw), self.spec.half_m
        rx = (math.copysign(half, hx) - x) / hx if abs(hx) > 1e-9 else math.inf
        ry = (math.copysign(half, hy) - y) / hy if abs(hy) > 1e-9 else math.inf
        return max(0.0, min(rx, ry)) - GUARD_PAD_M

    def _guard_cmd(self, x: float, y: float, yaw: float, v: float, cmd) -> tuple[float, float]:
        """The edge brake: `cmd` with its throttle limited so the car can stop before the tape."""
        self.guarding = False
        thr, steer = cmd
        if not self.guard:
            return cmd
        a, lat = BRAKE_PLAN_MPS2, GUARD_LATENCY_S
        if abs(v) >= GUARD_MIN_MPS:                       # moving: can it still stop?
            d = math.copysign(1.0, v)
            stop = abs(v) * lat + v * v / (2.0 * a)
            brake = min(max(abs(v), 0.25), 1.0)
            if stop > self._room(x, y, yaw, d) and thr * d > -brake:
                self.guarding = True
                return -d * brake, steer
        if abs(thr) >= self.spec.deadband:                # cap the throttle toward the tape
            d = math.copysign(1.0, thr)
            room = max(self._room(x, y, yaw, d), 0.0)
            safe = -a * lat + math.sqrt(a * a * lat * lat + 2.0 * a * room)
            cap = max(safe / self.spec.top_speed_mps, GUARD_CREEP)
            if abs(thr) > cap:
                self.guarding = True
                return d * cap, steer
        return cmd

    # -- one control tick ---------------------------------------------------------------------
    def step(self, x: float, y: float, yaw: float, v: float, seen: bool, cmd_in,
             unseen_s: float | None = None) -> tuple[float, float]:
        """The command to send this tick: `cmd_in` (throttle, steer) unless the car is recovering.
        When not seen, x / y / yaw / v are ignored (the last seen values are used). unseen_s:
        how long the cameras have not seen it, when the caller knows (the pose stream's lost_s,
        which includes perception's own hold time); otherwise counted here in ticks."""
        dt, half = self.dt, self.spec.half_m
        if seen or self.last is None:
            self.last = (float(x), float(y), float(yaw), float(v))
            self.dr = [float(x), float(y), float(v)]
        if seen:
            self.t_unseen = 0.0
        else:
            # Dead reckoning: straight along the last seen heading at the remembered speed, which
            # the commands sent since then have been running down (see the end of this method).
            first = self.t_unseen == 0.0
            self.t_unseen = max(self.t_unseen + dt, float(unseen_s or 0.0))
            if self.t_unseen <= DEAD_RECKON_MAX_S:
                gap = self.t_unseen if first else dt       # the first unseen tick covers the pose's whole age
                self.dr[0] += self.dr[2] * gap * math.cos(self.last[2])
                self.dr[1] += self.dr[2] * gap * math.sin(self.last[2])
        lx, ly, lyaw, lv = self.last
        ex, ey, ev = self.dr
        outside = math.hypot(max(abs(ex) - half, 0.0), max(abs(ey) - half, 0.0))

        enter = not self.recovering and (
            outside > (ENTER_OUTSIDE_M if seen else 0.0)
            or (not seen and self.t_unseen >= BLIND_AFTER_S - 0.5 * dt))
        inside = (half - max(abs(lx), abs(ly))) >= REJOIN_INSIDE_M
        if enter:
            self.recovering, self.stopped, self.lead = True, False, 0
        elif self.recovering and seen and inside:
            self.recovering, self.stopped, self.lead = False, False, 0

        blind = self.recovering and not seen
        if blind and (not self.blind or enter):                # the first blind tick of this recovery
            ticks = int(math.ceil(abs(ev) / (BRAKE_PLAN_MPS2 * dt) - 1e-6)) if abs(ev) > 0.1 else 0
            self.brake_left = min(max(ticks, 0), 8)
            self.rp_idx = self.rp_hold = 0
            self.rp_dir, self.rp_extra = 0.0, int(round(REPLAY_EXTRA_S / dt))
            self.t_blind = 0.0
        self.blind = blind
        self.gave_up = blind and self.t_blind >= GIVE_UP_S - 0.5 * dt

        if not self.recovering:
            # Not yet lost for good (or simply seen): the model drives, under the edge brake at
            # the dead-reckoned position.
            cmd = self._guard_cmd(ex, ey, lyaw, ev, (float(cmd_in[0]), float(cmd_in[1])))
        elif seen:
            cmd = self._seen_cmd(lx, ly, lyaw, lv)
        elif self.gave_up:
            cmd = (0.0, 0.0)
        else:
            cmd = self._blind_cmd()
        self.t_blind = self.t_blind + dt if blind else 0.0
        if not seen:                                           # what this command does to the remembered speed
            if abs(ev) > 1e-6 and cmd[0] * ev < 0.0 and abs(cmd[0]) >= self.spec.deadband:
                self.dr[2] = math.copysign(max(abs(ev) - BRAKE_PLAN_MPS2 * dt, 0.0), ev)      # braking
            elif abs(cmd[0]) < self.spec.deadband:
                self.dr[2] = math.copysign(max(abs(ev) - COAST_MPS2 * dt, 0.0), ev)           # coasting
        if seen:                                               # history: seen ticks only
            self.hist.insert(0, (cmd[0], cmd[1], lv))
            del self.hist[self.steps:]
        return cmd


class _Speed:
    """Signed forward speed from successive SEEN poses (loops.SignedSpeed, kept local so this
    module imports nothing from the loop)."""

    def __init__(self) -> None:
        self.prev, self.v = None, 0.0

    def update(self, pose) -> float:
        if self.prev is not None:
            dt = pose.timestamp_s - self.prev.timestamp_s
            if dt > 0.35:
                # Seen again after a gap: the straight line between the two sightings says nothing
                # about how fast it is moving now. Unknown reads as standing; the next pose fixes it.
                self.v = 0.0
            elif dt > 1e-6:
                self.v = ((pose.x - self.prev.x) * math.cos(pose.heading)
                          + (pose.y - self.prev.y) * math.sin(pose.heading)) / dt
        self.prev = pose
        return self.v


class DeadZone:
    """Per-tick supervisor of unseen and out-of-bounds cars for the real-time loop.

    spec None: no recovery is possible (an older model whose manifest lacks the car's numbers);
    an unseen car is then simply sent zero. runtime_handles_unseen: the model's runtime does its
    own recovery (it was given `seen`); `runtime_status` then returns its (recovering, gave_up)
    lists and this class only supervises."""

    def __init__(self, n: int, *, spec: CarSpec | None, runtime_handles_unseen: bool = False,
                 runtime_status=None, dt: float = 0.1, min_throttle: float = 0.0,
                 binary_drive: bool = False, guard: bool = True) -> None:
        self.n = int(n)
        self.spec = spec
        self.runtime_handles_unseen = bool(runtime_handles_unseen)
        self.runtime_status = runtime_status
        self.dt = float(dt)
        self.min_throttle = max(0.0, float(min_throttle))
        self.binary_drive = bool(binary_drive)
        self.guard = bool(guard)
        self.recoveries = ([EdgeRecovery(spec, dt, guard=guard) for _ in range(self.n)]
                           if spec is not None and not self.runtime_handles_unseen else None)
        self.reset()

    def set_arena_half(self, half_m: float) -> bool:
        """Use the taped square perception was calibrated on (its half side, m) as "the tape" of the
        controller-side recovery and edge brake, in place of the model manifest's. True if it changed.
        A runtime that recovers its own cars keeps the arena it was trained on: nothing to set."""
        if self.recoveries is None or not (0.3 <= float(half_m) <= 5.0):
            return False
        if abs(float(half_m) - self.spec.half_m) < 0.02:
            return False
        self.spec = replace(self.spec, half_m=float(half_m))
        for r in self.recoveries:
            r.spec = self.spec
        return True

    @property
    def mode(self) -> str:
        if self.runtime_handles_unseen:
            return "runtime"
        return "controller" if self.recoveries is not None else "stop-only"

    def reset(self) -> None:
        """After a pose stall or a re-arm: nothing is remembered."""
        n = self.n
        self.speeds = [_Speed() for _ in range(n)]
        self.unseen_s = [0.0] * n
        self.recovering = [False] * n
        self.gave_up = [False] * n
        self.blind = [False] * n
        self.guarding = [False] * n             # the edge brake is overriding the model's throttle
        self.evader_unseen_s = 0.0
        self.evader_hold = False
        self._sd = [0.0] * n                    # sigma-delta remainder per car (binary drive)
        for r in self.recoveries or []:
            r.reset()

    def _shape(self, i: int, thr: float) -> float:
        """A recovery throttle as this car's drive can execute it (see the module docstring)."""
        dead = self.spec.deadband if self.spec is not None else 0.05
        if abs(thr) < dead:
            return 0.0
        if self.binary_drive:
            self._sd[i] += min(abs(thr), 1.0)
            if self._sd[i] >= 1.0 - 1e-9:
                self._sd[i] -= 1.0
                return math.copysign(BINARY_PULSE, thr)
            return 0.0
        if abs(thr) < self.min_throttle:
            return math.copysign(self.min_throttle, thr)
        return thr

    def apply(self, action, pursuers, evader, seen, lost_s=None, log=None):
        """`action` (throttle, steer per pursuer) as it should be SENT. `seen`: one bool per
        pursuer, then the evader (None = everything seen); `lost_s`: the pose stream's seconds
        since each was last seen, in the same order. Returns `action` itself when nothing is
        overridden, otherwise a copy."""
        n = self.n
        seen = [True] * (n + 1) if seen is None else [bool(s) for s in seen]
        lost_s = [0.0] * (n + 1) if lost_s is None else [float(x) for x in lost_s]
        out = action
        rt_rec, rt_gave = (self.runtime_status() if self.runtime_handles_unseen and self.runtime_status
                           else ([False] * n, [False] * n))

        def put(i: int, thr: float, steer: float) -> None:
            nonlocal out
            if out is action:
                out = np.array(action, dtype=np.float64)
            out[2 * i], out[2 * i + 1] = thr, steer

        for i, pose in enumerate(pursuers):
            was = (self.recovering[i], self.gave_up[i], self.blind[i])
            self.unseen_s[i] = 0.0 if seen[i] else max(self.unseen_s[i] + self.dt, lost_s[i])
            if self.recoveries is not None:
                r = self.recoveries[i]
                v = self.speeds[i].update(pose) if seen[i] else self.speeds[i].v
                thr, steer = r.step(pose.x, pose.y, pose.heading, v, seen[i],
                                    (action[2 * i], action[2 * i + 1]), unseen_s=lost_s[i])
                self.recovering[i], self.gave_up[i], self.blind[i] = r.recovering, r.gave_up, r.blind
                self.guarding[i] = r.guarding and not r.recovering
                if r.recovering:
                    put(i, self._shape(i, thr), steer)
                elif r.guarding:
                    put(i, thr, steer)
            elif self.runtime_handles_unseen:
                rec, gave = bool(rt_rec[i]), bool(rt_gave[i])
                stale = not seen[i] and not rec and self.unseen_s[i] >= BLIND_STALE_S
                self.recovering[i] = rec
                self.blind[i] = rec and not seen[i]
                self.gave_up[i] = gave or (not seen[i] and self.unseen_s[i] >= GIVE_UP_S + BLIND_STALE_S)
                if stale or self.gave_up[i]:
                    put(i, 0.0, 0.0)
                elif rec:
                    put(i, self._shape(i, float(action[2 * i])), float(action[2 * i + 1]))
            else:
                self.recovering[i] = self.blind[i] = False
                self.gave_up[i] = not seen[i] and self.unseen_s[i] >= GIVE_UP_S
                if not seen[i]:
                    put(i, 0.0, 0.0)
            if log is not None and (self.recovering[i], self.gave_up[i], self.blind[i]) != was:
                where = f"last seen ({pose.x:+.2f}, {pose.y:+.2f})"
                if self.gave_up[i]:
                    log(f"car {i} LOST: not seen for {self.unseen_s[i]:.1f} s, {where}: stopped")
                elif self.blind[i]:
                    log(f"car {i} OUT OF VIEW, {where}: blind recovery (brake, then back along its own track)")
                elif self.recovering[i]:
                    log(f"car {i} PAST THE TAPE at ({pose.x:+.2f}, {pose.y:+.2f}): driving back in")
                elif was[0] or was[1]:
                    log(f"car {i} BACK at ({pose.x:+.2f}, {pose.y:+.2f}): the model has it again")

        # The evader. Without it there is nothing to chase: every car stands still, armed.
        self.evader_unseen_s = 0.0 if seen[n] else max(self.evader_unseen_s + self.dt, lost_s[n])
        grace = EVADER_BLIND_S if self.runtime_handles_unseen else 0.0
        hold = (not seen[n]) and self.evader_unseen_s > grace
        if hold != self.evader_hold:
            self.evader_hold = hold
            if log is not None:
                log(f"EVADER OUT OF VIEW, last seen ({evader.x:+.2f}, {evader.y:+.2f}): cars standing still, "
                    f"still armed" if hold else "EVADER BACK: driving again")
        if hold:
            for i in range(n):
                if not self.recovering[i]:
                    put(i, 0.0, 0.0)
        return out

    def status(self) -> dict:
        """For telemetry: one entry per pursuer."""
        return {"mode": self.mode,
                "unseen_s": [round(u, 2) for u in self.unseen_s],
                "recovering": list(self.recovering), "blind": list(self.blind),
                "gave_up": list(self.gave_up), "edge_brake": list(self.guarding),
                "evader_unseen_s": round(self.evader_unseen_s, 2), "evader_hold": self.evader_hold}
