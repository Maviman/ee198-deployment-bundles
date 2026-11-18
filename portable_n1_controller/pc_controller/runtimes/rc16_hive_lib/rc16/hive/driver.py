"""Pawn drivers (rc16/hive/CONTRACT.md, "driver.py"): take each pursuer to its slot. Hand-built, not learned.

Pure torch, batched (N envs x 3 pawns), stateless: the command depends only on this tick's inputs. A pawn uses
its own pose and speed, its own slot, and the evader's pose. It uses NOTHING about the other pawns.

`slots` is duck-typed (plays.Slots is not imported): any object with
    .xy (N,3,2) m      the target point of each pawn
    .yaw (N,3) rad     the heading wanted at that point
    .speed (N,3) m/s   the speed wanted on arrival (0 = stop there)

How a pawn drives (metres, m/s, radians):
- Gear. Forward, except: REVERSE when the slot is a stop slot (speed < 0.05), is behind the car (negative
  along-nose offset), within REVERSE_RANGE_M, and the slot heading is within 90 deg of the car heading.
  The gear flips only once the slot is GEAR_BAND_M (plus SHUFFLE_PER_M x the side offset from the slot axis)
  behind or ahead, and the current speed sign is used as the memory, so a car that arrives beside the slot or
  crooked shuffles past it and back onto the axis instead of stopping crooked.
- Approach axis. A line through the slot. If the car is within CONE_IN_RAD (45 deg) of straight-behind the slot,
  or nearer than AXIS_NEAR_M, it is the slot axis (the line along the slot heading), so the car arrives with the
  slot heading. If the car comes from more than CONE_OUT_RAD (75 deg) off, it is the straight line from the car
  to the slot: position first, the heading is then corrected by the shuffle above. Blended in between.
- Steering: pure pursuit of the AIM POINT on the approach axis LOOKAHEAD_M ahead of the car (behind it in
  reverse), kept AIM_INSET_M inside the tape. The same formula serves both gears (the car follows the same
  circle).
- Three-point turn. If the room ahead to the tape (minus TAPE_INSET_M) is under KTURN_ROOM_M and the aim point is
  more than 30 deg off the nose, the car backs up with opposite lock at KTURN_MPS until the aim point is within
  45 deg or the room behind runs out.
- Turn on the slot. A stop slot nearer than ON_SLOT_M whose heading is more than ON_SLOT_RAD (35 deg) off the
  car's is not approached any more: the car turns on the spot, to and fro across the slot on full lock (forward
  with the lock the way it has to turn, back with the opposite lock, each leg until it is ON_SLOT_FLIP_M past
  the slot) at ON_SLOT_MPS, until the heading is within 35 deg; the approach above then finishes the job.
  (Review 2026-11-17: without this a car that reached its slot facing about 90 deg off drove full-lock circles
  through it for ever: 1.5 % of random start/slot poses never settled; with it all did.)
- Room. The driver uses the SAME room as the shield (shield.rooms on the tape minus TAPE_INSET_M, straight
  ray): the distance along the nose (forward) or the tail (reverse) to that line, less the shield's pad; 0 for a
  car past the line and pointing further out; and, for a car past the line, at least the shield's crawl room in
  the direction that brings it back. So the driver never asks for a move the shield will refuse. `turn_radius`
  is handed to shield.rooms: leave it None for the straight ray; if the shield is run with a turn radius, give
  the driver the same one.
- Pinned at the line. With less than PINNED_M of room ahead the car cannot usefully drive forward, whatever the
  aim: it backs up with opposite lock until it has UNPIN_M ahead. Reverse gear and the three-point turn are not
  used with less than PINNED_M of room behind: the car then goes forward on full lock instead. With less than
  PINNED_M both ways (wedged across a corner of the line) it creeps at CREEP_MPS toward the roomier end.
  (Review 2026-11-17: without this a car on or over the line stood still for ever when its wanted gear pointed
  outward: 10 % of starts in the last 0.07 m inside the tape, 25 % of starts beyond it.)
- Speed wanted = the smallest of: V_MAX_DRIVER; the speed from which it can slow to the slot speed in the
  remaining distance with BRAKE_PLAN and 0.15 s latency (shield.speed_for_room); slot speed + K_NEAR x distance
  (a gentle final approach); sqrt(LAT_PLAN / |curvature|) for the commanded steering (v^2 / r <= 4.0); and the
  ramming limit below; not less than CREEP_MPS until it has arrived; then never above the speed from which it can
  stop before the tape (minus TAPE_INSET_M) in its direction of travel. 0 once a stop slot is within ARRIVE_M and
  ARRIVE_RAD.
- Throttle: this car only COASTS at zero throttle, so when it is more than BRAKE_OVER_MPS too fast (or moving the
  wrong way) the throttle is REVERSED against the motion; otherwise speed/TOP_SPEED plus a proportional term.
  |throttle| never exceeds V_MAX_DRIVER / TOP_SPEED.
- Ramming rule. "Box gap" = geometry.obb_separation of the pawn box and the evader box. "Closing speed" = the
  component of the pawn velocity (its own, along its heading) on the line from its centre to the evader centre;
  the driver is not given the evader velocity. While the gap is under CONTACT_GAP_M the wanted closing speed is
  CLOSE_TARGET_MPS (below CONTACT_SPEED to leave room for tracking error); further out it is the speed from which
  the pawn can still slow to that by CONTACT_GAP_M + CLOSE_MARGIN_M. Because a car on full lock swings its nose
  20 deg per tick at 1 m/s, the closing speed is judged on the worst direction of travel it will have while it
  slows down (the present one swept over its stopping distance at the commanded steering), not only the
  present one. (Review 2026-11-17: judged on the present direction only, a pawn circling beside the evader
  reached 0.50 m/s of closing speed inside the 0.25 m gap before the limit bit.)

Added 2026-11-17 after the hold forensics (artifacts/rc16/diag/hive_holds/squeeze_static: against a PARKED
evader SQUEEZE won only 74 %; in the rest the pawns stood pressed against the evader or each other, a median
0.17 m short of their slots, still sending 0.2 throttle). All three need `state` (a DriverState); with
state=None the driver is the stateless one described above.
- Detour. A pawn does not drive THROUGH the evader. When the straight line from the pawn to its slot crosses
  the evader's box grown by ORBIT_GROW_M on every side (a pawn's half width + 0.02 m; the SQUEEZE slots lie just
  outside that box) and the slot lies more than ORBIT_RELEASE_RAD round the evader from
  the pawn, the pawn is sent to a VIA POINT instead: on the circle of radius ORBIT_R_M about the evader,
  ORBIT_STEP_RAD further round toward the slot, pulled in to ORBIT_INSET_M inside the tape if it falls beyond
  that (so next to the tape the pawn slides in along it), heading along the circle, at ORBIT_MPS. The short way
  round is taken unless its via point then lies inside the grown evader box (no room between the evader and the
  tape) and the other way's does not; if both do, the pawn goes straight for the slot. So it circles the evader and comes onto its
  slot from the side. The SQUEEZE slots all lie along such a circle, so the pawn arrives with the slot heading.
- Arrival latch. A pawn that has reached a stop slot (ARRIVE_M, ARRIVE_RAD) stays "arrived" (throttle 0) until
  it is more than LEAVE_M or LEAVE_RAD off, instead of creeping to and fro about the 0.05 m arrival circle.
- Hold. Where slots.hold is True (plays.py, the SQUEEZE hold) the pawn is treated as arrived: it stops where
  it is, whatever its distance to the slot point, and is never sent on a detour.
  It stops ACTIVELY: this car only coasts at zero throttle (0.06 m of roll from 0.3 m/s), so while it still
  moves faster than HOLD_BRAKE_MIN_MPS it is braked with reverse throttle (HOLD_BRAKE_GAIN x speed).
- Unjam. A pawn that has sent |throttle| >= STUCK_THROTTLE with the SAME SIGN on each of the last STUCK_TICKS
  ticks (a to-and-fro shuffle is not a jam) and moved less than STUCK_MOVE_M over them is STUCK (against the evader or a teammate it cannot see). It backs off: the
  opposite gear, wheels straight, BACKOFF_THROTTLE, for BACKOFF_TICKS + BACKOFF_PER_JOB x its slot job (so two
  pawns jammed on each other back off for different times: by job, never by car number).
"""

from __future__ import annotations

import math

import torch

from rc16 import specs as S
from rc16.hive import geometry as G
from rc16.hive.shield import rooms, speed_for_room

T_MAX = G.V_MAX_DRIVER / S.TOP_SPEED_MPS    # the largest |throttle| the driver ever sends
LAT_PLAN = 4.0              # m/s^2: lateral acceleration allowed when slowing for a turn (limit is 5.2)
REVERSE_RANGE_M = 0.60      # m: a stop slot behind the car and nearer than this is reached in reverse
STOP_SLOT_SPEED = 0.05      # m/s: a slot speed below this means "stop there"
ARRIVE_M = 0.05             # m: a stop slot this close, and ...
ARRIVE_RAD = math.radians(12.0)   # ... with the heading this close, counts as reached (the driver asks for 0 m/s)
LOOKAHEAD_M = 0.30          # m: pursuit distance: the aim point is this far ahead of the car along the approach axis
GEAR_BAND_M = 0.03          # m: the slot must be this far behind (ahead) before the gear flips to reverse (forward)
SHUFFLE_PER_M = 2.5         # that band grows by this many m per m of side offset from the slot axis, so a car
SHUFFLE_MAX_M = 0.30        # that arrives beside the slot runs past it far enough to steer onto the axis (max band)
CONE_IN_RAD = math.radians(45.0)    # car within this angle of straight-behind the slot: approach on the slot axis
CONE_OUT_RAD = math.radians(75.0)   # car beyond this angle: approach on the straight line to the slot
AXIS_NEAR_M = 0.30          # m: nearer than this the slot axis is always used ...
AXIS_FAR_M = 0.45           # m: ... blended in from this distance
TAPE_INSET_M = 0.05         # m: the driver plans its own stops against the tape minus this (the shield line)
KTURN_ROOM_M = 0.30         # m: room ahead under which a car that must turn round backs up instead
KTURN_MPS = 0.5             # m/s: speed while backing up in a three-point turn
ON_SLOT_M = 0.30            # m: a stop slot nearer than this with the heading more than ON_SLOT_RAD off is not
ON_SLOT_RAD = math.radians(35.0)    # approached any more: the car turns on the spot, to and fro on full lock
ON_SLOT_FLIP_M = 0.16       # m: each leg of that turn runs until the car is this far past the slot
ON_SLOT_MPS = 0.40          # m/s: speed of those legs
PINNED_M = 0.06             # m: with less room than this in a direction the car cannot usefully drive that way
UNPIN_M = 0.15              # m: a car backing away from the line it was pinned at keeps backing to this room ahead
AIM_INSET_M = 0.10          # m: the aim point is kept this far inside the tape
K_NEAR = 2.5                # 1/s: final-approach speed per metre of remaining distance
CREEP_MPS = 0.15            # m/s: slowest speed asked for while not yet arrived (above the command deadband)
KP_SPEED = 1.0              # proportional speed gain (throttle = (v_des + KP * (v_des - v)) / TOP_SPEED)
BRAKE_OVER_MPS = 0.12       # m/s: reverse-brake when the car is this much faster than wanted ...
BRAKE_MIN_SPEED = 0.25      # m/s: ... and at least this fast (slower than this it just coasts down)
BRAKE_THROTTLE_MIN = 0.25   # weakest brake command (throttle units)
CONTACT_GAP_M = 0.25        # m: box gap to the evader under which the closing speed is limited
CLOSE_MARGIN_M = 0.10       # m: the closing speed is already down to the target this far before CONTACT_GAP_M
CLOSE_TARGET_MPS = 0.7 * G.CONTACT_SPEED   # m/s: closing speed asked for near the evader
CLOSE_BRAKE = 0.6 * G.BRAKE_PLAN           # m/s^2: deceleration assumed when slowing for the evader
ORBIT_GROW_M = G.HALF_WIDTH_M + 0.02    # m: a straight run to the slot that crosses the evader box grown by this
#                             is a run through the evader (first version: nearer than 0.27 m to its centre, which
#                             every nose / tail slot is, so a pawn 60 deg round from its slot never went in)
ORBIT_R_M = 0.40            # m: radius of the circle about the evader on which the via points lie
ORBIT_STEP_RAD = math.radians(50.0)     # the via point is this much further round the evader than the pawn
ORBIT_RELEASE_RAD = math.radians(45.0)  # nearer than this (angle round the evader) the pawn goes for the slot itself
ORBIT_MPS = 0.6             # m/s: speed wanted at a via point
ORBIT_INSET_M = 0.08        # m: a via point is pulled in to at least this far inside the tape
LEAVE_M = 0.10              # m: an arrived pawn stays arrived until it is this far from its stop slot ...
LEAVE_RAD = math.radians(25.0)          # ... or this far off its heading
STUCK_TICKS = 5             # ticks (0.5 s) of pushing without moving that make a pawn "stuck"
STUCK_THROTTLE = 0.10       # |throttle| that counts as pushing
STUCK_MOVE_M = 0.02         # m: moved less than this over STUCK_TICKS = not moving (a creep covers 0.075 m)
BACKOFF_TICKS = 5           # ticks a stuck pawn backs off, plus ...
BACKOFF_PER_JOB = 3         # ... this many per slot job index (0, 1, 2)
BACKOFF_THROTTLE = 0.20     # |throttle| while backing off (settles near 0.5 m/s, the contact speed)
HOLD_BRAKE_FLOOR = 0.06     # weakest hold brake command (the car ignores |throttle| < 0.05)
HOLD_BRAKE_MIN_MPS = 0.12   # m/s: a holding pawn faster than this is braked with reverse throttle
HOLD_BRAKE_GAIN = 0.3       # throttle per m/s of that brake. One 0.1 s tick of throttle u changes the speed by about
#                             0.75 x u m/s, so this removes ~22 % of the speed per tick: stable with the 2 to 3 ticks of
#                             loop delay (1.5 oscillated at +-0.5 m/s, artifacts/rc16/diag/hive_holds/squeeze_static_drv4)


class DriverState:
    """Per-pawn memory of the driver, batch (N,3): the arrival latch and the unjam timer (module docstring)."""

    def __init__(self, n, device="cpu"):
        self.n, self.device = n, device
        self.arrived = torch.zeros(n, 3, dtype=torch.bool, device=device)
        self.hist_xy = torch.zeros(n, 3, STUCK_TICKS, 2, device=device)     # positions of the last ticks, oldest first
        self.pushing = torch.zeros(n, 3, dtype=torch.long, device=device)   # consecutive ticks of |throttle| >= STUCK_THROTTLE
        self.push_sign = torch.zeros(n, 3, device=device)                   # ... all with this sign
        self.backoff = torch.zeros(n, 3, dtype=torch.long, device=device)   # ticks of backing off still to run
        self.back_dir = torch.zeros(n, 3, device=device)                    # +1 / -1: the gear of the back-off
        self.unjams = torch.zeros(n, 3, dtype=torch.long, device=device)    # back-offs started (for the diagnostics)
        self.via = torch.zeros(n, 3, dtype=torch.bool, device=device)       # this tick the pawn was sent to a via point
        self.fresh = torch.ones(n, 3, dtype=torch.bool, device=device)      # no position seen yet since the reset

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.arrived[ids] = False
        self.hist_xy[ids] = 0.0
        self.pushing[ids] = 0
        self.push_sign[ids] = 0.0
        self.backoff[ids] = 0
        self.back_dir[ids] = 0.0
        self.unjams[ids] = 0
        self.via[ids] = False
        self.fresh[ids] = True


def detour(p_xy, s_xy, s_yaw, s_speed, e_xy, e_yaw):
    """Go round the evader, not through it: (xy, yaw, speed, used) with the slot replaced by a via point where
    `used` (N,3) is True. See the module docstring (Detour)."""
    e = e_xy.unsqueeze(-2)
    rp, rs = p_xy - e, s_xy - e
    # does the straight run pawn -> slot cross the grown evader box? (slab test in the evader body frame)
    yaw = e_yaw.unsqueeze(-1)
    a, b = G.body_frame(rp, yaw), G.body_frame(rs, yaw)
    half = torch.tensor([G.HALF_LENGTH_M + ORBIT_GROW_M, G.HALF_WIDTH_M + ORBIT_GROW_M], dtype=a.dtype, device=a.device)
    dseg = b - a
    par = dseg.abs() < 1e-6
    safe = torch.where(par, torch.ones_like(dseg), dseg)
    t1, t2 = (-half - a) / safe, (half - a) / safe
    lo = torch.where(par, torch.full_like(t1, -1e9), torch.minimum(t1, t2))
    hi = torch.where(par, torch.full_like(t1, 1e9), torch.maximum(t1, t2))
    miss = (par & (a.abs() > half)).any(-1)
    t_in, t_out = lo.max(-1).values, hi.min(-1).values
    through = ~miss & (t_out >= t_in.clamp(min=0.0)) & (t_in <= 1.0)
    ang_p = torch.atan2(rp[..., 1], rp[..., 0])
    dth = G.wrap(torch.atan2(rs[..., 1], rs[..., 0]) - ang_p)         # the short way round to the slot, + = CCW
    sgn = torch.where(dth >= 0.0, torch.ones_like(dth), -torch.ones_like(dth))
    lim = S.ARENA_HALF_M - ORBIT_INSET_M

    def via(sign):
        ang = ang_p + sign * ORBIT_STEP_RAD
        xy = (e + ORBIT_R_M * torch.stack([torch.cos(ang), torch.sin(ang)], -1)).clamp(-lim, lim)
        inside = (G.body_frame(xy - e, yaw).abs() < half).all(-1)     # pulled onto the evader: no way through
        return xy, ang + sign * 0.5 * math.pi, ~inside

    xy1, yaw1, ok1 = via(sgn)
    xy2, yaw2, ok2 = via(-sgn)
    first = ok1 | ~ok2
    v_xy = torch.where(first.unsqueeze(-1), xy1, xy2)
    v_yaw = torch.where(first, yaw1, yaw2)
    used = through & (dth.abs() > ORBIT_RELEASE_RAD) & (ok1 | ok2)
    return (torch.where(used.unsqueeze(-1), v_xy, s_xy), torch.where(used, v_yaw, s_yaw),
            torch.where(used, torch.full_like(s_speed, ORBIT_MPS), s_speed), used)


def speed_to_throttle(v_des, v):
    """Throttle in [-T_MAX, T_MAX] that takes the signed forward speed v (m/s) to the wanted v_des (m/s).

    Brakes by REVERSE throttle: when the car moves against v_des, or is more than BRAKE_OVER_MPS faster than
    |v_des| while above BRAKE_MIN_SPEED. Otherwise feed-forward v_des / TOP_SPEED plus a proportional term,
    never negative along the wanted direction (so a small overspeed coasts).
    """
    mag = v_des.abs()
    g = torch.where(mag > 1e-3, torch.sign(v_des), torch.zeros_like(v_des))
    speed = v.abs()
    push = g * ((mag + KP_SPEED * (mag - g * v)) / S.TOP_SPEED_MPS).clamp(0.0, T_MAX)
    over = speed - mag
    brake = -torch.sign(v) * (over / 0.5).clamp(BRAKE_THROTTLE_MIN, T_MAX)
    thr = torch.where((over > BRAKE_OVER_MPS) & (speed > BRAKE_MIN_SPEED), brake, push)
    against = (g * v) < -0.05
    thr = torch.where(against, g * speed.clamp(BRAKE_THROTTLE_MIN, T_MAX), thr)
    return thr


def pursuit_steer(aim_body, gear):
    """Steer command in [-1, 1] (+ = left) that puts the car on the circle through an aim point.

    aim_body (...,2): the aim point in the car body frame (x along the nose, y to the left), m.
    gear (...,): +1 forward, -1 reverse. The circle through the car, tangent to its heading, through the aim
    point has curvature 2 * y / (x^2 + y^2); the car follows that same circle in either gear. When the aim point
    is on the wrong side for the gear (behind in forward, ahead in reverse) the answer is full lock toward it.
    Returns (steer, curvature_cmd) with curvature_cmd (1/m) the curvature of the commanded steering angle.
    """
    ax, ay = aim_body[..., 0], aim_body[..., 1]
    kappa = 2.0 * ay / (ax * ax + ay * ay).clamp(min=1e-6)
    delta = torch.atan(S.WHEELBASE_M * kappa).clamp(-S.MAX_STEER_RAD, S.MAX_STEER_RAD)
    side = torch.where(ay >= 0.0, torch.ones_like(ay), -torch.ones_like(ay))
    delta = torch.where(gear * ax < 0.0, side * S.MAX_STEER_RAD, delta)
    return delta / S.MAX_STEER_RAD, torch.tan(delta) / S.WHEELBASE_M


def drive(p_xy, p_yaw, p_v, slots, e_xy, e_yaw, turn_radius=None, state=None):
    """Commands (N,3,2) = (throttle, steer) that take each pawn to its slot. See the module docstring.

    p_xy (N,3,2) m, p_yaw (N,3) rad, p_v (N,3) signed forward speed m/s; slots: .xy (N,3,2), .yaw (N,3),
    .speed (N,3) (duck-typed); e_xy (N,2) m, e_yaw (N,) rad: the evader pose. turn_radius: the value the shield
    is run with (None = its default, the straight ray), so that both count the room to the tape the same way.
    state: a DriverState (detour, arrival latch and unjam; uses slots.job if present) or None (the stateless
    driver as it was: straight to the slot, no latch, no unjam).
    """
    s_xy, s_yaw, s_speed = slots.xy, slots.yaw, slots.speed.clamp(min=0.0)
    slot_dist = (s_xy - p_xy).norm(dim=-1)
    slot_head_err = G.wrap(s_yaw - p_yaw).abs()
    slot_stop = s_speed < STOP_SLOT_SPEED
    hold = getattr(slots, "hold", None)
    if state is not None:
        d_xy, d_yaw, d_speed, via = detour(p_xy, s_xy, s_yaw, s_speed, e_xy, e_yaw)
        if hold is not None:
            via = via & ~hold
        s_xy = torch.where(via.unsqueeze(-1), d_xy, s_xy)
        s_yaw = torch.where(via, d_yaw, s_yaw)
        s_speed = torch.where(via, d_speed, s_speed)
        state.via = via
    rel = s_xy - p_xy
    dist = rel.norm(dim=-1)
    rel_b = G.body_frame(rel, p_yaw)
    bx = rel_b[..., 0]
    head_err = G.wrap(s_yaw - p_yaw).abs()
    stop_slot = s_speed < STOP_SLOT_SPEED

    # room (m) along the nose and along the tail, exactly as the shield counts it (see the module docstring)
    heading = torch.stack([torch.cos(p_yaw), torch.sin(p_yaw)], -1)
    room_f, room_r = rooms(p_xy, p_yaw, S.ARENA_HALF_M - TAPE_INSET_M, turn_radius)
    can_f, can_r = room_f >= PINNED_M, room_r >= PINNED_M

    # gear: reverse for a stop slot that is close behind and roughly along the car heading
    reversing = p_v < -0.05
    s_dir = torch.stack([torch.cos(s_yaw), torch.sin(s_yaw)], -1)
    cross = (rel[..., 0] * s_dir[..., 1] - rel[..., 1] * s_dir[..., 0]).abs()     # side offset from the slot axis, m
    band = (GEAR_BAND_M + SHUFFLE_PER_M * cross).clamp(max=SHUFFLE_MAX_M)
    bx_thr = torch.where(reversing, band, -band)
    rng = torch.where(reversing, torch.full_like(bx, 1.25 * REVERSE_RANGE_M), torch.full_like(bx, REVERSE_RANGE_M))
    ang = torch.where(reversing, torch.full_like(bx, math.radians(100.0)), torch.full_like(bx, math.radians(90.0)))
    reverse = stop_slot & (bx < bx_thr) & (dist < rng) & (head_err < ang) & can_r
    gear = torch.where(reverse, -torch.ones_like(bx), torch.ones_like(bx))

    # approach axis: the slot axis when the car comes from behind the slot or is close; the straight line from
    # the car to the slot when it comes from the side or the front (position first, heading fixed at the end)
    off = G.wrap(torch.atan2(rel[..., 1], rel[..., 0]) - s_yaw)       # 0 = the car is straight behind the slot
    w_ang = ((CONE_OUT_RAD - off.abs()) / (CONE_OUT_RAD - CONE_IN_RAD)).clamp(0.0, 1.0)
    w_near = ((AXIS_FAR_M - dist) / (AXIS_FAR_M - AXIS_NEAR_M)).clamp(0.0, 1.0)
    w = torch.where(reverse, torch.ones_like(w_ang), torch.maximum(w_ang, w_near))
    ax_yaw = s_yaw + (1.0 - w) * off
    ax_dir = torch.stack([torch.cos(ax_yaw), torch.sin(ax_yaw)], -1)
    along = -(rel * ax_dir).sum(-1)                                   # car position along the axis, 0 at the slot
    lim = S.ARENA_HALF_M - AIM_INSET_M
    aim = (s_xy + (along + gear * LOOKAHEAD_M).unsqueeze(-1) * ax_dir).clamp(-lim, lim)
    aim_b = G.body_frame(aim - p_xy, p_yaw)
    steer, kappa = pursuit_steer(aim_b, gear)

    # three-point turn: no room ahead to swing the nose round to the aim point -> back up with opposite lock
    bearing = torch.atan2(aim_b[..., 1], aim_b[..., 0])
    start_k = (room_f < KTURN_ROOM_M) & (bearing.abs() > math.radians(30.0))
    keep_k = reversing & (room_r > KTURN_ROOM_M) & (bearing.abs() > math.radians(45.0))
    # pinned at the line: no room ahead at all -> back away from it whatever the aim, until there is some.
    # Backing up needs room behind: without it the car goes forward on the pursuit steering instead.
    pinned = ~can_f & (room_r > room_f)
    keep_p = reversing & (room_f < UNPIN_M)
    kturn = (gear > 0) & (((start_k | keep_k | keep_p) & can_r) | pinned)
    k_side = torch.where(bearing >= 0.0, torch.ones_like(bearing), -torch.ones_like(bearing))
    steer = torch.where(kturn, -k_side, steer)
    kappa = torch.where(kturn, torch.full_like(kappa, 1.0 / S.MIN_TURN_RADIUS_M), kappa)
    gear = torch.where(kturn, -torch.ones_like(gear), gear)

    # turn on the slot: standing on a stop slot but facing the wrong way -> to and fro across it on full lock
    need_ccw = G.wrap(s_yaw - p_yaw) >= 0.0                           # the heading still has to turn left
    on_slot = stop_slot & (dist < ON_SLOT_M) & (head_err > ON_SLOT_RAD)
    one = torch.ones_like(bx)
    toward = torch.where(bx >= 0.0, one, -one)                        # the gear that takes the car to the slot
    moving_gear = torch.where(p_v > 0.05, one, torch.where(reversing, -one, toward))
    t_gear = torch.where(dist < ON_SLOT_FLIP_M, moving_gear, toward)  # keep going until this far past the slot
    t_room = torch.where(t_gear > 0, room_f, room_r)
    t_other = torch.where(t_gear > 0, room_r, room_f)
    t_gear = torch.where((t_room < PINNED_M) & (t_other > t_room), -t_gear, t_gear)
    t_steer = t_gear * torch.where(need_ccw, one, -one)               # forward: lock the way to turn; back: opposite
    steer = torch.where(on_slot, t_steer, steer)
    kappa = torch.where(on_slot, torch.full_like(kappa, 1.0 / S.MIN_TURN_RADIUS_M), kappa)
    gear = torch.where(on_slot, t_gear, gear)

    # speed plan
    v_plan = speed_for_room(dist, G.BRAKE_PLAN, v_end=s_speed)
    v_near = s_speed + K_NEAR * dist
    v_turn = torch.sqrt(LAT_PLAN / kappa.abs().clamp(min=1e-3))
    v_mag = torch.minimum(torch.minimum(v_plan, v_near), v_turn).clamp(max=G.V_MAX_DRIVER)
    v_mag = torch.where(kturn, v_mag.clamp(max=KTURN_MPS), v_mag).clamp(min=CREEP_MPS)
    v_mag = torch.where(on_slot, torch.full_like(v_mag, ON_SLOT_MPS), v_mag)
    # the tape: never faster than the car can stop before it, in the direction it is driving. A car wedged
    # across a corner (no usable room either way) still asks for a creep, or it would never turn itself out;
    # the shield lets a wedged car crawl (its rule 5).
    room_g = torch.where(gear > 0, room_f, room_r)
    v_tape = speed_for_room(room_g, G.BRAKE_PLAN)
    v_tape = torch.where(~can_f & ~can_r, v_tape.clamp(min=CREEP_MPS), v_tape)
    v_mag = torch.minimum(v_mag, v_tape)

    # ramming rule: limit the pawn's own velocity component toward the evader
    e_c = e_xy.unsqueeze(-2)
    e_h = e_yaw.unsqueeze(-1)
    gap = G.obb_separation(p_xy, p_yaw, e_c, e_h)
    to_e = e_c - p_xy
    # closing speed per m/s driven in this gear = cos(phi), phi = the bearing of the evader centre off the
    # direction of travel. A turning car swings its direction of travel while it slows down: the direction is
    # swept over the stopping distance at the commanded curvature and the worst cos(phi) on the way is used.
    travel = torch.where(gear > 0, p_yaw, p_yaw + math.pi)
    phi = G.wrap(torch.atan2(to_e[..., 1], to_e[..., 0]) - travel)
    sweep = gear * torch.tan(steer * S.MAX_STEER_RAD) / S.WHEELBASE_M * G.stop_distance(p_v, G.BRAKE_PLAN)
    across = (phi * sweep > 0.0) & (phi.abs() <= sweep.abs())         # the sweep passes over the evader
    closing_per_speed = torch.maximum(torch.cos(phi), torch.cos(phi - sweep))
    closing_per_speed = torch.where(across, torch.ones_like(phi), closing_per_speed)
    close_cap = speed_for_room(gap - CONTACT_GAP_M - CLOSE_MARGIN_M, CLOSE_BRAKE, v_end=CLOSE_TARGET_MPS)
    capped = torch.minimum(v_mag, close_cap / closing_per_speed.clamp(min=1e-3))
    v_mag = torch.where(closing_per_speed > 0.05, capped, v_mag)

    arrived = stop_slot & (dist < ARRIVE_M) & (head_err < ARRIVE_RAD)
    if state is not None:
        # the latch is about the real slot (never a via point): once there, stay until clearly off it
        arrived = slot_stop & (arrived | (state.arrived & (slot_dist < LEAVE_M) & (slot_head_err < LEAVE_RAD)))
        state.arrived = arrived
    if hold is not None:
        arrived = arrived | hold
    v_mag = torch.where(arrived, torch.zeros_like(v_mag), v_mag)
    steer = torch.where(arrived, torch.zeros_like(steer), steer)

    thr = speed_to_throttle(gear * v_mag, p_v).clamp(-T_MAX, T_MAX)
    steer = steer.clamp(-1.0, 1.0)
    if hold is not None:
        brake = -torch.sign(p_v) * (HOLD_BRAKE_GAIN * p_v.abs()).clamp(HOLD_BRAKE_FLOOR, T_MAX)
        brake = torch.where(p_v.abs() > HOLD_BRAKE_MIN_MPS, brake, torch.zeros_like(brake))
        thr = torch.where(hold, brake, thr)
    if state is not None:
        st = state
        st.hist_xy = torch.where(st.fresh[..., None, None], p_xy.unsqueeze(2).expand_as(st.hist_xy), st.hist_xy)
        st.fresh = torch.zeros_like(st.fresh)
        st.hist_xy = torch.cat([st.hist_xy[:, :, 1:], p_xy.unsqueeze(2)], 2)
        # the furthest the pawn was from the oldest position of the window (not the net displacement)
        moved = (st.hist_xy - st.hist_xy[:, :, :1]).norm(dim=-1).amax(-1)
        if hold is not None:
            st.backoff = torch.where(hold, torch.zeros_like(st.backoff), st.backoff)
        backing = st.backoff > 0
        pushing = (thr.abs() >= STUCK_THROTTLE) & ~backing
        if hold is not None:
            pushing = pushing & ~hold
        same = torch.sign(thr) == st.push_sign
        st.pushing = torch.where(pushing, torch.where(same, st.pushing + 1, torch.ones_like(st.pushing)),
                                 torch.zeros_like(st.pushing))
        st.push_sign = torch.sign(thr)
        stuck = (st.pushing >= STUCK_TICKS) & (moved < STUCK_MOVE_M)
        job = getattr(slots, "job", None)
        extra = BACKOFF_PER_JOB * job.long() if job is not None else torch.zeros_like(st.backoff)
        st.back_dir = torch.where(stuck, -torch.sign(thr), st.back_dir)
        st.backoff = torch.where(stuck, BACKOFF_TICKS + extra, (st.backoff - 1).clamp(min=0))
        st.unjams = st.unjams + stuck.long()
        st.pushing = torch.where(stuck, torch.zeros_like(st.pushing), st.pushing)
        # no room to back off that way (the tape): the back-off is dropped; the shield would refuse it anyway
        room_b = torch.where(st.back_dir > 0, room_f, room_r)
        backing = (st.backoff > 0) & (room_b >= PINNED_M)
        thr = torch.where(backing, st.back_dir * BACKOFF_THROTTLE, thr)
        steer = torch.where(backing, torch.zeros_like(steer), steer)
    return torch.stack([thr, steer], -1)
