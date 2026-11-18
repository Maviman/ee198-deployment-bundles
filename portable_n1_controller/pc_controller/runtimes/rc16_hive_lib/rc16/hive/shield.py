"""The distance limiter (rc16/hive/CONTRACT.md, "shield.py"): the last thing a command passes before a car.

It keeps a car's body CENTRE inside the line by overriding the THROTTLE only; steering is never changed. It runs
on the Orin for the real cars and, in the simulator, on every pursuer and on the evader. Pure torch, broadcasts
over any leading dimensions, no state.

Terms (metres, m/s, radians):
- line: the square |x|, |y| <= line that the centre must stay in (default: the tape minus 0.05 m).
- direction of travel: the heading for forward motion, heading + pi for reverse.
- room: how far the car can travel before its centre reaches the line, less ROOM_PAD_M. Two definitions:
  * turn_radius=None (the DEFAULT, the contract's wording): the STRAIGHT ray along the direction of travel
    (geometry.room_to_tape). The steering angle is not used. This does NOT hold a car that steers toward the
    line: see the AUDIT note below.
  * turn_radius=r (opt-in, TURN_RADIUS_M = the full-lock radius 0.294 m): the SHORTEST path on which the car
    could reach the line whatever the steering does next: a turn toward the nearest line at radius r, then
    straight at it (`turn_room`). Heading straight at a line this is the same distance ahead; running PARALLEL
    to a line it is much shorter than the ray.
- safe speed for a room: the highest speed from which the car still stops inside that room with BRAKE_PLAN
  (4.0 m/s^2) after 0.15 s of latency; the inverse of geometry.stop_distance (`speed_for_room`).

Rule, per car and per tick:
1. moving (|v| > 0.05 m/s) faster than the safe speed for the room ahead of its motion: the throttle becomes a
   brake (opposite to the motion; magnitude |v| in m/s clamped to [BRAKE_THROTTLE_MIN, 1.0]) unless the command
   already brakes at least that hard: such a command is left alone. The brake grows with the speed so that a
   slow car is stopped, not thrown back the other way: on this car a reverse command c against a speed v slows
   it by about 7.5 c + 2.5 v + 0.8 m/s^2 (the motor model of rc16/sim/car.py), so c = |v| stops it in about one
   0.1 s tick below 0.4 m/s and is the full tyre limit (4.8 m/s^2) above;
2. otherwise the throttle is capped, in each direction, at safe speed / TOP_SPEED (a command of 1.0 settles at
   TOP_SPEED and speed is proportional to the command, so a capped command cannot settle above the safe speed);
   a cap below the command deadband becomes 0. A throttle that opposes the current motion (a brake) is always
   let through up to magnitude max(that cap, |v| clamped to [BRAKE_THROTTLE_MIN, 1.0]);
3. a car on or over the line is never trapped: for a throttle direction that does not take the centre further
   from the square (does not increase geometry.dist_outside), the room is taken as at least the stopping
   distance from INWARD_SPEED (0.5 m/s), so that throttle is never zeroed or reversed and is capped no lower
   than INWARD_SPEED / TOP_SPEED. Known cost: next to a corner the way back in across one line can run straight
   out over the next one, and this floor lets the car do that at up to 0.5 m/s (it then stops up to 0.08 m
   past that second line; a car judged on the room really there instead took three times as long to return);
4. a car whose pose or speed is not finite (not seen, no estimate) gets its command back unchanged: there is
   nothing to limit it against;
5. a WEDGED car is let out. Wedged = slower than WEDGE_SPEED with rules 2 and 3 leaving it no throttle in
   either direction: it sits across a corner of the line with both ends within about 5 cm of it (or, with the
   turn-aware room, alongside the line within a few millimetres), and would stay there for ever. It gets a
   crawl cap of WEDGE_THROTTLE in both directions and rule 1 leaves it alone up to WEDGE_SPEED, so it can creep
   while the steering (which the shield never touches) turns it out. If the creep takes it over the line, rule
   3 lets it only come back. The line is 0.05 m inside the tape; under a blind to-and-fro command a wedged car
   has been measured up to 0.044 m past the line (0.006 m inside the tape) on its way out.

AUDIT note (module review 2026-11-17; bicycle models, 4000 cars, 30 s of RANDOM throttle and steering through
the shield): with the straight ray 27 to 55 % of the cars crossed the TAPE at least once (up to 0.33 m past
it), because a car running along the tape has a long ray and no room left to stop once it steers in. With
turn_radius=TURN_RADIUS_M none crossed the tape (worst 0.04 m past the line, 0.01 m inside the tape). The price
is speed near the tape: running parallel to the line the safe speed is 0.63 / 0.86 / 1.15 / 1.37 / 1.74 m/s at
0.05 / 0.10 / 0.20 / 0.30 / 0.50 m from it. The default stays the straight ray because the pawn driver and the
SmartEvader plan their own speed on that ray and were tuned against it; with the driver's own commands (not
random ones) about 0.1 % of cars crossed the tape in 30 s on the ray and none with the turn-aware room.
tests/test_rc16_hive_shield.py pins the direction of both (the ray leaks under steering, the turn room holds).
"""

from __future__ import annotations

import math

import torch

from rc16 import specs as S
from rc16.hive import geometry as G

LATENCY_S = 0.15            # s: delay assumed between the decision to brake and the brake biting
MOVING_EPS = 0.05           # m/s: below this |v| the car is treated as standing still
INWARD_SPEED = 0.5          # m/s: the speed always allowed for a car outside the line that is heading back in
ROOM_PAD_M = 0.03           # m taken off the room before it is used: covers one 10 Hz tick of creep at the deadband
BRAKE_THROTTLE_MIN = 0.1    # the weakest brake command the shield issues (throttle units, opposite to motion)
LINE_INSET_M = 0.05         # m: the default line is this far inside the tape
TURN_RADIUS_M = S.MIN_TURN_RADIUS_M     # m: the radius to pass as turn_radius for the turn-aware room (full lock)
WEDGE_THROTTLE = 0.10       # crawl cap for a wedged car (rule 5): twice the command deadband
WEDGE_SPEED = 0.10          # m/s: a wedged car is one slower than this; up to this speed rule 1 leaves it alone


def speed_for_room(room, brake: float = G.BRAKE_PLAN, latency_s: float = LATENCY_S, v_end=0.0):
    """Highest speed (m/s, >= v_end) from which a car slows to `v_end` (m/s) within `room` metres.

    Inverse of geometry.stop_distance: solves v * latency_s + (v^2 - v_end^2) / (2 * brake) = room for v.
    room <= 0 returns v_end. Accepts tensors or floats.
    """
    room = torch.as_tensor(room, dtype=torch.get_default_dtype()).clamp(min=0.0)
    v_end = torch.as_tensor(v_end, dtype=room.dtype, device=room.device)
    bl = brake * latency_s
    v = -bl + torch.sqrt(bl * bl + 2.0 * brake * room + v_end * v_end)
    return torch.maximum(v, v_end)


def turn_room(xy, heading, line: float = S.ARENA_HALF_M - LINE_INSET_M, turn_radius: float = TURN_RADIUS_M):
    """Shortest path length (m, >= 0) on which a point at xy (...,2), travelling along `heading` (...,) rad and
    turning no tighter than `turn_radius` m, can reach the edge of the square |x|, |y| <= line.

    For each of the four lines, with d = the perpendicular distance to it and a = the angle of the direction of
    travel above that line (a = +90 deg straight at it, 0 parallel to it, negative going away), the shortest way
    to it is a full-lock arc that turns the travel direction to face the line, then a straight run:
        d >= r cos a:   r (pi/2 - a) + d - r cos a          (the arc is finished before the line)
        d <  r cos a:   r (acos(cos a - d / r) - a)         (the line is reached during the arc)
    A point already past a line gets 0 for it when it travels further out, and the arc back out (d = 0 in the
    formula) otherwise. The result is the smallest of the four. For reverse motion pass yaw + pi.
    As turn_radius grows this tends to geometry.room_to_tape (a car that cannot turn).
    """
    r = float(turn_radius)
    h = torch.stack([torch.cos(heading), torch.sin(heading)], -1)
    best = None
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            d = line - sign * xy[..., axis]                 # distance to this line, negative when past it
            s_n = (sign * h[..., axis]).clamp(-1.0, 1.0)    # share of the travel that goes toward this line
            a = torch.asin(s_n)
            c = torch.cos(a)
            dd = d.clamp(min=0.0)
            s_far = r * (0.5 * math.pi - a) + (dd - r * c)
            s_arc = r * (torch.acos((c - dd / r).clamp(-1.0, 1.0)) - a)
            s = torch.where(dd >= r * c, s_far, s_arc)
            s = torch.where((d <= 0.0) & (s_n > 0.0), torch.zeros_like(s), s)
            best = s if best is None else torch.minimum(best, s)
    return best.clamp(min=0.0)


def rooms(xy, yaw, line: float = S.ARENA_HALF_M - LINE_INSET_M, turn_radius=None):
    """(room_forward, room_reverse) in metres, each (...,): the room the shield uses for a car at xy (...,2)
    with heading yaw (...,): the straight ray to the line (turn_radius None) or the shortest way to it
    (turn_room, turn_radius in m), less ROOM_PAD_M, and, for a car outside the line, at least the stopping
    distance from INWARD_SPEED in a direction that does not take it further out (rule 3 of the module docstring).
    """
    if turn_radius is None:
        room_f = G.room_to_tape(xy, yaw, line)
        room_r = G.room_to_tape(xy, yaw + math.pi, line)
    else:
        room_f = turn_room(xy, yaw, line, turn_radius)
        room_r = turn_room(xy, yaw + math.pi, line, turn_radius)
    room_f = (room_f - ROOM_PAD_M).clamp(min=0.0)
    room_r = (room_r - ROOM_PAD_M).clamp(min=0.0)
    # A car on/over the line: a direction that does not increase its distance outside always gets crawl room.
    out_vec = xy - xy.clamp(-line, line)                    # zero inside the square, points outward outside
    outside = out_vec.abs().sum(-1) > 0.0
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    along = (out_vec * heading).sum(-1)                     # > 0: forward motion goes further out
    floor = float(G.stop_distance(INWARD_SPEED, G.BRAKE_PLAN, LATENCY_S))
    room_f = torch.where(outside & (along <= 0.0), room_f.clamp(min=floor), room_f)
    room_r = torch.where(outside & (along >= 0.0), room_r.clamp(min=floor), room_r)
    return room_f, room_r


def safe_speeds(xy, yaw, line: float = S.ARENA_HALF_M - LINE_INSET_M, turn_radius=None):
    """(safe_forward, safe_reverse) in m/s, each (...,) >= 0: the highest speed in each direction of travel that
    the shield lets a car at this pose keep (speed_for_room of `rooms`). A controller that plans its speed under
    these is not overridden."""
    room_f, room_r = rooms(xy, yaw, line, turn_radius)
    return speed_for_room(room_f), speed_for_room(room_r)


def shield(cmd, xy, yaw, v, line: float = S.ARENA_HALF_M - LINE_INSET_M, turn_radius=None):
    """Limit the throttle so the car can always stop before `line`. See the module docstring for the rule.

    cmd (...,2) = (throttle, steer) in [-1, 1]; xy (...,2) centre, m; yaw (...,) rad; v (...,) signed forward
    speed, m/s (negative = reversing). turn_radius: None = the straight ray along the heading (the contract's
    rule); a radius in m (TURN_RADIUS_M) = the turn-aware room. Returns (cmd_out (...,2), intervened (...,)
    bool); intervened is True where the throttle was changed. The steer column is returned untouched.
    """
    thr, steer = cmd[..., 0], cmd[..., 1]
    safe_f, safe_r = safe_speeds(xy, yaw, line, turn_radius)
    cap_f = safe_f / S.TOP_SPEED_MPS
    cap_r = safe_r / S.TOP_SPEED_MPS
    cap_f = torch.where(cap_f < S.PWM_DEADBAND, torch.zeros_like(cap_f), cap_f).clamp(max=1.0)
    cap_r = torch.where(cap_r < S.PWM_DEADBAND, torch.zeros_like(cap_r), cap_r).clamp(max=1.0)

    speed = v.abs()
    moving = speed > MOVING_EPS
    # 5. wedged (slow, no throttle left either way): a crawl both ways, and no brake below the crawl speed
    wedged = (speed <= WEDGE_SPEED) & (cap_f == 0.0) & (cap_r == 0.0)
    cap_f = torch.where(wedged, torch.full_like(cap_f, WEDGE_THROTTLE), cap_f)
    cap_r = torch.where(wedged, torch.full_like(cap_r, WEDGE_THROTTLE), cap_r)

    m = torch.sign(v)                                        # direction of motion, +1 forward
    safe_m = torch.where(v > 0.0, safe_f, safe_r)
    safe_m = torch.where(wedged, safe_m.clamp(min=WEDGE_SPEED), safe_m)
    brake_mag = speed.clamp(BRAKE_THROTTLE_MIN, 1.0)

    # 2. caps per direction; a brake (throttle against the motion) keeps at least brake_mag of authority
    lo = -cap_r
    hi = cap_f
    lo = torch.where(moving & (v > 0.0), torch.minimum(lo, -brake_mag), lo)
    hi = torch.where(moving & (v < 0.0), torch.maximum(hi, brake_mag), hi)
    out = torch.maximum(torch.minimum(thr, hi), lo)

    # 1. too fast for the room ahead: brake, unless the command already brakes at least as hard.
    #    (A harder brake is passed whole, although once the car has stopped it is a drive the other way for the
    #    rest of the tick. Capping it at the cap of that direction was tried in the 2026-11-17 review and left
    #    out: 0.2 % of recovering pawns then hung at a corner of the line.)
    must_brake = moving & (speed > safe_m)
    already = (thr * m) <= -brake_mag
    out = torch.where(must_brake & ~already, -m * brake_mag, out)
    out = torch.where(must_brake & already, thr, out)

    # 4. no finite pose or speed: nothing to limit against, the command goes through as it came
    known = torch.isfinite(xy).all(-1) & torch.isfinite(yaw) & torch.isfinite(v)
    out = torch.where(known, out, thr)

    intervened = (out - thr).abs() > 1e-6
    return torch.stack([out, steer], -1), intervened
