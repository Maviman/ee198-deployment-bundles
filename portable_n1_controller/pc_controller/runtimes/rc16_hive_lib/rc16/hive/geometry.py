"""Shared state and plane geometry for the N=3 playbook commander (rc16/hive/CONTRACT.md, "Shared state: World").

Pure torch, no Isaac imports. Every function broadcasts over leading dimensions and runs on the device of its
inputs (CPU or CUDA).

Frames and units: metres, radians, seconds. Arena frame = origin at the centre of the square, +x right, +y up,
yaw counter-clockwise from +x. A car's pose is its body centre (mid wheelbase); its body frame has +x out of the
nose and +y to its left. A "car box" is the body rectangle S.LENGTH_M (along the nose) x S.WIDTH_M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rc16 import specs as S

# --- HIVE constants -------------------------------------------------------------------------------------------
BRAKE_PLAN = 4.0        # m/s^2: deceleration assumed when planning a stop (measured reverse-brake is ~4.8)
V_MAX_DRIVER = 2.0      # m/s: the fastest speed a pawn driver ever asks for
CONTACT_SPEED = 0.5     # m/s: the highest closing speed allowed when touching the evader (the ramming rule)

HALF_LENGTH_M = 0.5 * S.LENGTH_M    # car box half-extent along the nose
HALF_WIDTH_M = 0.5 * S.WIDTH_M      # car box half-extent to the side


@dataclass
class World:
    """What the cameras report for one control tick, batched over N envs. Car order: pursuers p0, p1, p2.

    p_xy (N,3,2) m     p_yaw (N,3) rad    p_v (N,3) signed forward speed m/s (negative = reversing)
    e_xy (N,2) m       e_yaw (N,) rad     e_v (N,)  signed forward speed m/s
    p_seen (N,3) bool, e_seen (N,) bool   whether the cameras currently see the car
    """

    p_xy: torch.Tensor
    p_yaw: torch.Tensor
    p_v: torch.Tensor
    e_xy: torch.Tensor
    e_yaw: torch.Tensor
    e_v: torch.Tensor
    p_seen: torch.Tensor
    e_seen: torch.Tensor


def wrap(a):
    """Angle (rad) wrapped to (-pi, pi]."""
    a = torch.as_tensor(a)
    return math.pi - torch.remainder(math.pi - a, 2.0 * math.pi)


def body_frame(vec_w, yaw):
    """World -> body: rotate a world vector (...,2) into the frame of a car heading `yaw` (...,).

    Result (...,2): [0] = component along the nose, [1] = component to the car's left.
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = vec_w[..., 0], vec_w[..., 1]
    return torch.stack([c * x + s * y, -s * x + c * y], -1)


def _sat(c1, y1, c2, y2, inflate=0.0):
    """Largest separation (m) over the four face normals of two car boxes; box 1 grown by `inflate` per side."""
    hx1, hy1 = HALF_LENGTH_M + inflate, HALF_WIDTH_M + inflate
    hx2, hy2 = HALF_LENGTH_M, HALF_WIDTH_M
    u1 = torch.stack([torch.cos(y1), torch.sin(y1)], -1)
    v1 = torch.stack([-torch.sin(y1), torch.cos(y1)], -1)
    u2 = torch.stack([torch.cos(y2), torch.sin(y2)], -1)
    v2 = torch.stack([-torch.sin(y2), torch.cos(y2)], -1)
    d = c2 - c1
    sep = None
    for ax in (u1, v1, u2, v2):
        s = ((d * ax).sum(-1).abs() - hx1 * (u1 * ax).sum(-1).abs() - hy1 * (v1 * ax).sum(-1).abs()
             - hx2 * (u2 * ax).sum(-1).abs() - hy2 * (v2 * ax).sum(-1).abs())
        sep = s if sep is None else torch.maximum(sep, s)
    return sep


def obb_separation(c1, y1, c2, y2):
    """Separating-axis separation (m) of two car boxes: centres c (...,2), headings y (...,).

    > 0: a lower bound on the gap between the boxes (exact when a face of one looks at the other, an
    under-estimate corner to corner). < 0: they overlap, by this penetration depth. Same maths as
    car_separation in rc16/train/chase_env.py.
    """
    return _sat(c1, y1, c2, y2)


def obb_overlap(c1, y1, c2, y2, inflate=0.0):
    """True where two car boxes overlap (separating-axis test, exact for rectangles). Broadcastable.

    `inflate` (m) grows box 1 (c1, y1) by that much on every side before the test; box 2 is not grown.
    Boxes that only touch (separation exactly 0) do not count as overlapping.
    """
    return _sat(c1, y1, c2, y2, inflate) < 0.0


def room_to_tape(xy, yaw, line=S.ARENA_HALF_M):
    """Room (m, >= 0) ahead of the point xy (...,2) along the heading yaw (...,) before it leaves the square.

    The square is |x| <= line, |y| <= line. Per axis, the ray travels toward one of the two lines (x = +line when
    it moves in +x, x = -line when it moves in -x; same for y); the room on that axis is the distance along the
    ray to that line, or 0 when the point is already past it. The result is the smaller of the two axes.

    - inside the square: the distance along the heading to the first line the ray crosses;
    - already past a line and heading further out: 0;
    - past a line but heading back in: positive (the distance to the next line it would leave through,
      NOT the short distance back to the line it is past).

    For reverse motion pass yaw + pi.
    """
    d = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    moving = d.abs() > 1e-9
    safe = torch.where(moving, d, torch.ones_like(d))
    t = ((torch.sign(d) * line - xy) / safe).clamp(min=0.0)
    t = torch.where(moving, t, torch.full_like(t, float("inf")))
    return t.min(dim=-1).values


def dist_outside(xy, line=S.ARENA_HALF_M):
    """Metres the point xy (...,2) is beyond the square |x|,|y| <= line (0 inside or on it).

    Euclidean distance to the square, the same definition as rc16.sim.arena.distance_outside.
    """
    return (xy.abs() - line).clamp(min=0.0).norm(dim=-1)


def stop_distance(v, brake=BRAKE_PLAN, latency_s=0.15):
    """Metres needed to stop from speed |v| (m/s): |v| * latency_s travelled before the brake bites, then
    v^2 / (2 * brake) at a constant deceleration `brake` (m/s^2)."""
    v = torch.as_tensor(v).abs()
    return v * latency_s + v * v / (2.0 * brake)
