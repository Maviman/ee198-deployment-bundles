"""Box-stop: the win condition of the N=3 playbook commander (rc16/hive/CONTRACT.md, "boxstop.py").

Pure torch, batched over N envs, no Isaac imports. Source of the definition: artifacts/rc16/report/workflow/plan.md
lines 56-135, with this car's numbers (DECISIONS D22, D25).

Terms (each is defined here before any number is quoted for it):

escape arc      One of 18 paths the evader could drive from where it stands: 9 constant curvatures, evenly spaced
                from full lock right (-1/0.302 per metre) to full lock left (+1/0.302 per metre), each driven
                forward and in reverse. Arc index k = 0..8 forward (k = 0 full right, 4 straight, 8 full left),
                k = 9..17 the same curvatures in reverse. The path is rear-axle Ackermann: the mid-point of the rear
                axle (S.REAR_AXLE_X_M behind the body centre) follows the circle and the body stays tangent to it.
station         A point on an arc, measured as path length of the rear axle: 0.01, 0.02, ... 0.40 m (40 stations).
free travel h   For one arc: the path length (m) of the first station at which the evader's box, grown by 0.01 m on
                every side, overlaps a pursuer's box (a pursuer already touching the evader counts only where the
                arc pushes further into it: free_travel, resting_contact_rule). inf when no station to 0.40 m
                overlaps one, and inf when the evader's body centre is beyond the out-line at an earlier station
                (the tape never blocks: an arc that leaves the arena before it meets a car is FREE).
out-line        Where a car counts as "out": its body centre is more than 0.30 m (rc16.train.envs.OUT_MARGIN_M)
                beyond the tape square, measured as rc16/train/envs.py measures it (geometry.dist_outside, the
                straight-line distance to the square |x|, |y| <= S.ARENA_HALF_M). Along a side this is the line
                tape + 0.30 m; past a corner of the tape it is a quarter circle of radius 0.30 m, not a square
                corner.
Boxed(g)        All 18 free travels are < g metres.
blocked_frac    The share (0..1) of the 18 arcs whose free travel is < g. Boxed(g) is blocked_frac == 1.
hold            Consecutive control steps (10 Hz) for which the evader stayed boxed. It starts on a step with
                Boxed(0.15 m), which anchors the evader's xy; it continues on every later step with Boxed(0.20 m)
                AND the evader within 0.15 m of the anchor AND `trying`; any failed step sets it back to 0.
trying          Supplied by the caller: the evader is commanding real throttle (plan: |throttle| >= 0.3 on at least
                5 of the last 10 steps). A parked evader is not "stopped by the pursuers".
capture         hold >= hold_steps (10 steps = 1.0 s).

Ceiling / floor of the measure (CPU checks in tests/test_rc16_hive_boxstop.py):
- One car can never box. The straight-ahead arc and the straight-back arc are one straight line through where the
  evader stands, the boxes are convex, so a single car cannot be met in both directions (with the
  resting-contact rule; without it a car resting against the flank scores Boxed). The most one car reaches is
  17 of the 18 arcs (blocked_frac 0.944): a car alongside the flank, just over 1 cm away and a few degrees askew.
  So blocked_frac near 1 is NOT evidence of a box; only Boxed (all 18) is.
- Two cars are enough: side-on across the nose and the tail at a 0.05 m gap they already block all 18.
"""

from __future__ import annotations

import torch

from rc16 import specs as S
from rc16.hive import geometry as G

# --- the definition's numbers ---------------------------------------------------------------------------------
FULL_LOCK_RADIUS_M = 0.302   # m: this car's measured full-lock turn radius at <= 1 m/s (CONTRACT "Facts")
N_CURVATURES = 9             # curvatures per direction
N_ARCS = 2 * N_CURVATURES    # 18: forward then reverse
STATION_STEP_M = 0.01        # m between stations
ARC_LENGTH_M = 0.40          # m: the last station
N_STATIONS = 40              # ARC_LENGTH_M / STATION_STEP_M
EVADER_INFLATE_M = 0.01      # m the evader's box is grown on every side for the overlap test
OUT_MARGIN_M = 0.30          # m beyond the tape where the out-line lies (rc16.train.envs.OUT_MARGIN_M)
G_START_M = 0.15             # m: Boxed(g) threshold that STARTS a hold
G_HOLD_M = 0.20              # m: Boxed(g) threshold that CONTINUES a hold
ANCHOR_RADIUS_M = 0.15       # m the evader may drift from the hold's anchor point
HOLD_STEPS = 10              # control steps of hold that make a capture (1.0 s at 10 Hz)

# m: centre-to-centre distance beyond which a pursuer cannot touch any station. The evader's centre moves at most
# 0.40 * sqrt(1 + (0.085 / 0.302)^2) = 0.416 m; the inflated evader box reaches 0.173 m from its centre and a
# pursuer box 0.159 m from its own: 0.748 m in all.
REACH_M = 0.80
CONTACT_DEEPER_M = 0.005     # m: see free_travel, resting_contact_rule
_CHUNK_ENVS = 2048           # envs per SAT chunk (bounds the peak memory of free_travel)
_TABLE_CACHE: dict = {}


def arc_table(device=None, dtype=torch.float32):
    """Poses of the evader's BODY CENTRE along the 18 escape arcs, in the evader's own starting body frame.

    Returns (xy (18, 40, 2) m, yaw (18, 40) rad): [..., 0] along the starting nose, [..., 1] to the left; yaw is
    the heading change. Constant (it does not depend on where the evader is), so it is built once per device.
    """
    key = (str(device), dtype)
    if key not in _TABLE_CACHE:
        kappa = torch.linspace(-1.0 / FULL_LOCK_RADIUS_M, 1.0 / FULL_LOCK_RADIUS_M, N_CURVATURES,
                               dtype=torch.float64)
        s = torch.arange(1, N_STATIONS + 1, dtype=torch.float64) * STATION_STEP_M
        sign = torch.tensor([1.0, -1.0], dtype=torch.float64)
        d = (sign[:, None, None] * s[None, None, :]).expand(2, N_CURVATURES, N_STATIONS)   # signed path length
        k = kappa[None, :, None].expand_as(d)
        th = k * d                                              # heading change
        straight = k.abs() < 1e-9
        ks = torch.where(straight, torch.ones_like(k), k)
        rx = torch.where(straight, d, torch.sin(th) / ks)       # rear-axle mid-point, relative to its start
        ry = torch.where(straight, torch.zeros_like(d), (1.0 - torch.cos(th)) / ks)
        a = -S.REAR_AXLE_X_M                                    # body centre is this far ahead of the rear axle
        cx = -a + rx + a * torch.cos(th)
        cy = ry + a * torch.sin(th)
        xy = torch.stack([cx, cy], -1).reshape(N_ARCS, N_STATIONS, 2)
        yaw = th.reshape(N_ARCS, N_STATIONS)
        _TABLE_CACHE[key] = (xy.to(device=device, dtype=dtype), yaw.to(device=device, dtype=dtype))
    return _TABLE_CACHE[key]


def free_travel(e_pose, p_pose, mask, prefilter=True, resting_contact_rule=True):
    """Free travel (m) of the evader along each of its 18 escape arcs.

    e_pose (N,3) = evader x, y, yaw;  p_pose (N,3,3) = each pursuer's x, y, yaw;  mask (N,3) bool = which pursuers
    count as blockers (False = ignored; a 0/1 number mask is accepted). A pursuer or evader pose that is NaN (a
    car the cameras lost) blocks nothing. Returns (N,18) float32: the path length of the first station whose
    inflated evader box overlaps a counted pursuer box, in 0.01 .. 0.40; inf = free (nothing met within 0.40 m,
    or the evader's centre passed the out-line first: more than 0.30 m beyond the tape square, the env's "out").
    Columns 0..8 forward (full right .. full left), 9..17 reverse (same curvature order: 9 = reversing with full
    right lock, which swings the tail to the right).

    resting_contact_rule (default True; a DEVIATION from the letter of plan.md, see below): a pursuer whose box
    is already inside the evader's 0.01 m skin where the evader stands (touching it, or pressed into it) blocks a
    station only if the evader's grown box is at least CONTACT_DEEPER_M (5 mm) deeper into that pursuer there than
    it is now, i.e. only if the arc moves INTO the car. Without the rule one car leaning on the evader's flank
    "overlaps" at every station of every arc (the evader just slides along it) and a single pursuer scores
    Boxed. False gives the literal plan.md test (overlap at the station, nothing else).

    prefilter=True skips the overlap tests of envs whose counted pursuers are all further than REACH_M from the
    evader (their result is all-inf either way); False tests every env (same answer, used by the tests).
    """
    n = e_pose.shape[0]
    dev = e_pose.device
    e_pose = e_pose.to(torch.float32)
    p_pose = p_pose.to(torch.float32)
    t_xy, t_yaw = arc_table(dev)
    m = N_ARCS * N_STATIONS
    t_xy = t_xy.reshape(1, m, 1, 2)
    t_yaw_f = t_yaw.reshape(1, m, 1)
    # float64 then rounded, so station 15 is exactly the float32 nearest to 0.15 (0.15 is then NOT < g = 0.15)
    s_len = (torch.arange(1, N_STATIONS + 1, dtype=torch.float64) * STATION_STEP_M).to(dev, torch.float32)
    zero_xy = torch.zeros(1, 1, 2, device=dev)
    zero_yaw = torch.zeros(1, 1, device=dev)
    inf = torch.tensor(float("inf"), device=dev)
    mask = torch.as_tensor(mask, device=dev).bool()         # tolerate a 0/1 mask or one built on another device
    out = torch.full((n, N_ARCS), float("inf"), device=dev, dtype=torch.float32)
    # Envs with no counted pursuer within reach cannot have a blocked arc: skip their overlap tests.
    centre_d = (p_pose[..., :2] - e_pose[:, None, :2]).norm(dim=-1)
    near = ((centre_d < REACH_M) & mask).any(-1)
    if prefilter and not bool(near.all()):
        rows = near.nonzero().squeeze(-1)
        e_pose, p_pose, mask = e_pose[rows], p_pose[rows], mask[rows]
    else:
        rows = None
    res = out if rows is None else torch.empty(e_pose.shape[0], N_ARCS, device=dev, dtype=torch.float32)
    for a in range(0, e_pose.shape[0], _CHUNK_ENVS):      # chunks of envs, not a loop over envs
        ep, pp, mk = e_pose[a:a + _CHUNK_ENVS], p_pose[a:a + _CHUNK_ENVS], mask[a:a + _CHUNK_ENVS]
        b = ep.shape[0]
        # pursuers in the evader's body frame: the arc table is constant there
        rel = G.body_frame(pp[..., :2] - ep[:, None, :2], ep[:, None, 2])      # (b,3,2)
        ryaw = pp[..., 2] - ep[:, None, 2]                                     # (b,3)
        sep = G._sat(t_xy, t_yaw_f, rel[:, None], ryaw[:, None], EVADER_INFLATE_M)      # (b,m,3) metres
        if resting_contact_rule:
            # separation where the evader stands now (station 0); a pursuer already inside the 0.01 m skin
            # blocks a station only when the arc has pushed at least 5 mm deeper into it than that
            sep0 = G._sat(zero_xy, zero_yaw, rel, ryaw, EVADER_INFLATE_M)               # (b,3)
            limit = torch.where(sep0 < 0.0, sep0 - CONTACT_DEEPER_M, torch.zeros_like(sep0))
            hit = sep < limit[:, None, :]
        else:
            hit = sep < 0.0
        hit = (hit & mk[:, None, :]).any(-1).reshape(b, N_ARCS, N_STATIONS)
        # world position of the evader's centre at each station, for the out-line
        c, s = torch.cos(ep[:, 2])[:, None], torch.sin(ep[:, 2])[:, None]
        lx, ly = t_xy[0, :, 0, 0][None], t_xy[0, :, 0, 1][None]
        wx = ep[:, 0:1] + c * lx - s * ly
        wy = ep[:, 1:2] + s * lx + c * ly
        # "out" exactly as the env decides it (rc16/train/envs.py: distance_outside > OUT_MARGIN_M)
        beyond = (G.dist_outside(torch.stack([wx, wy], -1)) > OUT_MARGIN_M).reshape(b, N_ARCS, N_STATIONS)
        fh = torch.where(hit, s_len, inf).amin(-1)          # first station that meets a pursuer (inf = none)
        fo = torch.where(beyond, s_len, inf).amin(-1)       # first station beyond the out-line (inf = none)
        res[a:a + b] = torch.where(fo < fh, inf, fh)
    if rows is not None:
        out[rows] = res
    return out


TAPE_DEEPER_M = 0.005        # m: see tape_travel
EDGE_MIN_NEAR = 2            # edge rule: pursuers that must be within g of the evader's body


def tape_travel(e_pose):
    """EDGE RULE (optional, OFF by default; proposed to the user 2026-11-17, not part of plan.md): the travel
    (m) along each of the 18 escape arcs before the evader's body goes over the TAPE.

    e_pose (N,3). Returns (N,18) float32: the path length of the first station at which some corner of the
    evader's own box (0.2921 x 0.127 m, not grown) is more than TAPE_DEEPER_M (5 mm) further beyond the tape
    square than the furthest corner is where the evader stands now (0 if it stands wholly inside); inf = the
    arc stays inside for its whole 0.40 m. "Beyond" is measured per axis (the larger of |x|, |y| minus the
    half side). So an evader with its nose on the tape has all its forward arcs stopped by the tape at once,
    and one already over it is stopped only by arcs that take it further out, never by backing in.
    Why it exists: with plan.md's "the tape never blocks", an evader parked nose-out at the tape cannot be
    boxed at all (no pursuer fits between its nose and the tape), artifacts/rc16/diag/hive_holds/squeeze_default.
    """
    dev = e_pose.device
    e_pose = e_pose.to(torch.float32)
    t_xy, t_yaw = arc_table(dev)                              # (18,40,2), (18,40)
    s_len = (torch.arange(1, N_STATIONS + 1, dtype=torch.float64) * STATION_STEP_M).to(dev, torch.float32)
    corners = torch.tensor([[G.HALF_LENGTH_M, G.HALF_WIDTH_M], [G.HALF_LENGTH_M, -G.HALF_WIDTH_M],
                            [-G.HALF_LENGTH_M, G.HALF_WIDTH_M], [-G.HALF_LENGTH_M, -G.HALF_WIDTH_M]],
                           device=dev, dtype=torch.float32)   # (4,2)

    def overhang(cx, cy, yaw):                                # (...,) centre and heading -> (...,) metres beyond
        co, si = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        x = cx[..., None] + co * corners[:, 0] - si * corners[:, 1]
        y = cy[..., None] + si * corners[:, 0] + co * corners[:, 1]
        return torch.maximum(x.abs(), y.abs()).amax(-1) - S.ARENA_HALF_M

    c, s = torch.cos(e_pose[:, 2])[:, None, None], torch.sin(e_pose[:, 2])[:, None, None]
    wx = e_pose[:, 0, None, None] + c * t_xy[None, ..., 0] - s * t_xy[None, ..., 1]        # (N,18,40)
    wy = e_pose[:, 1, None, None] + s * t_xy[None, ..., 0] + c * t_xy[None, ..., 1]
    over = overhang(wx, wy, e_pose[:, 2, None, None] + t_yaw[None])
    over0 = overhang(e_pose[:, 0], e_pose[:, 1], e_pose[:, 2]).clamp(min=0.0)
    hit = over > over0[:, None, None] + TAPE_DEEPER_M
    inf = torch.tensor(float("inf"), device=dev)
    return torch.nan_to_num(torch.where(hit, s_len, inf).amin(-1), nan=float("inf"), posinf=float("inf"))


class BoxStopTracker:
    """Counts how long the evader has been box-stopped and says when that is a capture. One per batch of N envs.

    tape_wall=True switches on the EDGE RULE (off by default = plan.md): an arc also counts as blocked when the
    evader's body would go over the tape within g (tape_travel), provided at least EDGE_MIN_NEAR (2) counted
    pursuers have their box within g of the evader's box (obb_separation). free_pawn then holds the
    pursuer-only free travel and near (N,) the number of pursuers that close.

    After step(): hold (N,) long = consecutive boxed steps; boxed (N,) bool = Boxed(g) on this step, g = 0.15 m
    for an env with no hold running and 0.20 m for one with a hold running (both x g_scale); blocked_frac (N,)
    float in [0,1] = share of the 18 arcs with free travel < that same g; free (N,18) = free travel (m, inf =
    free); anchor (N,2) = the evader xy where the running hold started.
    g_scale (>= 1) loosens both thresholds for a curriculum; 1.0 is the real definition.
    """

    def __init__(self, n, device, hold_steps=HOLD_STEPS, g_scale=1.0, tape_wall=False, min_near=EDGE_MIN_NEAR):
        self.n = int(n)
        self.device = device
        self.hold_steps = int(hold_steps)
        self.g_scale = float(g_scale)
        self.tape_wall = bool(tape_wall)
        self.min_near = int(min_near)       # edge rule: pursuers that must be within g (2 = default; 3 = all three,
        #                                     the "all three capture" model of 2026-11-18)
        self.free_pawn = torch.full((self.n, N_ARCS), float("inf"), dtype=torch.float32, device=device)
        self.near = torch.zeros(self.n, dtype=torch.long, device=device)
        self.hold = torch.zeros(self.n, dtype=torch.long, device=device)
        self.anchor = torch.zeros(self.n, 2, dtype=torch.float32, device=device)
        self.boxed = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.blocked_frac = torch.zeros(self.n, dtype=torch.float32, device=device)
        self.free = torch.full((self.n, N_ARCS), float("inf"), dtype=torch.float32, device=device)

    def reset(self, ids=None):
        """Forget the hold of the envs `ids` (index tensor / list / bool mask; None = all)."""
        if ids is None:
            ids = slice(None)
        self.hold[ids] = 0
        self.anchor[ids] = 0.0
        self.boxed[ids] = False
        self.blocked_frac[ids] = 0.0
        self.free[ids] = float("inf")
        self.free_pawn[ids] = float("inf")
        self.near[ids] = 0

    def step(self, e_pose, p_pose, trying, mask=None):
        """Advance one control step. e_pose (N,3), p_pose (N,3,3), trying (N,) bool, mask (N,3) bool or None
        (= all three pursuers count). Returns captured (N,) bool: hold >= hold_steps. It stays True on later
        steps for as long as the hold keeps running; the caller ends the episode and calls reset(ids).

        `trying` is NOT worked out here: the caller keeps the evader's last 10 commanded throttles and passes
        True where |throttle| >= 0.3 on at least 5 of them (module docstring, "trying"). Zero throttle is not
        trying even while the car still rolls: this car coasts."""
        if mask is None:
            mask = torch.ones(p_pose.shape[:2], dtype=torch.bool, device=p_pose.device)
        running = self.hold > 0
        g = torch.where(running, G_HOLD_M, G_START_M).to(torch.float32) * self.g_scale
        xy = e_pose[:, :2].to(torch.float32)
        self.anchor = torch.where(running[:, None], self.anchor, xy)
        self.free_pawn = free_travel(e_pose, p_pose, mask)
        self.free = self.free_pawn
        if self.tape_wall:
            self.free = torch.minimum(self.free_pawn, tape_travel(e_pose))
            gap = G.obb_separation(p_pose[..., :2].to(torch.float32), p_pose[..., 2].to(torch.float32),
                                   xy[:, None], e_pose[:, None, 2].to(torch.float32))
            self.near = ((torch.nan_to_num(gap, nan=1e9) < g[:, None]) & torch.as_tensor(mask, device=gap.device).bool()).sum(-1)
        under = self.free < g[:, None]
        self.blocked_frac = under.float().mean(-1)
        self.boxed = under.all(-1)
        if self.tape_wall:
            self.boxed = self.boxed & (self.near >= self.min_near)
        ok = self.boxed & ((xy - self.anchor).norm(dim=-1) <= ANCHOR_RADIUS_M) & trying.bool()
        self.hold = torch.where(ok, self.hold + 1, torch.zeros_like(self.hold))
        return self.hold >= self.hold_steps
