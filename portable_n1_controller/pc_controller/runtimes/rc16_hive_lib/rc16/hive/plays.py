"""The playbook of the N=3 commander (rc16/hive/CONTRACT.md "plays.py", rc16/COMMANDER_DESIGN.md table).

A PLAY is a team strategy. It defines three SLOTS; a slot is a target pose for one pawn: a point (m, arena
frame), a heading to have there (rad) and a speed to have on arrival (m/s, 0 = stop there). Slots are numbered by
importance: with only two pawns available the play uses slots 0 and 1, with one pawn slot 0. Pawns are matched
to slots by least total straight-line travel, never by car index.

Pure torch, batched over N envs, no Isaac imports. Imports geometry.py and, for the SQUEEZE hold only,
boxstop.py (the escape-arc test).

SQUEEZE HOLD (2026-11-17, after artifacts/rc16/diag/hive_holds/squeeze_static*: against a PARKED evader the
pawns reached all-18-arcs-blocked in 79 % of episodes and then drove off again, because they were still hunting
for a slot point 0.05 m / 12 deg wide; 134 of 141 lost holds were a blocker driving away). A SQUEEZE pawn is
told to HOLD (Slots.hold: stop where it is) as soon as ITS OWN body does its job, on the poses the Orin has:
    job 0 (nose)  all 9 forward escape arcs of the evader are blocked by this pawn alone within HOLD_IN_M
    job 1 (tail)  all 9 reverse arcs, likewise
    job 2 (side)  it is within HOLD_SIDE_M and HOLD_SIDE_RAD of its slot
The hold is kept until that free travel exceeds HOLD_OUT_M on some arc (job 2: until 1.5 x the distances), the
pawn gets another job, the play changes or the pawn is not active. While any pawn of an env holds, the
pawn-to-slot matching is not changed (a holding nose car is not re-labelled "side"). The test uses one pawn at
a time, so it needs nothing about teammates.
SQUEEZE STAND-OFF (same forensics, run squeeze_static_drv6: in 31 % of the ticks of lost episodes two pawns
were touching each other; the side car, already parked against the evader, stood in the way of the nose or
tail car). The side car now waits SQUEEZE_STANDOFF_M abeam of the evader, on the open side, and only moves in
to its place beside the evader once the nose car AND the tail car both hold.

Frame used by every play (all in the arena frame, metres and radians)
    f       unit vector out of the evader's nose;  l = unit vector to the evader's LEFT (body left, always)
    E       the evader's centre PREDICT_S = 0.3 s ahead at its present signed speed: E = e_xy + 0.3 * e_v * f
    t       the evader's direction of TRAVEL: +f while it drives forward, -f while it reverses. The sign only
            changes when |e_v| passes TRAVEL_DEADBAND = 0.15 m/s the other way (it starts as +f), so a stopped
            evader does not flip it. "ahead" / "behind" in the plays mean along +t / -t.
    side    +1 = the evader's left, -1 = its right (the parameter of AMBUSH and FEINT)
    open side  the body side (+1 left / -1 right) with more room from E sideways to the tape; chosen when a play
            starts and changed only if the other side becomes more than OPEN_SIDE_HYST = 0.10 m roomier. When the
            two rooms are equal (both 0 once E is past a corner of the tape) it is the side the arena centre is on
    moving  how much the evader counts as moving, 0..1: 0 while |e_v| <= TRAVEL_DEADBAND (0.15 m/s), rising
            in a straight line to 1 at |e_v| = 2 x TRAVEL_DEADBAND (0.30 m/s). Every non-zero arrival speed is
            multiplied by it, so around a stopped evader all slots are STOP slots. Reason: driver.py only
            stops at a slot whose speed is 0; it drives THROUGH any other slot, and the slots that ask for
            speed point at or past the evader (measured on the kinematic car with a parked evader: the pawns
            never settled and kept driving into the evader's box).
    chase speed  moving x clamp(|e_v| + 0.3, 0.5, V_MAX_DRIVER = 2.0) m/s: a little faster than the evader
    anchor  E, t and l as they were on the first tick of the running play (t_in_play = 0 or a new choice);
            AMBUSH, FEINT and NET use it so that "waiting" points and the feint axis stay put in the arena

Every slot point is finally clamped to |x|, |y| <= ARENA_HALF - SLOT_MARGIN_M (0.20 m inside the tape);
SQUEEZE slots to SQUEEZE_MARGIN_M (0.09 m) instead. Reason (artifacts/rc16/diag/hive_holds/squeeze_static_drv5):
a car lying across the nose or tail of an evader needs its centre 0.24 m from the evader centre, so with the
0.20 m margin no box could be closed on an evader whose centre was within 0.44 m of the tape ahead or behind
(win 24 % under 0.40 m, 57 % at 0.40 to 0.50 m, 87 to 90 % beyond, against a PARKED evader). At 0.09 m a pawn
lying along the tape has its outer side 0.026 m inside it and its centre 0.04 m inside the limiter line.
KEEP-OUT rule: when E is itself in that 0.20 m margin band, a slot that belongs on the tape side of E is
clamped across to the ARENA side of E, 0 to 0.2 m from its centre: on top of the evader, where a pawn sent
there shoves the evader toward the tape. Such a slot (moved by the clamp on one axis, across E, to nearer than
min(its designed distance from E, KEEP_OUT_M = 0.32 m)) is slid along the clamp line, away from E, until it is
that far from E again (see _clamp_keep_out). 0.32 m is the centre distance at which two car boxes cannot overlap
whatever their headings. A clamped slot that stays on the tape side of E is left where the clamp put it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rc16 import specs as S
from rc16.hive import boxstop as BX
from rc16.hive import geometry as G

PLAYS = ("CHASE_FLANK", "AMBUSH", "HERD", "FEINT", "NET", "SQUEEZE", "REGROUP")
CHOICES = [
    ("CHASE_FLANK", None),
    ("AMBUSH", "left"), ("AMBUSH", "right"),
    ("HERD", 0), ("HERD", 1), ("HERD", 2), ("HERD", 3),
    ("FEINT", "left"), ("FEINT", "right"),
    ("NET", 0.60), ("NET", 0.40),
    ("SQUEEZE", None),
    ("NET", 0.80),          # NOT in the contract's list, see the note below
    ("REGROUP", None),
]
N_CHOICES = len(CHOICES)
# The contract fixes N_CHOICES = 14 (the commander's action and observation widths depend on it) but its list
# names only 13 choices (1 + 2 + 4 + 2 + 2 + 1 + 1). Index 12 is therefore an ADDED choice, a wide ring NET r=0.80:
# indices 0..11 are exactly the contract's order and REGROUP stays the last choice (13). Look choices up by
# value (CHOICES.index(("REGROUP", None))), not by a remembered number.
assert N_CHOICES == 14
HERD_CORNERS = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))     # corner 0..3 = (++, -+, --, +-)

PREDICT_S = 0.3             # s: slots are built on the evader's position this far ahead
SLOT_MARGIN_M = 0.20        # m: every slot point stays at least this far inside the tape
SQUEEZE_MARGIN_M = 0.09     # m: ... except SQUEEZE slots (stop slots beside the evader): this far
SQUEEZE_FLANK_SHORT_M = 0.02 # m: edge rule: a nose / tail slot the clamp left this much nearer the evader than designed
#                             (no room between the evader and the tape) is moved to the evader's flank instead
SQUEEZE_CLOSE_GAP_M = 0.25  # m: the side car also moves in when ONE end car holds and the other end car's box is
#                             within this of the evader's box (h1: no win had all three cars closing an arc)
SQUEEZE_STANDOFF_M = 0.65   # m: where the SQUEEZE side car waits (centre to centre, abeam) until nose and tail hold
HYSTERESIS_M = 0.15         # m: a new pawn-to-slot matching must save more than this total travel to be taken
TRAVEL_DEADBAND = 0.15      # m/s: |e_v| above which the travel direction t follows the sign of e_v
OPEN_SIDE_HYST = 0.10       # m: extra sideways room the other side needs before the "open side" changes
KEEP_OUT_M = 0.32           # m: centre distance from E a CLAMPED slot is kept at (2 x the box half-diagonal 0.159)
SQUEEZE_SLIDE_BIAS_M = 0.10 # m: see _solve (a displaced nose / tail car prefers the side the side car is not on)
ARRIVED_M = 0.15            # m: a pawn this close to its target counts as in place (progress of REGROUP)
NEAR_TAPE_M = 0.10          # m: a pawn whose centre is nearer the tape than this (or outside) is "near the tape"
SQUEEZE_GAP_M = 0.03        # m: box-to-box gap between a SQUEEZE slot and the evader's body
PROGRESS_SPEED = 1.0        # m/s: the speed that maps to progress 1.0 (CONTACT_SPEED is used for SQUEEZE)
HOLD_IN_M = 0.11            # m: SQUEEZE hold starts when the pawn alone blocks its 9 arcs within this (win needs 0.15)
HOLD_OUT_M = 0.14           # m: ... and ends when one of them is free for this far again
HOLD_SIDE_M = 0.10          # m: the side pawn holds within this distance of its slot ...
HOLD_SIDE_RAD = math.radians(25.0)      # ... and this heading error

_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))    # pawn i -> slot _PERMS[q][i]


@dataclass
class Slots:
    """Targets PER PAWN (already matched). xy (N,3,2) m, yaw (N,3) rad heading wanted there, speed (N,3) m/s
    wanted on arrival (0 = stop there), job (N,3) long = the play's slot index 0..2 that pawn holds."""

    xy: torch.Tensor
    yaw: torch.Tensor
    speed: torch.Tensor
    job: torch.Tensor
    hold: torch.Tensor | None = None    # (N,3) bool: stop where you are (the SQUEEZE hold), or None


def _unit(v, fallback):
    """v (...,2) scaled to length 1; `fallback` (a unit vector) where v is shorter than 0.05 m."""
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n > 0.05, v / n.clamp(min=1e-6), fallback)


def _dir(a):
    """Unit vector (...,2) at angle a (rad, CCW from +x)."""
    return torch.stack([torch.cos(a), torch.sin(a)], -1)


def _ang(v):
    """Angle (rad) of vector v (...,2)."""
    return torch.atan2(v[..., 1], v[..., 0])


def _clamp_keep_out(xy, E, bias=None, margin=None):
    """Slot points xy (N,3,2) clamped into the margin square, without dropping them onto the evader.

    Plain clamp first: |x|, |y| <= ARENA_HALF - SLOT_MARGIN_M. Then a slot is slid when all of these hold:
      - the clamp moved it on exactly ONE axis (call the other axis, along that tape line, the tangent);
      - that put it on the other side of E (N,2) from where it was designed, or level with E: this happens
        only when E itself is at or beyond the clamp line (the evader is in the 0.20 m margin band);
      - it is now nearer to E than want = min(its designed distance from E, KEEP_OUT_M).
    It is slid along the tangent until its distance from E is `want` again (and clamped on the tangent too, so
    next to a corner it can stay short). Which way: the way it already leans from E along the tangent
    (+ `bias` (N,K,2) m, added to that lean; None = 0); with no lean (under 0.1 mm) toward the arena centre.
    Everything else (a corner, a slot the clamp did not move, a clamped slot still on its own side of E) is
    the plain clamp. xy may be (N,K,2) for any K. margin: (N,) m per env, None = SLOT_MARGIN_M.
    """
    if margin is None:
        margin = torch.full(xy.shape[:1], SLOT_MARGIN_M, dtype=xy.dtype, device=xy.device)
    lim1 = (S.ARENA_HALF_M - margin)[:, None]                     # (N,1)
    lim = lim1[..., None]                                         # (N,1,1)
    c = torch.maximum(torch.minimum(xy, lim), -lim)
    Eb = E[:, None].expand_as(xy)
    over = xy.abs() > lim                                         # (N,3,2): the axis the clamp moved
    b = over[..., :1].long()                                      # tangent axis: 1 (y) when x was clamped
    a = 1 - b
    want = (xy - Eb).norm(dim=-1).clamp(max=KEEP_OUT_M)
    da = (c - Eb).gather(-1, a).squeeze(-1)                       # offset from E across the tape line (fixed)
    crossed = da * (xy - Eb).gather(-1, a).squeeze(-1) <= 0.0     # the clamp carried it across E
    need = (over[..., 0] ^ over[..., 1]) & crossed & ((c - Eb).norm(dim=-1) < want - 1e-4)
    lean = (xy - Eb).gather(-1, b).squeeze(-1)
    if bias is not None:
        lean = lean + bias.gather(-1, b).squeeze(-1)
    e_b = Eb.gather(-1, b).squeeze(-1)
    to_centre = torch.where(e_b > 0, -torch.ones_like(e_b), torch.ones_like(e_b))
    sgn = torch.where(lean.abs() > 1e-4, torch.sign(lean), to_centre)
    reach = (want * want - da * da).clamp(min=0.0).sqrt()
    new_b = torch.maximum(torch.minimum(e_b + sgn * reach, lim1), -lim1)
    slid = c.scatter(-1, b, new_b[..., None])
    return torch.where(need[..., None], slid, c)


def _stack3(a, b, c):
    return torch.stack([a, b, c], 1)


# --- the seven plays: each returns slot points (N,3,2), headings (N,3), arrival speeds (N,3) BY SLOT INDEX ------
# `fr` is the frame dict built by Playbook._frame (see the module docstring for the symbols).

def _chase_flank(fr):
    """CHASE_FLANK: one follows, two run wide to get level with the evader.

    slot 0 (chaser)       0.45 m behind E on the travel line
    slot 1 (flank, open)  0.10 m ahead of E and 0.50 m out on the OPEN side
    slot 2 (flank, other) 0.10 m ahead of E and 0.50 m out on the other side
    All three head the way the evader travels and arrive at the chase speed. With two pawns: chaser + the
    flanker on the open side (the side the evader has room to turn into).
    """
    E, t, l, os_ = fr["E"], fr["t"], fr["l"], fr["open"][:, None]
    xy = _stack3(E - 0.45 * t, E + 0.10 * t + 0.50 * os_ * l, E + 0.10 * t - 0.50 * os_ * l)
    yaw = _ang(t)[:, None].expand(-1, 3)
    return xy, yaw, fr["chase"][:, None].expand(-1, 3)


def _ambush(fr):
    """AMBUSH(side): one pushes from behind on `side`, two wait where that push sends the evader.

    slot 0 (pusher)  0.40 m behind E and 0.25 m toward `side`, heading at the midpoint of the two waiting
                     points, arriving at the chase speed. It follows the evader.
    slot 1 (waiter)  FIXED in the arena: 0.90 m ahead of the anchor and 0.20 m to the side AWAY from the push
    slot 2 (waiter)  FIXED in the arena: 0.70 m ahead of the anchor and 0.60 m to the side away from the push
    The waiters face back at the anchor (where the evader was when the play began) and stop there (speed 0).
    The waiting points are clamped into the margin square when the play begins, with the keep-out rule applied
    about the anchor (not about the evader's later positions).
    With two pawns: the pusher and the waiter on the path.
    """
    E, t, l, sd = fr["E"], fr["t"], fr["l"], fr["side"][:, None]
    a, at, al = fr["a_xy"], fr["a_t"], fr["a_l"]
    wait = _clamp_keep_out(torch.stack([a + 0.90 * at - 0.20 * sd * al, a + 0.70 * at - 0.60 * sd * al], 1), a)
    w1, w2 = wait[:, 0], wait[:, 1]
    s0 = E - 0.40 * t + 0.25 * sd * l
    yaw = _stack3(_ang(0.5 * (w1 + w2) - s0), _ang(a - w1), _ang(a - w2))
    zero = torch.zeros_like(fr["chase"])
    return _stack3(s0, w1, w2), yaw, _stack3(fr["chase"], zero, zero)


def _herd(fr):
    """HERD(corner): an arc of pawns behind the evader that leaves the way to the chosen corner open.

    u = the direction from E to the corner (corner 0..3 = ++, -+, --, +- at the tape corners). The slots sit
    on a circle of radius 0.55 m round E, at angles measured from u:
        three pawns: slot 0 at 180 deg (straight behind), slot 1 at +110 deg, slot 2 at -110 deg
                     (so the arc covers 220 deg and the 140 deg toward the corner is open)
        two pawns:   slot 0 at +135 deg, slot 1 at -135 deg (slot 2, unused, at 180 deg)
    Each faces the evader and arrives at moving x clamp(|e_v|, 0.3, 1.0) m/s: it keeps pace with an evader
    that moves, and stops on the arc (speed 0) round one that stands still. It does not push a parked evader.
    """
    E, n = fr["E"], fr["E"].shape[0]
    u = _unit(fr["corner"] * S.ARENA_HALF_M - E, fr["corner"] / math.sqrt(2.0))
    d3 = torch.tensor([180.0, 110.0, -110.0], device=E.device).deg2rad()
    d2 = torch.tensor([135.0, -135.0, 180.0], device=E.device).deg2rad()
    ang = torch.where((fr["k"] == 2)[:, None], d2, d3) + _ang(u)[:, None]
    xy = E[:, None] + 0.55 * _dir(ang)
    speed = (fr["moving"] * fr["espeed"].clamp(0.3, 1.0))[:, None].expand(n, 3)
    return xy, _ang(E[:, None] - xy), speed


def _feint(fr):
    """FEINT(side): one pawn shows pressure on `side` to turn the evader into the other two.

    slot 0 (feinter)   0.20 m ahead of E and 0.40 m out on `side`, heading 60 deg across the evader's travel
                       line toward the other side, arriving at the chase speed
    slot 1 (receiver)  0.55 m ahead of E and 0.45 m out on the OTHER side
    slot 2 (receiver)  0.05 m behind E and 0.55 m out on the other side
    The receivers face the evader and arrive stopped (speed 0). With two pawns: feinter + the forward receiver.
    """
    E, t, l, sd = fr["E"], fr["t"], fr["l"], fr["side"][:, None]
    xy = _stack3(E + 0.20 * t + 0.40 * sd * l, E + 0.55 * t - 0.45 * sd * l, E - 0.05 * t - 0.55 * sd * l)
    # turning from the travel direction toward the far side: with l = body left and t = sgn * f, "toward -side"
    # is a rotation by -side * sgn (counter-clockwise positive)
    yaw0 = _ang(t) - fr["side"] * fr["sgn"] * math.radians(60.0)
    yaw = _stack3(yaw0, _ang(E - xy[:, 1]), _ang(E - xy[:, 2]))
    zero = torch.zeros_like(fr["chase"])
    return xy, yaw, _stack3(fr["chase"], zero, zero)


def _net(fr):
    """NET(r): equal spacing on a ring of radius r (0.60 or 0.40 m; 0.80 m is the added wide ring) round E.

    The ring's zero direction is the evader's travel direction when the play began (the anchor), so the ring
    moves with the evader but does not spin when it turns.
        three pawns: slot 0 at 0 deg (ahead), slot 1 at +120 deg, slot 2 at -120 deg
        two pawns:   slot 0 at +60 deg, slot 1 at -60 deg (both ahead; slot 2, unused, at 180 deg)
    Each faces the evader and stops on the ring (speed 0). Shrinking the ring = choosing the smaller r.
    """
    E = fr["E"]
    d3 = torch.tensor([0.0, 120.0, -120.0], device=E.device).deg2rad()
    d2 = torch.tensor([60.0, -60.0, 180.0], device=E.device).deg2rad()
    ang = torch.where((fr["k"] == 2)[:, None], d2, d3) + _ang(fr["a_t"])[:, None]
    xy = E[:, None] + fr["radius"][:, None, None] * _dir(ang)
    return xy, _ang(E[:, None] - xy), torch.zeros_like(ang)


def _squeeze(fr):
    """SQUEEZE: the box-stop poses. Car box = 0.2921 x 0.127 m; SQUEEZE_GAP_M = 0.03 m of air to the evader.

    Uses the evader's BODY axes (nose f, left l), not its travel direction.
    slot 0 (nose)  centred 0.2396 m ahead of E (half length 0.146 + half width 0.0635 + 0.03), lying ACROSS
                   the nose (heading e_yaw + 90 deg): blocks the 9 forward arcs
    slot 1 (tail)  the same distance behind E, also across: blocks the 9 reverse arcs
    slot 2 (side)  alongside on the OPEN side, parallel to the evader, centre 0.157 m out (0.127 + 0.03):
                   closes the side the evader's nose or tail would swing into if slot 0 or 1 is a little off
    All stop there (speed 0); the driver limits the closing speed to CONTACT_SPEED. A car box is the same
    turned 180 deg, so each pawn is given whichever of the two headings is nearer its present one.
    Arc definition this is built for (boxstop.py): 9 curvatures in +-1/0.302 1/m, forward and reverse,
    0.40 m long, blocked when free travel < 0.15 m. With two pawns: nose and tail.
    """
    E, f, l, yaw = fr["E"], fr["f"], fr["l"], fr["e_yaw"]
    d = G.HALF_LENGTH_M + G.HALF_WIDTH_M + SQUEEZE_GAP_M
    side = 2.0 * G.HALF_WIDTH_M + SQUEEZE_GAP_M
    side = torch.where(fr["sq_close"], torch.full_like(fr["open"], side), torch.full_like(fr["open"], SQUEEZE_STANDOFF_M))
    xy = _stack3(E + d * f, E - d * f, E + (side * fr["open"])[:, None] * l)
    yaws = _stack3(yaw + 0.5 * math.pi, yaw + 0.5 * math.pi, yaw)
    return xy, yaws, torch.zeros_like(yaws)


def _regroup(fr):
    """REGROUP: back off to a loose triangle between the evader and open space.

    o = the direction from E to the arena centre (the open space); if E is within 0.05 m of the centre, the
    direction opposite to the evader's travel.
    slot 0  0.60 m from E along o
    slot 1  0.80 m from E along o turned +45 deg (counter-clockwise)
    slot 2  0.80 m from E along o turned -45 deg
    Each faces the evader and stops there (speed 0).
    """
    E = fr["E"]
    o = _ang(_unit(-E, -fr["t"]))
    ang = _stack3(o, o + 0.25 * math.pi, o - 0.25 * math.pi)
    r = torch.tensor([0.60, 0.80, 0.80], device=E.device)
    xy = E[:, None] + r[None, :, None] * _dir(ang)
    return xy, _ang(E[:, None] - xy), torch.zeros_like(ang)


_PLAY_FNS = (_chase_flank, _ambush, _herd, _feint, _net, _squeeze, _regroup)


class Playbook:
    """Turns the commander's choice into three per-pawn targets, and scores whether the play is working.

    State kept per env (cleared by reset): the running choice, the pawn-to-slot matching (for the hysteresis),
    the travel sign, the open side, the anchor, and the `active` mask of the last slots() call.
    """

    def __init__(self, n, device="cpu", tape_wall=False):
        self.n, self.device = int(n), torch.device(device)
        self.tape_wall = bool(tape_wall)    # the EDGE RULE of boxstop.py is in force (SQUEEZE hold then counts the tape)
        dev = self.device
        self._ch_play = torch.tensor([PLAYS.index(p) for p, _ in CHOICES], device=dev)
        side = [(1.0 if a == "left" else -1.0) if isinstance(a, str) else 0.0 for _, a in CHOICES]
        self._ch_side = torch.tensor(side, device=dev)
        corner = [HERD_CORNERS[a] if p == "HERD" else (1.0, 1.0) for p, a in CHOICES]
        self._ch_corner = torch.tensor(corner, device=dev)
        radius = [float(a) if p == "NET" else 0.6 for p, a in CHOICES]
        self._ch_radius = torch.tensor(radius, device=dev)
        self._perms = torch.tensor(_PERMS, device=dev)
        self._choice = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._perm = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._sgn = torch.ones(n, device=dev)
        self._open = torch.ones(n, device=dev)
        self._a_xy = torch.zeros(n, 2, device=dev)
        self._a_t = torch.tensor([1.0, 0.0], device=dev).repeat(n, 1)
        self._a_l = torch.tensor([0.0, 1.0], device=dev).repeat(n, 1)
        self._active = torch.ones(n, 3, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._job = torch.full((n, 3), -1, dtype=torch.long, device=dev)

    def reset(self, ids=None):
        """Forget the running play, matching and anchor of envs `ids` (None = all)."""
        ids = slice(None) if ids is None else ids
        self._choice[ids] = -1
        self._perm[ids] = -1
        self._sgn[ids] = 1.0
        self._open[ids] = 1.0
        self._a_xy[ids] = 0.0
        self._a_t[ids] = torch.tensor([1.0, 0.0], device=self.device)
        self._a_l[ids] = torch.tensor([0.0, 1.0], device=self.device)
        self._active[ids] = True
        self._hold[ids] = False
        self._job[ids] = -1

    # --- internals ---------------------------------------------------------------------------------------------
    def _frame(self, w, choice, active, relatch):
        """The symbols of the module docstring for this tick. Reads the state, does not change it."""
        f = _dir(w.e_yaw)
        l = torch.stack([-f[:, 1], f[:, 0]], -1)
        sgn = torch.where(w.e_v > TRAVEL_DEADBAND, torch.ones_like(self._sgn),
                          torch.where(w.e_v < -TRAVEL_DEADBAND, -torch.ones_like(self._sgn), self._sgn))
        t = sgn[:, None] * f
        E = w.e_xy + PREDICT_S * w.e_v[:, None] * f
        diff = G.room_to_tape(E, w.e_yaw + 0.5 * math.pi) - G.room_to_tape(E, w.e_yaw - 0.5 * math.pi)
        # equal rooms (E past a corner: both 0): the side the arena centre is on, not "always left"
        tie = torch.where(-(E * l).sum(-1) >= 0, torch.ones_like(diff), -torch.ones_like(diff))
        fresh = torch.where(diff.abs() > 1e-6, torch.sign(diff), tie)
        kept = torch.where(diff * self._open < -OPEN_SIDE_HYST, -self._open, self._open)
        espeed = w.e_v.abs()
        moving = ((espeed - TRAVEL_DEADBAND) / TRAVEL_DEADBAND).clamp(0.0, 1.0)
        r2 = relatch[:, None]
        return {
            "E": E, "f": f, "l": l, "t": t, "sgn": sgn, "e_yaw": w.e_yaw, "espeed": espeed,
            "moving": moving, "chase": moving * (espeed + 0.3).clamp(0.5, G.V_MAX_DRIVER),
            "open": torch.where(relatch, fresh, kept),
            "a_xy": torch.where(r2, E, self._a_xy), "a_t": torch.where(r2, t, self._a_t),
            "a_l": torch.where(r2, l, self._a_l),
            "side": self._ch_side[choice], "corner": self._ch_corner[choice], "radius": self._ch_radius[choice],
            # SQUEEZE: the nose car and the tail car both held on the previous tick -> the side car may close in
            "sq_close": self._sq_close(w) & ~relatch,
            "k": active.sum(1),
        }

    def _sq_close(self, w):
        """SQUEEZE: may the side car move in? Both end cars held on the previous call, or one held and the
        other's box is within SQUEEZE_CLOSE_GAP_M of the evader's box now."""
        nose = self._hold & (self._job == 0)
        tail = self._hold & (self._job == 1)
        gap = torch.nan_to_num(G.obb_separation(w.e_xy[:, None], w.e_yaw[:, None], w.p_xy, w.p_yaw), nan=1e9)
        near = gap < SQUEEZE_CLOSE_GAP_M
        nose_near = (near & (self._job == 0)).any(1)
        tail_near = (near & (self._job == 1)).any(1)
        return (nose.any(1) & (tail.any(1) | tail_near)) | (tail.any(1) & nose_near)

    def _solve(self, w, choice, active, relatch):
        """Slots by slot index, then the matching. Returns a dict; changes no state."""
        n = choice.shape[0]
        fr = self._frame(w, choice, active, relatch)
        outs = [fn(fr) for fn in _PLAY_FNS]
        play = self._ch_play[choice]
        rows = torch.arange(n, device=choice.device)
        raw_xy = torch.stack([o[0] for o in outs])[play, rows]                     # (N,3,2) before the clamp
        # SQUEEZE: a nose / tail car that has to slide along the tape goes to the side the side car is NOT on,
        # unless it already leans more than SQUEEZE_SLIDE_BIAS_M the other way
        is_sq = (play == PLAYS.index("SQUEEZE")).to(raw_xy.dtype)
        bias = -SQUEEZE_SLIDE_BIAS_M * (is_sq * fr["open"])[:, None, None] * fr["l"][:, None].expand(n, 3, 2)
        margin = SLOT_MARGIN_M + (SQUEEZE_MARGIN_M - SLOT_MARGIN_M) * is_sq           # (N,)
        s_xy = _clamp_keep_out(raw_xy, fr["E"], bias, margin)                      # (N,3,2)
        s_yaw_all = torch.stack([o[1] for o in outs])[play, rows]
        if self.tape_wall:
            # edge rule: the tape closes that end, so a nose / tail car with no room there goes ALONGSIDE the
            # evader on the side the side car is not on (heading along the evader), instead of onto its box
            d_nt = G.HALF_LENGTH_M + G.HALF_WIDTH_M + SQUEEZE_GAP_M
            short = (s_xy[:, :2] - fr["E"][:, None]).norm(dim=-1) < d_nt - SQUEEZE_FLANK_SHORT_M     # (N,2)
            lat = 2.0 * G.HALF_WIDTH_M + SQUEEZE_GAP_M
            flank = fr["E"] - (lat * fr["open"])[:, None] * fr["l"]
            flank = _clamp_keep_out(flank[:, None], fr["E"], None, margin)[:, 0]
            # no room on that side either (the evader lies at an angle to the tape): the open side, outside
            # where the side car closes in (one more car width out)
            outer = fr["E"] + ((2.0 * lat) * fr["open"])[:, None] * fr["l"]
            outer = _clamp_keep_out(outer[:, None], fr["E"], None, margin)[:, 0]
            cramped = (flank - fr["E"]).norm(dim=-1) < lat - SQUEEZE_FLANK_SHORT_M
            flank = torch.where(cramped[:, None], outer, flank)
            use = short & (is_sq > 0)[:, None]
            s_xy = torch.cat([torch.where(use[..., None], flank[:, None].expand(n, 2, 2), s_xy[:, :2]), s_xy[:, 2:]], 1)
            s_yaw_all = torch.cat([torch.where(use, fr["e_yaw"][:, None].expand(n, 2), s_yaw_all[:, :2]),
                                   s_yaw_all[:, 2:]], 1)
        s_yaw = G.wrap(s_yaw_all)                                                  # (N,3)
        s_speed = torch.stack([o[2] for o in outs])[play, rows]                    # (N,3)

        # matching: cost of each of the 6 pawn->slot maps = total straight-line travel of the active pawns;
        # a map is allowed only if the active pawns take the k most important slots (0..k-1)
        dist = (w.p_xy[:, :, None] - s_xy[:, None]).norm(dim=-1)                   # (N, pawn, slot)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0)     # a pawn with no known position costs nothing
        per = self._perms[None].expand(n, 6, 3)                                    # (N,6,3) slot of pawn i
        d = dist[:, None].expand(n, 6, 3, 3).gather(3, per[..., None]).squeeze(3)  # (N,6,3) travel of pawn i
        act = active[:, None]
        cost = (d * act).sum(-1)
        valid = (~act | (per < fr["k"][:, None, None])).all(-1)
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        best_cost, best = cost.min(1)
        same_play = choice == self._choice
        prev = self._perm.clamp(min=0)
        prev_cost = cost.gather(1, prev[:, None]).squeeze(1)
        keep = same_play & (self._perm >= 0) & torch.isfinite(prev_cost) & (prev_cost - best_cost <= HYSTERESIS_M)
        # a matching is not changed while a pawn of that env is holding its SQUEEZE position
        # (only a nose / tail hold counts, and only while the same pawns are active as on the previous call)
        frozen = (self._hold & (self._job < 2)).any(1) & (active == self._active).all(1)
        keep = keep | (same_play & (self._perm >= 0) & torch.isfinite(prev_cost) & frozen)
        idx = torch.where(keep, prev, best)
        job = self._perms[idx]                                                     # (N,3)

        xy = s_xy.gather(1, job[..., None].expand(n, 3, 2))
        yaw = s_yaw.gather(1, job)
        speed = s_speed.gather(1, job)
        # SQUEEZE: a box turned 180 deg is the same box, so take the heading nearer the pawn's own
        flip = (play == PLAYS.index("SQUEEZE"))[:, None] & (torch.cos(yaw - w.p_yaw) < 0)
        yaw = G.wrap(torch.where(flip, yaw + math.pi, yaw))
        # an inactive pawn keeps its own pose and stops
        xy = torch.where(active[..., None], xy, w.p_xy)
        yaw = torch.where(active, yaw, w.p_yaw)
        speed = torch.where(active, speed, torch.zeros_like(speed))
        return {"fr": fr, "play": play, "s_xy": s_xy, "s_yaw": s_yaw, "s_speed": s_speed, "idx": idx,
                "slots": Slots(xy=xy, yaw=yaw, speed=speed, job=job)}

    # --- public ------------------------------------------------------------------------------------------------
    def slots(self, world, choice, t_in_play, active):
        """Targets per pawn for this tick. choice (N,) long in 0..13, t_in_play (N,) s since the play began,
        active (N,3) bool (False = that pawn is recovering: it gets its own pose and speed 0).

        The anchor and the open side are (re)taken when the choice differs from the last call or
        t_in_play <= 0; the matching hysteresis restarts only when the choice differs.
        """
        active = active.bool()
        relatch = (choice != self._choice) | (t_in_play <= 1e-6)
        out = self._solve(world, choice, active, relatch)
        fr = out["fr"]
        self._choice = choice.clone()
        self._perm = out["idx"]
        self._sgn, self._open = fr["sgn"], fr["open"]
        self._a_xy, self._a_t, self._a_l = fr["a_xy"], fr["a_t"], fr["a_l"]
        self._active = active.clone()
        slots = out["slots"]
        slots.hold = self._squeeze_hold(world, out["play"], slots, active, relatch)
        return slots

    def _squeeze_hold(self, w, play, slots, active, relatch):
        """Which pawns hold their SQUEEZE position this tick (module docstring, SQUEEZE HOLD). Updates the latch."""
        n = play.shape[0]
        is_sq = play == PLAYS.index("SQUEEZE")
        job = slots.job
        latched = self._hold & (job == self._job) & ~relatch[:, None] & is_sq[:, None] & active
        prev = self._hold & (job == self._job)
        closing = self._sq_close(w) & ~relatch                                       # the side car may be in
        hold = torch.zeros_like(latched)
        if bool(is_sq.any()):
            e_pose = torch.cat([w.e_xy, w.e_yaw[:, None]], -1)
            p_pose = torch.cat([w.p_xy, w.p_yaw[..., None]], -1)
            eye = torch.eye(3, dtype=torch.bool, device=play.device)
            alone = torch.stack([BX.free_travel(e_pose, p_pose, eye[k][None].expand(n, 3)) for k in range(3)], 1)
            gap_ok = torch.ones_like(latched)
            if self.tape_wall:
                # edge rule: the tape may do the blocking, but the pawn must then stand close to the evader
                alone = torch.minimum(alone, BX.tape_travel(e_pose)[:, None])
                gap = G.obb_separation(w.p_xy, w.p_yaw, w.e_xy[:, None], w.e_yaw[:, None])
                gap_ok = gap < torch.where(latched, torch.full_like(gap, HOLD_OUT_M), torch.full_like(gap, HOLD_IN_M))
            worst_f = alone[..., :BX.N_CURVATURES].max(-1).values        # (N,3) largest free travel, forward arcs
            worst_r = alone[..., BX.N_CURVATURES:].max(-1).values
            worst = torch.where(job == 0, worst_f, worst_r)
            g = torch.where(latched, torch.full_like(worst, HOLD_OUT_M), torch.full_like(worst, HOLD_IN_M))
            blocks = (worst < g) & gap_ok
            dist = (slots.xy - w.p_xy).norm(dim=-1)
            herr = G.wrap(slots.yaw - w.p_yaw).abs()
            k = torch.where(latched, torch.full_like(dist, 1.5), torch.ones_like(dist))
            beside = (dist < k * HOLD_SIDE_M) & (herr < k * HOLD_SIDE_RAD)
            hold = torch.where(job == 2, beside & closing[:, None], blocks) & is_sq[:, None] & active
        self._hold = hold
        self._job = job.clone()
        return hold

    def raw_slots(self, world, choice, active=None):
        """The play's three slots BY SLOT INDEX (not matched to pawns), after the tape clamp:
        (xy (N,3,2), yaw (N,3), speed (N,3)). Changes no state. For tests and drawings."""
        if active is None:
            active = torch.ones_like(world.p_seen, dtype=torch.bool)
        out = self._solve(world, choice, active.bool(), choice != self._choice)
        return out["s_xy"], out["s_yaw"], out["s_speed"]

    def progress(self, world, choice):
        """Signed "is it working" number per env, in [-1, 1], larger = working. Changes no state.

        It is a rate: a speed (m/s) divided by PROGRESS_SPEED = 1.0 m/s (CONTACT_SPEED = 0.5 for SQUEEZE) and
        clipped. v_e = the evader's velocity vector, v_p = a pawn's. Uses the matching and anchor of the last
        slots() call when that call was for the same choice, otherwise the present world with all pawns active.

        CHASE_FLANK  per active pawn: 1 if its box is within 0.15 m of the evader's box, otherwise its closing
                     speed on the evader / 1.0; mean over the active pawns. (Until 2026-11-17 it was the
                     flankers' gain along the evader's travel direction, which read "broke_down" on 95 % of
                     decisions, also in episodes the play won: run h1, DECISIONS D28.)
        AMBUSH       v_e toward the midpoint of the two waiting points: + when the evader runs into the trap
        HERD         v_e toward the chosen corner
        FEINT        v_e sideways AWAY from the feint side, on the anchor's left axis: + once it has turned away
        NET          mean closing speed of the ACTIVE pawns on the evader, minus (largest angular gap between
                     active pawns seen from the evader - 180 deg) / 180 deg when that gap is over 180 deg (an
                     open net; one active pawn or none = fully open, -1)
        SQUEEZE      per active pawn: 1 if its box is within 0.15 m of the evader's box (in place), or it is
                     the side car within 0.15 m of its stand-off point; otherwise its closing speed on the
                     evader / 0.5; mean over the active pawns (0 if none)
        REGROUP      per pawn: -1 if its centre is within NEAR_TAPE_M = 0.10 m of the tape or outside (so a
                     recovering pawn outside the tape counts -1); else 1 if within 0.15 m of its target; else
                     its speed toward its target; mean of the three
        An inactive (recovering) pawn's own motion never counts for CHASE_FLANK, NET or SQUEEZE.
        """
        same = choice == self._choice
        active = self._active | ~same[:, None]
        out = self._solve(world, choice, active, ~same)
        fr, sl, play = out["fr"], out["slots"], out["play"]
        w = world
        one = lambda x: x.clamp(-1.0, 1.0)                                        # noqa: E731
        evel = w.e_v[:, None] * fr["f"]
        pvel = w.p_v[..., None] * _dir(w.p_yaw)
        rel = pvel - evel[:, None]
        to_e = w.e_xy[:, None] - w.p_xy
        close = (rel * (to_e / to_e.norm(dim=-1, keepdim=True).clamp(min=1e-6))).sum(-1)       # (N,3) m/s

        gap = G.obb_separation(w.e_xy[:, None], w.e_yaw[:, None], w.p_xy, w.p_yaw)
        act_f = active.float()
        n_act = act_f.sum(1).clamp(min=1.0)
        p_chase = torch.where(gap <= 0.15, torch.ones_like(gap), one(close / PROGRESS_SPEED))
        p_chase = (p_chase * act_f).sum(1) / n_act

        wmid = out["s_xy"][:, 1:3].mean(1)
        p_ambush = one((evel * _unit(wmid - w.e_xy, fr["t"])).sum(-1) / PROGRESS_SPEED)

        u = _unit(fr["corner"] * S.ARENA_HALF_M - w.e_xy, fr["corner"] / math.sqrt(2.0))
        p_herd = one((evel * u).sum(-1) / PROGRESS_SPEED)

        p_feint = one(-fr["side"] * (evel * fr["a_l"]).sum(-1) / PROGRESS_SPEED)

        bearing = _ang(-to_e)
        first = bearing.gather(1, act_f.argmax(1, keepdim=True))        # the bearing of one active pawn
        bearing = torch.sort(torch.where(active, bearing, first), dim=1).values    # inactive: adds no gap
        gaps = torch.cat([bearing[:, 1:] - bearing[:, :-1], bearing[:, :1] + 2.0 * math.pi - bearing[:, -1:]], 1)
        hole = ((gaps.max(1).values - math.pi) / math.pi).clamp(0.0, 1.0)
        hole = torch.where(active.any(1), hole, torch.ones_like(hole))
        p_net = one(one((close * act_f).sum(1) / n_act / PROGRESS_SPEED) - hole)

        waiting = (sl.job == 2) & ((sl.xy - w.p_xy).norm(dim=-1) < ARRIVED_M)
        p_squeeze = torch.where((gap <= 0.15) | waiting, torch.ones_like(gap), one(close / G.CONTACT_SPEED))
        p_squeeze = (p_squeeze * act_f).sum(1) / n_act

        to_t = sl.xy - w.p_xy
        dist = to_t.norm(dim=-1)
        rate = one((pvel * (to_t / dist[..., None].clamp(min=1e-6))).sum(-1) / PROGRESS_SPEED)
        term = torch.where(dist < ARRIVED_M, torch.ones_like(rate), rate)
        room = S.ARENA_HALF_M - w.p_xy.abs().max(-1).values
        p_regroup = torch.where(room < NEAR_TAPE_M, -torch.ones_like(term), term).mean(1)

        allp = torch.stack([p_chase, p_ambush, p_herd, p_feint, p_net, p_squeeze, p_regroup])
        return allp[play, torch.arange(choice.shape[0], device=choice.device)]
