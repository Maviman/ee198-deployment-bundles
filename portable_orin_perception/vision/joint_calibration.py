"""Calibrate several cameras into ONE arena frame, together.

Two cameras at opposite corners of the arena both look along its diagonal. Each sees
its own (near) corner tag and the two side corner tags, but the far corner is past the
top of its frame, so each has three of the four points a homography needs and neither
can be calibrated alone: `arena scan` used to stop there ("need 4 corner tags").

What pins the fourth point is that the cameras look at the same floor: every tag BOTH
of them see must come out at the same arena position from either camera. For a camera
that sees three corners the unknown is where its missing corner would sit in its own
(extended) image plane, two numbers. Its homography is the exact one through the three
corners it sees and that virtual fourth one, so the corners it sees keep a zero
residual. The two numbers per camera are fitted by least squares over every point the
cameras share (the four corners of each tag they both see), each weighted by how well
each camera can locate it: a tag corner is good to ~``sigma_px`` px, which is
``sigma_px * tag_m / side_px`` metres, so a small far-away tag counts for little and the
camera that sees a tag up close decides where it is. A camera that sees all four corners
keeps its own homography and anchors the others.

How many shared tags: each point both cameras see gives two equations and each
three-corner camera has two unknowns, so two tags are the minimum; three, spread along
the band both cameras see, pin each camera to ~1.3 cm on the real rig (2026-11-12) and
give the fit a check it did not use (``leave_one_out``): the honest answer to "do the
two cameras agree?". The tags must lie ON THE FLOOR: the fit is of the floor plane, and a
car's roof tag (~9 cm up) is pushed away from each camera, in opposite directions.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

SIGMA_PX = 0.5          # how well a tag corner is located in the scan's median frame
# A three-corner camera's virtual corner starts on the line from its near corner through
# its side corners' midpoint: s = 1 is the affine guess (a parallelogram), and perspective
# shortens the far side, so the truth is usually inside it.
INIT_S = (0.6, 0.7, 0.8, 0.9, 1.0, 1.15)
LOO_MAX_UNCERTAINTY_M = 0.05   # a leave-one-out refit looser than this checks nothing


@dataclass
class TagView:
    corners: np.ndarray          # (4, 2) px, ArUco canonical order, median over the scan
    side_px: float
    tag_m: float                 # printed size: only weights the fit


@dataclass
class CameraView:
    slot: int
    corners_px: dict             # corner tag id -> centre px, the ones reliably seen
    tags: dict                   # tag id -> TagView, every reliably seen tag (corners too)


@dataclass
class CameraFit:
    slot: int
    H: np.ndarray | None = None              # pixel -> arena metres
    seen: list = field(default_factory=list)            # corner ids it saw
    virtual: dict = field(default_factory=dict)         # corner id -> fitted px, not seen
    uncertainty_m: float | None = None       # worst 1-sigma position error the fit leaves
    error: str = ""


@dataclass
class SharedTag:
    tag_id: int
    poses: dict                  # slot -> (x, y, heading)
    spread_m: float              # largest distance between two cameras' positions
    heading_spread_deg: float
    loo_m: float | None = None   # spread when this tag was left out of the fit


@dataclass
class JointFit:
    cameras: dict                # slot -> CameraFit
    shared: list                 # SharedTag, every non-corner tag 2+ cameras see
    free_params: int = 0
    equations: int = 0
    rms: float | None = None     # normalised residual (1.0 = the noise SIGMA_PX predicts)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and all(c.H is not None for c in self.cameras.values())


# ------------------------------------------------------------------ geometry
def homography_4pt(src_px, dst_m) -> np.ndarray:
    """The exact homography taking four pixel points onto four arena points."""
    src = np.asarray(src_px, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(dst_m, dtype=np.float64).reshape(4, 2)
    # Normalise the pixels (~1000) to ~1 so the 8x8 solve is well conditioned.
    c, s = src.mean(axis=0), max(float(np.abs(src - src.mean(axis=0)).max()), 1e-9)
    T = np.array([[1 / s, 0, -c[0] / s], [0, 1 / s, -c[1] / s], [0, 0, 1]])
    n = (src - c) / s
    A, b = np.zeros((8, 8)), np.zeros(8)
    for i, ((x, y), (X, Y)) in enumerate(zip(n, dst)):
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * X, -y * X]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * Y, -y * Y]
        b[2 * i], b[2 * i + 1] = X, Y
    h = np.linalg.solve(A, b)
    return np.append(h, 1.0).reshape(3, 3) @ T


def project(H, pts) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) points through H -> ((N,2) mapped points, (N,) homogeneous w)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(H, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def mirrored_triples(centers_px: dict, points: dict) -> list:
    """Triples of corner tags that wind round the image the other way than round the
    floor. Any camera above the floor keeps every triangle's winding (image up = +y), so
    one such triple means two tags are swapped on the floor: a MIRRORED arena frame."""
    ids = [k for k in centers_px if k in points]
    bad = []
    for a, b, c in itertools.combinations(sorted(ids), 3):
        img = [(centers_px[k][0], -centers_px[k][1]) for k in (a, b, c)]
        flo = [points[k] for k in (a, b, c)]
        si, sf = _cross(*img), _cross(*flo)
        if si * sf < 0:
            bad.append((a, b, c))
    return bad


def tag_pose(H, corners_px) -> tuple[float, float, float]:
    c = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    m, _ = project(H, np.vstack([c.mean(axis=0), (c[0] + c[1]) / 2.0]))
    return float(m[0, 0]), float(m[0, 1]), float(math.atan2(m[1, 1] - m[0, 1], m[1, 0] - m[0, 0]))


def agreement(Hs: dict, views: list, skip_ids=()) -> list:
    """Every tag at least two cameras with a homography see (except ``skip_ids``): where
    each camera puts it, and how far apart those are."""
    by_tag: dict[int, dict[int, tuple]] = {}
    for v in views:
        if Hs.get(v.slot) is None:
            continue
        for tid, tv in v.tags.items():
            if tid not in skip_ids:
                by_tag.setdefault(tid, {})[v.slot] = tag_pose(Hs[v.slot], tv.corners)
    out = []
    for tid, poses in sorted(by_tag.items()):
        if len(poses) < 2:
            continue
        spread = max(math.dist(a[:2], b[:2]) for a, b in itertools.combinations(poses.values(), 2))
        hs = max(abs(math.degrees(math.remainder(a[2] - b[2], 2 * math.pi)))
                 for a, b in itertools.combinations(poses.values(), 2))
        out.append(SharedTag(tid, poses, spread, hs))
    return out


# ------------------------------------------------------------------- the fit
class _Problem:
    def __init__(self, views: list, points: dict, sigma_px: float) -> None:
        self.views = {v.slot: v for v in views}
        self.points = points
        self.sigma_px = sigma_px
        self.fixed: dict[int, np.ndarray] = {}
        self.free: dict[int, tuple[list, int]] = {}      # slot -> (seen ids, missing id)
        self.order: list[int] = []                         # free slots, parameter order
        # shared points: (tag id, corner index, {slot: (px, sigma_m)})
        self.shared: list[tuple[int, int, dict]] = []

    def H(self, slot: int, theta: np.ndarray) -> np.ndarray:
        if slot in self.fixed:
            return self.fixed[slot]
        seen, missing = self.free[slot]
        i = self.order.index(slot)
        v = self.views[slot]
        src = [v.corners_px[k] for k in seen] + [theta[2 * i:2 * i + 2]]
        dst = [self.points[k] for k in seen] + [self.points[missing]]
        return homography_4pt(src, dst)

    def _valid(self, slot: int, H: np.ndarray, theta: np.ndarray) -> bool:
        """The four corners must stay a convex quad with the floor's winding, and every
        shared point must sit on the same side of the horizon as the corners."""
        seen, missing = self.free[slot]
        i = self.order.index(slot)
        v = self.views[slot]
        ids = seen + [missing]
        px = {k: np.asarray(v.corners_px[k], float) for k in seen}
        px[missing] = theta[2 * i:2 * i + 2]
        c = np.mean([self.points[k] for k in ids], axis=0)
        ring = sorted(ids, key=lambda k: math.atan2(self.points[k][1] - c[1], self.points[k][0] - c[0]))
        for j in range(4):
            a, b, d = (px[ring[(j + q) % 4]] for q in range(3))
            if _cross((a[0], -a[1]), (b[0], -b[1]), (d[0], -d[1])) <= 0:
                return False
        _, w_quad = project(H, [px[k] for k in ids])
        pts = [p for _, _, obs in self.shared if slot in obs for p in [obs[slot][0]]]
        w_all = np.concatenate([w_quad, project(H, pts)[1]]) if pts else w_quad
        return bool(np.all(w_all > 0) or np.all(w_all < 0))

    def residuals(self, theta: np.ndarray) -> np.ndarray | None:
        Hs = {}
        for slot in self.views:
            try:
                Hs[slot] = self.H(slot, theta)
            except np.linalg.LinAlgError:
                return None
            if slot in self.free and not self._valid(slot, Hs[slot], theta):
                return None
        r = []
        for _tid, _q, obs in self.shared:
            est = {s: project(Hs[s], [px])[0][0] for s, (px, _sig) in obs.items()}
            w = {s: 1.0 / obs[s][1] ** 2 for s in obs}
            mean = sum(w[s] * est[s] for s in obs) / sum(w.values())
            for s in obs:
                r.extend((est[s] - mean) / obs[s][1])
        return np.asarray(r, dtype=np.float64)


def _jacobian(fun, theta: np.ndarray, r0: np.ndarray, h: float = 0.1) -> np.ndarray | None:
    J = np.zeros((len(r0), len(theta)))
    for k in range(len(theta)):
        d = np.zeros_like(theta)
        d[k] = h
        rp, rm = fun(theta + d), fun(theta - d)
        if rp is not None and rm is not None:
            J[:, k] = (rp - rm) / (2 * h)
        elif rp is not None:
            J[:, k] = (rp - r0) / h
        elif rm is not None:
            J[:, k] = (r0 - rm) / h
        else:
            return None
    return J


def _levenberg_marquardt(fun, theta: np.ndarray, max_iter: int = 200):
    r = fun(theta)
    if r is None:
        return None, math.inf
    cost, lam = float(r @ r), 1e-3
    for _ in range(max_iter):
        J = _jacobian(fun, theta, r)
        if J is None:
            break
        A, g = J.T @ J, J.T @ r
        stepped = False
        while lam < 1e12:
            try:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r_new = fun(theta + step)
            if r_new is not None and float(r_new @ r_new) < cost:
                done = cost - float(r_new @ r_new) < 1e-12 * (1.0 + cost) or np.abs(step).max() < 1e-6
                theta, r, cost = theta + step, r_new, float(r_new @ r_new)
                lam = max(lam / 5, 1e-12)
                stepped = True
                break
            lam *= 10
        if not stepped or done:
            break
    return theta, cost


def _initial_guesses(view: CameraView, seen: list, missing: int, points: dict) -> list:
    """Affine extrapolation of the missing corner, shortened by several amounts."""
    P = np.array([points[k] for k in seen], dtype=np.float64)
    U = np.array([view.corners_px[k] for k in seen], dtype=np.float64)
    A = np.linalg.solve(np.hstack([P, np.ones((3, 1))]), U)          # arena -> px, exact on 3
    guess = np.append(points[missing], 1.0) @ A
    near = U[int(np.argmax([math.dist(points[k], points[missing]) for k in seen]))]
    return [near + s * (guess - near) for s in INIT_S]


def fit_joint(views: list, points: dict, *, half: tuple[float, float],
              sigma_px: float = SIGMA_PX, corner_ids=(), _start: np.ndarray | None = None,
              _loo: bool = True) -> JointFit:
    """Each camera's pixel -> arena homography, fitted together. ``points``: corner tag id
    -> arena position (m). ``corner_ids``: tags that are calibration corners (left out of
    the agreement report; their centres are exact by construction)."""
    from hive_perception.core import arena_frame

    pb = _Problem(views, points, sigma_px)
    cams = {v.slot: CameraFit(v.slot, seen=sorted(k for k in v.corners_px if k in points)) for v in views}
    errors = []
    for v in views:
        cf = cams[v.slot]
        bad = mirrored_triples(v.corners_px, points)
        if bad:
            cf.error = (f"corner tags {', '.join(map(str, bad[0]))} wind round the image the other way than "
                        "round the floor: a MIRRORED frame (two tags swapped). Tape each tag in the corner "
                        "arena_test_6ft.yaml gives it")
        elif len(cf.seen) >= 4:
            H, _res = arena_frame.fit_homography([v.corners_px[k] for k in cf.seen], [points[k] for k in cf.seen])
            pb.fixed[v.slot] = H
        elif len(cf.seen) == 3 and len(points) >= 4:
            missing = next(k for k in points if k not in cf.seen)
            pb.free[v.slot] = (cf.seen, missing)
            pb.order.append(v.slot)
        else:
            cf.error = (f"sees {len(cf.seen)} corner tag(s) {cf.seen}: a camera needs at least 3 "
                        "(4 on its own, 3 with another camera that sees the 4th)")
        if cf.error:
            errors.append(f"camera {v.slot}: {cf.error}")
    if errors:
        return JointFit(cams, [], error="; ".join(errors))

    # A corner no camera sees leaves a projective freedom every camera shares: unfixable.
    seen_any = set().union(*(set(c.seen) for c in cams.values()))
    for slot, (_seen, missing) in pb.free.items():
        if missing not in seen_any:
            return JointFit(cams, [], error=f"no camera sees corner tag {missing}: at least one camera must "
                                            "see each corner")

    by_tag: dict[int, dict] = {}
    for v in views:
        for tid, tv in v.tags.items():
            by_tag.setdefault(tid, {})[v.slot] = tv
    for tid, tvs in sorted(by_tag.items()):
        if len(tvs) < 2:
            continue
        for q in range(4):
            pb.shared.append((tid, q, {s: (np.asarray(tv.corners, float)[q],
                                           sigma_px * tv.tag_m / max(tv.side_px, 1.0)) for s, tv in tvs.items()}))
    n_params = 2 * len(pb.order)
    n_eq = sum(2 * (len(obs) - 1) for _, _, obs in pb.shared)

    if pb.order:
        non_corner = [t for t in by_tag if len(by_tag[t]) >= 2 and t not in corner_ids]
        if not non_corner:
            return JointFit(cams, [], n_params, n_eq,
                            error="no floor tag is in view of two cameras: lay the evader or a spare tag "
                                  "flat where both cameras see it (the band between the side corner tags)")
        if _start is not None:
            starts = [np.asarray(_start, dtype=np.float64)]
        else:
            per_cam = [_initial_guesses(pb.views[s], *pb.free[s], points) for s in pb.order]
            scored = []
            for combo in itertools.product(*per_cam):
                th = np.concatenate(combo)
                r = pb.residuals(th)
                if r is not None:
                    scored.append((float(r @ r), th))
            scored.sort(key=lambda x: x[0])
            starts = [th for _, th in scored[:4]]
        best, best_cost = None, math.inf
        for th0 in starts:
            th, cost = _levenberg_marquardt(pb.residuals, th0)
            if th is not None and cost < best_cost:
                best, best_cost = th, cost
        if best is None:
            return JointFit(cams, [], n_params, n_eq,
                            error="the joint fit found no valid solution: check the tags both cameras see")
        theta = best
    else:
        theta, best_cost = np.zeros(0), 0.0
        r = pb.residuals(theta)
        best_cost = float(r @ r) if r is not None else 0.0

    for slot in cams:
        cams[slot].H = pb.H(slot, theta)
        if slot in pb.free:
            i = pb.order.index(slot)
            cams[slot].virtual = {pb.free[slot][1]: theta[2 * i:2 * i + 2].copy()}
    dof = n_eq - n_params
    rms = math.sqrt(best_cost / dof) if dof > 0 else None

    # How well the shared tags pin each free camera: parameter covariance, propagated to
    # the worst point of a grid over the arena.
    if pb.order:
        r = pb.residuals(theta)
        J = _jacobian(pb.residuals, theta, r)
        A = J.T @ J if J is not None else None
        if A is None or np.linalg.cond(A) > 1e14:
            return JointFit(cams, [], n_params, n_eq, rms,
                            error="the tags both cameras see do not pin each camera's missing corner: lay more "
                                  "floor tags where both see them, spread out along that band")
        cov = np.linalg.inv(A) * (max(1.0, rms ** 2) if rms is not None else 1.0)
        hw, hh = half
        grid = [(x * hw * 0.95, y * hh * 0.95) for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 5)]
        for slot in pb.order:
            Hs = cams[slot].H
            px, _ = project(np.linalg.inv(Hs), grid)
            worst = 0.0
            for p in px:
                Jp = np.zeros((2, len(theta)))
                for k in range(len(theta)):
                    d = np.zeros_like(theta)
                    d[k] = 0.1
                    Jp[:, k] = (project(pb.H(slot, theta + d), [p])[0][0]
                                - project(pb.H(slot, theta - d), [p])[0][0]) / 0.2
                worst = max(worst, math.sqrt(max(np.linalg.eigvalsh(Jp @ cov @ Jp.T))))
            cams[slot].uncertainty_m = worst

    fit = JointFit(cams, agreement({s: c.H for s, c in cams.items()}, views, skip_ids=corner_ids),
                   n_params, n_eq, rms)

    # Leave-one-out: refit without each shared tag and see where the cameras then put it.
    # Only where the rest still pin every free camera (one shared tag per free camera, and
    # a fit that is not left wobbling), or the number measures the gap, not the cameras.
    if pb.order and _loo:
        for st in fit.shared:
            reduced = [CameraView(v.slot, v.corners_px, {t: tv for t, tv in v.tags.items() if t != st.tag_id})
                       for v in views]
            left = [t for t in by_tag if t != st.tag_id and t not in corner_ids and len(by_tag[t]) >= 2]
            if len(left) < len(pb.order):
                continue
            sub = fit_joint(reduced, points, half=half, sigma_px=sigma_px, corner_ids=corner_ids,
                            _start=theta, _loo=False)
            if not sub.ok or max(c.uncertainty_m or 0.0 for c in sub.cameras.values()) > LOO_MAX_UNCERTAINTY_M:
                continue
            again = agreement({s: c.H for s, c in sub.cameras.items()},
                              [CameraView(v.slot, v.corners_px, {st.tag_id: v.tags[st.tag_id]})
                               for v in views if st.tag_id in v.tags])
            if again:
                st.loo_m = again[0].spread_m
    return fit
