"""Lens distortion from the arena's own boundary tape (the plumb-line method).

A homography maps a flat floor to a PINHOLE image. A wide webcam is not one: the
Webcam X1 bends the arena's straight tape edges by up to 12 px (2026-11-12 snapshot),
and on a corner camera that bend lands where the camera can least afford it, at the
far side and in the corner it cannot see. A synthetic pair of corner cameras with that
much barrel distortion disagreed by ~25 cm; without it, ~1 cm. So each camera's lens
is measured before its homography is fitted, from what is always in view anyway: the
tape between two adjacent corner tags is a straight line on the floor, so it must be a
straight line in an undistorted image.

One radial term (k1, OpenCV's model, about the image centre) is fitted so that every
tape edge the camera sees comes out straight. The focal length is a nominal one: with
a single k1 the model only depends on k1 / f^2, and the homography absorbs the rest,
so the pair (f, k1) written to camera_info.<slot>.yaml is exact for this pipeline even
though f is not the lens's true focal length. A checkerboard calibration
(CALIBRATION.md) measures the full model, and `arena scan` never overwrites one.
"""

from __future__ import annotations

import math

import numpy as np

MARK = "tape-plumb-line"          # camera_name of a camera_info this module wrote
K1_RANGE = (-0.45, 0.25)


def nominal_K(size) -> np.ndarray:
    w, h = size
    f = 0.7 * w
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def undistort(points, K, k1: float) -> np.ndarray:
    import cv2

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if not k1:
        return pts.reshape(-1, 2)
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    # the same converged inversion as the live path (marker_math.undistort_points)
    return cv2.undistortPointsIter(pts, K, D, None, K, (3, 40, 1e-6)).reshape(-1, 2)


def _sample(gray, pts) -> np.ndarray:
    import cv2

    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.remap(gray, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    return out.reshape(-1).astype(np.float64)


def tape_points(gray, pa, pb, size_a: float, size_b: float, inside, *, n: int = 48,
                min_contrast: float = 20.0) -> np.ndarray | None:
    """Points on the boundary tape between corner tags at ``pa`` and ``pb`` (pixel centres,
    ``size_*`` their side in px; ``inside`` is any pixel inside the arena, e.g. a third
    corner tag, to tell which side is out). The tape is the darkest ridge running beside the
    line between the two tags, just outside it. Returns (N, 2) px or None when no single
    ridge runs along most of the edge."""
    import cv2

    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    pa, pb = np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)
    d = pb - pa
    L = float(np.linalg.norm(d))
    if L < 50:
        return None
    u = d / L
    out = np.array([-u[1], u[0]])
    if float(np.dot(np.asarray(inside, dtype=np.float64) - pa, out)) > 0:
        out = -out
    cands = []                                   # (t, offset px, depth, local tag px)
    step = 0.5
    for t in np.linspace(0.12, 0.88, n):
        s = (1 - t) * size_a + t * size_b
        offs = np.arange(-0.5 * s, max(16.0, 3.5 * s), step)
        prof = _sample(g, (pa + t * d)[None, :] + offs[:, None] * out[None, :])
        if np.any(prof < 0):                     # runs off the frame
            continue
        sm = cv2.blur(prof.reshape(-1, 1).astype(np.float32), (1, 5)).reshape(-1)
        floor = float(np.median(sm))
        dark = sm < floor - min_contrast
        # Each dark band's centre = midway between its two half-depth edges. The darkest
        # pixel wanders across a textured tape; its two edges do not.
        i = 0
        while i < len(sm):
            if not dark[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(sm) and dark[j + 1]:
                j += 1
            lo = float(sm[i:j + 1].min())
            half = (floor + lo) / 2.0
            a = i
            while a > 0 and sm[a] < half:
                a -= 1
            b = j
            while b < len(sm) - 1 and sm[b] < half:
                b += 1
            if a == 0 or b == len(sm) - 1:     # band runs off the profile: not a whole tape
                i = j + 1
                continue
            left = offs[a] + step * (half - sm[a]) / (sm[a + 1] - sm[a]) if sm[a + 1] != sm[a] else offs[a]
            right = offs[b - 1] + step * (sm[b - 1] - half) / (sm[b - 1] - sm[b]) if sm[b - 1] != sm[b] else offs[b]
            width = right - left
            if max(2.0, 0.1 * s) <= width <= 1.5 * s + 4:
                cands.append((t, 0.5 * (left + right), floor - lo, s))
            i = j + 1
    if len(cands) < n // 3:
        return None
    C = np.array(cands)
    # The tape: one smooth ridge (offset quadratic in t) through most samples. RANSAC.
    rng = np.random.default_rng(0)
    best, best_in = None, None
    ts = np.unique(C[:, 0])
    for _ in range(400):
        pick = rng.choice(len(C), 3, replace=False)
        if len(set(C[pick, 0])) < 3:
            continue
        coef = np.polyfit(C[pick, 0], C[pick, 1], 2)
        tol = np.maximum(1.5, 0.12 * C[:, 3])
        inl = np.abs(np.polyval(coef, C[:, 0]) - C[:, 1]) <= tol
        # one point per sample position: the deepest inlier
        score = len(set(C[inl, 0]))
        if best is None or score > best[0]:
            best = (score, coef)
            best_in = inl
    if best is None or best[0] < 0.6 * len(ts):
        return None
    pts = []
    for t in ts:
        m = best_in & (C[:, 0] == t)
        if not np.any(m):
            continue
        i = np.flatnonzero(m)[np.argmax(C[m, 2])]
        pts.append(pa + t * d + C[i, 1] * out)
    return np.array(pts)


def straightness_px(edges, K, k1: float) -> float:
    """RMS distance (px) of the undistorted points of each edge from that edge's line."""
    tot, n = 0.0, 0
    for P in edges:
        Q = undistort(P, K, k1)
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        dist = (Q - c) @ vt[1]
        tot += float(dist @ dist)
        n += len(dist)
    return math.sqrt(tot / max(n, 1))


def _trim(edges, K, k1: float) -> list:
    """Each edge without its points > 3 rms off its line (a cable or a scuff on the tape)."""
    out = []
    for P in edges:
        Q = undistort(P, K, k1)
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        dist = np.abs((Q - c) @ vt[1])
        keep = dist <= max(3.0 * math.sqrt(float(np.mean(dist ** 2))), 0.5)
        out.append(P[keep] if keep.sum() >= 10 else P)
    return out


def fit_k1(edges, size) -> tuple[float, float, float, np.ndarray]:
    """(k1, rms before, rms after, K): the radial term that straightens ``edges``."""
    k1, before, _after, K = _fit_k1(edges, size)
    k1, _b, after, K = _fit_k1(_trim(edges, nominal_K(size), k1), size)
    return k1, before, after, K


def _fit_k1(edges, size) -> tuple[float, float, float, np.ndarray]:
    K = nominal_K(size)
    ks = np.arange(K1_RANGE[0], K1_RANGE[1] + 1e-9, 0.01)
    costs = [straightness_px(edges, K, k) for k in ks]
    i = int(np.argmin(costs))
    lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(30):                         # golden-section refine
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if straightness_px(edges, K, c1) < straightness_px(edges, K, c2):
            b = c2
        else:
            a = c1
    k1 = (a + b) / 2
    return float(k1), straightness_px(edges, K, 0.0), straightness_px(edges, K, k1), K


def write_camera_info(path, size, K, k1: float, device: str, note: str) -> None:
    import yaml

    w, h = size
    data = {
        "image_width": int(w), "image_height": int(h), "camera_name": MARK, "device": str(device),
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in np.asarray(K).reshape(-1)]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [round(float(k1), 5), 0.0, 0.0, 0.0, 0.0]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]},
        "projection_matrix": {"rows": 3, "cols": 4,
                              "data": [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0, 0.0, float(K[1, 1]),
                                       float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]},
    }
    from pathlib import Path

    Path(path).write_text(
        f"# Lens distortion of {device}, measured by `arena scan` from the arena's boundary\n"
        f"# tape ({note}). f is nominal: with one radial term only k1/f^2 matters.\n"
        "# Delete this file to re-measure; a checkerboard calibration replaces it.\n"
        + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_camera_info(path) -> dict | None:
    import yaml

    try:
        return yaml.safe_load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
