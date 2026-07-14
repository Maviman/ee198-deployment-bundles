"""Pixel -> arena-frame conversion via a planar floor homography.

The arena frame is the training/controller frame from the main repo's
docs/perception_interface_contract.md: origin at the arena CENTER, +x right,
+y up, meters; heading in radians, 0 = facing +x, counter-clockwise positive
(math convention). Image pixel y grows DOWNWARD — that flip is absorbed by the
homography because it is fitted from pixel<->arena correspondences that already
encode it; nothing downstream special-cases it.

Heading is computed by mapping TWO pixel points (marker center and its
top-edge midpoint) through the homography and taking atan2 in ARENA
coordinates — never from image-space angles, which are wrong under both the
y flip and perspective tilt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def fit_homography(pixel_points, arena_points_m) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography mapping >=4 pixel points onto their measured
    arena-frame positions (meters). Returns (H, per_point_residuals_m).

    method=0 (plain least squares, no RANSAC) on purpose: calibration points
    are few and hand-measured, not outlier-prone, and a mismeasured one should
    show up as a LOUD residual, not be silently voted away."""
    import cv2

    src = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(arena_points_m, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4:
        raise ValueError(f"need >=4 correspondences to fit a homography, got {src.shape[0]}")
    if src.shape != dst.shape:
        raise ValueError(f"pixel/arena point counts differ: {src.shape} vs {dst.shape}")
    H, _mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed (collinear or degenerate points?)")
    residuals = np.linalg.norm(project_px_to_arena(H, src) - dst, axis=1)
    return H, residuals


def project_px_to_arena(H, points_px) -> np.ndarray:
    """(N,2) pixel points -> (N,2) arena-frame meters (perspective divide)."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ np.asarray(H, dtype=np.float64).T
    return homog[:, :2] / homog[:, 2:3]


def pose_from_marker(H, center_px, top_midpoint_px) -> tuple[float, float, float]:
    """(x_m, y_m, heading_rad) for one marker, from its center and top-edge
    midpoint in (already undistorted) pixels."""
    mapped = project_px_to_arena(H, np.vstack([center_px, top_midpoint_px]))
    x, y = mapped[0]
    dx, dy = mapped[1] - mapped[0]
    return float(x), float(y), float(np.arctan2(dy, dx))


def save_homography_yaml(path, H, residuals_m, *, arena_config: str = "",
                         image_size=None, notes: str = "") -> None:
    payload = {
        "homography": np.asarray(H, dtype=float).reshape(3, 3).tolist(),
        "max_residual_m": float(np.max(residuals_m)),
        "mean_residual_m": float(np.mean(residuals_m)),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arena_config": str(arena_config),
        "image_size": list(image_size) if image_size is not None else None,
        "notes": notes,
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_homography_yaml(path) -> np.ndarray:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    H = np.asarray(payload["homography"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"{path}: homography must be 3x3, got {H.shape}")
    return H
