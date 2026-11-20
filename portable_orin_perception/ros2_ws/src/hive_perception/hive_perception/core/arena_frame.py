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


def camera_position(H, camera_matrix) -> tuple[float, float, float] | None:
    """Where the camera is in the arena frame: (x_m, y_m, height above the floor in m), from the
    floor homography ``H`` (undistorted pixels -> arena metres) and the lens matrix K.

    inv(H) maps the floor to the image, so inv(K) inv(H) = s [r1 r2 t]: two columns of the
    camera's rotation and its translation, and the camera centre is -R^T t. Returns None when
    the two do not describe one real camera (no K, r1 and r2 not the same length or not at
    right angles: a nominal K, or a homography fitted on distorted pixels), so a caller never
    corrects with a made-up position."""
    if camera_matrix is None:
        return None
    try:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        M = np.linalg.inv(K) @ np.linalg.inv(np.asarray(H, dtype=np.float64).reshape(3, 3))
    except (np.linalg.LinAlgError, ValueError):
        return None
    n1, n2 = float(np.linalg.norm(M[:, 0])), float(np.linalg.norm(M[:, 1]))
    if not (n1 > 0.0 and n2 > 0.0 and 0.87 < n1 / n2 < 1.15):
        return None
    r1, r2, t = M[:, 0] / n1, M[:, 1] / n2, M[:, 2] / (0.5 * (n1 + n2))
    if abs(float(r1 @ r2)) > 0.12:
        return None
    U, _s, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt                                         # the nearest true rotation
    c = -R.T @ t
    # The homography's sign is free; flipping it flips only the height.
    x, y, z = float(c[0]), float(c[1]), abs(float(c[2]))
    return (x, y, z) if np.isfinite([x, y, z]).all() and z > 0.2 else None


def lift_to_height(xy, camera_xyz, height_m: float) -> tuple[float, float]:
    """The arena position of a point ``height_m`` above the floor whose FLOOR-plane projection
    (what the homography returns) is ``xy``. A tag on a car roof is seen along a ray that meets
    the floor further from the camera than the car is: the error is height / camera height of
    the horizontal distance to the camera (9 cm up, camera 1.6 m up, car 3.5 m away: 0.20 m),
    always pointing away from the camera. Similar triangles undo it."""
    cx, cy, cz = camera_xyz
    k = 1.0 - float(height_m) / float(cz)
    return cx + (float(xy[0]) - cx) * k, cy + (float(xy[1]) - cy) * k


def save_homography_yaml(path, H, residuals_m, *, arena_config: str = "",
                         image_size=None, notes: str = "", corner_px=None,
                         extra: dict | None = None) -> None:
    """``corner_px`` ({tag id: [x, y]}) records where each calibration corner
    sat in the image, so the vision service can tell when the camera has been
    bumped since (localizer.corner_drift_px)."""
    payload = {
        "homography": np.asarray(H, dtype=float).reshape(3, 3).tolist(),
        "max_residual_m": float(np.max(residuals_m)),
        "mean_residual_m": float(np.mean(residuals_m)),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arena_config": str(arena_config),
        "image_size": list(image_size) if image_size is not None else None,
        "notes": notes,
    }
    if corner_px:
        payload["corner_px"] = {int(k): [round(float(v[0]), 2), round(float(v[1]), 2)]
                                for k, v in corner_px.items()}
    if extra:
        payload.update(extra)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_calibration_yaml(path) -> dict:
    """The whole calibration record: ``homography`` as a 3x3 array plus
    ``image_size`` / ``corner_px`` / residuals when present."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    payload["homography"] = load_homography_yaml(path)
    payload["corner_px"] = {int(k): v for k, v in (payload.get("corner_px") or {}).items()}
    return payload


def load_homography_yaml(path) -> np.ndarray:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    H = np.asarray(payload["homography"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"{path}: homography must be 3x3, got {H.shape}")
    return H
