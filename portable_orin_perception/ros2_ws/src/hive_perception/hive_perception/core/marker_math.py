"""Marker-level geometry in PIXEL space: ArUco corner arrays -> marker center
and top-edge midpoint (the heading reference), plus lens-distortion removal.
Arena-frame conversion (meters, radians) lives in arena_frame.py.

OpenCV ArUco corner order is [top-left, top-right, bottom-right, bottom-left]
of the marker's CANONICAL orientation (as generated/printed), regardless of how
the marker currently appears in the image. Mounting rule that follows: tape the
marker so its canonical top edge faces the car's NOSE; the vector from the
marker center to that edge's midpoint is then the car's forward direction and
no per-car angle offset is ever needed.
"""

from __future__ import annotations

import numpy as np


def as_corner_array(corners) -> np.ndarray:
    """Normalize any cv2.aruco corner blob ((1,4,2), (4,2), nested list) to a
    float64 (4,2) array."""
    return np.asarray(corners, dtype=np.float64).reshape(4, 2)


def marker_center_px(corners) -> np.ndarray:
    """Marker center in pixels: mean of the 4 corners. (The projected centroid
    differs from the true center under perspective skew by far less than our
    noise floor at overhead-camera geometry.)"""
    return as_corner_array(corners).mean(axis=0)


def marker_top_midpoint_px(corners) -> np.ndarray:
    """Midpoint of the canonical TOP edge (corners 0 and 1) in pixels — the
    point 'ahead of' the marker center along the car's nose direction."""
    arr = as_corner_array(corners)
    return (arr[0] + arr[1]) / 2.0


def undistort_points(points_px, camera_matrix, dist_coeffs) -> np.ndarray:
    """Remove lens distortion from (N,2) pixel points, returning (N,2) points
    still in pixel units (cv2.undistortPoints with P=camera_matrix). A
    homography cannot absorb lens distortion — distortion is not a projective
    transform — so this must run before any arena_frame projection. No-op when
    the calibration is absent or all-zero (the uncalibrated template)."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if camera_matrix is None or dist_coeffs is None:
        return pts
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dist.size == 0 or not np.any(dist):
        return pts
    import cv2  # local import: pure-geometry helpers above stay OpenCV-free

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    out = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K)
    return out.reshape(-1, 2)
