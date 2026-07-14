"""Colcon-side smoke tests for the rclpy-free core (run by `colcon test` on the
Orin). The heavyweight contract tests live in the main repo's pytest suite
(tests/test_orin_perception_core.py) — this file just proves the core imports
and behaves sanely inside the ROS environment."""

from __future__ import annotations

import json
import math

import numpy as np

from hive_perception.core import arena_frame, marker_math
from hive_perception.core.frame_builder import FrameBuilder


def make_square_homography() -> np.ndarray:
    px = [[100, 500], [500, 500], [500, 100], [100, 100]]
    arena = [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]]
    H, residuals = arena_frame.fit_homography(px, arena)
    assert float(residuals.max()) < 1e-6  # exact 4-point fit up to SVD float noise
    return H


def test_pose_from_marker_center_and_heading():
    H = make_square_homography()
    x, y, heading = arena_frame.pose_from_marker(H, [300.0, 300.0], [300.0, 250.0])
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert abs(heading - math.pi / 2) < 1e-6  # up in image = +y in arena here


def test_frame_builder_schema_and_staleness():
    fb = FrameBuilder(pursuer_ids=[0], evader_id=1, hold_max_age_s=0.25)
    fb.update({0: (1.0, -2.0, 0.5), 1: (3.0, 1.0, -1.2)}, t=10.0)
    frame = json.loads(fb.build_frame(10.05))
    assert frame["t"] == 10.0
    assert frame["pursuers"][0] == {"x": 1.0, "y": -2.0, "heading": 0.5}
    assert frame["evader"] == {"x": 3.0, "y": 1.0, "heading": -1.2}
    assert fb.build_frame(10.3) is None  # past the hold cap -> silence


def test_undistort_noop_without_calibration():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = marker_math.undistort_points(pts, None, None)
    assert np.allclose(out, pts)
