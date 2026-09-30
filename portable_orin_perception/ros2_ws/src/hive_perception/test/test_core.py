"""Colcon-side smoke tests for the rclpy-free core (run by `colcon test` on the
Orin). The heavyweight contract tests live in the main repo's pytest suite
(tests/test_orin_perception_core.py) — this file just proves the core imports
and behaves sanely inside the ROS environment."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hive_perception.core import arena_frame, marker_math, tag_family
from hive_perception.core.frame_builder import FrameBuilder, MarkerMap


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


# --------------------------------------------------------------------------
# Subset dictionary: the row-index/real-id translation. Getting this wrong
# swaps vehicle identities silently, so it is pinned hard.
# --------------------------------------------------------------------------

VEHICLE_MAP = MarkerMap(
    dictionary="DICT_APRILTAG_36h11",
    evader_id=0,
    pursuer_ids=[1, 2, 3],
    calibration_corner_ids=[10, 11, 12, 13],
)


def test_all_ids_order_is_vehicles_then_corners():
    assert VEHICLE_MAP.all_ids == [0, 1, 2, 3, 10, 11, 12, 13]


def test_tag_set_round_trips_row_index_to_printed_id():
    ts = VEHICLE_MAP.tag_set()
    assert [ts.to_real_id(i) for i in range(len(ts.real_ids))] == VEHICLE_MAP.all_ids


def test_tag_set_rejects_out_of_range_row():
    ts = VEHICLE_MAP.tag_set()
    with pytest.raises(IndexError):
        ts.to_real_id(len(ts.real_ids))


def test_subset_rows_are_bit_identical_to_the_real_codebook():
    """The printed tag must be a genuine 36h11 member, or cuAprilTags — which
    only knows the canonical codebook — will not read our sheets."""
    cv2 = pytest.importorskip("cv2")
    full = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    ts = VEHICLE_MAP.tag_set()
    sub = ts.build_opencv_dictionary()
    for row, real in enumerate(ts.real_ids):
        assert np.array_equal(sub.bytesList[row], full.bytesList[real])


def test_detection_of_full_codebook_tags_maps_back_to_printed_ids():
    """End-to-end: render from the FULL codebook at real ids (what the printer
    produces), detect with the SUBSET dictionary (what the node runs)."""
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2.aruco, "ArucoDetector"):
        pytest.skip("OpenCV < 4.7")
    full = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    ts = VEHICLE_MAP.tag_set()
    side, gap = 96, 36
    canvas = np.full((side + 2 * gap, len(ts.real_ids) * (side + gap) + gap), 255, np.uint8)
    for k, real in enumerate(ts.real_ids):
        x = gap + k * (side + gap)
        canvas[gap:gap + side, x:x + side] = cv2.aruco.generateImageMarker(full, real, side)
    detector = cv2.aruco.ArucoDetector(
        ts.build_opencv_dictionary(), cv2.aruco.DetectorParameters())
    _corners, ids, _rej = detector.detectMarkers(canvas)
    assert ids is not None, "subset dictionary detected nothing"
    assert sorted(ts.to_real_id(int(r)) for r in ids.flatten()) == list(ts.real_ids)


def test_error_correction_cannot_alias_two_printed_tags():
    ts = VEHICLE_MAP.tag_set()
    d = ts.min_hamming_distance()
    assert ts.max_correction_bits <= (d - 1) // 2, (
        f"correcting {ts.max_correction_bits} bits with min distance {d} can turn "
        "one vehicle's tag into another's")


def test_family_pixel_budget_penalises_apriltag_correctly():
    aruco = tag_family.family_spec("DICT_4X4_50")
    april = tag_family.family_spec("DICT_APRILTAG_36h11")
    assert (aruco.total_modules, april.total_modules) == (6, 8)
    # Same tag, same camera: 36h11 covers 3/4 the ground ArUco does.
    a = tag_family.max_ground_width_m(0.07, 1280, aruco)
    b = tag_family.max_ground_width_m(0.07, 1280, april)
    assert abs(b / a - 6 / 8) < 1e-9


def test_unknown_family_fails_loudly():
    with pytest.raises(ValueError):
        tag_family.family_spec("DICT_NOT_A_REAL_FAMILY")
