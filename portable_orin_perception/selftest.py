"""Prove this perception bundle works on THIS machine — no ROS, no camera:

  1. Fiducial end-to-end, using the family configured in marker_map.yaml:
     render a tag at its printed id, detect it through the SUBSET dictionary
     the node actually runs, translate the row index back to the printed id,
     and recover its arena pose (catches a broken/old OpenCV install — needs
     the 4.7+ ArucoDetector API — and a mis-wired subset id mapping).
  2. Homography math: fit from synthetic correspondences (including the image
     y-down -> arena y-up flip), check residuals ~0 and heading conventions.
  3. Frame builder: staleness/hold policy (the dead-man safety semantics).
  4. Wire-format round-trip: FrameBuilder JSON parsed by the CONTROLLER'S OWN
     parser (byte-identical vendored copy of portable_n1_controller's
     pose_stream.py) — proves this bundle speaks the controller's language.

Run:  python3 selftest.py     (works before setup_orin.sh — only needs
                               numpy + opencv + pyyaml, see requirements.txt)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))
sys.path.insert(0, str(ROOT / "vendored"))

from hive_perception.core import arena_frame, marker_math, tag_family  # noqa: E402
from hive_perception.core.frame_builder import FrameBuilder, load_marker_map  # noqa: E402
from pc_controller.pose_stream import parse_pose_frame  # noqa: E402 (vendored)

# Pure scale+flip homography: 800x800 px image of a 1.6x1.6 m arena patch,
# pixel (400,400) = arena (0,0), image up = arena +y.
CAL_PX = [[100, 700], [700, 700], [700, 100], [100, 100]]
CAL_ARENA = [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]]


def check_fiducial_detection() -> list[str]:
    """Exercise the REAL configured family through the REAL subset-dictionary
    path: render a tag from the full codebook at its printed id (what the
    printer produces), detect it with the subset dictionary (what the node
    runs), translate the row index back, and localize it.

    Testing the full predefined dictionary instead would pass even if the
    subset mapping were inverted — which is the failure that silently swaps
    the evader for a pursuer.
    """
    import cv2

    failures = []
    if not hasattr(cv2.aruco, "ArucoDetector"):
        return [f"OpenCV {cv2.__version__} lacks cv2.aruco.ArucoDetector (need >= 4.7); "
                "install opencv-contrib-python or a newer opencv-python"]

    marker_map = load_marker_map(ROOT / "config" / "marker_map.yaml")
    try:
        spec = tag_family.family_spec(marker_map.dictionary)
    except ValueError as exc:
        return [str(exc)]
    if not hasattr(cv2.aruco, spec.opencv_dict):
        return [f"OpenCV {cv2.__version__} has no {spec.opencv_dict}; "
                "install opencv-contrib-python"]

    tag_set = marker_map.tag_set()
    printed_id = marker_map.pursuer_ids[0]
    full = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.opencv_dict))
    marker = cv2.aruco.generateImageMarker(full, printed_id, 120)
    scene = np.full((800, 800), 255, dtype=np.uint8)
    scene[340:460, 340:460] = marker  # centered at px (400,400), unrotated

    detector = cv2.aruco.ArucoDetector(
        tag_set.build_opencv_dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(scene)
    if ids is None:
        return [f"subset dictionary detected no {marker_map.dictionary} tag "
                "(OpenCV install problem, or a bad family/id in marker_map.yaml)"]
    decoded = [tag_set.to_real_id(int(r)) for r in ids.flatten()]
    if printed_id not in decoded:
        return [f"printed id {printed_id} came back as {decoded} — the subset "
                "dictionary's row-index mapping is wrong"]
    blob = corners[decoded.index(printed_id)]

    H, _ = arena_frame.fit_homography(CAL_PX, CAL_ARENA)
    x, y, heading = arena_frame.pose_from_marker(
        H, marker_math.marker_center_px(blob), marker_math.marker_top_midpoint_px(blob))
    if abs(x) > 0.01 or abs(y) > 0.01:
        failures.append(f"synthetic tag center off: ({x:.4f}, {y:.4f}), expected (0, 0)")
    if abs(heading - math.pi / 2) > 0.01:
        failures.append(f"synthetic tag heading {heading:.4f}, expected pi/2 "
                        "(canonical top edge toward image top = arena +y)")
    if tag_set.max_correction_bits > (tag_set.min_hamming_distance() - 1) // 2:
        failures.append(
            f"max_correction_bits={tag_set.max_correction_bits} exceeds what the "
            f"chosen ids can safely correct (min Hamming distance "
            f"{tag_set.min_hamming_distance()}) — one vehicle's tag could be read "
            "as another's")
    print(f"  rendered, detected, and localized a real {marker_map.dictionary} tag "
          f"(id {printed_id}) through the {len(tag_set.real_ids)}-code subset")
    return failures


def check_homography_math() -> list[str]:
    failures = []
    H, residuals = arena_frame.fit_homography(CAL_PX, CAL_ARENA)
    if float(residuals.max()) > 1e-6:  # 4 points define H exactly; 1e-6 m leaves room for SVD float noise
        failures.append(f"exact-fit residual too big: {residuals.max():.3e}")

    x, y, heading = arena_frame.pose_from_marker(H, [400.0, 400.0], [450.0, 400.0])
    if abs(heading) > 1e-6:
        failures.append(f"+x heading wrong: {heading}")
    _, _, heading_up = arena_frame.pose_from_marker(H, [400.0, 400.0], [400.0, 350.0])
    if abs(heading_up - math.pi / 2) > 1e-6:
        failures.append(f"image-up must map to +pi/2 (y flip!), got {heading_up}")

    corner = arena_frame.project_px_to_arena(H, [[100.0, 100.0]])[0]
    if not np.allclose(corner, [-0.8, 0.8], atol=1e-6):
        failures.append(f"corner projection wrong: {corner}")

    pts = np.array([[10.0, 20.0], [630.0, 470.0]])
    if not np.allclose(marker_math.undistort_points(pts, None, None), pts):
        failures.append("undistort_points is not a no-op without calibration")
    print("  homography fit/projection/heading conventions checked")
    return failures


def check_frame_builder_policy() -> list[str]:
    failures = []
    fb = FrameBuilder(pursuer_ids=[0], evader_id=1, hold_max_age_s=0.25)
    if fb.build_frame(0.0) is not None:
        failures.append("frame built before any detections")
    fb.update({0: (1.0, -2.0, 0.5)}, t=10.0)
    if fb.build_frame(10.0) is not None:
        failures.append("frame built with the evader never seen")
    fb.update({1: (3.0, 1.0, -1.2)}, t=10.1)
    if fb.build_frame(10.2) is None:
        failures.append("complete frame not built while poses are fresh")
    if fb.build_frame(10.1 + 0.6) is not None:
        failures.append("stale poses (past hold cap) still produced a frame")
    fb.update({0: (1.0, -2.0, 0.5), 1: (3.0, 1.0, -1.2)}, t=11.0)
    frame = json.loads(fb.build_frame(11.05))
    if frame["t"] != 11.0:
        failures.append(f"frame t must be capture time (11.0), got {frame['t']}")
    print("  staleness/hold dead-man policy checked")
    return failures


def check_controller_roundtrip() -> list[str]:
    failures = []
    fb = FrameBuilder(pursuer_ids=[7, 8], evader_id=9)
    fb.update({7: (1.0, -2.0, 0.5), 8: (0.25, 0.75, -3.0), 9: (3.0, 1.0, -1.2)}, t=12.5)
    raw = fb.build_frame(12.55)
    pursuers, evader = parse_pose_frame(raw.encode("utf-8"), expected_pursuers=2)
    if (pursuers[0].x, pursuers[0].y, pursuers[0].heading, pursuers[0].timestamp_s) != (1.0, -2.0, 0.5, 12.5):
        failures.append(f"controller parsed pursuer 0 wrong: {pursuers[0]}")
    if (pursuers[1].x, pursuers[1].heading) != (0.25, -3.0):
        failures.append(f"controller parsed pursuer 1 wrong: {pursuers[1]}")
    if (evader.x, evader.y, evader.heading) != (3.0, 1.0, -1.2):
        failures.append(f"controller parsed evader wrong: {evader}")
    try:
        parse_pose_frame(raw.encode("utf-8"), expected_pursuers=3)
        failures.append("pursuer-count mismatch not rejected by the controller parser")
    except ValueError:
        pass
    print("  frame JSON accepted by the controller's own (vendored) parser")
    return failures


def main() -> None:
    failures: list[str] = []
    print("[1/4] fiducial render -> subset detect -> localize")
    failures += check_fiducial_detection()
    print("[2/4] homography + heading conventions")
    failures += check_homography_math()
    print("[3/4] frame-builder dead-man policy")
    failures += check_frame_builder_policy()
    print("[4/4] controller wire-format round-trip")
    failures += check_controller_roundtrip()

    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSELFTEST PASSED - perception bundle is healthy on this machine.")


if __name__ == "__main__":
    main()
