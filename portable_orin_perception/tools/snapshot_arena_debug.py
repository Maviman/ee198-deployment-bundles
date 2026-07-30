"""Bring-up aid: grab one frame from the overhead camera, run the SAME ArUco
detection the real aruco_detector_node uses, and draw what it saw — detected
marker boundaries/ids plus the calibrated arena rectangle (projected back into
pixels via the fitted homography). Answers "is the camera/marker/calibration
setup actually correct?" without spinning up ROS or trusting numbers blind.

Not a ROS node and not part of the live pipeline — aruco_detector_node itself
publishes no debug image, so this exists purely as a snapshot check. Camera
must be free (stop any running usb_cam_node_exe / rqt_image_view first).

    python3 tools/snapshot_arena_debug.py --device /dev/video0
    python3 tools/snapshot_arena_debug.py --out /tmp/check.png --warmup-frames 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))

from hive_perception.core import arena_frame  # noqa: E402
from hive_perception.core.frame_builder import load_marker_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640,
                        help="must match perception.launch.py's image_width for the overlay to line up")
    parser.add_argument("--height", type=int, default=480,
                        help="must match perception.launch.py's image_height for the overlay to line up")
    parser.add_argument("--warmup-frames", type=int, default=15,
                        help="frames to discard while auto-exposure settles")
    parser.add_argument("--marker-map", default=str(ROOT / "config" / "marker_map.yaml"))
    parser.add_argument("--arena-config", default=str(ROOT / "config" / "arena_test_6ft.yaml"))
    parser.add_argument("--homography", default=str(ROOT / "config" / "arena_homography.yaml"))
    parser.add_argument("--out", default=str(ROOT / "arena_debug_snapshot.png"))
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.device} — is another node "
                          "(usb_cam_node_exe, rqt_image_view) already using it?")

    frame = None
    for _ in range(max(1, args.warmup_frames)):
        ret, frame = cap.read()
        if not ret:
            raise SystemExit(f"failed to read from {args.device}")
    cap.release()

    marker_map = load_marker_map(args.marker_map)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_map.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    out = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

    if Path(args.homography).exists():
        arena_cfg = yaml.safe_load(Path(args.arena_config).read_text())
        half_w, half_h = arena_cfg["arena_width_m"] / 2, arena_cfg["arena_height_m"] / 2
        H = arena_frame.load_homography_yaml(args.homography)
        arena_corners_m = np.array([[-half_w, half_h], [half_w, half_h],
                                     [half_w, -half_h], [-half_w, -half_h]])
        arena_px = arena_frame.project_px_to_arena(np.linalg.inv(H), arena_corners_m)
        cv2.polylines(out, [arena_px.astype(np.int32)], isClosed=True,
                      color=(0, 255, 255), thickness=2)
        for (x, y) in arena_px.astype(np.int32):
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
        boundary_note = "yellow = calibrated arena boundary"
    else:
        boundary_note = "no arena_homography.yaml yet — run calibrate_arena.sh for the boundary overlay"

    n_found = 0 if ids is None else len(ids)
    found_ids = [] if ids is None else sorted(int(i) for i in ids.flatten())
    label = f"detected: {n_found} marker(s) {found_ids}  |  {boundary_note}"
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}")
    print(f"detected ids: {found_ids}")


if __name__ == "__main__":
    main()
