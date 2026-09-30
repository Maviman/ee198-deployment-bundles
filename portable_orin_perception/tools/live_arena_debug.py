"""Bring-up aid: live version of snapshot_arena_debug.py. Continuously reads
the overhead camera, runs the SAME detection aruco_detector_node uses (same subset
dictionary, so both what it finds and what it costs match the node),
and shows a window with detected marker boundaries/ids plus the calibrated
arena rectangle (projected back into pixels via the fitted homography) —
updated every frame, so you can watch it while physically adjusting the
camera, focus, or marker placement instead of re-running a snapshot each time.

Not a ROS node — opens the device directly with OpenCV. Needs a physical
display (X session) on the Orin. Camera must be free (stop any running
usb_cam_node_exe / rqt_image_view first — only one process can hold
/dev/video0 at a time).

    python3 tools/live_arena_debug.py --device /dev/video0

Press 'q' or Esc in the window, or Ctrl-C in the terminal, to quit.
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

WINDOW_NAME = "live arena debug (q or Esc to quit)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640,
                        help="must match perception.launch.py's image_width for the overlay to line up")
    parser.add_argument("--height", type=int, default=480,
                        help="must match perception.launch.py's image_height for the overlay to line up")
    parser.add_argument("--marker-map", default=str(ROOT / "config" / "marker_map.yaml"))
    parser.add_argument("--arena-config", default=str(ROOT / "config" / "arena_test_6ft.yaml"))
    parser.add_argument("--homography", default=str(ROOT / "config" / "arena_homography.yaml"))
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.device} — is another node "
                          "(usb_cam_node_exe, rqt_image_view) already using it?")

    marker_map = load_marker_map(args.marker_map)
    # Same SUBSET dictionary aruco_detector_node builds, so this tool matches the
    # node in both what it detects and what it costs. detectMarkers() therefore
    # returns row indices -> translate with tag_set.to_real_id() before use.
    tag_set = marker_map.tag_set()
    detector = cv2.aruco.ArucoDetector(
        tag_set.build_opencv_dictionary(), cv2.aruco.DetectorParameters())

    arena_px = None
    boundary_note = "no arena_homography.yaml yet — run calibrate_arena.sh for the boundary overlay"
    if Path(args.homography).exists():
        arena_cfg = yaml.safe_load(Path(args.arena_config).read_text())
        half_w, half_h = arena_cfg["arena_width_m"] / 2, arena_cfg["arena_height_m"] / 2
        H = arena_frame.load_homography_yaml(args.homography)
        arena_corners_m = np.array([[-half_w, half_h], [half_w, half_h],
                                     [half_w, -half_h], [-half_w, -half_h]])
        arena_px = arena_frame.project_px_to_arena(np.linalg.inv(H), arena_corners_m).astype(np.int32)
        boundary_note = "yellow = calibrated arena boundary"

    print(f"streaming from {args.device} — {boundary_note}")
    print("press 'q' or Esc in the window to quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("frame read failed — camera disconnected?")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rejected = detector.detectMarkers(gray)

            out = frame
            if ids is not None:
                # Label with PRINTED ids, not the subset's row indices.
                cv2.aruco.drawDetectedMarkers(out, corners, np.array(
                    [[tag_set.to_real_id(int(i))] for i in ids.flatten()], dtype=np.int32))
            if arena_px is not None:
                cv2.polylines(out, [arena_px], isClosed=True, color=(0, 255, 255), thickness=2)
                for (x, y) in arena_px:
                    cv2.circle(out, (x, y), 4, (0, 255, 255), -1)

            n_found = 0 if ids is None else len(ids)
            found_ids = [] if ids is None else sorted(
                tag_set.to_real_id(int(i)) for i in ids.flatten())
            label = f"detected: {n_found} marker(s) {found_ids}  |  {boundary_note}"
            cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = Esc
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
