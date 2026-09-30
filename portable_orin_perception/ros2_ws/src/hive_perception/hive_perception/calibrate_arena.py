"""Arena homography calibration CLI (deliberately ROS-free — works on any
machine with OpenCV and a camera or a saved image; also runs fine inside the
sourced ROS environment on the Orin via `ros2 run hive_perception calibrate_arena`).

Procedure (full checklist in the bundle's CALIBRATION.md):
  1. Place the 4 calibration markers (ids from marker_map.yaml) at measured
     arena positions listed in the arena config YAML (edit the YAML to match
     where you ACTUALLY put them — any 4 non-collinear points work).
  2. Run this tool. It grabs frames, averages each marker's center over the
     detections, undistorts using camera_info.yaml, fits the pixel->arena
     homography, and reports per-marker residuals.
  3. Residuals <= ~0.02 m: save and move on. Larger: re-measure the worst
     marker's position first — a bad tape-measure number looks exactly like this.

Usage:
    python -m hive_perception.calibrate_arena --device 0
    python -m hive_perception.calibrate_arena --image snapshot.png
Key args: --arena-config, --marker-map, --camera-info, --out, --frames.
Paths default to the bundle's config/ when run via calibrate_arena.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from .core import arena_frame, marker_math
from .core.frame_builder import load_marker_map


def load_arena_points(path) -> dict[int, tuple[float, float]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    points = data["calibration_points"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in points.items()}


def load_ros_camera_info(path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the ROS camera_info YAML format (what the camera_calibration tool
    writes and usb_cam loads). Returns (K, D); (None, None) if unreadable."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64)
        return K, D
    except (OSError, KeyError, ValueError) as exc:
        print(f"warning: could not read camera info from {path} ({exc}); "
              "proceeding WITHOUT undistortion")
        return None, None


def collect_marker_centers(images, tag_set, wanted_ids: list[int],
                           min_detections: int = 3) -> dict[int, np.ndarray]:
    """Detect calibration markers across frames; return id -> mean center px.

    ``tag_set`` must be the marker map's own TagSet (marker_map.tag_set()), the
    same one the detector node builds: detections come back as subset ROW
    INDICES, and translating them with a different tag set would silently
    mislabel which corner is which — a calibration that then looks fine but
    maps the arena inside out.
    """
    import cv2

    detector = cv2.aruco.ArucoDetector(
        tag_set.build_opencv_dictionary(), cv2.aruco.DetectorParameters())
    seen: dict[int, list[np.ndarray]] = {i: [] for i in wanted_ids}
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for marker_corners, row in zip(corners, ids.flatten()):
            marker_id = tag_set.to_real_id(int(row))
            if marker_id in seen:
                seen[marker_id].append(marker_math.marker_center_px(marker_corners))
    centers = {}
    for vid, hits in seen.items():
        if len(hits) >= min(min_detections, len(images)):
            centers[vid] = np.mean(hits, axis=0)
    return centers


def grab_frames(device: str, count: int, width: int, height: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
    if not cap.isOpened():
        sys.exit(f"could not open camera device {device!r}")
    # Must match the live pipeline's resolution (perception.launch.py sets
    # image_width/image_height explicitly) — a homography fitted at one
    # resolution is invalid pixel geometry at another, even same aspect ratio.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        sys.exit(f"camera {device!r} produced no frames")
    return frames


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--device", help="camera device (index like 0, or /dev/video0)")
    src.add_argument("--image", help="calibrate from a saved image instead of a live camera")
    parser.add_argument("--frames", type=int, default=10, help="frames to average over (live mode)")
    parser.add_argument("--width", type=int, default=640,
                        help="capture width — MUST match perception.launch.py's image_width")
    parser.add_argument("--height", type=int, default=480,
                        help="capture height — MUST match perception.launch.py's image_height")
    parser.add_argument("--arena-config", default="config/arena_test_6ft.yaml")
    parser.add_argument("--marker-map", default="config/marker_map.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--out", default="config/arena_homography.yaml")
    parser.add_argument("--residual-warn-m", type=float, default=0.02)
    args = parser.parse_args(argv)

    import cv2

    marker_map = load_marker_map(args.marker_map)
    arena_points = load_arena_points(args.arena_config)
    corner_ids = marker_map.calibration_corner_ids
    missing_cfg = [i for i in corner_ids if i not in arena_points]
    if missing_cfg:
        sys.exit(f"{args.arena_config} lacks calibration_points entries for ids {missing_cfg}")

    if args.image:
        images = [cv2.imread(args.image)]
        if images[0] is None:
            sys.exit(f"could not read image {args.image!r}")
    else:
        images = grab_frames(args.device, args.frames, args.width, args.height)
    print(f"collected {len(images)} frame(s) at {images[0].shape[1]}x{images[0].shape[0]}")

    centers = collect_marker_centers(images, marker_map.tag_set(), corner_ids)
    not_seen = [i for i in corner_ids if i not in centers]
    if not_seen:
        sys.exit(f"calibration markers not detected reliably: ids {not_seen} — "
                 "check lighting, focus, and that the right sheet ids are placed")

    K, D = load_ros_camera_info(args.camera_info)
    px = marker_math.undistort_points(
        np.vstack([centers[i] for i in corner_ids]), K, D)
    arena = np.array([arena_points[i] for i in corner_ids])

    H, residuals = arena_frame.fit_homography(px, arena)
    print("\nper-marker residuals (m):")
    for vid, r in zip(corner_ids, residuals):
        flag = "  <-- re-measure this one?" if r > args.residual_warn_m else ""
        print(f"  id {vid}: {r:.4f}{flag}")
    if float(residuals.max()) > args.residual_warn_m:
        print(f"\nWARNING: max residual {residuals.max():.4f} m exceeds "
              f"{args.residual_warn_m} m — usable, but re-measuring is recommended.")

    arena_frame.save_homography_yaml(
        args.out, H, residuals,
        arena_config=str(args.arena_config),
        image_size=(images[0].shape[1], images[0].shape[0]),
        notes="pixel -> arena-centered meters; fitted by calibrate_arena",
    )
    print(f"\nsaved {args.out} (max residual {residuals.max():.4f} m). "
          "Re-run whenever the camera moves, zooms, or refocuses.")


if __name__ == "__main__":
    main()
