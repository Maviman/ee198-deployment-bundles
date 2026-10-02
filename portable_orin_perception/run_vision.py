"""Low-latency vision service: camera -> tags -> UDP pose frames, one process.

This is what `arena up` runs on the vision Orin. No ROS on the hot path: the
ROS 2 pipeline (run_perception.sh) still works and shares all the detection
math, but it pays for an image hop through DDS and a 10 Hz resampling timer
that this path does not.

    python3 run_vision.py --pose-target 10.42.0.2:9870 --telemetry 10.42.0.2:9871
    python3 run_vision.py --synthetic --pose-target 127.0.0.1:9870     # no camera

Capture mode comes from config/camera.yaml; the arena calibration from
config/arena_homography.yaml (`arena scan` writes it). The service refuses to
start if the two disagree on resolution.
"""

from __future__ import annotations

import os

# Before OpenCV opens any camera: grab() gives up after 1 s instead of 10, so a
# hung webcam is detected and reopened in seconds (vision/capture.py).
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")

import argparse  # noqa: E402
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision.service import VisionService  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-target", default="127.0.0.1:9870",
                    help="controller host:port for pose frames (comma-separate several)")
    ap.add_argument("--telemetry", default=None, help="arena hub host:port (e.g. 10.42.0.2:9871)")
    ap.add_argument("--telemetry-hz", type=float, default=15.0)
    ap.add_argument("--preview-port", type=int, default=8090, help="MJPEG preview port, 0 = off")
    ap.add_argument("--preview-fps", type=float, default=10.0)
    ap.add_argument("--device", default=None, help="override config/camera.yaml device")
    ap.add_argument("--camera-config", default=None)
    ap.add_argument("--marker-map", default=None)
    ap.add_argument("--pursuers", type=int, default=None,
                    help="track the first N pursuer tags of config/marker_map.yaml (CAR_INDEX 0..N-1); "
                         "must match the model's num_pursuers. Default: all of them")
    ap.add_argument("--arena-config", default=None)
    ap.add_argument("--homography", default=None)
    ap.add_argument("--hold-max-age-s", type=float, default=0.25,
                    help="dead-man hold cap (do not raise for a driving session)")
    ap.add_argument("--synthetic", action="store_true", help="render tags instead of opening a camera")
    ap.add_argument("--world-port", type=int, default=None,
                    help="with --synthetic: take vehicle poses from tools/sim_world.py on this port")
    ap.add_argument("--no-roi", action="store_true", help="disable tracking windows (A/B testing)")
    ap.add_argument("--no-gating", action="store_true", help="disable plausibility gates (A/B testing)")
    ap.add_argument("--cv-threads", type=int, default=0, help="cv2.setNumThreads (0 = OpenCV default)")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[vision {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    return VisionService(args, log=log).run()


if __name__ == "__main__":
    raise SystemExit(main())
