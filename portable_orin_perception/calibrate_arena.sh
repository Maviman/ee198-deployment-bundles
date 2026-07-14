#!/usr/bin/env bash
# Fit the pixel -> arena homography from the 4 calibration markers.
# Full checklist: CALIBRATION.md. Typical use:
#
#   ./calibrate_arena.sh --device /dev/video0
#   ./calibrate_arena.sh --image snapshot.png
#
# Writes config/arena_homography.yaml. Re-run whenever the camera moves.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

export PYTHONPATH="$(pwd)/ros2_ws/src/hive_perception${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m hive_perception.calibrate_arena \
    --arena-config config/arena_test_6ft.yaml \
    --marker-map config/marker_map.yaml \
    --camera-info config/camera_info.yaml \
    --out config/arena_homography.yaml \
    "$@"
