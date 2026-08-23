#!/usr/bin/env bash
# Start the whole perception pipeline: camera -> ArUco detector -> UDP bridge.
# With debug:=true this is a single command that ALSO opens the live camera/
# detection window (rqt_image_view on /hive/debug_image) — no second command
# needed to see what the camera sees.
#
#   ./run_perception.sh <controller-pc-ip> [extra ros2 launch args...]
#   ./run_perception.sh 192.168.86.42 expected_pursuers:=1 video_device:=/dev/video0
#   ./run_perception.sh 127.0.0.1 debug:=true   # + opens the debug window
#
# GPU paths (measure first with tools/probe_orin_gpu.py):
#   camera_backend:=gst    hardware MJPEG decode on the NVJPG engine instead of
#                          usb_cam's CPU mjpeg2rgb
#   image_width:=1280 image_height:=720
#                          higher capture res — REQUIRES re-running
#                          ./calibrate_arena.sh at the same size
#
# Requires setup_orin.sh to have been run once, and (for real poses) a saved
# arena calibration in config/arena_homography.yaml (./calibrate_arena.sh).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

CONTROLLER_IP="${1:?usage: ./run_perception.sh <controller-pc-ip> [launch args...]}"
shift || true

if [ ! -f config/arena_homography.yaml ]; then
    echo "config/arena_homography.yaml not found — run ./calibrate_arena.sh first." >&2
    exit 1
fi

UBUNTU_VER="$(. /etc/os-release && echo "${VERSION_ID}")"
case "${UBUNTU_VER}" in
    22.04) ROS_DISTRO=humble ;;
    24.04) ROS_DISTRO=jazzy ;;
    *) echo "Unsupported Ubuntu ${UBUNTU_VER}"; exit 1 ;;
esac
# ROS 2's setup.bash references unset variables internally (e.g.
# AMENT_TRACE_SETUP_FILES) and isn't nounset-safe — relax -u just for sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
# shellcheck disable=SC1091
source "ros2_ws/install/setup.bash"
set -u

export HIVE_PERCEPTION_ROOT="$(pwd)"

VIEWER_PID=""
for arg in "$@"; do
    if [ "$arg" = "debug:=true" ]; then
        ros2 run rqt_image_view rqt_image_view /hive/debug_image &
        VIEWER_PID=$!
        trap '[ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null || true' EXIT INT TERM
        break
    fi
done

ros2 launch hive_perception perception.launch.py \
    "controller_ip:=${CONTROLLER_IP}" "$@"
