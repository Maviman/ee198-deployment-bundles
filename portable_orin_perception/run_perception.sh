#!/usr/bin/env bash
# Start the whole perception pipeline: camera -> ArUco detector -> UDP bridge.
#
#   ./run_perception.sh <controller-pc-ip> [extra ros2 launch args...]
#   ./run_perception.sh 192.168.86.42 expected_pursuers:=1 video_device:=/dev/video0
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
exec ros2 launch hive_perception perception.launch.py \
    "controller_ip:=${CONTROLLER_IP}" "$@"
