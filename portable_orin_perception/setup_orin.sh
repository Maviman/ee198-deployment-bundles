#!/usr/bin/env bash
# One-time Jetson Orin setup for the perception bundle. Idempotent — safe to
# re-run. Detects the JetPack Ubuntu release and installs the matching ROS 2
# distro (22.04 -> Humble, 24.04 -> Jazzy), the packages this bundle needs,
# and builds the colcon workspace.
#
#   cd portable_orin_perception && ./setup_orin.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

echo "== [1/6] Detecting platform =="
if [ -f /etc/nv_tegra_release ]; then
    echo "Jetson L4T: $(head -c 60 /etc/nv_tegra_release)"
else
    echo "note: /etc/nv_tegra_release not found — not a Jetson? Continuing anyway."
fi
UBUNTU_VER="$(. /etc/os-release && echo "${VERSION_ID}")"
case "${UBUNTU_VER}" in
    22.04) ROS_DISTRO=humble ;;
    24.04) ROS_DISTRO=jazzy ;;
    *) echo "Unsupported Ubuntu ${UBUNTU_VER} (need 22.04 or 24.04)"; exit 1 ;;
esac
echo "Ubuntu ${UBUNTU_VER} -> ROS 2 ${ROS_DISTRO}"

echo "== [2/6] ROS 2 apt repository =="
sudo apt-get update && sudo apt-get install -y curl gnupg lsb-release software-properties-common
sudo add-apt-repository -y universe
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
ARCH="$(dpkg --print-architecture)"
CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME}")"
echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update

echo "== [3/6] Installing ROS 2 ${ROS_DISTRO} + tools =="
sudo apt-get install -y \
    "ros-${ROS_DISTRO}-desktop" \
    ros-dev-tools \
    "ros-${ROS_DISTRO}-usb-cam" \
    "ros-${ROS_DISTRO}-cv-bridge" \
    "ros-${ROS_DISTRO}-camera-calibration" \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-opencv python3-numpy python3-yaml \
    chrony v4l-utils

echo "== [4/6] Checking OpenCV ArUco =="
python3 - <<'PY'
import cv2
ok = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
print(f"OpenCV {cv2.__version__}, ArucoDetector available: {ok}")
if not ok:
    raise SystemExit(
        "cv2.aruco.ArucoDetector missing (OpenCV < 4.7?). Fix with:\n"
        "  pip3 install --user opencv-contrib-python\n"
        "then re-run this script.")
PY

echo "== [5/6] rosdep =="
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update || true

echo "== [6/6] Building the workspace =="
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
(cd ros2_ws && colcon build --symlink-install)

echo
echo "Setup complete. Next:"
echo "  python3 selftest.py                      # must print SELFTEST PASSED"
echo "  ./calibrate_arena.sh --device /dev/video0   # after placing corner markers"
echo "  ./run_perception.sh <controller-pc-ip>      # start the pipeline"
echo "Clock sync (latency numbers need it): chrony is installed; verify with"
echo "  chronyc tracking    # offset should be a few ms or better"
