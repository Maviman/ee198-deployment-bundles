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

echo "== [4/6] Checking OpenCV fiducial support =="
python3 - <<'PY'
import cv2
ok = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
print(f"OpenCV {cv2.__version__}, ArucoDetector available: {ok}")
if not ok:
    raise SystemExit(
        "cv2.aruco.ArucoDetector missing (OpenCV < 4.7?). Fix with:\n"
        "  pip3 install --user opencv-contrib-python\n"
        "then re-run this script.")

# Check the family the marker map ACTUALLY names, not a hard-coded one -- the
# family is a config choice (ArUco today, AprilTag tag36h11 once the sheets are
# reprinted) and this script must not demand whichever one it was written for.
import pathlib, yaml
cfg = pathlib.Path("config/marker_map.yaml")
family = yaml.safe_load(cfg.read_text(encoding="utf-8"))["dictionary"] if cfg.exists() else None
if family:
    if not hasattr(cv2.aruco, family):
        raise SystemExit(
            f"cv2.aruco.{family} missing -- this OpenCV build cannot detect or generate\n"
            f"the family named in {cfg}. Fix with:\n"
            "  pip3 install --user opencv-contrib-python\n"
            "then re-run this script.")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    print(f"{family}: {d.bytesList.shape[0]} tags, {d.markerSize}x{d.markerSize} data bits")
    # The subset-dictionary constructor is the exact API the detector depends on
    # and is newer than ArucoDetector itself, so prove it here rather than at
    # the first camera frame.
    import numpy as np
    probe = cv2.aruco.Dictionary(np.ascontiguousarray(d.bytesList[:2, :, :]), d.markerSize, 0)
    print(f"subset-dictionary API: OK ({probe.bytesList.shape[0]} rows)")
PY

echo "== [5/6] rosdep =="
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update || true

echo "== [6/6] Building the workspace =="
# ROS 2's setup.bash references unset variables internally (e.g.
# AMENT_TRACE_SETUP_FILES) and isn't nounset-safe — relax -u just for sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
(cd ros2_ws && colcon build --symlink-install)

echo
echo "Setup complete. Next:"
echo "  python3 selftest.py                      # must print SELFTEST PASSED"
echo "  ./calibrate_arena.sh --device /dev/video0   # after placing corner markers"
echo "  ./run_perception.sh <controller-pc-ip>      # start the pipeline"
echo "Clock sync (latency numbers need it): chrony is installed; verify with"
echo "  chronyc tracking    # offset should be a few ms or better"
