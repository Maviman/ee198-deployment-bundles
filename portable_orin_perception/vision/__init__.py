"""The low-latency vision path: camera -> detect -> UDP pose frames, one process.

The ROS 2 pipeline (usb_cam -> aruco_detector -> pose_bridge) stays for
learning, rviz and rosbag. This package is what `arena up` runs on the vision
Orin. It shares the detection math with the ROS nodes (hive_perception.core),
so the two paths cannot disagree about where a car is.
"""

import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
_CORE = BUNDLE_ROOT / "ros2_ws" / "src" / "hive_perception"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
