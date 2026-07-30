"""Bring up the whole perception pipeline: usb_cam -> aruco_detector -> pose_bridge.

Config files are found through the HIVE_PERCEPTION_ROOT environment variable
(the bundle root). run_perception.sh sets it automatically; set it yourself if
launching by hand:

    export HIVE_PERCEPTION_ROOT=/path/to/portable_orin_perception
    ros2 launch hive_perception perception.launch.py controller_ip:=192.168.86.42

Launch arguments (ros2 launch hive_perception perception.launch.py --show-args):
    controller_ip      where UDP pose frames go (the PC running run_controller.py)
    controller_port    default 9870 (the controller's --pose-port)
    expected_pursuers  how many pursuer cars are marked up (default 1)
    video_device       default /dev/video0
    send_rate_hz       pose-frame rate to the controller (default 10.0)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BUNDLE_ROOT = os.environ.get("HIVE_PERCEPTION_ROOT", os.getcwd())


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(BUNDLE_ROOT, "config")
    return LaunchDescription([
        DeclareLaunchArgument("controller_ip", default_value="127.0.0.1"),
        DeclareLaunchArgument("controller_port", default_value="9870"),
        DeclareLaunchArgument("expected_pursuers", default_value="1"),
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("send_rate_hz", default_value="10.0"),
        # Safety dead-man cutoff (default matches the documented 0.25s hard
        # constraint — do not raise this for anything but a wheels-off,
        # confirm-the-wiring run; revert once real end-to-end latency is fixed).
        DeclareLaunchArgument("hold_max_age_s", default_value="0.25"),

        # Camera. 640x480@15 chosen so aruco_detector (pure-CPU, no GPU accel)
        # can actually keep up in real time on this hardware — 1280x720 was
        # measured pegging aruco_detector at ~150% CPU with frames queueing
        # 0.5-0.9s stale, permanently tripping the 0.25s dead-man even with
        # clean detections. Must match calibrate_arena.sh's --width/--height
        # (and re-run calibration) if you change this.
        Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            name="overhead_camera",
            parameters=[{
                "video_device": LaunchConfiguration("video_device"),
                "image_width": 640,
                "image_height": 480,
                "framerate": 15.0,
                "pixel_format": "mjpeg2rgb",
                "camera_name": "overhead",
                "camera_info_url": "file://" + os.path.join(config, "camera_info.yaml"),
            }],
        ),

        # Detector: images -> arena-frame PoseArray. Topic names below assume
        # usb_cam publishes /image_raw and /camera_info; if your version
        # namespaces them (ros2 topic list will show it), fix the remappings.
        Node(
            package="hive_perception",
            executable="aruco_detector",
            name="aruco_detector",
            parameters=[{
                "marker_map_path": os.path.join(config, "marker_map.yaml"),
                "homography_path": os.path.join(config, "arena_homography.yaml"),
            }],
            remappings=[
                ("image_raw", "/image_raw"),
                ("camera_info", "/camera_info"),
            ],
        ),

        # Bridge: PoseArray -> UDP JSON pose frames for the controller.
        Node(
            package="hive_perception",
            executable="pose_bridge",
            name="pose_bridge",
            parameters=[{
                "controller_ip": LaunchConfiguration("controller_ip"),
                "controller_port": LaunchConfiguration("controller_port"),
                "expected_pursuers": LaunchConfiguration("expected_pursuers"),
                "send_rate_hz": LaunchConfiguration("send_rate_hz"),
                "hold_max_age_s": LaunchConfiguration("hold_max_age_s"),
            }],
        ),
    ])
