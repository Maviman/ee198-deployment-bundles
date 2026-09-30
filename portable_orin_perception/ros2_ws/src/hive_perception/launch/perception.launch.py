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
    debug              publish an annotated /hive/debug_image (default false);
                        view with: ros2 run rqt_image_view rqt_image_view
    camera_backend     usb_cam (CPU MJPEG decode, default) | gst (hardware
                        NVJPG decode via GStreamer nvjpegdec). Measure both
                        with tools/probe_orin_gpu.py before switching.
    image_width        capture width  } defaults come from config/camera.yaml,
    image_height       capture height } the single source of truth shared with
    framerate          capture fps    } the fast path and `arena scan`
    video_device                      }
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

BUNDLE_ROOT = os.environ.get("HIVE_PERCEPTION_ROOT", os.getcwd())


def _camera_defaults():
    """config/camera.yaml (+ camera.local.yaml): the one capture-mode file every
    consumer reads, so the ROS path, the fast path and the calibration cannot
    disagree about resolution."""
    # No silent fallback: a guessed resolution against a real calibration is a
    # wrong pose on every frame. Fail the launch with the reason instead.
    try:
        from hive_perception.core.camera_config import load_camera_config
        return load_camera_config(os.path.join(BUNDLE_ROOT, "config", "camera.yaml"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cannot read {os.path.join(BUNDLE_ROOT, 'config', 'camera.yaml')}: {exc}. Fix the "
            "file, or rebuild the workspace (colcon build --symlink-install) if "
            "hive_perception.core.camera_config is missing") from exc


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(BUNDLE_ROOT, "config")
    cam = _camera_defaults()
    return LaunchDescription([
        DeclareLaunchArgument("controller_ip", default_value="127.0.0.1"),
        DeclareLaunchArgument("controller_port", default_value="9870"),
        DeclareLaunchArgument("expected_pursuers", default_value="1"),
        DeclareLaunchArgument("video_device", default_value=str(cam.device)),
        DeclareLaunchArgument("send_rate_hz", default_value="10.0"),
        # Safety dead-man cutoff (default matches the documented 0.25s hard
        # constraint — do not raise this for anything but a wheels-off,
        # confirm-the-wiring run; revert once real end-to-end latency is fixed).
        DeclareLaunchArgument("hold_max_age_s", default_value="0.25"),
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("camera_backend", default_value="usb_cam"),

        # Capture resolution. 640x480@15 was chosen when the detector was ArUco
        # on the CPU: 1280x720 @ 30 fps pegged it at ~150% CPU with frames
        # queueing 0.5-0.9 s stale, permanently tripping the 0.25 s dead-man.
        # Note that experiment changed resolution AND frame rate together — at
        # 15 fps the 720p budget is ~63% of a core, and pinning clocks
        # (`sudo jetson_clocks`, board profiled at 1.19 of 1.73 GHz) takes it
        # to ~43%. Re-measure before assuming 720p is out of reach.
        # Defaults come from config/camera.yaml. The arena calibration records
        # the resolution it was made at; change it there, then `arena scan`.
        # This ROS path moves every image through DDS, so at 720p it may want
        # framerate:=15 (the fast path, run_vision.py, has no such hop).
        DeclareLaunchArgument("image_width", default_value=str(cam.width)),
        DeclareLaunchArgument("image_height", default_value=str(cam.height)),
        DeclareLaunchArgument("framerate", default_value=str(cam.fps)),

        # Camera A: usb_cam, CPU MJPEG decode (mjpeg2rgb). The known-good path.
        Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            name="overhead_camera",
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("camera_backend"), "' != 'gst'"])),
            parameters=[{
                "video_device": LaunchConfiguration("video_device"),
                "image_width": LaunchConfiguration("image_width"),
                "image_height": LaunchConfiguration("image_height"),
                "framerate": LaunchConfiguration("framerate"),
                "pixel_format": "mjpeg2rgb",
                "camera_name": "overhead",
                "camera_info_url": "file://" + os.path.join(config, "camera_info.yaml"),
            }],
        ),

        # Camera B: hardware JPEG decode on the NVJPG engine, freeing the 2.8 ms
        # (640x480) / 7.5 ms (1280x720) per frame that CPU decode costs. Same
        # topics and message types as usb_cam, so nothing downstream changes.
        Node(
            package="hive_perception",
            executable="gst_camera",
            name="overhead_camera",
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("camera_backend"), "' == 'gst'"])),
            parameters=[{
                "video_device": LaunchConfiguration("video_device"),
                "image_width": LaunchConfiguration("image_width"),
                "image_height": LaunchConfiguration("image_height"),
                "framerate": LaunchConfiguration("framerate"),
                "decoder": "auto",
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
                "arena_config_path": os.path.join(config, "arena_test_6ft.yaml"),
                "debug": LaunchConfiguration("debug"),
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
