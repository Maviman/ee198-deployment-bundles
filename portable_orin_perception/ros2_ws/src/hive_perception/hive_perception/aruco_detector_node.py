"""ROS 2 node: /image_raw -> ArUco detection -> /hive/vehicle_poses.

Publishes one geometry_msgs/PoseArray per camera frame, stamped with the
IMAGE's capture time (never node wall time — the controller finite-differences
velocities from these stamps). Pose order is the slot contract consumed by
pose_bridge_node and documented in the bundle README:

    poses[0..N-1] = pursuers in marker_map.yaml order, poses[N] = evader

A vehicle whose marker was NOT detected this frame gets a NaN pose in its slot
(the PoseArray is raw per-frame truth, useful in rviz2/rosbag; the hold/
staleness policy lives downstream in pose_bridge_node, not here).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .core import arena_frame, marker_math
from .core.frame_builder import load_marker_map


class ArucoDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_detector")
        self.declare_parameter("marker_map_path", "config/marker_map.yaml")
        self.declare_parameter("homography_path", "config/arena_homography.yaml")
        self.declare_parameter("poses_topic", "/hive/vehicle_poses")

        marker_map_path = self.get_parameter("marker_map_path").value
        homography_path = self.get_parameter("homography_path").value
        self.marker_map = load_marker_map(marker_map_path)
        self.homography = arena_frame.load_homography_yaml(homography_path)
        self.slots = self.marker_map.pursuer_ids + [self.marker_map.evader_id]

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.marker_map.dictionary))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.bridge = CvBridge()
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        self.pose_pub = self.create_publisher(
            PoseArray, self.get_parameter("poses_topic").value, 10)
        self.create_subscription(CameraInfo, "camera_info", self.on_camera_info, 10)
        self.create_subscription(Image, "image_raw", self.on_image, 10)
        self.get_logger().info(
            f"tracking markers {self.slots} (evader={self.marker_map.evader_id}) "
            f"via {self.marker_map.dictionary}; homography loaded from {homography_path}")

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def on_image(self, msg: Image) -> None:
        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, "bgr8"), cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        found: dict[int, tuple[float, float, float]] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if int(marker_id) not in self.slots:
                    continue
                ref_px = np.vstack([
                    marker_math.marker_center_px(marker_corners),
                    marker_math.marker_top_midpoint_px(marker_corners),
                ])
                ref_px = marker_math.undistort_points(
                    ref_px, self.camera_matrix, self.dist_coeffs)
                found[int(marker_id)] = arena_frame.pose_from_marker(
                    self.homography, ref_px[0], ref_px[1])

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "arena"
        for vid in self.slots:
            pose = Pose()
            x, y, heading = found.get(vid, (math.nan, math.nan, math.nan))
            pose.position.x, pose.position.y = float(x), float(y)
            if math.isnan(heading):
                pose.orientation.w = math.nan
            else:
                pose.orientation.z = math.sin(heading / 2.0)
                pose.orientation.w = math.cos(heading / 2.0)
            out.poses.append(pose)
        self.pose_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
