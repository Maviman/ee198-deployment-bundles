"""ROS 2 node: /image_raw -> fiducial detection -> /hive/vehicle_poses.

Family-agnostic: ArUco DICT_4X4_50 or AprilTag DICT_APRILTAG_36h11, chosen by
``dictionary`` in config/marker_map.yaml. Detection runs against a SUBSET
dictionary holding only the ids this arena prints — see core.tag_family; with
the full 587-codeword 36h11 book this node costs ~37 ms/frame, with the 8-code
subset ~4.7 ms, i.e. cheaper than the ArUco baseline it replaces.

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
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_topic", "/hive/debug_image")

        marker_map_path = self.get_parameter("marker_map_path").value
        homography_path = self.get_parameter("homography_path").value
        self.marker_map = load_marker_map(marker_map_path)
        self.homography = arena_frame.load_homography_yaml(homography_path)
        self.slots = self.marker_map.pursuer_ids + [self.marker_map.evader_id]

        # Subset dictionary: only the ids this arena prints. detectMarkers()
        # then reports ROW INDICES into tag_set.real_ids, never the printed id
        # — every detection must go through tag_set.to_real_id().
        self.tag_set = self.marker_map.tag_set()
        self.detector = cv2.aruco.ArucoDetector(
            self.tag_set.build_opencv_dictionary(), cv2.aruco.DetectorParameters())
        self.bridge = CvBridge()
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        self.pose_pub = self.create_publisher(
            PoseArray, self.get_parameter("poses_topic").value, 10)
        self.debug = bool(self.get_parameter("debug").value)
        self.debug_pub = None
        if self.debug:
            self.debug_pub = self.create_publisher(
                Image, self.get_parameter("debug_topic").value, 10)
        self.create_subscription(CameraInfo, "camera_info", self.on_camera_info, 10)
        self.create_subscription(Image, "image_raw", self.on_image, 10)
        self.get_logger().info(
            f"tracking markers {self.slots} (evader={self.marker_map.evader_id}) "
            f"via {self.marker_map.dictionary} subset of {len(self.tag_set.real_ids)} "
            f"codes {list(self.tag_set.real_ids)} "
            f"(error correction {self.tag_set.max_correction_bits} bits); "
            f"homography loaded from {homography_path}"
            + (f"; debug overlay on {self.get_parameter('debug_topic').value}"
               if self.debug else ""))

    def _label_for(self, marker_id: int) -> str:
        if marker_id == self.marker_map.evader_id:
            return "EVADER"
        if marker_id in self.marker_map.pursuer_ids:
            return f"P{self.marker_map.pursuer_ids.index(marker_id) + 1}"
        if marker_id in self.marker_map.calibration_corner_ids:
            return "CORNER"
        return "?"

    def _publish_debug_image(self, bgr, detections, found, stamp) -> None:
        debug_img = bgr.copy()
        if detections:
            # Draw outlines ourselves rather than via drawDetectedMarkers: that
            # helper would label each tag with its subset ROW INDEX, and an
            # overlay showing the wrong id is worse than no overlay at all.
            outlines = [np.asarray(c, dtype=np.float32).reshape(1, 4, 2) for _i, c in detections]
            cv2.aruco.drawDetectedMarkers(debug_img, outlines)
            for mid, marker_corners in detections:
                center = marker_math.marker_center_px(marker_corners).astype(int)
                text = f"id{mid} {self._label_for(mid)}"
                if mid in found:
                    x, y, heading = found[mid]
                    text += f"  ({x:+.2f},{y:+.2f}m {math.degrees(heading):+.0f}deg)"
                color = (0, 255, 0) if mid in self.slots else (0, 165, 255)
                cv2.putText(debug_img, text, (center[0] + 8, center[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        missing = [vid for vid in self.slots if vid not in found]
        if missing:
            cv2.putText(debug_img, f"missing: {missing}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        out_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        out_msg.header.stamp = stamp
        out_msg.header.frame_id = "camera"
        self.debug_pub.publish(out_msg)

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def _decode(self, corners, ids) -> list[tuple[int, np.ndarray]]:
        """(printed_id, corners) pairs. Translates the subset dictionary's row
        indices into real tag ids; an out-of-range row means the dictionary and
        the marker map disagree, which is a configuration bug worth shouting
        about rather than silently dropping."""
        if ids is None:
            return []
        out = []
        for marker_corners, row in zip(corners, ids.flatten()):
            try:
                out.append((self.tag_set.to_real_id(int(row)), marker_corners))
            except IndexError as exc:
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)
        return out

    def on_image(self, msg: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        detections = self._decode(corners, ids)

        found: dict[int, tuple[float, float, float]] = {}
        for marker_id, marker_corners in detections:
            if marker_id not in self.slots:
                continue
            ref_px = np.vstack([
                marker_math.marker_center_px(marker_corners),
                marker_math.marker_top_midpoint_px(marker_corners),
            ])
            ref_px = marker_math.undistort_points(
                ref_px, self.camera_matrix, self.dist_coeffs)
            found[marker_id] = arena_frame.pose_from_marker(
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
        if self.debug:
            self._publish_debug_image(bgr, detections, found, msg.header.stamp)


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
