"""ROS 2 node: /hive/vehicle_poses -> UDP JSON pose frames to the controller.

Subscribes to the detector's PoseArray (slot order: pursuers then evader, NaN =
not detected this frame), feeds core.frame_builder, and sends the controller's
exact wire format to <controller_ip>:9870 from a fixed-rate timer (default
10 Hz — the contract rate; the camera may run faster, the controller only needs
the freshest pose each tick).

Safety semantics (inherited from FrameBuilder — do not soften): if any vehicle
has been missing longer than hold_max_age_s, NO frame is sent at all. The
controller's own 0.3 s pose-stall dead-man then fires an E-stop. Silence is the
E-stop signal; never send a guessed pose.
"""

from __future__ import annotations

import math
import socket

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node

from .core.frame_builder import FrameBuilder


class PoseBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_bridge")
        self.declare_parameter("controller_ip", "127.0.0.1")
        self.declare_parameter("controller_port", 9870)
        self.declare_parameter("expected_pursuers", 1)
        self.declare_parameter("send_rate_hz", 10.0)
        self.declare_parameter("hold_max_age_s", 0.25)
        self.declare_parameter("poses_topic", "/hive/vehicle_poses")

        self.expected_pursuers = int(self.get_parameter("expected_pursuers").value)
        # Slot indices stand in for vehicle ids here: the detector already
        # resolved marker ids into slot order, so the bridge is id-agnostic.
        self.builder = FrameBuilder(
            pursuer_ids=list(range(self.expected_pursuers)),
            evader_id=self.expected_pursuers,
            hold_max_age_s=float(self.get_parameter("hold_max_age_s").value),
        )
        self.target = (
            str(self.get_parameter("controller_ip").value),
            int(self.get_parameter("controller_port").value),
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.suppressed = 0

        self.create_subscription(
            PoseArray, self.get_parameter("poses_topic").value, self.on_poses, 10)
        period = 1.0 / float(self.get_parameter("send_rate_hz").value)
        self.create_timer(period, self.on_tick)
        self.get_logger().info(
            f"bridging {self.expected_pursuers} pursuer(s) + evader -> "
            f"udp://{self.target[0]}:{self.target[1]} at "
            f"{self.get_parameter('send_rate_hz').value} Hz")

    def on_poses(self, msg: PoseArray) -> None:
        if len(msg.poses) != self.expected_pursuers + 1:
            self.get_logger().error(
                f"PoseArray has {len(msg.poses)} poses, expected "
                f"{self.expected_pursuers + 1} — check expected_pursuers vs marker_map.yaml")
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        detections = {}
        for slot, pose in enumerate(msg.poses):
            if math.isnan(pose.position.x) or math.isnan(pose.orientation.w):
                continue
            heading = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            detections[slot] = (pose.position.x, pose.position.y, heading)
        self.builder.update(detections, t)

    def on_tick(self) -> None:
        now_t = self.get_clock().now().nanoseconds * 1e-9
        frame = self.builder.build_frame(now_t)
        if frame is None:
            self.suppressed += 1
            self.get_logger().warning(
                f"frame suppressed (stale/missing slots {self.builder.missing_ids(now_t)}) — "
                "controller dead-man will engage if this persists",
                throttle_duration_sec=1.0)
            return
        self.sock.sendto(frame.encode("utf-8"), self.target)
        self.sent += 1
        if self.sent % 100 == 0:
            self.get_logger().info(f"{self.sent} frames sent ({self.suppressed} suppressed)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
