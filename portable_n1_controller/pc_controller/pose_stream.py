"""Pose input for the real (non-simulated) mode: JSON pose frames over UDP,
matching the project's perception interface contract (overhead camera -> ArUco
-> x, y, heading at >= 10 Hz, arena-centered coordinates, meters/radians).

Frame format (one JSON object per UDP datagram):

    {"t": 12.34,
     "pursuers": [{"x": 1.0, "y": -2.0, "heading": 0.5}],
     "evader":   {"x": 3.0, "y":  1.0, "heading": -1.2}}

- t: perception timestamp in seconds (any monotonic clock; used for velocity
  finite-differencing, so it must be the CAPTURE time, not the send time).
- headings in radians, arena frame, same convention as training (0 = +x, CCW+).
"""

from __future__ import annotations

import json
import socket

from controller_runtime.pose_types import VehiclePose


def parse_pose_frame(raw: bytes | str, *, expected_pursuers: int) -> tuple[list[VehiclePose], VehiclePose]:
    frame = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    t = float(frame["t"])
    pursuers = [
        VehiclePose(float(p["x"]), float(p["y"]), float(p["heading"]), t)
        for p in frame["pursuers"]
    ]
    if len(pursuers) != expected_pursuers:
        raise ValueError(f"expected {expected_pursuers} pursuer poses, got {len(pursuers)}")
    ev = frame["evader"]
    evader = VehiclePose(float(ev["x"]), float(ev["y"]), float(ev["heading"]), t)
    return pursuers, evader


class UdpPoseSource:
    """Blocking-with-timeout UDP listener for perception frames. Returns None on
    timeout so the caller can trigger its dead-man stop instead of hanging."""

    def __init__(self, listen_port: int, *, expected_pursuers: int, timeout_s: float = 0.3) -> None:
        self.expected_pursuers = expected_pursuers
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", listen_port))
        self.sock.settimeout(timeout_s)

    def next_frame(self) -> tuple[list[VehiclePose], VehiclePose] | None:
        try:
            raw, _addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        return parse_pose_frame(raw, expected_pursuers=self.expected_pursuers)

    def close(self) -> None:
        self.sock.close()
