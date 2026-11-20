"""Pose input for the real (non-simulated) mode: JSON pose frames over UDP,
matching the project's perception interface contract (overhead camera -> ArUco
-> x, y, heading at >= 10 Hz, arena-centered coordinates, meters/radians).

Frame format (one JSON object per UDP datagram):

    {"t": 12.34,
     "pursuers": [{"x": 1.0, "y": -2.0, "heading": 0.5}],
     "evader":   {"x": 3.0, "y":  1.0, "heading": -1.2},
     "seq": 812, "lat": 0.0061}

- t: perception timestamp in seconds (any monotonic clock; used for velocity
  finite-differencing, so it must be the CAPTURE time, not the send time).
- headings in radians, arena frame, same convention as training (0 = +x, CCW+).
- seq, lat: OPTIONAL. The fast vision path (portable_orin_perception/
  run_vision.py) adds a frame counter and its own capture->send latency in
  seconds, measured on the vision host's clock. The controller adds its own
  receive->command time to that for an end-to-end figure that needs no
  cross-machine clock sync. Senders that omit them (the ROS bridge) still parse.
- A LOST vehicle (the cameras have not seen it for longer than its hold time)
  carries no pose, only how long it has been lost and where it was last seen:

      {"x": null, "y": null, "heading": null, "lost_s": 0.62,
       "last": {"x": 1.18, "y": 0.4, "heading": 0.1}}

  A pose that perception is HOLDING (seen a few frames ago) carries its own
  capture time as "t" in its entry; VehiclePose.timestamp_s is then that time, so
  a speed estimate does not read a held pose as a car standing still.
  parse_pose_frame_meta() returns a lost vehicle as a VehiclePose at the LAST SEEN place and
  says so in meta["seen"] (pursuers in order, then the evader) and
  meta["lost_s"]. parse_pose_frame() (callers that cannot handle a lost vehicle)
  raises ValueError for such a frame, which every caller treats as a bad frame.
"""

from __future__ import annotations

import json
import socket
import time

from controller_runtime.pose_types import VehiclePose


def _vehicle(entry: dict, t: float) -> tuple[VehiclePose, bool, float]:
    """(pose, seen, lost_s). A lost entry gives the last seen pose, seen False."""
    own_t = entry.get("t")
    t = t if own_t is None else float(own_t)       # a held or lost pose carries its own capture time
    if entry["x"] is None:
        last = entry["last"]
        return (VehiclePose(float(last["x"]), float(last["y"]), float(last["heading"]), t), False,
                float(entry.get("lost_s") or 0.0))
    return VehiclePose(float(entry["x"]), float(entry["y"]), float(entry["heading"]), t), True, 0.0


def _parse(frame: dict, expected_pursuers: int):
    """(pursuers, evader, seen, lost_s); seen / lost_s list the pursuers in order, then the evader."""
    t = float(frame["t"])
    parsed = [_vehicle(p, t) for p in frame["pursuers"]]
    if len(parsed) != expected_pursuers:
        raise ValueError(f"expected {expected_pursuers} pursuer poses, got {len(parsed)}")
    parsed.append(_vehicle(frame["evader"], t))
    poses = [p[0] for p in parsed]
    return poses[:-1], poses[-1], [p[1] for p in parsed], [p[2] for p in parsed]


def parse_pose_frame(raw: bytes | str, *, expected_pursuers: int) -> tuple[list[VehiclePose], VehiclePose]:
    frame = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    pursuers, evader, seen, _lost = _parse(frame, expected_pursuers)
    if not all(seen):
        raise ValueError("frame has a lost vehicle (this caller needs every pose)")
    return pursuers, evader


def parse_pose_frame_meta(raw: bytes | str, *, expected_pursuers: int) -> tuple[list[VehiclePose], VehiclePose, dict]:
    """The poses plus {"seq": int|None, "lat": float|None, "seen": [bool], "lost_s": [float]}.
    seen / lost_s list the pursuers in order, then the evader. A vehicle that is not seen is
    returned at its LAST SEEN pose: check meta["seen"] before trusting a pose."""
    frame = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    pursuers, evader, seen, lost_s = _parse(frame, expected_pursuers)
    lat = frame.get("lat")
    seq = frame.get("seq")
    return pursuers, evader, {"seq": None if seq is None else int(seq),
                              "lat": None if lat is None else float(lat),
                              "t": float(frame["t"]), "seen": seen, "lost_s": lost_s,
                              "arena": frame.get("arena")}


class UdpPoseSource:
    """Blocking-with-timeout UDP listener for perception frames. Returns None on
    timeout so the caller can trigger its dead-man stop instead of hanging."""

    def __init__(self, listen_port: int, *, expected_pursuers: int, timeout_s: float = 0.3) -> None:
        self.expected_pursuers = expected_pursuers
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Small receive buffer on purpose: a backlog here is stale poses. The
        # fast path drains to the newest frame anyway; this caps the worst case.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        except OSError:
            pass
        self.sock.bind(("0.0.0.0", listen_port))
        self.sock.settimeout(timeout_s)
        self.bad_frames = 0
        self.drained = 0
        self.last_error = ""

    def next_frame(self) -> tuple[list[VehiclePose], VehiclePose] | None:
        try:
            raw, _addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        try:
            return parse_pose_frame(raw, expected_pursuers=self.expected_pursuers)
        except ValueError:
            return None                      # a lost vehicle: for this caller the same as no frame

    def next_latest(self, timeout_s: float):
        """Newest valid frame as (pursuers, evader, meta), waiting up to
        ``timeout_s`` for one; None on timeout.

        Everything already queued is drained and only the newest frame is
        returned. Acting on an older frame when a newer one is sitting in
        the buffer is pure added latency. A malformed datagram is counted and
        skipped: one bad packet must not kill the controller. ``meta`` gains
        ``arrival`` (perf_counter at receipt of the frame returned)."""
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            self.sock.settimeout(max(remaining, 0.001))
            try:
                raw, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                return None
            except ConnectionResetError:     # Windows ICMP quirk; not a frame
                continue
            arrival = time.perf_counter()
            self.sock.setblocking(False)
            try:
                while True:
                    try:
                        raw2, _ = self.sock.recvfrom(65535)
                    except (BlockingIOError, ConnectionResetError):
                        break
                    raw, arrival = raw2, time.perf_counter()
                    self.drained += 1
            finally:
                self.sock.setblocking(True)
            try:
                pursuers, evader, meta = parse_pose_frame_meta(raw, expected_pursuers=self.expected_pursuers)
            except (ValueError, KeyError, TypeError) as exc:
                self.bad_frames += 1
                self.last_error = str(exc)
                continue                     # keep waiting for a good one, same deadline
            meta["arrival"] = arrival
            return pursuers, evader, meta

    def drain_latest(self):
        """Non-blocking next_latest: the newest valid frame already queued, or
        None. For callers that wait on several sockets with select()."""
        newest = None
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    raw, _ = self.sock.recvfrom(65535)
                except (BlockingIOError, ConnectionResetError):
                    break
                if newest is not None:
                    self.drained += 1
                newest = (raw, time.perf_counter())
        finally:
            self.sock.setblocking(True)
        if newest is None:
            return None
        try:
            pursuers, evader, meta = parse_pose_frame_meta(newest[0], expected_pursuers=self.expected_pursuers)
        except (ValueError, KeyError, TypeError) as exc:
            self.bad_frames += 1
            self.last_error = str(exc)
            return None
        meta["arrival"] = newest[1]
        return pursuers, evader, meta

    def close(self) -> None:
        self.sock.close()
