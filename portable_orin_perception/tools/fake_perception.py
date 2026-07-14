"""Stand-in for the whole Orin pipeline: streams synthetic pose frames to the
controller over UDP, built with the SAME FrameBuilder the real bridge uses —
so the controller sees byte-for-byte the traffic the Orin will send.

Lets you dry-run `run_controller.py --source udp` (+ mock_esp.py or a real
ESP32) with zero cameras, markers, or ROS. Also exercises the dead-man path:
with --stall it goes silent mid-run (the controller must E-stop within 0.3 s)
and then resumes (the controller must re-arm).

    python tools/fake_perception.py                       # 10 s of frames
    python tools/fake_perception.py --duration 5 --stall 1.2 --resume 3

Scene: evader idles near (3, 1) of the 14x10 m training arena while the
pursuer drives toward it from (-3, -1) at ~0.5 m/s — plausible-motion data,
not a physics sim (the controller's commands don't move anything here).
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT / "ros2_ws" / "src" / "hive_perception"))

from hive_perception.core.frame_builder import FrameBuilder  # noqa: E402


def stream(sock: socket.socket, target, fb: FrameBuilder, *, seconds: float,
           rate_hz: float, start_t: float, n_pursuers: int) -> int:
    period = 1.0 / rate_hz
    sent = 0
    end = time.time() + seconds
    while time.time() < end:
        now = time.time()
        elapsed = now - start_t
        detections = {}
        for i in range(n_pursuers):
            x = -3.0 + 0.5 * elapsed
            y = -1.0 + 0.8 * i
            detections[i] = (min(x, 2.0), y, 0.15 * math.sin(elapsed))
        detections[n_pursuers] = (3.0 + 0.05 * math.sin(0.5 * elapsed), 1.0, -1.2)
        fb.update(detections, now)
        frame = fb.build_frame(now)
        if frame is not None:
            sock.sendto(frame.encode("utf-8"), target)
            sent += 1
        time.sleep(max(0.0, period - (time.time() - now)))
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", default="127.0.0.1:9870",
                        help="where run_controller.py --source udp is listening")
    parser.add_argument("--pursuers", type=int, default=1)
    parser.add_argument("--rate", type=float, default=10.0, help="frames per second")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to stream")
    parser.add_argument("--stall", type=float, default=0.0,
                        help="then go SILENT this many seconds (dead-man test)")
    parser.add_argument("--resume", type=float, default=0.0,
                        help="then stream again this many seconds (re-arm test)")
    args = parser.parse_args()

    host, _, port = args.controller.partition(":")
    target = (host, int(port or 9870))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fb = FrameBuilder(pursuer_ids=list(range(args.pursuers)), evader_id=args.pursuers)
    start_t = time.time()

    print(f"streaming {args.pursuers} pursuer(s) + evader -> udp://{target[0]}:{target[1]} "
          f"at {args.rate:g} Hz for {args.duration:g}s")
    sent = stream(sock, target, fb, seconds=args.duration, rate_hz=args.rate,
                  start_t=start_t, n_pursuers=args.pursuers)
    print(f"  {sent} frames sent")

    if args.stall > 0:
        print(f"going SILENT for {args.stall:g}s (controller should E-stop within 0.3 s)")
        time.sleep(args.stall)
        if args.resume > 0:
            print(f"resuming for {args.resume:g}s (controller should re-arm)")
            sent = stream(sock, target, fb, seconds=args.resume, rate_hz=args.rate,
                          start_t=start_t, n_pursuers=args.pursuers)
            print(f"  {sent} frames sent after recovery")
    sock.close()
    print("done")


if __name__ == "__main__":
    main()
