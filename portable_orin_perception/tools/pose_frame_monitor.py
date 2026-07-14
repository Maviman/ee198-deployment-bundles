"""PC-side bring-up aid: listen on the controller's pose port and print what
the perception pipeline is actually sending — rate, freshness, and the poses
themselves — WITHOUT running the controller. Uses the controller's own
(vendored) parser, so if this tool accepts the frames, the controller will too.

Run on the machine that will host run_controller.py:

    python tools/pose_frame_monitor.py --expected-pursuers 1

Notes: 'capture->arrival' is only meaningful once the Orin and this PC are
clock-synced (chrony; see setup_orin.sh output). A negative or wild value
means clocks, not physics.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendored"))

from pc_controller.pose_stream import parse_pose_frame  # noqa: E402 (vendored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--expected-pursuers", type=int, default=1)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)
    print(f"listening on udp/:{args.port} for frames with "
          f"{args.expected_pursuers} pursuer(s) + evader (ctrl-c to stop)")

    count = 0
    window_start = time.time()
    window_count = 0
    last_arrival = None
    try:
        while True:
            try:
                raw, addr = sock.recvfrom(65535)
            except socket.timeout:
                if count:
                    print("... stream stalled >1 s (controller dead-man would have "
                          "E-stopped by 0.3 s)")
                continue
            arrival = time.time()
            try:
                pursuers, evader = parse_pose_frame(
                    raw, expected_pursuers=args.expected_pursuers)
            except (ValueError, KeyError) as exc:
                print(f"BAD FRAME from {addr[0]}: {exc}")
                continue
            count += 1
            window_count += 1
            gap = (arrival - last_arrival) if last_arrival else 0.0
            last_arrival = arrival

            if arrival - window_start >= 1.0:
                rate = window_count / (arrival - window_start)
                p = pursuers[0]
                print(f"[{count:6d}] {rate:5.1f} Hz  gap {gap * 1000:5.1f} ms  "
                      f"capture->arrival {(arrival - p.timestamp_s) * 1000:7.1f} ms  "
                      f"P1({p.x:+.2f},{p.y:+.2f},{p.heading:+.2f})  "
                      f"E({evader.x:+.2f},{evader.y:+.2f},{evader.heading:+.2f})")
                window_start = arrival
                window_count = 0
    except KeyboardInterrupt:
        print(f"\n{count} frames total")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
