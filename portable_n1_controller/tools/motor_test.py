"""Direct actuator test: bypasses perception AND the policy entirely, sends a
fixed step sequence of throttle/steer commands straight to the ESP. For
answering one question in isolation: does the car respond to commands at all
(the V3 firmware presses its RC remote's FWD/BACK/LEFT/RIGHT buttons),
independent of whether the perception/policy pipeline is healthy right now.

Unlike tools/link_test.py (deliberately zero-throttle, comms-only), this DOES
command the drive motor. Sends steadily at 10 Hz throughout (matching the
real control loop) so the ESP's own 300 ms failsafe never trips between
steps -- a gap here would look identical to a real failsafe, muddying the
one thing this tool exists to isolate.

    python tools/motor_test.py --esp 192.168.1.194:8888
    python tools/motor_test.py --esp 192.168.1.194:8888 --throttle 0.5 --hold 2.0

What to watch for, since neither sign convention is knowable from software:

- "forward" really drives forward and "steer left" really goes left. If a
  button is wrong, fix the order on the car's serial console with
  `pins <fwd> <back> <left> <right>` (its `test` command shows which is which).
- In the firmware's modulated mode, a small throttle should give a slower
  wheel than a big one. If every throttle looks the same (or nothing moves at
  small values), the remote is missing the short presses: raise `slot`.
- In binary mode, a command below 1/3 presses nothing: use --throttle 0.5.
- Every "neutral" step releases every button.

SAFETY: wheels OFF the ground for this. Default throttle magnitude is a
gentle 0.3 (normalized) -- raise with --throttle only once 0.3 is confirmed
safe. Ctrl-C at any time sends an explicit E-stop before exiting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.esp_link import EspLink

RATE_HZ = 10.0


def run_step(link: EspLink, label: str, throttle: float, steer: float,
             seconds: float, ack_timeout: float) -> None:
    print(f"-- {label}: throttle={throttle:+.2f} steer={steer:+.2f} for {seconds:g}s --")
    period = 1.0 / RATE_HZ
    end = time.perf_counter() + seconds
    n_lost = 0
    n_sent = 0
    while time.perf_counter() < end:
        tick_start = time.perf_counter()
        sent = link.send_commands(np.array([throttle, steer]))
        ack = link.wait_ack(sent["seq"], timeout_s=ack_timeout)
        n_sent += 1
        if ack is None:
            n_lost += 1
        time.sleep(max(0.0, period - (time.perf_counter() - tick_start)))
    flag = "" if n_lost == 0 else f"  ({n_lost}/{n_sent} packets lost -- check the link, not the motor)"
    print(f"   sent {n_sent} packets{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esp", required=True, help="ESP address as ip:port")
    parser.add_argument("--throttle", type=float, default=0.3,
                        help="normalized throttle magnitude to test, [-1, 1] (default 0.3, gentle)")
    parser.add_argument("--steer", type=float, default=0.5,
                        help="normalized steer magnitude to test, [-1, 1] (default 0.5)")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds to hold each step")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds of neutral between steps")
    parser.add_argument("--ack-timeout", type=float, default=0.2)
    args = parser.parse_args()

    host, _, port = args.esp.partition(":")
    link = EspLink(host, int(port or 8888))
    print(f"motor test vs udp://{link.addr[0]}:{link.addr[1]} -- WHEELS SHOULD BE OFF THE GROUND\n")

    steps = [
        ("neutral (settle/arm)", 0.0, 0.0, args.settle),
        ("forward", args.throttle, 0.0, args.hold),
        ("neutral", 0.0, 0.0, args.settle),
        ("reverse", -args.throttle, 0.0, args.hold),
        ("neutral", 0.0, 0.0, args.settle),
        ("steer left", 0.0, args.steer, args.hold),
        ("neutral", 0.0, 0.0, args.settle),
        ("steer right", 0.0, -args.steer, args.hold),
        ("neutral (final)", 0.0, 0.0, args.settle),
    ]
    try:
        for label, thr, steer, seconds in steps:
            run_step(link, label, thr, steer, seconds, args.ack_timeout)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        link.close()  # sends an explicit E-stop
        print("done -- sent final E-stop")


if __name__ == "__main__":
    main()
