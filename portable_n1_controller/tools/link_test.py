"""Command-link exchange test: send numbered packets to the ESP and require a
delivery confirmation (ACK) for every one.

Reports per-packet ACK status and round-trip time, then a summary: delivered /
lost / stale counts, RTT statistics, and the ESP's reported WiFi signal (RSSI).
Works identically against tools/mock_esp.py (no hardware) and the real board.

    python tools/link_test.py --esp 127.0.0.1:8888              # vs mock
    python tools/link_test.py --esp 192.168.4.23:8888           # vs real ESP32

SAFETY: by default packets carry zero throttle and a gentle steering sine wave,
so on a real car the wheels twitch left/right but the drive motor never runs.
Still: first runs with the car OFF the ground.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.esp_link import EspLink

RATE_HZ = 10.0  # same 10 Hz cadence the real controller uses


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP command-link ACK test")
    parser.add_argument("--esp", required=True, help="ESP address as ip:port")
    parser.add_argument("--count", type=int, default=100, help="packets to send")
    parser.add_argument("--ack-timeout", type=float, default=0.2, help="seconds to wait per ACK")
    parser.add_argument("--quiet", action="store_true", help="summary only, no per-packet lines")
    args = parser.parse_args()

    host, _, port = args.esp.partition(":")
    link = EspLink(host, int(port or 8888))
    print(f"sending {args.count} packets to udp://{link.addr[0]}:{link.addr[1]} at {RATE_HZ:.0f} Hz\n")

    rtts_ms: list[float] = []
    rssis: list[int] = []
    lost: list[int] = []
    stale = 0
    try:
        for i in range(args.count):
            tick_start = time.perf_counter()
            steer = 0.5 * math.sin(2.0 * math.pi * i / 40.0)  # slow, gentle sweep
            sent = link.send_commands(np.array([0.0, steer]))
            ack = link.wait_ack(sent["seq"], timeout_s=args.ack_timeout)
            rtt_ms = (time.perf_counter() - tick_start) * 1000.0
            if ack is None:
                lost.append(sent["seq"])
                if not args.quiet:
                    print(f"  #{sent['seq']:4d}  LOST (no ACK within {args.ack_timeout * 1000:.0f} ms)")
            else:
                rtts_ms.append(rtt_ms)
                if isinstance(ack.get("rssi"), int):
                    rssis.append(ack["rssi"])
                if not ack.get("applied", False):
                    stale += 1
                if not args.quiet:
                    extra = "" if ack.get("applied") else "  [received but NOT applied: stale seq]"
                    print(f"  #{sent['seq']:4d}  ACK  rtt {rtt_ms:6.1f} ms  rssi {ack.get('rssi')} dBm{extra}")
            time.sleep(max(0.0, (1.0 / RATE_HZ) - (time.perf_counter() - tick_start)))
    finally:
        link.close()

    delivered = args.count - len(lost)
    print(f"\n===== link test summary =====")
    print(f"sent      : {args.count}")
    print(f"delivered : {delivered}  ({100.0 * delivered / args.count:.1f}%)")
    print(f"lost      : {len(lost)}" + (f"  seqs {lost[:10]}{'...' if len(lost) > 10 else ''}" if lost else ""))
    print(f"stale     : {stale} (delivered but not applied)")
    if rtts_ms:
        arr = np.array(rtts_ms)
        print(f"rtt ms    : mean {arr.mean():.2f}  median {np.median(arr):.2f}  "
              f"p95 {np.percentile(arr, 95):.2f}  max {arr.max():.2f}")
        over_budget = int((arr > 100.0).sum())
        print(f"rtt >100ms: {over_budget}  (control-loop latency budget)")
    if rssis:
        print(f"rssi dBm  : mean {np.mean(rssis):.0f}  worst {min(rssis)}  "
              f"(> -70 good, < -80 expect trouble)")
    # Verdict tied to the actual control-loop requirements, not perfection:
    # a lone dropped packet is superseded 100 ms later by the next command, and
    # the car's failsafe only trips on ~3 consecutive losses (300 ms silence).
    # What actually endangers control: loss BURSTS (>=2 consecutive -> stale
    # commands approaching failsafe), overall loss > 2%, or p95 RTT over budget.
    burst = any(b - a == 1 for a, b in zip(lost, lost[1:]))
    loss_frac = len(lost) / args.count
    p95_ok = bool(rtts_ms) and float(np.percentile(np.array(rtts_ms), 95)) <= 100.0
    if not lost and p95_ok:
        verdict = "PASS"
    elif not burst and loss_frac <= 0.02 and p95_ok:
        verdict = "PASS (isolated losses only - fine for 10 Hz control)"
    else:
        verdict = "LOSSY - investigate before driving"
    print(f"verdict   : {verdict}")


if __name__ == "__main__":
    main()
