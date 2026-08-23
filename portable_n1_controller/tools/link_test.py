"""Command-link exchange test: send numbered packets to the ESP(s) and require a
delivery confirmation (ACK) for every one.

Reports per-packet ACK status and round-trip time, then a summary: delivered /
lost / stale counts, RTT statistics, and the ESP's reported WiFi signal (RSSI).
Works identically against tools/mock_esp.py (no hardware) and the real board.

    python tools/link_test.py --esp 127.0.0.1:8888              # vs mock
    python tools/link_test.py --esp 192.168.4.23:8888           # vs real ESP32

FLEET. Pass several addresses in CAR_INDEX order to test the whole fleet at
once. Every car receives the same joint packet and answers separately, so the
summary is per car -- which is the point: with three radios the interesting
question is not "did the link work" but "which car is worst".

    python tools/link_test.py --esp 192.168.4.10:8888,192.168.4.11:8888

Per-car stats also catch a duplicate CAR_INDEX, which is otherwise invisible:
two cars flashed with the same index produce ACKs that collide on one slot, so
one index shows double the expected ACKs and another shows silence.

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

from pc_controller.esp_link import EspLink, parse_targets

RATE_HZ = 10.0  # same 10 Hz cadence the real controller uses


def summarise(label: str, count: int, rtts_ms: list[float], rssis: list[int],
              lost: list[int], stale: int, slot_bad: int) -> str:
    delivered = count - len(lost)
    print(f"\n----- {label} -----")
    print(f"sent      : {count}")
    print(f"delivered : {delivered}  ({100.0 * delivered / count:.1f}%)")
    print(f"lost      : {len(lost)}"
          + (f"  seqs {lost[:10]}{'...' if len(lost) > 10 else ''}" if lost else ""))
    print(f"stale     : {stale} (delivered but not applied)")
    if slot_bad:
        print(f"NO SLOT   : {slot_bad}  <-- this car found no command at its own "
              "CAR_INDEX. Check the index and the --esp order.")
    if rtts_ms:
        arr = np.array(rtts_ms)
        print(f"rtt ms    : mean {arr.mean():.2f}  median {np.median(arr):.2f}  "
              f"p95 {np.percentile(arr, 95):.2f}  max {arr.max():.2f}")
        print(f"rtt >100ms: {int((arr > 100.0).sum())}  (control-loop latency budget)")
    if rssis:
        print(f"rssi dBm  : mean {np.mean(rssis):.0f}  worst {min(rssis)}  "
              f"(> -70 good, < -80 expect trouble)")

    # Verdict tied to the actual control-loop requirements, not perfection:
    # a lone dropped packet is superseded 100 ms later by the next command, and
    # the car's failsafe only trips on ~3 consecutive losses (300 ms silence).
    # What actually endangers control: loss BURSTS (>=2 consecutive -> stale
    # commands approaching failsafe), overall loss > 2%, or p95 RTT over budget.
    burst = any(b - a == 1 for a, b in zip(lost, lost[1:]))
    loss_frac = len(lost) / count
    p95_ok = bool(rtts_ms) and float(np.percentile(np.array(rtts_ms), 95)) <= 100.0
    if slot_bad:
        verdict = "MISADDRESSED - this car is not being commanded"
    elif not lost and p95_ok:
        verdict = "PASS"
    elif not burst and loss_frac <= 0.02 and p95_ok:
        verdict = "PASS (isolated losses only - fine for 10 Hz control)"
    else:
        verdict = "LOSSY - investigate before driving"
    print(f"verdict   : {verdict}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esp", required=True,
                        help="ESP address ip:port, or comma-separated in CAR_INDEX order")
    parser.add_argument("--count", type=int, default=100, help="packets to send")
    parser.add_argument("--ack-timeout", type=float, default=0.2, help="seconds to wait per ACK")
    parser.add_argument("--quiet", action="store_true", help="summary only, no per-packet lines")
    args = parser.parse_args()

    targets = parse_targets(args.esp)
    link = EspLink(targets)
    n = len(targets)
    print(f"sending {args.count} packets at {RATE_HZ:.0f} Hz to {n} car(s):")
    for i, (host, port) in enumerate(targets):
        print(f"  car index {i} (cmd[{i}]) -> udp://{host}:{port}")
    print()

    rtts_ms: dict[int, list[float]] = {i: [] for i in range(n)}
    rssis: dict[int, list[int]] = {i: [] for i in range(n)}
    lost: dict[int, list[int]] = {i: [] for i in range(n)}
    stale = {i: 0 for i in range(n)}
    slot_bad = {i: 0 for i in range(n)}
    unexpected: set[int] = set()

    try:
        for k in range(args.count):
            tick_start = time.perf_counter()
            steer = 0.5 * math.sin(2.0 * math.pi * k / 40.0)  # slow, gentle sweep
            # Zero throttle for every car; only steering moves. One pair per car,
            # because each ESP reads its own CAR_INDEX out of this array.
            actions = np.tile(np.array([0.0, steer]), n)
            sent = link.send_commands(actions)
            acks = link.wait_acks(sent["seq"], timeout_s=args.ack_timeout)
            rtt_ms = (time.perf_counter() - tick_start) * 1000.0

            for car, ack in acks.items():
                if car >= n:
                    unexpected.add(car)
                    continue
                rtts_ms[car].append(rtt_ms)
                if isinstance(ack.get("rssi"), int):
                    rssis[car].append(ack["rssi"])
                if not ack.get("applied", False):
                    stale[car] += 1
                if ack.get("slot") is False:
                    slot_bad[car] += 1
            for car in range(n):
                if car not in acks:
                    lost[car].append(sent["seq"])

            if not args.quiet:
                bits = []
                for car in range(n):
                    if car in acks:
                        a = acks[car]
                        mark = "ACK" if a.get("applied") else "old"
                        slot = "" if a.get("slot", True) else "!NOSLOT"
                        bits.append(f"c{car}:{mark}{slot} {a.get('rssi')}dBm")
                    else:
                        bits.append(f"c{car}:LOST")
                print(f"  #{sent['seq']:4d}  rtt {rtt_ms:6.1f} ms   " + "  ".join(bits))
            time.sleep(max(0.0, (1.0 / RATE_HZ) - (time.perf_counter() - tick_start)))
    finally:
        link.close()

    print("\n===== link test summary =====")
    verdicts = [summarise(f"car index {i}  ({targets[i][0]}:{targets[i][1]})",
                          args.count, rtts_ms[i], rssis[i], lost[i], stale[i], slot_bad[i])
                for i in range(n)]

    if unexpected:
        print(f"\nWARNING: ACKs from unknown car index {sorted(unexpected)} -- a car is "
              "flashed with an index outside this fleet, or two cars share one.")
    if n > 1:
        worst = max(range(n), key=lambda i: (len(lost[i]),
                                             np.percentile(rtts_ms[i], 95) if rtts_ms[i] else 1e9))
        print(f"\nfleet verdict: {'PASS' if all(v.startswith('PASS') for v in verdicts) else 'INVESTIGATE'}"
              f"   (worst car: index {worst})")


if __name__ == "__main__":
    main()
