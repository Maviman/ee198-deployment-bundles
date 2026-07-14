"""Pretend to be the car's ESP32: listen for command packets and print them.

Lets you verify the whole PC side (policy -> packet -> UDP) with zero hardware:

    terminal A:  python tools/mock_esp.py
    terminal B:  python run_controller.py --model models/n1_catch --esp 127.0.0.1:8888

Prints each packet plus running stats (rate, gaps, out-of-order), and flags the
same failsafe the real sketch would trip: no packet for >300 ms.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.esp_link import parse_command_packet

FAILSAFE_TIMEOUT_S = 0.3


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock ESP32 command receiver")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(FAILSAFE_TIMEOUT_S)
    print(f"mock ESP listening on udp :{args.port} (ctrl-C to stop)")

    last_seq = -1
    count = 0
    started = None
    armed = False
    try:
        while True:
            try:
                raw, addr = sock.recvfrom(65535)
            except socket.timeout:
                if armed:
                    print(f"!! FAILSAFE: no packet for {FAILSAFE_TIMEOUT_S}s -> neutral (real ESP stops the car here)")
                    armed = False
                continue
            packet = parse_command_packet(raw)
            now = time.time()
            started = started or now
            count += 1
            armed = not packet["estop"]
            # newer seq; or resync on controller restart: failsafe engaged
            # (not armed) or far-backwards jump (mirrors the real firmware)
            applied = (packet["seq"] > last_seq or not armed
                       or (last_seq - packet["seq"]) > 50)
            # ACK every parseable packet, exactly like the real firmware
            # (rssi is fake here; the real ESP reports its WiFi signal in dBm).
            ack = {"ack": packet["seq"], "applied": applied, "rssi": -42}
            sock.sendto(json.dumps(ack).encode("utf-8"), addr)
            flags = ""
            if not applied:
                flags += "  [OUT-OF-ORDER/DUP: dropped]"
            if packet["estop"]:
                flags += "  [E-STOP -> neutral]"
            if applied:
                last_seq = packet["seq"]
            pairs = " ".join(f"thr={c[0]:+.2f} str={c[1]:+.2f}" for c in packet["cmd"])
            age_ms = (now - packet["t"]) * 1000.0
            rate = count / max(now - started, 1e-9)
            print(f"#{packet['seq']:5d} from {addr[0]}  {pairs}  age {age_ms:5.1f} ms  ({rate:4.1f} pkt/s){flags}")
    except KeyboardInterrupt:
        print(f"\nreceived {count} packets")


if __name__ == "__main__":
    main()
