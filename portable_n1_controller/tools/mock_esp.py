"""Pretend to be the car's ESP32: listen for command packets and print them.

Lets you verify the whole PC side (policy -> packet -> UDP) with zero hardware:

    terminal A:  python tools/mock_esp.py
    terminal B:  python run_controller.py --model models/n1_catch --esp 127.0.0.1:8888

Prints each packet plus running stats (rate, gaps, out-of-order), and flags the
same failsafe the real sketch would trip: no packet for >300 ms.

FLEET MODE. --index is this mock car's CAR_INDEX, mirroring the real firmware:
it reads cmd[index] and ignores the rest, and reports the index in its ACK. To
rehearse three cars with no hardware, run three on different ports:

    python tools/mock_esp.py --index 0 --port 8888
    python tools/mock_esp.py --index 1 --port 8889
    python tools/mock_esp.py --index 2 --port 8890
    python run_controller.py --model <3-pursuer model>         --esp 127.0.0.1:8888,127.0.0.1:8889,127.0.0.1:8890

A packet with no entry at this index goes neutral and does NOT refresh the
failsafe timer -- same rule as the firmware, so a misaddressed fleet fails here
exactly as it would on the floor.
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
    parser.add_argument("--index", type=int, default=0,
                        help="this mock car's CAR_INDEX; it drives cmd[index]")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(FAILSAFE_TIMEOUT_S)
    print(f"mock ESP (car index {args.index}, drives cmd[{args.index}]) "
          f"listening on udp :{args.port} (ctrl-C to stop)")

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
            except ConnectionResetError:
                # Windows: when the controller closes its socket, our ACK bounces
                # and the NEXT recvfrom raises this (once per bounced datagram).
                # EspLink guards the same quirk. Without it the mock DIES on every
                # controller shutdown -- which matters more with a fleet, where
                # you leave three of these running across many controller restarts.
                continue
            try:
                probe = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if "probe" in probe:
                # Discovery (`--esp auto`, `arena cars`): answer like the firmware,
                # with no effect on the (mock) outputs.
                sock.sendto(json.dumps({"car": args.index, "fw": "mock", "mac": f"mock:{args.index}",
                                        "failsafe": not armed, "rssi": -42}).encode("utf-8"), addr)
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
            # Mirrors the firmware's ACK exactly, including "car" (whose ACK is
            # this?) and "slot" (did the packet actually address me?).
            slot_ok = len(packet["cmd"]) > args.index
            ack = {"ack": packet["seq"], "car": args.index, "applied": applied,
                   "slot": slot_ok, "rssi": -42}
            sock.sendto(json.dumps(ack).encode("utf-8"), addr)
            flags = ""
            if not applied:
                flags += "  [OUT-OF-ORDER/DUP: dropped]"
            if packet["estop"]:
                flags += "  [E-STOP -> neutral]"
            if not slot_ok and not packet["estop"]:
                flags += (f"  [NO COMMAND for index {args.index}: packet carries "
                          f"{len(packet['cmd'])} pair(s) -> neutral + failsafe]")
                armed = False
            if applied:
                last_seq = packet["seq"]
            mine = (packet["cmd"][args.index] if slot_ok else None)
            pairs = (f"thr={mine[0]:+.2f} str={mine[1]:+.2f}" if mine else "-- no slot --")
            if len(packet["cmd"]) > 1:
                pairs += f"  (of {len(packet['cmd'])} pairs)"
            age_ms = (now - packet["t"]) * 1000.0
            rate = count / max(now - started, 1e-9)
            print(f"#{packet['seq']:5d} from {addr[0]}  {pairs}  age {age_ms:5.1f} ms  ({rate:4.1f} pkt/s){flags}")
    except KeyboardInterrupt:
        print(f"\nreceived {count} packets")


if __name__ == "__main__":
    main()
