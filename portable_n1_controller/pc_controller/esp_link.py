"""UDP command link: PC -> ESP32.

Packet format (one JSON object per UDP datagram, newline-terminated):

    {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}

- seq: monotonically increasing packet counter (lets the ESP drop stale/reordered
  packets: ignore any seq <= the last one applied).
- t: PC wall-clock send time (seconds) -- for latency measurement, not control.
- estop: when true the ESP must go to neutral immediately regardless of cmd.
- cmd: one [throttle, steer] pair per pursuer, each normalized to [-1, 1]
  (SinglePursuerEnv's action layout: +throttle forward, +steer left).
  N=1 -> a single pair. The ESP maps throttle to L298N H-bridge PWM and steer
  to a steering-servo pulse width.

Failsafe contract (implemented ESP-side, see esp32/esp32_receiver): if no packet
arrives for FAILSAFE_TIMEOUT_MS the ESP goes to neutral on its own -- the PC does
not need to be trusted to keep the car safe.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np


class EspLink:
    def __init__(self, host: str, port: int = 8888) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # Windows quirk: an ICMP port-unreachable from any send makes the NEXT
        # recvfrom on this socket raise ConnectionResetError (once per bounced
        # datagram). wait_ack() swallows those and keeps listening, so a
        # receiver that isn't up yet / a WiFi hiccup can't kill the link.
        self.seq = 0
        self.packets_sent = 0

    def wait_ack(self, seq: int, *, timeout_s: float = 0.2) -> dict | None:
        """Block up to timeout_s for the ESP's ACK to packet `seq`. Returns the
        ack dict ({"ack": seq, "applied": bool, "rssi": dBm}) or None on timeout.
        ACKs for other (older) seqs arriving in between are discarded."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                raw, _addr = self.sock.recvfrom(512)
            except (BlockingIOError, ConnectionResetError):
                time.sleep(0.002)
                continue
            try:
                ack = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if ack.get("ack") == seq:
                return ack
        return None

    def send_commands(self, actions: np.ndarray, *, estop: bool = False) -> dict:
        """actions: flat array of 2*N normalized values (throttle, steer per car),
        exactly what the policy outputs. Clipped to [-1, 1] here -- the same clip
        the training env applies to incoming actions."""
        flat = np.clip(np.asarray(actions, dtype=np.float64).ravel(), -1.0, 1.0)
        if flat.size % 2 != 0:
            raise ValueError(f"action vector length {flat.size} is not throttle/steer pairs")
        packet = {
            "seq": self.seq,
            "t": round(time.time(), 4),
            "estop": bool(estop),
            "cmd": [
                [float(round(flat[i], 4)), float(round(flat[i + 1], 4))]
                for i in range(0, flat.size, 2)
            ],
        }
        self.sock.sendto((json.dumps(packet) + "\n").encode("utf-8"), self.addr)
        self.seq += 1
        self.packets_sent += 1
        return packet

    def send_stop(self) -> dict:
        """Explicit neutral/E-stop packet (also used as the shutdown message)."""
        return self.send_commands(np.zeros(2), estop=True)

    def close(self) -> None:
        try:
            self.send_stop()
        finally:
            self.sock.close()


def parse_command_packet(raw: bytes | str) -> dict:
    """Inverse of send_commands' wire format; used by tools/mock_esp.py and the
    selftest. The real ESP32 sketch implements the same parse in C++."""
    packet = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    for key in ("seq", "t", "estop", "cmd"):
        if key not in packet:
            raise ValueError(f"command packet missing '{key}': {packet}")
    for pair in packet["cmd"]:
        if len(pair) != 2 or any(abs(v) > 1.0 for v in pair):
            raise ValueError(f"bad throttle/steer pair {pair}")
    return packet
