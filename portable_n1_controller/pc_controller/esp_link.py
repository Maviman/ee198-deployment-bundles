"""UDP command link: PC -> one or more ESP32s.

Packet format (one JSON object per UDP datagram, newline-terminated):

    {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}

- seq: monotonically increasing packet counter (lets the ESP drop stale/reordered
  packets: ignore any seq <= the last one applied).
- t: PC wall-clock send time (seconds) -- for latency measurement, not control.
- estop: when true the ESP must go to neutral immediately regardless of cmd.
- cmd: one [throttle, steer] pair per pursuer, each normalized to [-1, 1]
  (SinglePursuerEnv's action layout: +throttle forward, +steer left).
  The ESP maps throttle to L298N H-bridge PWM and steer to a steering-servo
  pulse width.

Multi-vehicle addressing
------------------------
The policy is CENTRALISED: one forward pass emits 2*N values, the joint action
for the whole fleet. So every car receives the SAME packet, containing every
car's command, and each ESP picks out its own pair using the CAR_INDEX burned
into it (esp32/esp32_receiver). ``cmd[i]`` belongs to the car whose index is i,
which is also the policy's pursuer slot i and marker_map.yaml's pursuer_ids[i].

Broadcast was considered and rejected: 802.11 broadcast frames get no
MAC-layer retransmission and go out at the lowest basic rate, so it would trade
per-packet reliability for airtime the link does not need. At 10 Hz, three cars
is ~5.5 kB/s and ~6% airtime as unicast.

Sequence numbers are shared across the fleet -- one counter, one stream, so a
seq identifies a control tick rather than a per-car conversation. That is what
makes an ACK from any car comparable with the others.

Failsafe contract (implemented ESP-side): if no packet arrives for
FAILSAFE_TIMEOUT_MS the ESP goes to neutral on its own -- the PC does not need
to be trusted to keep the car safe. Note this is PER CAR: with N cars, one can
fall back to neutral while the others keep driving, and the policy has no
concept of a stalled teammate. See parse_targets' note on fleet-wide E-stop.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

DEFAULT_PORT = 8888


def parse_targets(spec: str, default_port: int = DEFAULT_PORT) -> list[tuple[str, int]]:
    """'ip', 'ip:port', or a comma-separated list of either -> [(host, port), ...].

    Order is significant and is the fleet order: the first address is the car
    that drives cmd[0], i.e. the policy's pursuer slot 0. Getting this order
    wrong swaps which physical car obeys which command, so it is validated
    against the policy's pursuer count at start-up rather than discovered on
    the floor.
    """
    targets: list[tuple[str, int]] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        host, sep, port = chunk.rpartition(":")
        if not sep:                      # bare host, no colon
            host, port = chunk, str(default_port)
        if not host:
            raise ValueError(f"missing host in ESP address {chunk!r}")
        try:
            port_i = int(port)
        except ValueError:
            raise ValueError(f"bad port in ESP address {chunk!r}") from None
        targets.append((host, port_i))
    if not targets:
        raise ValueError(f"no ESP addresses parsed from {spec!r}")
    dupes = {t for t in targets if targets.count(t) > 1}
    if dupes:
        raise ValueError(
            f"duplicate ESP address(es) {sorted(dupes)} -- two fleet slots pointing at "
            "one car means a car obeys the wrong slot's command")
    return targets


class EspLink:
    def __init__(self, targets, port: int = DEFAULT_PORT) -> None:
        """``targets``: a single host string, 'ip:port', a comma-separated list,
        or an explicit list of (host, port) tuples."""
        if isinstance(targets, str):
            self.targets = parse_targets(targets, port)
        else:
            self.targets = [(str(h), int(p)) for h, p in targets]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # Windows quirk: an ICMP port-unreachable from any send makes the NEXT
        # recvfrom on this socket raise ConnectionResetError (once per bounced
        # datagram). The ack readers swallow those and keep listening, so a
        # receiver that isn't up yet / a WiFi hiccup can't kill the link. With
        # several cars this matters more, not less: one car being off should
        # never disturb the others' traffic.
        self.seq = 0
        self.packets_sent = 0
        self.datagrams_sent = 0

    @property
    def addr(self) -> tuple[str, int]:
        """First target. Kept so single-car callers and log lines still read
        naturally; use ``targets`` for anything fleet-aware."""
        return self.targets[0]

    def __len__(self) -> int:
        return len(self.targets)

    def _read_acks(self, seq: int, *, expect: int, timeout_s: float) -> dict[int, dict]:
        """Collect ACKs for `seq`, keyed by the reporting car's index."""
        acks: dict[int, dict] = {}
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and len(acks) < expect:
            try:
                raw, _addr = self.sock.recvfrom(512)
            except (BlockingIOError, ConnectionResetError):
                time.sleep(0.002)
                continue
            try:
                ack = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if ack.get("ack") != seq:
                continue          # an older tick's ACK arriving late
            # Firmware predating CAR_INDEX omits "car"; treat it as slot 0 so a
            # mixed fleet during a rolling reflash still reports something.
            acks[int(ack.get("car", 0))] = ack
        return acks

    def wait_acks(self, seq: int, *, expect: int | None = None,
                  timeout_s: float = 0.2) -> dict[int, dict]:
        """ACKs for packet `seq` from every car, keyed by car index. Returns as
        soon as `expect` cars have answered, or at timeout with whatever
        arrived -- a missing key is a car that did not confirm."""
        return self._read_acks(seq, expect=expect or len(self.targets),
                               timeout_s=timeout_s)

    def wait_ack(self, seq: int, *, timeout_s: float = 0.2) -> dict | None:
        """Single-car convenience: the first ACK for `seq`, or None on timeout."""
        acks = self._read_acks(seq, expect=1, timeout_s=timeout_s)
        return next(iter(acks.values()), None)

    def build_packet(self, actions: np.ndarray, *, estop: bool = False) -> dict:
        """The exact dict send_commands puts on the wire, without sending it."""
        flat = np.clip(np.asarray(actions, dtype=np.float64).ravel(), -1.0, 1.0)
        if flat.size % 2 != 0:
            raise ValueError(f"action vector length {flat.size} is not throttle/steer pairs")
        return {
            "seq": self.seq,
            "t": round(time.time(), 4),
            "estop": bool(estop),
            "cmd": [
                [float(round(flat[i], 4)), float(round(flat[i + 1], 4))]
                for i in range(0, flat.size, 2)
            ],
        }

    def send_commands(self, actions: np.ndarray, *, estop: bool = False) -> dict:
        """actions: flat array of 2*N normalized values (throttle, steer per car),
        exactly what the policy outputs. Clipped to [-1, 1] here -- the same clip
        the training env applies to incoming actions.

        One packet, unicast to every target. Each ESP selects its own pair by
        CAR_INDEX, so all cars act on the same tick of the joint action rather
        than on commands assembled at slightly different times.
        """
        packet = self.build_packet(actions, estop=estop)
        payload = (json.dumps(packet) + "\n").encode("utf-8")
        pairs = len(packet["cmd"])
        if pairs < len(self.targets) and not estop:
            # Each ESP goes neutral when its slot is absent, so this is safe --
            # but it means some car is being commanded to stop every tick, which
            # is a configuration error rather than a driving decision.
            raise ValueError(
                f"{pairs} command pair(s) for {len(self.targets)} car(s): car(s) "
                f"{list(range(pairs, len(self.targets)))} would receive nothing. "
                "Check --esp against the model's num_pursuers.")
        for target in self.targets:
            self.sock.sendto(payload, target)
            self.datagrams_sent += 1
        self.seq += 1
        self.packets_sent += 1
        return packet

    def send_stop(self) -> dict:
        """Explicit neutral/E-stop packet to EVERY car (also the shutdown message).

        Sized for the whole fleet rather than one pair: estop:true already
        short-circuits ahead of the cmd lookup in firmware, but a stop packet
        that is the wrong shape is exactly the thing you do not want to discover
        was relying on a default.
        """
        return self.send_commands(np.zeros(2 * len(self.targets)), estop=True)

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
