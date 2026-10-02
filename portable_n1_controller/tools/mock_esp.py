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

SERIAL MODE. --serial makes the mock a car on a UART cable: it creates a
pseudo-terminal, prints its path, and serves the same packets there, one per
line, answering like the firmware does over its UART (ACKs and probe replies as
JSON lines, plus the firmware's own log lines such as FAILSAFE). No pyserial is
needed for the mock itself, only for the controller that talks to it:

    terminal A:  python tools/mock_esp.py --serial --link /tmp/mock-esp0
    terminal B:  python run_controller.py --model models/n1_catch --esp serial:/tmp/mock-esp0

--link makes a stable symlink to the pty (its /dev/pts/N number changes every
run). One mock per car for a fleet (--index 1 --link /tmp/mock-esp1, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import socket
import sys
import threading
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.esp_link import parse_command_packet

FAILSAFE_TIMEOUT_S = 0.3
SLOT_WARN_EVERY_S = 2.0       # the firmware rate-limits its NO COMMAND line the same way


class MockCar:
    """The firmware's packet rules, independent of how the bytes arrive.

    ``on_message`` takes one received packet (a UDP datagram, or one line from
    the pty) and returns the replies to send back, unframed. ``console`` collects
    the log lines the real firmware would print on its UART; only the serial
    front end puts them on a wire. ``check_failsafe`` is called whenever the
    receive loop wakes, with or without data.
    """

    def __init__(self, index: int = 0, *, rssi: int | None = -42, say=print) -> None:
        self.index = index
        self.rssi = rssi              # UDP reports a (fake) WiFi RSSI; a UART has no radio: null
        self.say = say
        self.console: list[str] = []
        self.ignore_probes = 0        # tests: stay silent for the first N probes (a board still booting)
        self.last_seq = -1
        self.count = 0
        self.started: float | None = None
        self.armed = False
        self.last_valid = 0.0         # when a command that parsed last arrived (probes don't count)
        self._last_slot_warn = -1e9

    def on_message(self, raw: bytes, source: str, now: float | None = None) -> list[bytes]:
        try:
            probe = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return []
        if not isinstance(probe, dict):
            return []
        if "probe" in probe:
            # Discovery (`--esp auto`, `arena cars`) and serial verification:
            # answer like the firmware, with no effect on the (mock) outputs.
            if self.ignore_probes > 0:
                self.ignore_probes -= 1
                return []
            return [json.dumps({"car": self.index, "fw": "mock", "mac": f"mock:{self.index}",
                                "failsafe": not self.armed, "rssi": self.rssi}).encode("utf-8")]
        try:
            packet = parse_command_packet(raw)
        except (ValueError, TypeError, KeyError) as exc:
            # The real firmware ignores a malformed packet outright (no ACK, no effect).
            self.say(f"[ignored malformed packet: {exc}]")
            return []
        now = time.time() if now is None else now
        self.started = self.started or now
        self.count += 1
        was_armed = self.armed
        self.armed = not packet["estop"]
        # newer seq; or resync on controller restart: failsafe engaged
        # (not armed) or far-backwards jump (mirrors the real firmware)
        applied = (packet["seq"] > self.last_seq or not self.armed
                   or (self.last_seq - packet["seq"]) > 50)
        # ACK every parseable packet, exactly like the real firmware
        # (rssi is fake here; the real ESP reports its WiFi signal in dBm).
        # Mirrors the firmware's ACK exactly, including "car" (whose ACK is
        # this?) and "slot" (did the packet actually address me?).
        slot_ok = len(packet["cmd"]) > self.index
        ack = {"ack": packet["seq"], "car": self.index, "applied": applied,
               "slot": slot_ok, "rssi": self.rssi}
        flags = ""
        if not applied:
            flags += "  [OUT-OF-ORDER/DUP: dropped]"
        if packet["estop"]:
            flags += "  [E-STOP -> neutral]"
            if was_armed:
                self.console.append(f"#{packet['seq']} E-STOP -> buttons released")
        if not slot_ok and not packet["estop"]:
            flags += (f"  [NO COMMAND for index {self.index}: packet carries "
                      f"{len(packet['cmd'])} pair(s) -> neutral + failsafe]")
            self.armed = False
            if time.monotonic() - self._last_slot_warn > SLOT_WARN_EVERY_S:
                self._last_slot_warn = time.monotonic()
                self.console.append(
                    f"#{packet['seq']} NO COMMAND for car index {self.index} (packet carries "
                    f"{len(packet['cmd'])} pair(s)) -> released. Check --esp order and this "
                    "car's index.")
        elif slot_ok and not packet["estop"]:
            self.last_valid = time.monotonic()
        if applied:
            self.last_seq = packet["seq"]
        mine = (packet["cmd"][self.index] if slot_ok else None)
        pairs = (f"thr={mine[0]:+.2f} str={mine[1]:+.2f}" if mine else "-- no slot --")
        if len(packet["cmd"]) > 1:
            pairs += f"  (of {len(packet['cmd'])} pairs)"
        age_ms = (now - packet["t"]) * 1000.0
        rate = self.count / max(now - self.started, 1e-9)
        self.say(f"#{packet['seq']:5d} from {source}  {pairs}  age {age_ms:5.1f} ms  ({rate:4.1f} pkt/s){flags}")
        return [json.dumps(ack).encode("utf-8")]

    def check_failsafe(self, mono_now: float | None = None) -> bool:
        """True (once) when the dead-man trips: armed, and no driving command
        for FAILSAFE_TIMEOUT_S. Probes and malformed packets do not refresh it."""
        mono_now = time.monotonic() if mono_now is None else mono_now
        if self.armed and mono_now - self.last_valid > FAILSAFE_TIMEOUT_S:
            self.say(f"!! FAILSAFE: no packet for {FAILSAFE_TIMEOUT_S}s -> neutral (real ESP stops the car here)")
            self.console.append("FAILSAFE: command stream stalled -> buttons released")
            self.armed = False
            return True
        return False


def serve_udp(car: MockCar, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(FAILSAFE_TIMEOUT_S)
    print(f"mock ESP (car index {car.index}, drives cmd[{car.index}]) "
          f"listening on udp :{port} (ctrl-C to stop)")
    try:
        while True:
            try:
                raw, addr = sock.recvfrom(65535)
            except socket.timeout:
                car.check_failsafe()
                continue
            except ConnectionResetError:
                # Windows: when the controller closes its socket, our ACK bounces
                # and the NEXT recvfrom raises this (once per bounced datagram).
                # EspLink guards the same quirk. Without it the mock DIES on every
                # controller shutdown -- which matters more with a fleet, where
                # you leave three of these running across many controller restarts.
                continue
            for reply in car.on_message(raw, addr[0]):
                sock.sendto(reply, addr)
            car.console.clear()      # a UDP board's log lines go to its own USB console
    except KeyboardInterrupt:
        print(f"\nreceived {car.count} packets")


class PtyServer:
    """A serial port with a mock car on the far end: a pseudo-terminal whose
    other side is opened by the controller (pyserial) as an ordinary device.

    Frames are the firmware's: one JSON object per line in, one per line out. A
    line that does not start with "{" is a console command, which the mock
    does not model. ``run`` is the whole receive loop (the CLI runs it on the
    main thread; tests run it on a thread and call ``stop``).
    """

    def __init__(self, car: MockCar, link: str | None = None, *, record: bool = False) -> None:
        self.car = car
        self.link = link
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)     # no echo, no CR/LF translation, before anything opens the other end
        os.set_blocking(self.master, False)
        # We keep `slave` open ourselves: the controller closing its handle is then no hangup.
        self.path = os.ttyname(self.slave)
        self.shown = self.path
        if link:
            try:
                os.unlink(link)
            except FileNotFoundError:
                pass
            os.symlink(self.path, link)
            self.shown = link
        self.lines: list[tuple[float, bytes]] | None = [] if record else None   # tests: (perf_counter, raw line)
        self._stop = threading.Event()

    def send(self, data: bytes) -> None:
        try:
            os.write(self.master, data + b"\n")
        except OSError:
            pass               # nobody is reading: a UART with no listener just loses the bytes

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        buf = b""
        while not self._stop.is_set():
            ready, _, _ = select.select([self.master], [], [], 0.05)
            if ready:
                try:
                    buf += os.read(self.master, 4096)
                except OSError:
                    pass
            while True:
                nl = min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0), default=-1)
                if nl < 0:
                    break
                line, buf = buf[:nl].strip(), buf[nl + 1:]
                if not line:
                    continue
                if self.lines is not None:
                    self.lines.append((time.perf_counter(), line))
                # Firmware routing: a line starting with "{" is a frame, anything
                # else is a console command (not modelled; ignored).
                if line.startswith(b"{"):
                    for reply in self.car.on_message(line, "serial"):
                        self.send(reply)
            self.car.check_failsafe()
            for text in self.car.console:
                self.send(text.encode("utf-8"))
            self.car.console.clear()
            if len(buf) > 4096:
                buf = b""      # garbage with no newline: resync, like the firmware's bounded buffer

    def close(self) -> None:
        """Idempotent. A second close must never touch the old fd numbers: by then they may be
        another pty's (fd numbers are reused), and closing those kills a live server."""
        if self.link:
            try:
                os.unlink(self.link)
            except OSError:
                pass
            self.link = None
        for name in ("master", "slave"):
            fd = getattr(self, name)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)


def serve_serial(car: MockCar, link: str | None) -> None:
    server = PtyServer(car, link)
    print(f"mock ESP (car index {car.index}, drives cmd[{car.index}]) on a serial pty: {server.shown}"
          + (f"  (-> {server.path})" if link else "") + "  (ctrl-C to stop)")
    print(f"  run_controller.py ... --esp serial:{server.shown}")
    try:
        server.run()
    except KeyboardInterrupt:
        print(f"\nreceived {car.count} packets")
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock ESP32 command receiver")
    parser.add_argument("--port", type=int, default=None, help="UDP port (default 8888)")
    parser.add_argument("--index", type=int, default=0,
                        help="this mock car's CAR_INDEX; it drives cmd[index]")
    parser.add_argument("--serial", action="store_true",
                        help="be a car on a UART cable: serve a pseudo-terminal instead of UDP")
    parser.add_argument("--link", metavar="PATH",
                        help="with --serial: a stable symlink to the pty, for --esp serial:PATH")
    args = parser.parse_args()
    if args.link and not args.serial:
        parser.error("--link applies only with --serial")
    if args.serial and args.port is not None:
        parser.error("--serial and --port are alternatives: a car is on a cable or on UDP")

    if args.serial:
        serve_serial(MockCar(args.index, rssi=None), args.link)
    else:
        serve_udp(MockCar(args.index), args.port or 8888)


if __name__ == "__main__":
    main()
