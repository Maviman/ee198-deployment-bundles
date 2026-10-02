"""UART command transport: the UDP link's command packets over a serial line.

The ESP32 that presses the RC remote's buttons can sit on a cable next to the
control Orin instead of on WiFi (a USB-UART adapter, or a header UART such as
/dev/ttyTHS1 wired to the ESP32's TX/RX pins). Nothing about the packets
changes. A command packet is the same JSON object, built by the same code
(``esp_link.command_packet``), and the bytes written to the port are the bytes
a UDP datagram would carry, trailing newline included:

    {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.1]]}\\n

A byte stream has no datagram boundaries, so that newline is the framing: one
line is one frame, and after any damage both ends resynchronise on the next
newline. Replies come back the same way, one JSON object per line, mixed in
with the firmware's ordinary log lines (which never start with "{"):

    {"ack":42,"car":0,"applied":true,"slot":true,"rssi":null}\\n   delivery confirmation
    {"car":0,"fw":"...","mac":"...", ...}\\n                        reply to {"probe": 1}
    FAILSAFE: command stream stalled -> buttons released\\n        log line

There is no checksum. A frame damaged in transit nearly always stops parsing
as JSON and is dropped; the firmware only refreshes its 300 ms dead-man on a
frame that parsed, so a dropped frame costs one tick, exactly as a lost UDP
datagram does. Failsafe behaviour is the firmware's and is unchanged.

Who is on the other end
-----------------------
UDP finds a car's CAR_INDEX by asking it (``--esp auto``). A serial port can
only be verified the same way, and it matters more here: /dev/ttyUSB0 and
/dev/ttyUSB1 swap places across reboots and replugs, and a swapped pair means
the wrong physical car obeys each command. So an endpoint is not "up" until the
board on the port has answered a ``{"probe": 1}`` with a CAR_INDEX equal to the
endpoint's position in the ``--esp`` list, and no command frame is written
before that. The same check runs after every reconnect. Prefer the stable
names under /dev/serial/by-id/ for USB adapters.

Opening a port can reset the board: USB-UART adapters assert DTR/RTS on open,
and on ESP32 dev boards those lines drive EN and IO0 (DTR=0 with RTS=1 holds EN
low). The first version of this module forced both lines low before opening;
that reset the DevKitC-1 on every open (measured 2026-10-02), because the
kernel raises both lines in one step (a no-op for that circuit) and clearing DTR
first then leaves RTS alone asserted. The lines are now left as the kernel set
them: two trials on the same board with pyserial's defaults did not reset it.
Either way a board that reboots is covered twice: every button is released while
it boots, and the probe is repeated until it answers.

pyserial is imported only when a port is opened, so a controller that never
uses the serial transport does not need it.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Callable, NamedTuple

SERIAL_PREFIX = "serial:"
DEFAULT_BAUD = 115200        # the firmware's console / command UART speed
MAX_BAUD = 4_000_000

VERIFY_TIMEOUT_S = 4.0       # a board that resets on open needs a few seconds to boot
PROBE_PERIOD_S = 0.25
RETRY_S = 1.0                # between reopen attempts after a link failure
WRITE_TIMEOUT_S = 0.1        # a stalled port must not stall the 10 Hz control loop
MAX_LINE = 1024              # firmware replies are < 400 bytes; longer is line noise
LOG_TAIL = 50
PROBE_FRAME = (json.dumps({"probe": 1}) + "\n").encode("utf-8")


class SerialLinkError(RuntimeError):
    """The serial transport cannot be used as configured (message says what to fix)."""


class SerialUnavailable(SerialLinkError):
    """pyserial is not installed (or the wrong package named 'serial' is)."""


class SerialTarget(NamedTuple):
    device: str
    baud: int = DEFAULT_BAUD


def parse_serial_spec(chunk: str) -> SerialTarget:
    """'serial:/dev/ttyUSB0' or 'serial:/dev/ttyTHS1@230400' -> SerialTarget.

    Everything after the prefix up to an optional trailing '@<baud>' is the
    device path, so by-path names containing ':' work unchanged.
    """
    spec = chunk.strip()
    if not spec.lower().startswith(SERIAL_PREFIX):
        raise ValueError(f"not a serial target: {chunk!r}")
    rest = spec[len(SERIAL_PREFIX):].strip()
    device, sep, baud_s = rest.rpartition("@")
    if sep:
        if not baud_s.isdigit():
            raise ValueError(f"bad baud rate {baud_s!r} in serial target {chunk!r}")
        baud = int(baud_s)
        if not 0 < baud <= MAX_BAUD:
            raise ValueError(f"baud rate {baud} out of range in serial target {chunk!r}")
    else:
        device, baud = rest, DEFAULT_BAUD
    if not device:
        raise ValueError(f"missing device in serial target {chunk!r} "
                         "(e.g. serial:/dev/ttyUSB0 or serial:/dev/ttyTHS1@115200)")
    if device.lower() == "auto":
        raise ValueError("serial:auto is not supported: probing arbitrary ports is not safe. "
                         "List the device(s) explicitly, in CAR_INDEX order "
                         "(stable names: /dev/serial/by-id/*)")
    return SerialTarget(device, baud)


def target_label(target: SerialTarget) -> str:
    return f"serial:{target.device}" + (f"@{target.baud}" if target.baud != DEFAULT_BAUD else "")


def _import_serial():
    try:
        import serial
    except ImportError as exc:
        raise SerialUnavailable(
            "the serial transport needs pyserial: pip install pyserial "
            "(it is listed in requirements.txt; only --esp serial:... uses it)") from exc
    if not hasattr(serial, "Serial"):
        raise SerialUnavailable(
            "a module named 'serial' is installed but it is not pyserial: "
            "pip uninstall serial && pip install pyserial")
    return serial


class LineAssembler:
    """Bytes in, complete lines out.

    A line ends at "\\n", "\\r\\n" or a bare "\\r" (the firmware accepts all three).
    Blank lines are dropped. The buffer is bounded: bytes with no newline within
    ``max_line`` are noise, so the partial line is discarded and everything up to
    the next newline is skipped, which resynchronises the stream.
    """

    def __init__(self, max_line: int = MAX_LINE) -> None:
        self.max_line = max_line
        self.overflows = 0
        self._buf = bytearray()
        self._skipping = False

    def feed(self, data: bytes) -> list[bytes]:
        lines: list[bytes] = []
        for b in data:
            if b == 0x0A or b == 0x0D:
                if self._skipping:
                    self._skipping = False
                elif self._buf:
                    lines.append(bytes(self._buf))
                self._buf.clear()
            elif not self._skipping:
                self._buf.append(b)
                if len(self._buf) > self.max_line:
                    self._buf.clear()
                    self._skipping = True
                    self.overflows += 1
        return lines


def parse_line(line: bytes) -> tuple[str, object]:
    """Classify one received line: ("ack", dict) for a delivery confirmation,
    ("info", dict) for a probe reply, otherwise ("log", text). Anything that
    merely looks like JSON but is not (a garbled reply) is a log line too."""
    text = line.decode("utf-8", "replace").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
        except ValueError:
            return "log", text
        if isinstance(obj, dict):
            if "ack" in obj:
                return "ack", obj
            if "car" in obj:
                return "info", obj
    return "log", text


def _open_hint(device: str, exc: BaseException) -> str:
    code = getattr(exc, "errno", None)
    text = str(exc)
    if code == 2 or "No such file" in text:
        return (f"{device} does not exist. USB adapters appear as /dev/ttyUSB* or /dev/ttyACM* "
                "(stable names under /dev/serial/by-id/); the Orin's 40-pin header UARTs are "
                "/dev/ttyTHS1 and /dev/ttyTHS2.")
    if code == 13 or "Permission denied" in text:
        return (f"permission denied opening {device}: add your user to the dialout group "
                "(sudo usermod -aG dialout $USER) and log in again.")
    if "exclusively lock" in text or code == 16:
        return (f"{device} is already open in another process (a serial monitor, or a second "
                "controller driving the same car). Close it first.")
    return f"cannot open {device}: {text}"


class SerialEndpoint:
    """One serial port, one fleet slot.

    States: ``closed`` (never opened, or closed), ``verifying`` (open, probing
    for the board's CAR_INDEX), ``up`` (verified: command frames flow) and
    ``down`` (failed: reopened after a back-off). Everything here is
    non-blocking; ``service`` and ``poll`` are called from the control loop.
    """

    def __init__(self, target: SerialTarget, index: int, *,
                 log: Callable[[str], None] | None = None) -> None:
        self.target = target
        self.index = index
        self.label = target_label(target)
        self._log = log or (lambda _msg: None)
        self._ser = None
        self._asm = LineAssembler()
        self.state = "closed"
        self.info: dict | None = None
        self.last_error: str | None = None
        self.log_tail: deque[str] = deque(maxlen=LOG_TAIL)
        self.rx_bytes = 0             # lifetime total
        self._rx_since_open = 0       # this attempt only: what the silent-board diagnosis reports
        self.log_lines = 0
        self.frames_sent = 0
        self.failures = 0
        self._resync = False          # prefix the next write with a newline
        self._retry_at = 0.0
        self._next_probe = 0.0
        self._verify_deadline = 0.0
        self._last_logged = ("", 0.0)

    def open(self, now: float | None = None) -> None:
        """Open the port and start verifying the board on it. Raises SerialLinkError."""
        now = time.monotonic() if now is None else now
        serial = _import_serial()
        ser = serial.Serial()
        ser.port = self.target.device
        ser.baudrate = self.target.baud
        ser.timeout = 0                       # reads never block the control loop
        ser.write_timeout = WRITE_TIMEOUT_S
        # DTR/RTS are deliberately NOT touched: see the module docstring. Presetting both low
        # clears DTR before RTS and pulses EN on the DevKitC-1's auto-reset circuit.
        ser.exclusive = True                  # two controllers on one wire is a hazard
        try:
            ser.open()
        except (OSError, ValueError) as exc:  # SerialException is an OSError
            raise SerialLinkError(_open_hint(self.target.device, exc)) from exc
        try:
            ser.reset_input_buffer()
        except OSError:
            pass
        self._ser = ser
        self._asm = LineAssembler()
        self.info = None
        self.state = "verifying"
        self._resync = True
        self._rx_since_open = 0
        self._next_probe = now
        self._verify_deadline = now + VERIFY_TIMEOUT_S

    def selectable(self):
        """The object to hand to select(), or None (closed; or not selectable on Windows)."""
        if os.name != "posix":
            return None
        ser = self._ser
        return ser if ser is not None and ser.is_open else None

    def service(self, now: float | None = None) -> None:
        """Reopen after a failure, repeat the probe, give up on a silent board."""
        now = time.monotonic() if now is None else now
        if self.state == "down" and now >= self._retry_at:
            try:
                self.open(now)
            except SerialLinkError as exc:
                self._fail(str(exc), now)
        elif self.state == "verifying":
            if now >= self._verify_deadline:
                self._fail(self._silent_message(), now)
            elif now >= self._next_probe:
                self._next_probe = now + PROBE_PERIOD_S
                self._write(b"\n" + PROBE_FRAME, now)

    def write(self, payload: bytes) -> bool:
        """Put one command frame on the wire. False means it was not sent
        (link not verified yet, or just failed); the caller carries on, and the
        board's own dead-man stops the car."""
        if self.state != "up":
            return False
        data = b"\n" + payload if self._resync else payload
        if not self._write(data, time.monotonic()):
            return False
        self._resync = False
        self.frames_sent += 1
        return True

    def poll(self) -> list[dict]:
        """Read whatever has arrived. Returns delivery confirmations; probe
        replies feed verification and log lines go to the log callback."""
        ser = self._ser
        if ser is None:
            return []
        try:
            data = ser.read(max(ser.in_waiting, 1))
        except OSError as exc:
            self._fail(f"read failed: {exc}", time.monotonic())
            return []
        if not data:
            return []
        self.rx_bytes += len(data)
        self._rx_since_open += len(data)
        acks: list[dict] = []
        for line in self._asm.feed(data):
            kind, obj = parse_line(line)
            if kind == "ack":
                # Who is on the wire was checked once, at open. An ACK names the board that
                # sent it, so a board whose CAR_INDEX changed since (a cfg command, a swap
                # behind a hub) is caught on its very next frame, not trusted forever.
                car = obj.get("car")
                if self.state == "up" and car is not None and car != self.index:
                    self._fail(self._wrong_car_message(car), time.monotonic())
                    break
                acks.append(obj)
            elif kind == "info":
                self.info = obj
                if self.state == "verifying":
                    car = obj.get("car")
                    if car == self.index:
                        self._mark_up(obj)
                    else:
                        self._fail(self._wrong_car_message(car), time.monotonic())
                        break
            else:
                self.log_lines += 1
                self.log_tail.append(obj)
                self._log(f"car {self.index} ({self.label}): {obj}")
        return acks

    def close(self, drain_s: float = 0.25) -> None:
        """Close the port, first waiting briefly for queued output (the final
        E-stop) to leave the wire."""
        self._drop_port(drain_s)
        self.state = "closed"

    def _write(self, data: bytes, now: float) -> bool:
        ser = self._ser
        if ser is None:
            return False
        try:
            ser.write(data)
            return True
        except OSError as exc:      # includes SerialTimeoutException
            self._fail(f"write failed: {exc}", now)
            return False

    def _drop_port(self, drain_s: float = 0.0) -> None:
        ser, self._ser = self._ser, None
        if ser is None:
            return
        if drain_s > 0:
            deadline = time.monotonic() + drain_s
            try:
                while ser.out_waiting and time.monotonic() < deadline:
                    time.sleep(0.002)
            except (OSError, AttributeError):
                pass
        try:
            ser.close()
        except OSError:
            pass

    def _mark_up(self, info: dict) -> None:
        self.state = "up"
        self.last_error = None
        self._resync = True
        self._log(f"car {self.index} ({self.label}): link up, board reports fw "
                  f"{info.get('fw', '?')}")

    def _fail(self, reason: str, now: float) -> None:
        if self.state == "closed":
            return                  # closed on purpose: a late failure must not reopen it
        self.failures += 1
        self.last_error = reason
        self._drop_port()
        self.state = "down"
        self._retry_at = now + RETRY_S
        text, when = self._last_logged
        if text != reason or now - when > 30.0:     # a dead port must not flood the log
            self._last_logged = (reason, now)
            self._log(f"car {self.index} ({self.label}): LINK DOWN: {reason} "
                      f"(retrying every {RETRY_S:g} s; the board's failsafe holds the car stopped)")

    def _wrong_car_message(self, car) -> str:
        return (f"the board on {self.target.device} answered as car index {car}, but it is "
                f"listed at position {self.index} in --esp. Position N must be the board "
                f"flashed with CAR_INDEX N: reorder --esp, or set the board's index "
                f"(serial console: `index {self.index}`).")

    def _silent_message(self) -> str:
        heard = (f"{self._rx_since_open} byte(s) arrived but none was a probe reply, so the wiring "
                 "works and the firmware is probably too old to take commands over UART "
                 "(needs 'uart' in its fw string)"
                 if self._rx_since_open else
                 "nothing was received at all: check TX/RX are crossed, the grounds are "
                 "common, and that the baud matches the firmware")
        return (f"no reply from {self.target.device} at {self.target.baud} baud within "
                f"{VERIFY_TIMEOUT_S:g} s: {heard}.")
