"""Bluetooth LE command transport: the UDP link's command packets over BLE.

The car's ESP32-S3 has Bluetooth Low Energy (no classic Bluetooth), and the
firmware exposes the Nordic UART Service (NUS): the host WRITES bytes to the
service's RX characteristic and the car NOTIFIES bytes back on TX. On the wire
this is a byte stream exactly like the UART (serial_link.py), so the frames are
the same bytes, newline-terminated::

    {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.1]]}\\n

and the replies come back the same way: one JSON object per line (ACK, probe
reply), reassembled from whatever chunks the radio delivered.

Target syntax: ``ble:<advertised name>`` (the firmware advertises
``hive-car<CAR_INDEX>``) or ``ble:<AA:BB:CC:DD:EE:FF>`` (the car's Bluetooth
address, printed at boot). A name is found by scanning; an address connects
directly, which is faster and does not depend on the name being unique.

Who is on the other end is verified the same way as a serial port: the
endpoint is not "up" until the car has answered ``{"probe": 1}`` with a
CAR_INDEX equal to its position in the ``--esp`` list, and the check repeats
on every ACK. A connection that drops is reopened after a back-off; while it is
down the car gets nothing and its own 300 ms failsafe stops it.

How it is wired into the control loop
-------------------------------------
bleak (the BLE client library) is asyncio-based; the controller is a
synchronous select() loop. So each endpoint runs bleak on its own thread with
its own event loop, and hands bytes to the control loop through a socketpair:
the thread writes every notification into one end, the control loop selects
on and reads the other, like a serial port's file descriptor. Outgoing frames
go the other way through the loop's thread-safe call queue. Nothing in the
control loop ever blocks on the radio.

bleak is imported only when a BLE target is opened, so a controller that never
uses the transport does not need it.
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import threading
import time
from collections import deque
from typing import Callable, NamedTuple

from .serial_link import (LOG_TAIL, PROBE_FRAME, PROBE_PERIOD_S, RETRY_S, LineAssembler,
                          SerialLinkError, parse_line)

BLE_PREFIX = "ble:"
SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"      # host -> car
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"      # car -> host

SCAN_TIMEOUT_S = 8.0          # finding a car by name
CONNECT_TIMEOUT_S = 10.0
VERIFY_TIMEOUT_S = SCAN_TIMEOUT_S + CONNECT_TIMEOUT_S + 4.0   # scan + connect + probe
DRAIN_S = 0.4                 # on close: let the final E-stop leave the radio
MIN_CHUNK = 20                # ATT default MTU 23 - 3

_ADDR_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


class BleLinkError(SerialLinkError):
    """The BLE transport cannot be used as configured. A SerialLinkError so that
    callers which already handle the UART transport's failures handle this one."""


class BleUnavailable(BleLinkError):
    """bleak is not installed."""


class BleTarget(NamedTuple):
    peer: str          # advertised name, or a Bluetooth address

    @property
    def is_address(self) -> bool:
        return bool(_ADDR_RE.match(self.peer))


def parse_ble_spec(chunk: str) -> BleTarget:
    """'ble:hive-car0' or 'ble:1C:DB:D4:40:4B:AA' -> BleTarget."""
    spec = chunk.strip()
    if not spec.lower().startswith(BLE_PREFIX):
        raise ValueError(f"not a BLE target: {chunk!r}")
    peer = spec[len(BLE_PREFIX):].strip()
    if not peer:
        raise ValueError(f"missing name or address in BLE target {chunk!r} "
                         "(e.g. ble:hive-car0 or ble:1C:DB:D4:40:4B:AA)")
    if peer.lower() == "auto":
        raise ValueError("ble:auto is not supported: list each car as ble:hive-car<N> "
                         "(or its address), in CAR_INDEX order")
    if _ADDR_RE.match(peer):
        peer = peer.upper()
    return BleTarget(peer)


def target_label(target: BleTarget) -> str:
    return f"ble:{target.peer}"


def _import_bleak():
    try:
        import bleak
    except ImportError as exc:
        raise BleUnavailable(
            "the BLE transport needs bleak: pip install bleak "
            "(only --esp ble:... uses it)") from exc
    return bleak


class BleEndpoint:
    """One BLE connection, one fleet slot. Same surface as SerialEndpoint:
    ``open``, ``selectable``, ``service``, ``write``, ``poll``, ``close``, and
    the ``state`` / ``info`` / counters the dashboard reads.

    States: ``closed``, ``verifying`` (scanning / connecting / probing),
    ``up`` (frames flow), ``down`` (failed; reopened after a back-off).
    """

    def __init__(self, target: BleTarget, index: int, *,
                 log: Callable[[str], None] | None = None,
                 connector: Callable | None = None) -> None:
        self.target = target
        self.index = index
        self.label = target_label(target)
        self._log = log or (lambda _msg: None)
        # Test seam: a coroutine function (endpoint, stop_event) that plays the
        # radio. The default is the bleak one below.
        self._connector = connector or _bleak_session
        self._asm = LineAssembler()
        self.state = "closed"
        self.info: dict | None = None
        self.last_error: str | None = None
        self.log_tail: deque[str] = deque(maxlen=LOG_TAIL)
        self.rx_bytes = 0
        self.log_lines = 0
        self.frames_sent = 0
        self.failures = 0
        self.mtu: int | None = None
        self.connected = False
        self._rx_sock: socket.socket | None = None      # control-loop end
        self._tx_sock: socket.socket | None = None      # radio-thread end
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._outq: asyncio.Queue | None = None
        self._thread_error: str | None = None
        self._retry_at = 0.0
        self._next_probe = 0.0
        self._verify_deadline = 0.0
        self._last_logged = ("", 0.0)
        self._gen = 0        # which radio thread is current; stale ones are ignored

    # -- control-loop side -----------------------------------------------------
    def open(self, now: float | None = None) -> None:
        """Start the radio thread (scan / connect / subscribe) and begin verifying."""
        now = time.monotonic() if now is None else now
        if self._connector is _bleak_session:
            _import_bleak()
        self._rx_sock, self._tx_sock = socket.socketpair()
        self._rx_sock.setblocking(False)
        self._tx_sock.setblocking(False)
        self._asm = LineAssembler()
        self.info = None
        self.mtu = None
        self.connected = False
        self._thread_error = None
        self._gen += 1
        self.state = "verifying"
        self._next_probe = now           # probes start once the thread reports connected
        self._verify_deadline = now + VERIFY_TIMEOUT_S
        self._thread = threading.Thread(target=self._thread_main, args=(self._gen,),
                                        name=f"ble-car{self.index}", daemon=True)
        self._thread.start()

    def selectable(self):
        """The control-loop end of the socketpair (readable when bytes arrived), or None."""
        return self._rx_sock if os.name == "posix" and self._rx_sock is not None else None

    def service(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state == "down" and now >= self._retry_at:
            try:
                self.open(now)
            except BleLinkError as exc:
                self._fail(str(exc), now)
            return
        if self.state in ("verifying", "up") and self._thread_error is not None:
            self._fail(self._thread_error, now)
            return
        if self.state == "verifying":
            if now >= self._verify_deadline:
                self._fail(self._silent_message(), now)
            elif self.connected and now >= self._next_probe:
                self._next_probe = now + PROBE_PERIOD_S
                self._submit(b"\n" + PROBE_FRAME)

    def write(self, payload: bytes) -> bool:
        """Queue one command frame for the radio thread. False: link not up."""
        if self.state != "up":
            return False
        if not self._submit(payload):
            return False
        self.frames_sent += 1
        return True

    def poll(self) -> list[dict]:
        sock = self._rx_sock
        if sock is None:
            return []
        try:
            data = sock.recv(4096)
        except BlockingIOError:
            return []
        except OSError as exc:
            self._fail(f"read failed: {exc}", time.monotonic())
            return []
        if not data:
            return []
        self.rx_bytes += len(data)
        acks: list[dict] = []
        for line in self._asm.feed(data):
            kind, obj = parse_line(line)
            if kind == "ack":
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

    def close(self, drain_s: float = DRAIN_S) -> None:
        """Stop the radio thread, first giving queued output (the final E-stop)
        a moment to leave."""
        self._stop_thread(drain_s)
        self.state = "closed"

    # -- internals -------------------------------------------------------------
    def _submit(self, data: bytes) -> bool:
        loop, q = self._loop, self._outq
        if loop is None or q is None or not self.connected:
            return False
        try:
            loop.call_soon_threadsafe(q.put_nowait, data)
            return True
        except RuntimeError:          # loop closed under us
            return False

    def _thread_main(self, gen: int) -> None:
        try:
            asyncio.run(self._session(gen))
        except Exception as exc:        # noqa: BLE001 -- any radio failure is a link failure
            if gen == self._gen:
                self._thread_error = f"{type(exc).__name__}: {exc}"
        finally:
            if gen == self._gen:
                self.connected = False
                if self._thread_error is None and self.state in ("verifying", "up"):
                    self._thread_error = "BLE connection closed"

    async def _session(self, gen: int) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._outq = asyncio.Queue()
        await self._connector(self, self._stop)

    # Called by the connector on the radio thread.
    def _on_connected(self, mtu: int | None) -> None:
        self.mtu = mtu
        self.connected = True

    def _on_rx(self, data: bytes) -> None:
        sock = self._tx_sock
        if sock is None:
            return
        try:
            sock.send(data)
        except (BlockingIOError, OSError):
            pass                       # the control loop is not draining: drop, like a full UART buffer

    def _stop_thread(self, drain_s: float) -> None:
        loop, stop, thread = self._loop, self._stop, self._thread
        self._gen += 1                 # anything the old thread reports from now on is stale
        if loop is not None and stop is not None:
            if drain_s > 0:
                time.sleep(min(drain_s, 1.0))
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=CONNECT_TIMEOUT_S + 2.0)
        self._thread = None
        self._loop = None
        self._stop = None
        self._outq = None
        self.connected = False
        for s in (self._rx_sock, self._tx_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._rx_sock = self._tx_sock = None

    def _mark_up(self, info: dict) -> None:
        self.state = "up"
        self.last_error = None
        self._log(f"car {self.index} ({self.label}): link up, board reports fw "
                  f"{info.get('fw', '?')}, MTU {self.mtu}")

    def _fail(self, reason: str, now: float) -> None:
        if self.state == "closed":
            return
        self.failures += 1
        self.last_error = reason
        self._stop_thread(0.0)
        self.state = "down"
        self._retry_at = now + RETRY_S
        text, when = self._last_logged
        if text != reason or now - when > 30.0:
            self._last_logged = (reason, now)
            self._log(f"car {self.index} ({self.label}): LINK DOWN: {reason} "
                      f"(retrying every {RETRY_S:g} s; the board's failsafe holds the car stopped)")

    def _wrong_car_message(self, car) -> str:
        return (f"the car at {self.label} answered as car index {car}, but it is listed at "
                f"position {self.index} in --esp. Position N must be the car flashed with "
                f"CAR_INDEX N: reorder --esp, or set the car's index (serial console: "
                f"`index {self.index}`).")

    def _silent_message(self) -> str:
        if not self.connected:
            return (f"{self.label} was not found / did not connect within {VERIFY_TIMEOUT_S:g} s: "
                    "is the car powered, in range, advertising (serial banner says 'BLE: "
                    "advertising as ...'), and not already connected to another host?")
        return (f"connected to {self.label} but no probe reply within {VERIFY_TIMEOUT_S:g} s: "
                "the firmware is probably too old to take commands over BLE "
                "(needs 'ble' in its fw string)")


async def _bleak_session(ep: BleEndpoint, stop: asyncio.Event) -> None:
    """The radio, with bleak: find the car, connect, subscribe to TX, then pump
    the outgoing queue into RX until told to stop or the car disconnects."""
    from bleak import BleakClient, BleakScanner

    target = ep.target
    if target.is_address:
        device = target.peer
    else:
        device = await BleakScanner.find_device_by_name(target.peer, timeout=SCAN_TIMEOUT_S)
        if device is None:
            raise BleLinkError(f"no BLE device advertising as {target.peer!r} seen in "
                               f"{SCAN_TIMEOUT_S:g} s")
    gone = asyncio.Event()

    def on_disconnect(_client) -> None:
        gone.set()

    client = BleakClient(device, timeout=CONNECT_TIMEOUT_S, disconnected_callback=on_disconnect)
    await client.connect()
    try:
        mtu = getattr(client, "mtu_size", None)
        await client.start_notify(TX_UUID, lambda _h, data: ep._on_rx(bytes(data)))
        ep._on_connected(mtu)
        chunk = max(MIN_CHUNK, (mtu or 23) - 3)
        q = ep._outq
        assert q is not None
        while not stop.is_set() and not gone.is_set():
            get = asyncio.ensure_future(q.get())
            halt = asyncio.ensure_future(stop.wait())
            lost = asyncio.ensure_future(gone.wait())
            done, pending = await asyncio.wait({get, halt, lost}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            if get in done:
                data = get.result()
                for off in range(0, len(data), chunk):
                    await client.write_gatt_char(RX_UUID, data[off:off + chunk], response=False)
        if gone.is_set() and not stop.is_set():
            raise BleLinkError("the car disconnected")
    finally:
        if client.is_connected:
            try:
                await client.disconnect()
            except Exception:       # noqa: BLE001
                pass
