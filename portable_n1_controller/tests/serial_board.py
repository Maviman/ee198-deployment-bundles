"""A firmware stand-in on a pseudo-terminal, for the serial-transport tests.

No hardware and no pyserial on this side: the board is tools/mock_esp.py's
MockCar behind its PtyServer, run on a thread, with every line it receives
recorded. The controller under test opens ``board.path`` with pyserial exactly
as it would open /dev/ttyTHS1.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mock_esp import MockCar, PtyServer  # noqa: E402


class PtyBoard:
    def __init__(self, index: int = 0, *, ignore_probes: int = 0, link: str | None = None) -> None:
        self.car = MockCar(index, rssi=None, say=lambda _m: None)
        self.car.ignore_probes = ignore_probes     # a board still booting: probes go unanswered
        self.server = PtyServer(self.car, link, record=True)
        self.path = self.server.shown
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

    def say(self, text: str) -> None:
        """Print a line on the board's console, as its firmware would."""
        self.server.send(text.encode("utf-8"))

    @property
    def lines(self) -> list[tuple[float, bytes]]:
        return self.server.lines

    @property
    def probes(self) -> list[bytes]:
        return [raw for _t, raw in list(self.lines) if b'"probe"' in raw]

    @property
    def packets(self) -> list[tuple[float, dict]]:
        """(arrival time, packet) for every command packet received. Probes are not commands."""
        out = []
        for t, raw in list(self.lines):
            try:
                pkt = json.loads(raw)
            except ValueError:
                continue
            if isinstance(pkt, dict) and "seq" in pkt:
                out.append((t, pkt))
        return out

    def raw_frames(self) -> list[bytes]:
        return [raw for _t, raw in list(self.lines) if b'"seq"' in raw]

    def wait_for(self, predicate, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    def close(self) -> None:
        self.server.stop()
        self.thread.join(timeout=1.0)
        self.server.close()
