"""The BLE transport without a radio: target parsing, and the endpoint's state
machine driven by a fake radio session in place of bleak. The real radio is
exercised on the car (tools/link_test.py --esp ble:hive-car0)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pc_controller.ble_link import (BleEndpoint, BleTarget, parse_ble_spec, target_label)
from pc_controller.esp_link import EspLink, parse_targets


# ---------------------------------------------------------------- parsing
def test_ble_spec_by_name_and_by_address():
    assert parse_ble_spec("ble:hive-car0") == BleTarget("hive-car0")
    assert not parse_ble_spec("ble:hive-car0").is_address
    t = parse_ble_spec("ble:1c:db:d4:40:4b:a9")
    assert t.is_address and t.peer == "1C:DB:D4:40:4B:A9"
    assert target_label(t) == "ble:1C:DB:D4:40:4B:A9"


@pytest.mark.parametrize("bad", ["ble:", "ble:auto", "serial:/dev/x"])
def test_ble_spec_rejects_what_it_cannot_use(bad):
    with pytest.raises(ValueError):
        parse_ble_spec(bad)


def test_ble_targets_mix_with_the_other_transports_in_fleet_order():
    ts = parse_targets("ble:hive-car0,serial:/dev/ttyX,10.0.0.5")
    assert isinstance(ts[0], BleTarget) and ts[0].peer == "hive-car0"
    assert ts[2] == ("10.0.0.5", 8888)
    with pytest.raises(ValueError, match="duplicate BLE"):
        parse_targets("ble:hive-car0,ble:hive-car0")


# ---------------------------------------------------------------- a fake radio
class FakeCar:
    """A coroutine standing in for _bleak_session: 'connects' at once, answers
    probes as the given CAR_INDEX, ACKs every command frame, and can be told
    to drop the link."""

    def __init__(self, car=0, mtu=185, connect_delay=0.0):
        self.car = car
        self.mtu = mtu
        self.connect_delay = connect_delay
        self.frames: list[dict] = []
        self.drop = asyncio.Event()

    async def __call__(self, ep: BleEndpoint, stop: asyncio.Event):
        await asyncio.sleep(self.connect_delay)
        ep._on_connected(self.mtu)
        q = ep._outq
        buf = b""
        while not stop.is_set() and not self.drop.is_set():
            try:
                data = await asyncio.wait_for(q.get(), timeout=0.02)
            except asyncio.TimeoutError:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "probe" in msg:
                    reply = {"car": self.car, "fw": "test-ble", "via": "ble"}
                else:
                    self.frames.append(msg)
                    reply = {"ack": msg["seq"], "car": self.car, "applied": True,
                             "slot": True, "rssi": None}
                ep._on_rx((json.dumps(reply) + "\n").encode())
        if self.drop.is_set():
            raise RuntimeError("radio dropped")


def pump(ep: BleEndpoint, seconds: float, acks: list | None = None) -> None:
    """Run the endpoint's control-loop side for a while, like the controller does."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ep.service()
        got = ep.poll()
        if acks is not None:
            acks.extend(got)
        time.sleep(0.005)


# ---------------------------------------------------------------- the endpoint
def test_endpoint_verifies_the_car_index_before_it_is_up():
    car = FakeCar(car=0)
    logs: list[str] = []
    ep = BleEndpoint(BleTarget("hive-car0"), 0, log=logs.append, connector=car)
    ep.open()
    assert ep.state == "verifying"
    assert ep.write(b"{}\n") is False, "no frame before the probe is answered"
    pump(ep, 0.5)
    assert ep.state == "up", logs
    assert ep.mtu == 185
    assert ep.info["car"] == 0
    assert any("link up" in m for m in logs)
    ep.close(drain_s=0.0)
    assert ep.state == "closed"


def test_endpoint_refuses_a_car_with_the_wrong_index():
    car = FakeCar(car=1)
    logs: list[str] = []
    ep = BleEndpoint(BleTarget("hive-car0"), 0, log=logs.append, connector=car)
    ep.open()
    pump(ep, 0.5)
    assert ep.state == "down"
    assert "answered as car index 1" in ep.last_error
    ep.close(drain_s=0.0)


def test_frames_flow_and_acks_come_back():
    car = FakeCar()
    ep = BleEndpoint(BleTarget("hive-car0"), 0, connector=car)
    ep.open()
    pump(ep, 0.5)
    assert ep.state == "up"
    acks: list[dict] = []
    for seq in range(5):
        frame = json.dumps({"seq": seq, "t": 0, "estop": False, "cmd": [[0.5, 0.0]]}) + "\n"
        assert ep.write(frame.encode())
        pump(ep, 0.05, acks)
    pump(ep, 0.2, acks)
    assert [a["ack"] for a in acks] == [0, 1, 2, 3, 4]
    assert [f["seq"] for f in car.frames] == [0, 1, 2, 3, 4]
    assert ep.frames_sent == 5
    ep.close(drain_s=0.0)


def test_a_dropped_link_goes_down_and_reconnects_after_the_backoff():
    car = FakeCar()
    logs: list[str] = []
    ep = BleEndpoint(BleTarget("hive-car0"), 0, log=logs.append, connector=car)
    ep.open()
    pump(ep, 0.5)
    assert ep.state == "up"
    car.drop.set()
    pump(ep, 0.5)
    assert ep.state == "down", logs
    assert ep.write(b"{}\n") is False
    car.drop.clear()
    pump(ep, 1.8)                      # RETRY_S is 1 s, then verify again
    assert ep.state == "up", logs
    assert ep.failures == 1
    ep.close(drain_s=0.0)


def test_esplink_drives_a_ble_car_like_a_serial_one():
    import numpy as np
    from pc_controller import esp_link

    car = FakeCar()
    orig = esp_link.BleEndpoint
    # Inject the fake radio into every endpoint EspLink builds.
    esp_link.BleEndpoint = lambda t, i, log=None: orig(t, i, log=log, connector=car)
    try:
        link = EspLink("ble:hive-car0")
        assert link.transport(0) == "ble"
        link.wait_ready(timeout_s=3.0)
        assert link.link_up(0) is True
        for _ in range(3):
            link.send_commands(np.zeros(2))
            time.sleep(0.03)
            link.poll_acks()
        time.sleep(0.1)
        link.poll_acks()
        assert link.car_stats[0]["acks"] >= 2
        link.close()
        # The final E-stop reached the car.
        assert car.frames[-1]["estop"] is True
    finally:
        esp_link.BleEndpoint = orig
