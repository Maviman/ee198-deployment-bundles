"""tools/wifi_ap_test.py without a hotspot: fleet discovery (every index, no
duplicates, nothing outside the fleet), and reading link_test.py's verdicts from
a real run against three mock cars."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # portable_n1_controller
sys.path.insert(0, str(ROOT / "tools"))

import wifi_ap_test as w  # noqa: E402

PREFIX = "10.43.0."


def car(idx: int, ip: str) -> dict:
    return {"car": idx, "ip": ip, "mac": f"AA:BB:CC:00:00:0{idx}", "fw": "t", "radio": "wifi"}


class Script:
    """discover_cars stand-in: one scripted round per call, and a clock that a round advances."""

    def __init__(self, *rounds: dict) -> None:
        self.rounds, self.calls, self.now = list(rounds), 0, 0.0

    def discover(self, timeout_s: float, extra_targets: list[str]) -> dict:
        assert extra_targets == [PREFIX + "255"], "the hotspot's broadcast address is probed"
        self.calls += 1
        self.now += timeout_s
        return dict(self.rounds.pop(0)) if self.rounds else {}

    def clock(self) -> float:
        return self.now


def find(script: Script, n: int, deadline: float = 60.0):
    return w.find_cars(PREFIX, n, deadline, discover=script.discover, clock=script.clock)


def test_three_cars_found_as_they_join_over_several_rounds():
    s = Script({0: car(0, "10.43.0.11")},
               {},
               {0: car(0, "10.43.0.11"), 2: car(2, "10.43.0.13")},
               {1: car(1, "10.43.0.12")},
               {9: car(9, "10.43.0.99")})           # never reached: all three are in
    cars, problems = find(s, 3)
    assert problems == []
    assert {i: c["ip"] for i, c in cars.items()} == {0: "10.43.0.11", 1: "10.43.0.12", 2: "10.43.0.13"}
    assert s.calls == 4, "stops probing the moment the fleet is complete"


def test_a_missing_car_is_named_after_the_timeout():
    s = Script(*[{0: car(0, "10.43.0.11"), 2: car(2, "10.43.0.13")}] * 100)
    cars, problems = find(s, 3, deadline=5.0)
    assert sorted(cars) == [0, 2]
    assert problems == ["car index 1 never answered"]
    assert s.now >= 5.0


def test_two_cars_on_one_index_fail_whether_seen_in_one_round_or_two():
    same_round = Script({0: car(0, "10.43.0.11"), -1: [car(1, "10.43.0.12"), car(1, "10.43.0.14")]})
    _, problems = find(same_round, 3)
    assert problems[0].startswith("car index 1 answered from 10.43.0.12, 10.43.0.14: two cars share it")
    across = Script({1: car(1, "10.43.0.12")}, {1: car(1, "10.43.0.14")})
    _, problems = find(across, 3)
    assert problems[0].startswith("car index 1 answered from 10.43.0.12, 10.43.0.14")


def test_an_index_outside_the_fleet_fails_and_other_networks_are_ignored():
    s = Script({0: car(0, "10.43.0.11"), 3: car(3, "10.43.0.15"),
                1: car(1, "192.168.1.40")})          # a car on some other interface: not ours
    cars, problems = find(s, 3)
    assert "car index 3 at 10.43.0.15 is outside 0..2" in problems[0]
    assert 1 not in cars and "car index 1 never answered" in problems


def test_station_signals_per_mac():
    dump = ("Station 1c:db:d4:40:44:e0 (on wlP1p1s0)\n\tinactive time:\t8 ms\n\tsignal:  \t-57 dBm\n"
            "Station AA:BB:CC:00:00:01 (on wlP1p1s0)\n\tsignal:  \t-71 [-71] dBm\n")
    assert w.station_signals(dump) == {"1c:db:d4:40:44:e0": "-57 dBm", "aa:bb:cc:00:00:01": "-71 [-71] dBm"}


def test_verdicts_are_read_per_car_from_a_real_three_car_link_test():
    ports = [18888, 18889, 18890]
    mocks = [subprocess.Popen([sys.executable, str(ROOT / "tools/mock_esp.py"), "--index", str(i),
                               "--port", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i, p in enumerate(ports)]
    try:
        time.sleep(1.0)
        lt = subprocess.run([sys.executable, str(ROOT / "tools/link_test.py"), "--esp",
                             ",".join(f"127.0.0.1:{p}" for p in ports), "--count", "20", "--quiet"],
                            capture_output=True, text=True, timeout=60)
    finally:
        for m in mocks:
            m.terminate()
            m.wait(timeout=5)
    verdicts = w.link_verdicts(lt.stdout)
    assert len(verdicts) == 3 and all(v.startswith("PASS") for v in verdicts), lt.stdout
    assert w.link_verdicts("verdict   : LOSSY - investigate before driving\n") == [
        "LOSSY - investigate before driving"]
