"""The dashboard's START / INITIALIZE buttons: deploy/hub.py's /api/arm and /api/init.

    cd portable_n1_controller && python -m pytest tests/test_hub_controls.py -q

A real hub (real HTTP server on localhost) is fed synthetic telemetry by a thread, and a fake
controller on a UDP socket plays the control port (arm / disarm), so every refusal and every
automatic stop is checked end to end, not just the readiness arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import stat
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("arena_hub", ROOT.parent / "deploy" / "hub.py")
hubmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hubmod)


class FakeController:
    """The controller's localhost control port: records commands, answers like the real one."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.cmds: list[str] = []
        self.armed = False
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.alive:
            try:
                raw, addr = self.sock.recvfrom(1024)
            except (socket.timeout, OSError):
                continue
            cmd = json.loads(raw)["cmd"]
            self.cmds.append(cmd)
            if cmd == "arm":
                self.armed = True
            elif cmd == "disarm":
                self.armed = False
            self.sock.sendto(json.dumps({"armed": self.armed, "state": "RUNNING", "ticks": 1}).encode(), addr)


def healthy():
    """Telemetry of a ready system: the car inside the arena facing the evader, which is up and to the left."""
    vision = {"src": "vision", "status": "streaming", "mode": "1280x720@30", "fps": 30.0,
              "arena": [0.915, 0.915],
              "vehicles": [{"id": 1, "role": "P1", "seen": True, "vis": 100.0, "x": 0.1, "y": -0.2, "h": 1.5708},
                           {"id": 0, "role": "EVADER", "seen": True, "vis": 100.0, "x": -0.4, "y": 0.5, "h": 0.0}]}
    control = {"src": "control", "state": "RUNNING", "armed": False, "n": 1, "model": "m",
               "cars": [{"idx": 0, "addr": "serial:x", "rtt_ms": 16.0, "ack_age_s": 0.05, "slot": True}],
               "lat_ms": {"total": {"p50": 50.0, "p95": 60.0}},
               "poses": {"pursuers": [[0.1, -0.2, 1.5708]], "evader": [-0.4, 0.5, 0.0]},
               "cmd": [1.0, 1.0], "hold_radius": 0.6, "hold": [False], "stall_reason": ""}
    return vision, control


class Rig:
    def __init__(self, tmp_path, *, cap_s=120.0, arena_cli="", ctl=True) -> None:
        self.fc = FakeController()
        args = types.SimpleNamespace(telemetry_port=0, control_port=self.fc.port, preview="",
                                     run_dir=str(tmp_path / "runs"), arena_cli=arena_cli, run_cap_s=cap_s)
        self.hub = hubmod.Hub(args)
        self.vision, self.control = healthy()
        self.alive = True
        threading.Thread(target=self._feed, daemon=True).start()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), hubmod.make_handler(self.hub))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        time.sleep(0.25)

    def _feed(self) -> None:
        while self.alive:
            self.control["armed"] = self.fc.armed
            self.hub.ingest(dict(self.vision))
            self.hub.ingest(dict(self.control))
            time.sleep(0.1)

    def post(self, path, headers=None):
        h = {"X-Arena": "control"} if headers is None else headers
        req = urllib.request.Request(self.base + path, method="POST", headers=h, data=b"")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def state(self):
        with urllib.request.urlopen(self.base + "/api/state", timeout=5) as r:
            return json.loads(r.read())

    def failing(self):
        return [c["name"] for c in self.hub.readiness()["checks"] if not c["ok"]]

    def close(self) -> None:
        self.alive = False
        self.fc.alive = False
        self.hub._watch_gen += 1
        self.httpd.shutdown()
        self.hub.log_fh.close()


@pytest.fixture
def rig(tmp_path):
    made = []

    def make(**kw):
        r = Rig(tmp_path, **kw)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


def test_a_healthy_system_is_ready(rig):
    r = rig()
    assert r.hub.readiness()["ok"], r.failing()
    s = r.state()
    assert s["ready"]["ok"] and s["can_control"] is True and s["ready"]["hold_radius"] == 0.6


@pytest.mark.parametrize("what, expect", [
    (lambda r: r.vision["vehicles"][0].update(seen=False), "car and evader both seen"),
    (lambda r: r.control["cars"][0].update(ack_age_s=3.0), "every car link is answering"),
    (lambda r: r.control.update(state="STALLED", stall_reason="no pose frames"), "controller is RUNNING and disarmed"),
    (lambda r: r.control["lat_ms"]["total"].update(p95=180.0), "camera to command latency within budget"),
    (lambda r: r.control["poses"].update(pursuers=[[0.88, 0.0, 1.5708]]), "car starts inside the arena"),
    (lambda r: r.control["poses"].update(pursuers=[[0.75, 0.0, 0.0]]), "car faces inward if it starts near an edge"),
    (lambda r: r.control.update(cmd=[1.0, -1.0]), "steering sign agrees with the evader's side (frame not mirrored)"),
    (lambda r: r.vision.update(drift_px=40.0), "no error alerts"),
])
def test_each_failure_is_named_and_blocks_start(rig, what, expect):
    r = rig()
    what(r)
    time.sleep(0.35)
    assert expect in r.failing()
    code, body = r.post("/api/arm")
    assert code == 409 and expect in body["error"]
    assert "arm" not in r.fc.cmds, "the controller must never see an arm the hub refused"


def test_start_arms_the_controller_and_reports_the_run(rig):
    r = rig()
    code, body = r.post("/api/arm")
    assert code == 200 and body["armed"] is True and r.fc.cmds == ["arm"]
    time.sleep(0.3)
    run = r.state()["run"]
    assert run["armed_at"] is not None and run["remaining_s"] is not None and run["halt_reason"] is None
    assert not r.hub.readiness()["ok"], "an armed system must not offer a second START"


def test_a_stranger_on_the_network_can_stop_but_not_start_or_initialize(rig, monkeypatch):
    r = rig()
    monkeypatch.setattr(hubmod, "is_local", lambda addr: False)
    assert r.post("/api/arm")[0] == 403 and r.post("/api/init")[0] == 403
    assert r.fc.cmds == []
    assert r.post("/api/stop", headers={})[0] == 200 and r.fc.cmds == ["disarm"], "STOP must stay open to everyone"
    assert r.state()["can_control"] is False


def test_start_needs_the_dashboards_header_and_a_same_origin_request(rig):
    r = rig()
    assert r.post("/api/arm", headers={})[0] == 403
    assert r.post("/api/arm", headers={"X-Arena": "control", "Origin": "http://evil.example"})[0] == 403
    assert r.fc.cmds == []
    assert r.post("/api/arm", headers={"X-Arena": "control", "Origin": r.base})[0] == 200


def test_watchdog_disarms_at_the_time_cap(rig):
    r = rig(cap_s=0.6)
    assert r.post("/api/arm")[0] == 200
    time.sleep(1.6)
    assert r.fc.cmds[0] == "arm" and r.fc.cmds.count("disarm") >= 1
    assert "time cap" in r.state()["run"]["halt_reason"]


def test_watchdog_disarms_when_the_car_leaves_the_arena(rig):
    r = rig()
    assert r.post("/api/arm")[0] == 200
    time.sleep(0.4)
    r.control["poses"]["pursuers"] = [[-1.05, 0.1, 3.1]]
    time.sleep(0.6)
    assert "disarm" in r.fc.cmds
    assert "left the arena" in r.state()["run"]["halt_reason"]


def test_watchdog_ends_the_run_when_the_pose_feed_is_lost_rather_than_resuming(rig):
    r = rig()
    assert r.post("/api/arm")[0] == 200
    time.sleep(0.4)
    r.control.update(state="STALLED", stall_reason="no pose frames")
    time.sleep(1.8)
    assert "disarm" in r.fc.cmds and "pose feed lost" in r.state()["run"]["halt_reason"]


def test_an_operator_stop_ends_the_run_quietly(rig):
    r = rig()
    assert r.post("/api/arm")[0] == 200
    time.sleep(0.4)
    assert r.post("/api/stop", headers={})[0] == 200
    time.sleep(2.0)
    assert "STOP" in r.state()["run"]["halt_reason"]


def script(tmp_path, body, name="arena"):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_initialize_runs_arena_up_and_reports_its_output(rig, tmp_path):
    cli = script(tmp_path, 'echo "$1 started"; echo "  \x1b[32mok\x1b[0m   camera"; exit 0\n')
    r = rig(arena_cli=cli)
    assert r.state()["can_init"] is True
    assert r.post("/api/init") == (200, {"started": True})
    for _ in range(40):
        job = r.state()["init"]
        if not job["running"]:
            break
        time.sleep(0.1)
    assert job["ok"] is True and job["rc"] == 0
    assert job["lines"] == ["up started", "  ok   camera"], "output is kept, ANSI colours stripped"


def test_initialize_that_fails_says_so(rig, tmp_path):
    r = rig(arena_cli=script(tmp_path, 'echo "FAIL pipeline not healthy"; exit 1\n'))
    assert r.post("/api/init")[0] == 200
    time.sleep(0.8)
    job = r.state()["init"]
    assert job["ok"] is False and job["rc"] == 1 and "FAIL" in job["lines"][-1]


def test_initialize_refuses_while_armed_or_already_running_or_unconfigured(rig, tmp_path):
    r = rig(arena_cli=script(tmp_path, "sleep 1.5\n"))
    assert r.post("/api/init")[0] == 200
    assert r.post("/api/init")[0] == 409, "one arena up at a time"
    time.sleep(1.9)
    assert r.post("/api/arm")[0] == 200
    time.sleep(0.4)
    code, body = r.post("/api/init")
    assert code == 409 and "ARMED" in body["error"], "never restart services under a driving car"
    assert rig().post("/api/init")[0] == 501, "no --arena-cli: the button must say so, not guess"
