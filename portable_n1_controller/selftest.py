"""Prove this portable bundle works on THIS machine, no hardware needed:

  1. Golden-vector parity: feed recorded pose frames through the vendored
     observation adapter + ONNX policy and require bit-close agreement with the
     reference vectors frozen at bundle-creation time. Catches a broken install,
     a drifted vendored file, or a swapped/corrupted model in one shot.
  2. Command packet round-trip: EspLink -> real UDP socket -> parse.
  3. Perception frame parsing (the --source udp input format).

Run:  python selftest.py
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from controller_runtime.pose_types import VehiclePose  # noqa: E402
from pc_controller.esp_link import EspLink, parse_command_packet  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402
from pc_controller.pose_stream import parse_pose_frame  # noqa: E402

OBS_ATOL = 1e-9   # same numpy code path -> should be exact; tolerance for safety
ACT_ATOL = 1e-5   # onnxruntime kernel variation across machines/versions


def check_golden_vectors() -> list[str]:
    failures = []
    reference = json.loads((ROOT / "reference_vectors.json").read_text(encoding="utf-8"))
    for model_name, cases in reference["models"].items():
        loop = PortableLoop(str(ROOT / "models" / model_name))
        for i, case in enumerate(cases):
            loop.reset()
            obs = action = None
            for frame in case["frames"]:
                loop.adapter.ingest(
                    pursuer_poses=[
                        VehiclePose(p["x"], p["y"], p["heading"], p["t"]) for p in frame["pursuers"]
                    ],
                    evader_pose=VehiclePose(
                        frame["evader"]["x"], frame["evader"]["y"],
                        frame["evader"]["heading"], frame["evader"]["t"],
                    ),
                )
                obs = loop.adapter.observation()
                action = loop.policy.act(obs)
            obs_err = float(np.abs(np.asarray(obs) - np.asarray(case["final_observation"])).max())
            act_err = float(np.abs(np.asarray(action) - np.asarray(case["final_action"])).max())
            if obs_err > OBS_ATOL:
                failures.append(f"{model_name} case {i}: observation drift, max err {obs_err:.3e}")
            if act_err > ACT_ATOL:
                failures.append(f"{model_name} case {i}: action drift, max err {act_err:.3e}")
        print(f"  {model_name}: {len(cases)} golden cases checked")
    return failures


def check_packet_roundtrip() -> list[str]:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]
    link = EspLink("127.0.0.1", port)
    sent = link.send_commands(np.array([0.437, -1.7]))  # -1.7 must clip to -1.0
    raw, _ = rx.recvfrom(65535)
    stop = link.send_stop()
    raw_stop, _ = rx.recvfrom(65535)
    link.sock.close()
    rx.close()

    failures = []
    got = parse_command_packet(raw)
    if got != sent:
        failures.append(f"packet round-trip mismatch: sent {sent}, parsed {got}")
    if got["cmd"][0] != [0.437, -1.0]:
        failures.append(f"expected clipped cmd [0.437, -1.0], got {got['cmd'][0]}")
    got_stop = parse_command_packet(raw_stop)
    if not got_stop["estop"] or got_stop["cmd"][0] != [0.0, 0.0] or got_stop["seq"] != stop["seq"]:
        failures.append(f"bad stop packet: {got_stop}")
    print("  packet round-trip + clip + e-stop checked")
    return failures


def check_pose_frame_parsing() -> list[str]:
    frame = ('{"t": 12.5, "pursuers": [{"x": 1.0, "y": -2.0, "heading": 0.5}], '
             '"evader": {"x": 3.0, "y": 1.0, "heading": -1.2}}')
    pursuers, evader = parse_pose_frame(frame.encode("utf-8"), expected_pursuers=1)
    failures = []
    if (pursuers[0].x, pursuers[0].y, pursuers[0].heading, pursuers[0].timestamp_s) != (1.0, -2.0, 0.5, 12.5):
        failures.append(f"bad pursuer parse: {pursuers[0]}")
    if (evader.x, evader.heading) != (3.0, -1.2):
        failures.append(f"bad evader parse: {evader}")
    try:
        parse_pose_frame(frame.encode("utf-8"), expected_pursuers=3)
        failures.append("pursuer-count mismatch not rejected")
    except ValueError:
        pass
    print("  perception frame parsing checked")
    return failures


def main() -> None:
    failures: list[str] = []
    print("[1/3] golden-vector parity (obs contract + ONNX models)")
    failures += check_golden_vectors()
    print("[2/3] ESP command packet round-trip over UDP")
    failures += check_packet_roundtrip()
    print("[3/3] perception pose frame parsing")
    failures += check_pose_frame_parsing()

    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSELFTEST PASSED - bundle is healthy on this machine.")


if __name__ == "__main__":
    main()
