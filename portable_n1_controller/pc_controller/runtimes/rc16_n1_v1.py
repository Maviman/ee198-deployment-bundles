"""One-pursuer chase on the Orin: numpy + onnxruntime only (runtime "rc16_n1_v1").

Source of truth for the deployment bundle's pc_controller/runtimes/rc16_n1_v1.py (copied verbatim). It reproduces,
operation for operation, the observation that rc16/train/chase_env.py builds from camera poses in training:

    step(pursuers, evader, dt)
      pursuers  (1, 4) [x, y, heading, v]   arena metres, heading CCW from +x (rad), v = SIGNED forward speed (m/s)
      evader    (4,)   [x, y, heading, v]   as the deployment loop estimates it (SignedSpeed: displacement since the
                                            last pose projected on the heading / dt)
      ->        (1, 2) [throttle, steer] in [-1, 1]: throttle = PWM duty (+ forward), steer = proportional steering
                                            target as a fraction of full lock (+ left)

Internal state (reset() clears it): the previous poses (yaw rate = wrapped heading change / dt; a pose that did not
change keeps the previous yaw rate, as in training), the steering target the car is holding (replicated from the
commands sent: deadband, then the per-tick slew limit), and the last command.

Nothing here imports torch or Isaac. tests/test_rc16_n1_runtime.py (training repo) pins this file to the torch
observation; golden_vectors.json in the model folder pins it end to end (poses -> command).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RUNTIME = "rc16_n1_v1"


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def body(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx + s * vy, -s * vx + c * vy


def build_obs(m: dict, p, p_v: float, p_yr: float, steer: float, e, e_v: float, last_cmd) -> np.ndarray:
    """The 16-entry observation (rc16/train/chase_env.py build_obs). p, e = (x, y, heading)."""
    top, half, side, lock = m["top_speed_mps"], m["arena_half_m"], m["arena_side_m"], m["max_steer_rad"]
    rx, ry = body(e[0] - p[0], e[1] - p[1], p[2])
    dist = math.hypot(rx, ry)
    dh = e[2] - p[2]
    evx, evy = body(math.cos(e[2]) * e_v, math.sin(e[2]) * e_v, p[2])
    obs = [p_v / top, p_yr / m["yaw_rate_scale"], steer / lock,
           p[0] / half, p[1] / half, math.cos(p[2]), math.sin(p[2]),
           rx / side, ry / side, dist / side, math.cos(dh), math.sin(dh),
           evx / top, evy / top, float(last_cmd[0]), float(last_cmd[1])]
    return np.nan_to_num(np.asarray(obs, dtype=np.float32))[None, :]


class CommanderRuntime:
    """Name kept for the deployment loop (pc_controller/loops.py HiveLoop instantiates CommanderRuntime)."""

    def __init__(self, model_dir, seed=None, session=None):
        model_dir = Path(model_dir)
        self.manifest = json.loads((model_dir / "policy.onnx.manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("runtime") != RUNTIME:
            raise ValueError(f"{model_dir}: manifest runtime {self.manifest.get('runtime')!r} is not {RUNTIME!r}")
        self.m = self.manifest["observation"]
        self.throttle_limit = float(self.manifest.get("deploy_throttle_limit", 1.0))
        if session is None:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.reset()

    def reset(self, roles=None) -> None:
        self.prev = None
        self.yaw_rate = [0.0, 0.0]
        self.steer = 0.0
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.last = {}

    @property
    def roles(self) -> list[str]:
        return ["CHASE"]

    def policy(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(None, {self.input_name: obs})[0], dtype=np.float32).reshape(-1)

    def step(self, pursuers, evader, dt: float = 0.1):
        p = np.asarray(pursuers, dtype=np.float64).reshape(-1, 4)[0]
        e = np.asarray(evader, dtype=np.float64).reshape(4)
        poses = [(float(p[0]), float(p[1]), float(p[2])), (float(e[0]), float(e[1]), float(e[2]))]
        dt = float(dt) if dt and dt > 1e-6 else 0.1
        if self.prev is not None:
            for i in range(2):
                if poses[i] != self.prev[i]:                      # an unchanged pose keeps the previous yaw rate
                    self.yaw_rate[i] = wrap(poses[i][2] - self.prev[i][2]) / dt
        self.prev = poses
        obs = build_obs(self.m, poses[0], float(p[3]), self.yaw_rate[0], self.steer, poses[1], float(e[3]),
                        self.last_cmd)
        cmd = np.clip(self.policy(obs), -1.0, 1.0).astype(np.float32)
        # the drive's steering target after this command: deadband, then the slew limit per control tick
        s = float(cmd[1])
        goal = (0.0 if abs(s) < self.m["deadband"] else s) * self.m["max_steer_rad"]
        step = self.m["steer_step_rad"]
        self.steer = self.steer + max(-step, min(step, goal - self.steer))
        self.last_cmd = cmd.copy()
        out = cmd.copy()
        out[0] = float(np.clip(out[0], -self.throttle_limit, self.throttle_limit))
        self.last = {"obs": obs[0].tolist(), "command": out.tolist(), "roles": self.roles}
        return out.reshape(1, 2)
