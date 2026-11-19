"""Three-pursuer playbook commander on the controller (runtime "rc16_hive_v1").

Source of truth for the deployment bundle's pc_controller/runtimes/rc16_hive_v1.py (copied verbatim by
rc16/deploy/export_hive.py, together with the rc16 hive library it needs as pc_controller/runtimes/rc16_hive_lib/).

Unlike rc16_n1_v1 this runtime NEEDS TORCH (CPU is enough): the playbook, pawn drivers, distance limiter and
out-of-bounds recovery are the training code itself (rc16/hive/*.py), run here on one "env" on the CPU. Only the
commander network goes through onnxruntime.

    step(pursuers, evader, dt)
      pursuers  (3, 4) [x, y, heading, v]   arena metres, heading CCW from +x (rad), v = SIGNED forward speed (m/s),
                                            order = the order the loop lists the cars (p0, p1, p2)
      evader    (4,)   [x, y, heading, v]
      ->        (3, 2) [throttle, steer] in [-1, 1] per pursuer (same meaning as rc16_n1_v1)

What it replays of rc16/hive/hive_env.py, in the env's order, once per 0.1 s call:
  1. (not on the first call) the running play's progress tick: playbook.progress(world) -> CommanderState.tick
  2. every 5th call (0.5 s, the first call included): the commander decides -- build_obs -> ONNX -> argmax ->
     CommanderState.apply (0 = keep the play, 1..14 = switch, honoured after the 1.0 s minimum commitment)
  3. the control tick: playbook.slots -> driver.drive -> recovery.step -> shield -> clamp to [-1, 1]
Every car counts as seen on every call (the loop's dead-man stop handles a pose stall). The evader-side logic of
the env (the scripted evader, the box-stop tracker, rewards) is not part of deployment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUNTIME = "rc16_hive_v1"

_LIB = Path(__file__).resolve().parent / "rc16_hive_lib"
if _LIB.is_dir() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))          # the vendored copy wins over anything else named rc16

import torch  # noqa: E402

from rc16.hive import boxstop as BX  # noqa: E402
from rc16.hive import commander_io as CIO  # noqa: E402
from rc16.hive import driver as DRV  # noqa: E402
from rc16.hive import geometry as G  # noqa: E402
from rc16.hive import plays as PL  # noqa: E402
from rc16.hive import rules as RU  # noqa: E402
from rc16.hive import shield as SH  # noqa: E402
from rc16.hive.recovery import Recovery  # noqa: E402

CHOICE_NAMES = tuple(p if a is None else f"{p}_{a}" for p, a in PL.CHOICES)
REGROUP = PL.CHOICES.index(("REGROUP", None))


class CommanderRuntime:
    """Name kept for the deployment loop (pc_controller/loops.py HiveLoop instantiates CommanderRuntime)."""

    def __init__(self, model_dir, seed=None, session=None, policy=None):
        model_dir = Path(model_dir)
        self.manifest = json.loads((model_dir / "policy.onnx.manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("runtime") != RUNTIME:
            raise ValueError(f"{model_dir}: manifest runtime {self.manifest.get('runtime')!r} is not {RUNTIME!r}")
        self.tape_wall = bool(self.manifest.get("tape_wall", True))
        self.throttle_limit = float(self.manifest.get("deploy_throttle_limit", 1.0))
        self.blank_memory = bool(self.manifest.get("blank_memory", False))
        self._policy = policy               # optional torch callable obs (1,118) -> logits (1,15) (tests, export)
        if policy is None:
            if session is None:
                import onnxruntime as ort
                session = ort.InferenceSession(str(model_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
            self.session = session
            self.input_name = session.get_inputs()[0].name
        torch.set_num_threads(1)
        self.dev = torch.device("cpu")
        self.playbook = PL.Playbook(1, self.dev, tape_wall=self.tape_wall)
        self.recovery = Recovery(1, self.dev)
        self.driver_state = DRV.DriverState(1, self.dev)
        self.cs = CIO.CommanderState(1, self.dev)
        self.reset()

    def reset(self, roles=None) -> None:
        ids = torch.arange(1)
        self.playbook.reset(ids)
        self.recovery.reset(ids)
        self.driver_state.reset(ids)
        self.cs.reset(ids, first_choice=REGROUP)
        self.k = 0
        self.ticks = torch.zeros(1, dtype=torch.long)
        self.t = torch.zeros(1)
        self.recovering = torch.zeros(1, 3, dtype=torch.bool)
        self.action = 0
        self.last = {}

    @property
    def roles(self) -> list[str]:
        """The running play's name, once per pursuer (the loop shows these)."""
        name = CHOICE_NAMES[int(self.cs.choice[0])]
        return [f"{name}{' (recovering)' if bool(r) else ''}" for r in self.recovering[0]]

    def logits(self, obs: torch.Tensor) -> np.ndarray:
        if self._policy is not None:
            with torch.no_grad():
                return self._policy(obs).float().numpy().reshape(-1)
        x = obs.numpy().astype(np.float32)
        return np.asarray(self.session.run(None, {self.input_name: x})[0], dtype=np.float32).reshape(-1)

    def _world(self, p, e) -> G.World:
        f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))     # noqa: E731
        return G.World(p_xy=f(p[None, :, :2]), p_yaw=f(p[None, :, 2]), p_v=f(p[None, :, 3]),
                       e_xy=f(e[None, :2]), e_yaw=f(e[None, 2]), e_v=f(e[None, 3]),
                       p_seen=torch.ones(1, 3, dtype=torch.bool), e_seen=torch.ones(1, dtype=torch.bool))

    def step(self, pursuers, evader, dt: float = 0.1):
        p = np.asarray(pursuers, dtype=np.float64).reshape(3, 4)
        e = np.asarray(evader, dtype=np.float64).reshape(4)
        dt = float(dt) if dt and dt > 1e-6 else RU.TICK_S
        w = self._world(p, e)
        with torch.no_grad():
            if self.k > 0:                                       # the env's _book: progress of the 0.1 s just run
                self.ticks += 1
                self.t = self.ticks.float() * RU.TICK_S
                self.cs.tick(self.playbook.progress(w, self.cs.choice), torch.full((1,), dt))
            decided = self.k % RU.DECISION_TICKS == 0
            obs = None
            if decided:                                          # the env's _get_observations + _pre_physics_step
                e_pose = torch.cat([w.e_xy, w.e_yaw[:, None]], -1)
                free = BX.free_travel(e_pose, torch.cat([w.p_xy, w.p_yaw[..., None]], -1), torch.ones_like(w.p_seen))
                if self.tape_wall:
                    free = torch.minimum(free, BX.tape_travel(e_pose))
                blocked = (free < BX.G_START_M).float().mean(-1)
                obs = CIO.build_obs(w, self.cs, self.recovering, blocked, blank_memory=self.blank_memory)
                lg = self.logits(obs)
                self.action = int(np.argmax(lg))
                self.cs.apply(torch.tensor([self.action]), self.t)
                self.last_logits = lg
            # the env's _control
            slots = self.playbook.slots(w, self.cs.choice, self.cs.t_in_play, ~self.recovering)
            cmd = DRV.drive(w.p_xy, w.p_yaw, w.p_v, slots, w.e_xy, w.e_yaw, state=self.driver_state)
            cmd, rec, _ = self.recovery.step(w.p_xy, w.p_yaw, w.p_v, w.p_seen, cmd, dt=RU.TICK_S)
            cmd, _ = SH.shield(cmd, w.p_xy, w.p_yaw, RU.stale_speed(w.p_v, torch.zeros(1, 3, dtype=torch.long), RU.TICK_S))
            cmd = torch.nan_to_num(cmd).clamp(-1.0, 1.0)
            self.recovering = rec
        self.k += 1
        out = cmd[0].numpy().astype(np.float32).copy()
        out[:, 0] = np.clip(out[:, 0], -self.throttle_limit, self.throttle_limit)
        self.last = {"command": out.tolist(), "roles": self.roles, "choice": int(self.cs.choice[0]),
                     "action": self.action, "decided": decided}
        if obs is not None:
            self.last["obs"] = obs[0].tolist()
        return out
