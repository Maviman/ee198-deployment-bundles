"""Pick the control loop a model needs, from its manifest.

Two kinds of model run here:

* A flat policy (no "runtime" key in the manifest; n1_catch, n1_pin). The
  observation comes from the vendored SinglePursuerEnv, one ONNX pass runs, and
  the actions come straight out. That is PortableLoop.
* A role commander ("runtime": "hive_commander_v1", the Isaac hive model). One
  shared attention network picks a role, two settings and a correction for each
  car, and a scripted playbook turns those into throttle/steer. Everything
  between kinematic state and commands lives in the vendored runtime module
  (pc_controller/runtimes/). HiveLoop feeds that module and exposes the same
  interface as PortableLoop, so the real-time loop, the sim and the selftest
  drive either kind.

The runtime code is vendored in this repo on purpose, not loaded from the model
folder: the code that drives the cars is versioned and reviewed here, and a
model folder stays data only (weights, manifest, metrics, golden vectors). A
retrain that keeps the same runtime only changes the model folder.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from controller_runtime.latency import LatencyBudget, LatencyTracker
from controller_runtime.pose_types import VehiclePose

# manifest "runtime" -> the vendored module that implements it
RUNTIMES = {"hive_commander_v1": "pc_controller.runtimes.hive_commander_v1"}


def read_manifest(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "policy.onnx.manifest.json").read_text(encoding="utf-8"))


def make_loop(model_dir: str | Path, *, rate_limit_speed: bool = False):
    """The control loop for this model: PortableLoop, or the runtime its manifest names."""
    manifest = read_manifest(model_dir)
    # The cars carry no cameras (decided 2026-10-02): every input comes from the
    # overhead camera. A model trained with per-car camera features would be fed
    # values synthesized from overhead poses instead, so it is refused.
    if manifest.get("use_car_cameras"):
        raise SystemExit(f"{model_dir}: trained with car cameras (use_car_cameras: true), but the cars "
                         "have none. Use a model trained on the overhead camera only.")
    runtime = manifest.get("runtime")
    if not runtime:
        from pc_controller.portable_loop import PortableLoop
        return PortableLoop(str(model_dir), rate_limit_speed=rate_limit_speed)
    if runtime not in RUNTIMES:
        raise SystemExit(f"{model_dir}: model runtime {runtime!r} is not supported by this code "
                         f"(supported: {', '.join(RUNTIMES)}). Update the deployment code.")
    return HiveLoop(model_dir, manifest)


class SignedSpeed:
    """Forward speed from successive poses: the displacement projected on the car's
    heading, so driving backwards reads negative. The hive model was trained on
    signed speed. The same caveat as PoseHistory applies: a car shoved by
    contact reads as moving."""

    def __init__(self) -> None:
        self._prev: VehiclePose | None = None
        self.v = 0.0

    def reset(self) -> None:
        self._prev = None
        self.v = 0.0

    def update(self, pose: VehiclePose) -> None:
        prev = self._prev
        if prev is not None:
            dt = pose.timestamp_s - prev.timestamp_s
            if dt > 1e-6:
                dx, dy = pose.x - prev.x, pose.y - prev.y
                self.v = (dx * math.cos(pose.heading) + dy * math.sin(pose.heading)) / dt
        self._prev = pose


class HiveLoop:
    """A role-commander model behind PortableLoop's interface."""

    def __init__(self, model_dir: str | Path, manifest: dict | None = None) -> None:
        model_dir = Path(model_dir)
        manifest = manifest or read_manifest(model_dir)
        module = importlib.import_module(RUNTIMES[manifest["runtime"]])
        self.runtime = module.CommanderRuntime(str(model_dir))
        n = int(manifest["num_pursuers"])
        metrics_path = model_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        # What run_controller and the real-time loop read from a loop's policy.
        self.policy = SimpleNamespace(num_pursuers=n, input_dim=int(manifest.get("input_dim", 0)),
                                      action_dim=2 * n, metrics=metrics, manifest=manifest)
        self.capture_radius = float(manifest.get("capture_radius_m", manifest.get("capture_radius", 0.0)))
        self.latency_budget = LatencyBudget(0.1)
        self._speeds = [SignedSpeed() for _ in range(n + 1)]   # pursuers, then the evader
        self._last_t: float | None = None

    @property
    def roles(self) -> list[str]:
        return [str(r) for r in (getattr(self.runtime, "roles", None) or [])]

    def reset(self) -> None:
        """Episode start, or a re-arm after a pose stall: stale speeds and roles go."""
        self.runtime.reset()
        for s in self._speeds:
            s.reset()
        self._last_t = None

    def tick(self, *, pursuer_poses: list[VehiclePose], evader_pose: VehiclePose,
             tracker: LatencyTracker | None = None) -> np.ndarray:
        tracker = tracker or LatencyTracker()
        tracker.mark("pose_ingested")
        poses = list(pursuer_poses) + [evader_pose]
        if len(poses) != len(self._speeds):
            raise ValueError(f"expected {len(self._speeds) - 1} pursuer poses, got {len(pursuer_poses)}")
        for speed, pose in zip(self._speeds, poses):
            speed.update(pose)
        state = np.array([[p.x, p.y, p.heading, s.v] for p, s in zip(poses, self._speeds)], dtype=np.float64)
        t = evader_pose.timestamp_s
        # The model was trained at a fixed 0.1 s step; a real tick is close to that.
        dt = 0.1 if self._last_t is None else min(max(t - self._last_t, 0.02), 0.5)
        self._last_t = t
        tracker.mark("observation_built")
        action = np.asarray(self.runtime.step(state[:-1], state[-1], dt=dt), dtype=np.float64).reshape(-1)
        tracker.mark("action_computed")
        self.latency_budget.record(tracker.finish())
        return action
