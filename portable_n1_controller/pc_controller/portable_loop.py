"""The portable equivalent of controller_runtime.runtime_loop.RuntimeLoop:
pose ingest -> observation -> (optional action history) -> ONNX inference ->
latency tracking, with the torch-based PolicyRunner swapped for OnnxPolicy.

The observation is built by the VENDORED ObservationAdapter, which delegates to
the vendored SinglePursuerEnv._observation() -- the exact feature layout every
checkpoint was trained against. Parity of this vendored copy is pinned by
selftest.py's golden vectors (generated against the main repo).
"""

from __future__ import annotations

import numpy as np

from single_pursuer.env import SinglePursuerEnv
from controller_runtime.action_history_buffer import ActionHistoryBuffer
from controller_runtime.latency import LatencyBudget, LatencyTracker
from controller_runtime.observation_adapter import ObservationAdapter
from controller_runtime.pose_types import VehiclePose

from pc_controller.onnx_policy import OnnxPolicy


class PortableLoop:
    def __init__(
        self,
        model_dir: str,
        *,
        latency_budget_s: float = 0.1,
        rate_limit_speed: bool = False,
    ) -> None:
        self.policy = OnnxPolicy(model_dir)
        template_env = SinglePursuerEnv(
            num_pursuers=self.policy.num_pursuers,
            use_car_cameras=self.policy.use_car_cameras,
            seed=0,
        )
        self._rate_limit_speed = rate_limit_speed
        self.adapter = ObservationAdapter(template_env, rate_limit_speed=rate_limit_speed)
        self.action_history = (
            ActionHistoryBuffer(self.policy.action_history_k, self.policy.action_dim)
            if self.policy.action_history_k > 0
            else None
        )
        self.latency_budget = LatencyBudget(latency_budget_s)

    @property
    def capture_radius(self) -> float:
        return float(getattr(self.adapter.env, "capture_radius", 0.0))

    def reset(self) -> None:
        """Clear pose/action history at episode start or after a real-world re-arm."""
        self.adapter = ObservationAdapter(self.adapter.env, rate_limit_speed=self._rate_limit_speed)
        if self.action_history is not None:
            self.action_history.reset()

    def tick(
        self,
        *,
        pursuer_poses: list[VehiclePose],
        evader_pose: VehiclePose,
        tracker: LatencyTracker | None = None,
    ) -> np.ndarray:
        """One control cycle: ingest this frame's poses, return the commanded
        action (normalized, throttle then steer per pursuer)."""
        tracker = tracker or LatencyTracker()
        tracker.mark("pose_ingested")
        self.adapter.ingest(pursuer_poses=pursuer_poses, evader_pose=evader_pose)
        obs = self.adapter.observation()
        tracker.mark("observation_built")

        if self.action_history is not None:
            obs = self.action_history.augment(obs)

        action = self.policy.act(obs)
        tracker.mark("action_computed")

        if self.action_history is not None:
            self.action_history.record(action)

        self.latency_budget.record(tracker.finish())
        return action
