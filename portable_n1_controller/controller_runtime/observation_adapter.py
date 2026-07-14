"""Build the trained policy's exact observation vector from raw perception poses.

Deliberately does NOT reimplement ``SinglePursuerEnv``'s feature layout (global
block + per-pursuer overhead block + optional POV block): reimplementing it here
would create a second copy of the contract that could silently drift from the one
every checkpoint was actually trained against -- exactly the failure mode this
project's ``tests/test_torch_batch_parity.py`` exists to prevent for the GPU core.
Instead, ``ObservationAdapter`` owns a template ``SinglePursuerEnv`` purely as a
config holder + the ``_observation()`` method, writes the latest estimated
poses/speeds into its ``pursuers``/``evader`` state, and calls that same method --
the identical pattern already used by ``PerturbedEnv._noisy_observation()``.

The template env is NEVER reset or stepped in real deployment; only its static
config (arena size, speed limits, FOV, lead time, capture radius) and its
``_observation()`` method are used.
"""

from __future__ import annotations

from single_pursuer.env import SinglePursuerEnv
from controller_runtime.pose_types import PoseHistory, VehiclePose


class ObservationAdapter:
    def __init__(self, template_env: SinglePursuerEnv, *, rate_limit_speed: bool = False) -> None:
        """``rate_limit_speed``: clamp each vehicle's estimated speed to change no
        faster than its own known accel/brake limits per tick. Off by default
        (matches the originally tested exact-delegation behavior). Mitigates -- does
        NOT fully fix -- the contact-push speed-spike limitation documented on
        ``PoseHistory``; see that docstring before relying on this for anything
        safety-critical."""
        self.env = template_env
        e = template_env
        if rate_limit_speed:
            self.pursuer_histories = [
                PoseHistory(max_accel=e.max_accel, max_decel=e.max_brake) for _ in range(e.num_pursuers)
            ]
            self.evader_history = PoseHistory(max_accel=e.evader_max_accel, max_decel=e.evader_max_brake)
        else:
            self.pursuer_histories = [PoseHistory() for _ in range(e.num_pursuers)]
            self.evader_history = PoseHistory()

    def ingest(
        self,
        *,
        pursuer_poses: list[VehiclePose],
        evader_pose: VehiclePose,
    ) -> None:
        """Feed one perception frame. Call once per control tick before observe()."""
        if len(pursuer_poses) != self.env.num_pursuers:
            raise ValueError(
                f"expected {self.env.num_pursuers} pursuer poses, got {len(pursuer_poses)}"
            )
        for history, pose in zip(self.pursuer_histories, pursuer_poses):
            history.update(pose)
        self.evader_history.update(evader_pose)

    @property
    def ready(self) -> bool:
        """False until at least one frame has been ingested for every vehicle
        (speed estimates need a second sample; the first tick necessarily reports
        zero speed for everyone, which is a safe/correct assumption at startup)."""
        return self.evader_history.has_pose and all(h.has_pose for h in self.pursuer_histories)

    def observation(self):
        """Returns the observation vector, identical in layout/scale to
        SinglePursuerEnv._observation(). Raises if ingest() was never called."""
        if not self.ready:
            raise RuntimeError("ingest() must be called for every vehicle before observation()")

        e = self.env
        e.pursuers = [
            {"x": h.pose.x, "y": h.pose.y, "heading": h.pose.heading, "speed": h.speed}
            for h in self.pursuer_histories
        ]
        ev = self.evader_history.pose
        e.evader = {"x": ev.x, "y": ev.y, "heading": ev.heading, "speed": self.evader_history.speed}
        return e._observation()
