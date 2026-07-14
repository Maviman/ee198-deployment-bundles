"""Runtime twin of single_pursuer.action_history.ActionHistoryWrapper.

The wrapper version is a gym-style env wrapper (reset/step); the controller runtime
has no gym env to wrap (poses arrive from perception, not env.step()), so this is
the same buffer logic exposed as a plain object. Semantics are pinned bit-for-bit
identical to the wrapper by ``tests/test_action_history_buffer.py`` -- an
action-history-hardened policy (e.g. coop_n3_D_ah1) must see the exact same
augmentation shape on hardware as it did in training.
"""

from __future__ import annotations

import numpy as np


class ActionHistoryBuffer:
    def __init__(self, k: int, action_dim: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.action_dim = int(action_dim)
        self._history = np.zeros((self.k, self.action_dim), dtype=np.float64)

    def reset(self) -> None:
        self._history[:] = 0.0

    def record(self, commanded_action) -> None:
        """Call once per tick with the action just sent to the vehicles (BEFORE
        any actuator lag/latency a real radio link adds -- the controller always
        knows what it itself commanded)."""
        commanded = np.asarray(commanded_action, dtype=np.float64).reshape(self.action_dim)
        self._history = np.roll(self._history, 1, axis=0)
        self._history[0] = np.clip(commanded, -1.0, 1.0)

    def augment(self, obs: np.ndarray) -> np.ndarray:
        """Append the history (most-recent-first) to a base observation."""
        return np.concatenate([obs, self._history.reshape(-1)]).astype(obs.dtype)
