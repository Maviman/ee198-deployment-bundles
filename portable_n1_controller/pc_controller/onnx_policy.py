"""ONNX inference for a frozen V1 policy checkpoint.

Replaces controller_runtime.inference.PolicyRunner (which needs torch + skrl) with
onnxruntime only. The .onnx files were exported by controller_runtime/export_onnx.py
in the main repo and verified against the torch policy to <=1e-5 at export time;
the manifest records that verification plus everything needed to run inference
without touching the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxPolicy:
    def __init__(self, model_dir: str | Path) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        manifest = json.loads((model_dir / "policy.onnx.manifest.json").read_text(encoding="utf-8"))
        self.input_dim = int(manifest["input_dim"])
        self.action_dim = int(manifest["action_dim"])
        self.num_pursuers = int(manifest["num_pursuers"])
        self.use_car_cameras = bool(manifest["use_car_cameras"])
        self.action_history_k = int(manifest["action_history_k"])
        self.manifest = manifest
        self.metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(model_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Deterministic mean action for one observation vector. Output is the
        policy's raw mean, NOT clipped -- clip at the transport layer, matching
        how the env clips actions on ingest during training."""
        obs = np.asarray(observation, dtype=np.float32).reshape(1, self.input_dim)
        (action,) = self.session.run(["actions"], {"observations": obs})
        return action[0].astype(np.float64)
