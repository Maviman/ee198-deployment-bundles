"""Regenerate reference_vectors.json (the golden parity vectors selftest.py
checks against).

Only needs to be re-run if a model is re-exported or the observation contract
legitimately changes IN THE MAIN REPO -- in that case regenerate there first,
then re-vendor. Running it inside the portable folder just re-pins against the
vendored copies (still useful after edits, but it is not an independent check).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller_runtime.pose_types import VehiclePose  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402

FRAMES_PER_CASE = 3
CASES_PER_MODEL = 5


def _random_pose(rng: np.random.Generator, t: float) -> dict:
    # Stay comfortably inside the trained 14m x 10m arena.
    return {
        "x": float(rng.uniform(-6.0, 6.0)),
        "y": float(rng.uniform(-4.0, 4.0)),
        "heading": float(rng.uniform(-np.pi, np.pi)),
        "t": t,
    }


def _drift(pose: dict, rng: np.random.Generator, t: float) -> dict:
    # Small per-frame motion so finite-difference speed estimates are non-zero.
    return {
        "x": pose["x"] + float(rng.uniform(-0.25, 0.25)),
        "y": pose["y"] + float(rng.uniform(-0.25, 0.25)),
        "heading": pose["heading"] + float(rng.uniform(-0.2, 0.2)),
        "t": t,
    }


def main() -> None:
    reference: dict = {"description": "golden pose->observation->action vectors; see selftest.py", "models": {}}
    for model_name in ("n1_catch", "n1_pin"):
        rng = np.random.default_rng(42)
        loop = PortableLoop(str(ROOT / "models" / model_name))
        cases = []
        for _case in range(CASES_PER_MODEL):
            loop.reset()
            pursuers = [_random_pose(rng, 0.0) for _ in range(loop.policy.num_pursuers)]
            evader = _random_pose(rng, 0.0)
            frames = []
            obs = action = None
            for i in range(FRAMES_PER_CASE):
                t = i * 0.1
                if i > 0:
                    pursuers = [_drift(p, rng, t) for p in pursuers]
                    evader = _drift(evader, rng, t)
                frames.append({"pursuers": [dict(p) for p in pursuers], "evader": dict(evader)})
                loop.adapter.ingest(
                    pursuer_poses=[VehiclePose(p["x"], p["y"], p["heading"], p["t"]) for p in pursuers],
                    evader_pose=VehiclePose(evader["x"], evader["y"], evader["heading"], evader["t"]),
                )
                obs = loop.adapter.observation()
                action = loop.policy.act(obs)
            cases.append({
                "frames": frames,
                "final_observation": [float(v) for v in obs],
                "final_action": [float(v) for v in action],
            })
        reference["models"][model_name] = cases
        print(f"{model_name}: {len(cases)} cases")

    out = ROOT / "reference_vectors.json"
    out.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
