"""Per-hop latency instrumentation.

Phase 2's robustness grid found command latency is the dominant sim-to-real risk
(0.1s tolerable at ~90% capture, 0.2s collapses to ~7% for the un-hardened policy;
even the hardened champion caps out around 0.1s of real margin). This tracker times
every hop from camera frame to command transmission so the real pipeline's latency
budget can be measured, not assumed, against that ~0.1s requirement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class LatencySample:
    hops: dict[str, float] = field(default_factory=dict)  # hop name -> wall-clock timestamp

    def deltas(self) -> dict[str, float]:
        """Seconds elapsed from the FIRST recorded hop to each subsequent one."""
        if not self.hops:
            return {}
        ordered = sorted(self.hops.items(), key=lambda kv: kv[1])
        t0 = ordered[0][1]
        return {name: ts - t0 for name, ts in ordered}

    @property
    def total_s(self) -> float:
        d = self.deltas()
        return max(d.values()) if d else 0.0


class LatencyTracker:
    """One tracker per control tick: call mark(hop_name) at each pipeline stage
    (e.g. 'frame_captured', 'pose_estimated', 'observation_built', 'action_computed',
    'command_sent'), then finish() to get the completed sample."""

    def __init__(self) -> None:
        self._sample = LatencySample()

    def mark(self, hop_name: str) -> None:
        self._sample.hops[hop_name] = time.perf_counter()

    def finish(self) -> LatencySample:
        return self._sample


class LatencyBudget:
    """Rolling check against the deployment latency requirement."""

    def __init__(self, budget_s: float = 0.1) -> None:
        self.budget_s = budget_s
        self.samples: list[LatencySample] = []

    def record(self, sample: LatencySample) -> None:
        self.samples.append(sample)

    def within_budget_fraction(self) -> float:
        if not self.samples:
            return 1.0
        ok = sum(1 for s in self.samples if s.total_s <= self.budget_s)
        return ok / len(self.samples)

    def summary(self) -> dict[str, float]:
        totals = [s.total_s for s in self.samples]
        if not totals:
            return {"count": 0, "mean_s": 0.0, "max_s": 0.0, "within_budget_fraction": 1.0}
        return {
            "count": len(totals),
            "mean_s": sum(totals) / len(totals),
            "max_s": max(totals),
            "within_budget_fraction": self.within_budget_fraction(),
        }
