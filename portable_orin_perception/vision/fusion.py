"""Fuse one vehicle's pose from several cameras.

Two cameras at opposite corners of the arena each see the near half well (big
tags) and the far half badly (small, foreshortened tags). Each camera has its
own homography, fitted on the same four corner tags, so the two agree at the
corners and disagree by lens distortion and perspective error in between.

The fused pose is the weighted mean of the fresh estimates, weighted by the
square of the tag's on-screen side: localisation error in pixels is roughly
constant, so error in metres falls with tag size, and variance with its square.
A tag that is 50 px in one camera and 15 px in the other is 11:1 in favour of
the near camera, which is what you want; where both see it at a similar size
the average halves the noise. Heading is a circular mean.

Only estimates within ``window_s`` of each other are mixed. The cameras are not
synchronised, so the other camera's matching frame is up to one frame interval
old; mixing in anything older would drag a moving car's pose backwards.
"""

from __future__ import annotations

import math
from typing import Mapping


class PoseFuser:
    def __init__(self, vehicle_ids, window_s: float = 0.05) -> None:
        self.window_s = float(window_s)
        # vehicle id -> camera index -> (x, y, heading, side_px, t)
        self._est: dict[int, dict[int, tuple[float, float, float, float, float]]] = {
            int(v): {} for v in vehicle_ids}

    def update(self, cam: int, found: Mapping[int, tuple[float, float, float]],
               sizes: Mapping[int, float], t: float) -> dict[int, tuple[float, float, float, float, dict]]:
        """Record camera ``cam``'s detections at capture time ``t`` and return the fused
        pose of every vehicle with a fresh estimate: id -> (x, y, heading, t, share),
        where ``share`` is each contributing camera's weight fraction."""
        for vid, pose in found.items():
            if vid in self._est:
                self._est[vid][int(cam)] = (float(pose[0]), float(pose[1]), float(pose[2]),
                                            max(float(sizes.get(vid, 1.0)), 1.0), float(t))
        out = {}
        for vid, by_cam in self._est.items():
            fresh = {c: e for c, e in by_cam.items() if abs(t - e[4]) <= self.window_s}
            if not fresh:
                continue
            weights = {c: e[3] ** 2 for c, e in fresh.items()}
            total = sum(weights.values())
            x = sum(w * fresh[c][0] for c, w in weights.items()) / total
            y = sum(w * fresh[c][1] for c, w in weights.items()) / total
            s = sum(w * math.sin(fresh[c][2]) for c, w in weights.items())
            k = sum(w * math.cos(fresh[c][2]) for c, w in weights.items())
            heading = math.atan2(s, k)
            t_est = max(e[4] for e in fresh.values())
            out[vid] = (x, y, heading, t_est, {c: w / total for c, w in weights.items()})
        return out

    def forget(self, cam: int) -> None:
        for by_cam in self._est.values():
            by_cam.pop(int(cam), None)
