"""Synthetic overhead camera: renders real printed-tag images at known arena
poses through a homography, so the whole pipeline (decode -> detect -> pose ->
controller) can run and be measured with no camera.

Used by `arena sim`, the fast-path tests, and benchmarks. The tags are genuine
codebook images warped with a true perspective transform, and the background
has tape, cables and clutter for the quad detector to reject. Detection cost on
these frames is a lower bound on a real carpet.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from hive_perception.core import arena_frame
from hive_perception.core.tag_family import family_spec


def overhead_homography(image_size, arena_half_extent, fill: float = 0.9) -> np.ndarray:
    """Pixel -> arena homography for a camera looking straight down, arena
    centered, the arena's larger dimension filling ``fill`` of the image's
    matching dimension (the short side, for a square arena)."""
    w, h = image_size
    hw, hh = arena_half_extent
    ppm = fill * min(w / (2 * hw), h / (2 * hh))
    px = [[w / 2 - hw * ppm, h / 2 + hh * ppm], [w / 2 + hw * ppm, h / 2 + hh * ppm],
          [w / 2 + hw * ppm, h / 2 - hh * ppm], [w / 2 - hw * ppm, h / 2 - hh * ppm]]
    arena = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]
    H, _ = arena_frame.fit_homography(px, arena)
    return H


def tag_corners_arena(x: float, y: float, heading: float, side_m: float) -> np.ndarray:
    """Arena-frame corners [TL, TR, BR, BL] of a tag whose canonical top edge
    faces ``heading`` (the mounting rule: top edge toward the car's nose)."""
    f = np.array([math.cos(heading), math.sin(heading)]) * side_m / 2
    r = np.array([math.sin(heading), -math.cos(heading)]) * side_m / 2
    c = np.array([x, y])
    return np.array([c + f - r, c + f + r, c - f + r, c - f - r])


class SceneRenderer:
    def __init__(self, marker_map, homography, image_size, arena_half_extent, *, seed: int = 0,
                 blur_px: int = 3, noise: float = 4.0) -> None:
        self.marker_map = marker_map
        self.H = np.asarray(homography, dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.size = (int(image_size[0]), int(image_size[1]))
        self.half = arena_half_extent
        self.blur_px = int(blur_px)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        spec = family_spec(marker_map.dictionary)
        self.full_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.opencv_dict))
        self._tiles: dict[int, np.ndarray] = {}
        self.background = self._background()

    def _background(self) -> np.ndarray:
        w, h = self.size
        rng = self.rng
        img = rng.normal(118, 22, (h, w)).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 2.5)
        for _ in range(40):                               # cables, seams
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p1, p2, float(rng.integers(20, 235)), int(rng.integers(1, 4)))
        for _ in range(18):                               # boxes, tape scraps
            x, y = int(rng.integers(0, w - 50)), int(rng.integers(0, h - 50))
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(8, 50))),
                          float(rng.integers(20, 235)), -1)
        hw, hh = self.half
        edge = arena_frame.project_px_to_arena(
            self.Hinv, [[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]]).astype(np.int32)
        cv2.polylines(img, [edge], True, 240.0, max(2, w // 300))   # boundary tape
        return np.clip(img, 0, 255).astype(np.uint8)

    def _tile(self, marker_id: int) -> np.ndarray:
        if marker_id not in self._tiles:
            n = 240
            quiet = n // 6
            tile = np.full((n + 2 * quiet, n + 2 * quiet), 255, np.uint8)
            tile[quiet:quiet + n, quiet:quiet + n] = cv2.aruco.generateImageMarker(self.full_dict, marker_id, n)
            self._tiles[marker_id] = tile
        return self._tiles[marker_id]

    def _stamp(self, img, marker_id: int, corners_arena: np.ndarray) -> None:
        tile = self._tile(marker_id)
        n_total = tile.shape[0]
        quiet = (n_total - 240) / 2
        # Tile pixels of the BLACK square's corners -> image pixels of the tag corners.
        src = np.float32([[quiet, quiet], [n_total - quiet, quiet],
                          [n_total - quiet, n_total - quiet], [quiet, n_total - quiet]])
        dst = arena_frame.project_px_to_arena(self.Hinv, corners_arena).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        tile_corners = cv2.perspectiveTransform(
            np.float32([[[0, 0], [n_total, 0], [n_total, n_total], [0, n_total]]]), M)[0]
        x0, y0 = np.floor(tile_corners.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(tile_corners.max(axis=0)).astype(int) + 2
        h, w = img.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64) @ M
        patch = cv2.warpPerspective(tile, T, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tile, 255), T, (x1 - x0, y1 - y0),
                                   flags=cv2.INTER_NEAREST, borderValue=0)
        roi = img[y0:y1, x0:x1]
        roi[mask > 0] = patch[mask > 0]

    def render(self, vehicles: dict, *, corners: dict | None = None) -> np.ndarray:
        """``vehicles``: {tag id: (x, y, heading)}; ``corners``: {tag id: (x, y)}."""
        img = self.background.copy()
        mm = self.marker_map
        for cid, (cx, cy) in (corners or {}).items():
            self._stamp(img, cid, tag_corners_arena(cx, cy, math.pi / 2, mm.corner_tag_m))
        for vid, (x, y, heading) in vehicles.items():
            self._stamp(img, vid, tag_corners_arena(x, y, heading, mm.vehicle_tag_m))
        if self.blur_px > 1:
            k = np.zeros((self.blur_px, self.blur_px), np.float32)
            k[self.blur_px // 2, :] = 1.0 / self.blur_px
            img = cv2.filter2D(img, -1, k)
        if self.noise > 0:
            img = cv2.add(img, self.rng.normal(0, self.noise, img.shape).astype(np.int16),
                          dtype=cv2.CV_8U)
        return img

    @staticmethod
    def to_jpeg(gray: np.ndarray, quality: int = 85) -> bytes:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()


class ChaseScript:
    """Plausible motion for a demo/sim: the evader drives a slow figure-eight
    and each pursuer drives a lagged copy of it. Not physics, and it ignores
    the controller's commands. It exists to exercise the pipeline with moving
    tags at realistic speeds (<= ~0.6 m/s)."""

    def __init__(self, n_pursuers: int, half_extent, speed: float = 0.35) -> None:
        self.n = n_pursuers
        self.hw, self.hh = half_extent
        self.speed = speed

    def _path(self, s: float):
        ax, ay = self.hw * 0.55, self.hh * 0.45
        x, y = ax * math.sin(s), ay * math.sin(2 * s)
        dx, dy = ax * math.cos(s), 2 * ay * math.cos(2 * s)
        return x, y, math.atan2(dy, dx)

    def poses(self, t: float, marker_map) -> dict:
        omega = self.speed / max(self.hw, 0.3)
        out = {marker_map.evader_id: self._path(omega * t)}
        for i, pid in enumerate(marker_map.pursuer_ids):
            out[pid] = self._path(omega * t - 0.9 - 0.5 * i)
        return out
