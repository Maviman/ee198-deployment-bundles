"""Fiducial family metadata + the subset-dictionary trick that makes AprilTag
36h11 affordable on the CPU.

Why this module exists
----------------------
Moving from ArUco DICT_4X4_50 to AprilTag tag36h11 buys a real CUDA detector
(NVIDIA cuAprilTags) and per-tag 6-DOF pose, but naively it costs ~8x more CPU
time. Measured on an x86 dev box, 1280x720, identical frame:

    ArUco DICT_4X4_50            5.99 ms
    AprilTag 36h11 (587 codes)  37.34 ms      <-- unusable
    AprilTag 36h11 (8 codes)     4.65 ms      <-- faster than the ArUco baseline

The cost is NOT quad detection (an empty frame with no tags at all still costs
22.8 ms): it is the identification step matching every candidate quad against
all 587 codewords with up to 5-bit error correction. This project uses eight
tags, not 587.

So we build a cv2.aruco.Dictionary holding only the rows we actually print. The
rows are copied bit-for-bit out of the canonical 36h11 codebook, so the printed
tag is a genuine tag36h11 that NVIDIA cuAprilTags reads identically — the subset
is purely a CPU-side search-space reduction, never a different marker.

The trap this module exists to contain
--------------------------------------
A subset dictionary RENUMBERS ids: OpenCV reports the ROW INDEX, not the tag id.
Print real ids [0,1,2,3,10,11,12,13] and detectMarkers() returns [0..7]. Getting
this wrong silently swaps vehicle identities — the evader would be driven as a
pursuer. ``TagSet.to_real_id()`` is the only sanctioned way to translate, and
``test_core.py`` pins the round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FamilySpec:
    """Static facts about a fiducial family."""

    opencv_dict: str        # cv2.aruco predefined dictionary name
    cuapriltags_family: str | None   # NVIDIA cuAprilTags family, None if unsupported
    data_modules: int       # data bits per side
    border_modules: int     # black border rings (each side)

    @property
    def total_modules(self) -> int:
        """Modules across the printed black square — what sets the pixel floor."""
        return self.data_modules + 2 * self.border_modules

    def min_detect_px(self, px_per_module: float = 3.5) -> float:
        """Minimum on-screen tag size, in pixels per side, for reliable detection.

        The 3.5 px/module default is calibrated against a measured sweep
        (rotated synthetic tags, subset dictionaries, detection rate over 24
        trials per size):

            tag px |  ArUco 4x4  |  36h11 sharp  |  36h11 + motion blur
              14   |    100%     |      42%      |        0%
              18   |    100%     |      88%      |       29%
              20   |    100%     |     100%      |       79%
              24   |    100%     |     100%      |       96%
              28   |    100%     |     100%      |      100%

        36h11 needs 20 px sharp but 28 px once blurred — and blurred is the
        real case for a moving car, so 28 px (= 8 modules x 3.5) is the number
        to design against. It also reproduces HANDOFF.md's independent ArUco
        figure of ~20-25 px for a 6-module tag.

        Still synthetic: real carpet, cables and glare give the quad detector
        more candidates and less margin, so treat this as a floor, not a
        target, and confirm against real arena frames before spending money on
        the strength of it.
        """
        return self.total_modules * float(px_per_module)


FAMILIES: dict[str, FamilySpec] = {
    # The incumbent. Cheapest on CPU, no CUDA detector exists for it, 2D only.
    "DICT_4X4_50": FamilySpec("DICT_4X4_50", None, data_modules=4, border_modules=1),
    # The target. cuAprilTags accelerates it; per-tag 6-DOF pose enables 3D.
    "DICT_APRILTAG_36h11": FamilySpec("DICT_APRILTAG_36h11", "tag36h11",
                                      data_modules=6, border_modules=1),
    "DICT_APRILTAG_25h9": FamilySpec("DICT_APRILTAG_25h9", "tag25h9",
                                     data_modules=5, border_modules=1),
    "DICT_APRILTAG_16h5": FamilySpec("DICT_APRILTAG_16h5", "tag16h5",
                                     data_modules=4, border_modules=1),
}


def family_spec(dictionary: str) -> FamilySpec:
    try:
        return FAMILIES[dictionary]
    except KeyError:
        known = ", ".join(sorted(FAMILIES))
        raise ValueError(f"unknown fiducial family {dictionary!r}; known: {known}") from None


def max_ground_width_m(tag_size_m: float, image_width_px: int,
                       spec: FamilySpec, px_per_module: float = 3.5) -> float:
    """Widest arena this family/tag-size/resolution can still detect across.

    ground_width = tag_size * image_width / min_detect_px. This is the ceiling
    HANDOFF.md derives for ArUco, generalized: switching to 36h11 raises
    min_detect_px from 6 to 8 modules, shrinking coverage by 25% at equal
    resolution — which is exactly why the AprilTag move only makes sense
    together with the resolution increase that cuAprilTags pays for.
    """
    return float(tag_size_m) * float(image_width_px) / spec.min_detect_px(px_per_module)


def tag_px_at(tag_size_m: float, ground_width_m: float, image_width_px: int) -> float:
    """On-screen size (px/side) of a tag of ``tag_size_m`` in a scene spanning
    ``ground_width_m``. Compare against ``FamilySpec.min_detect_px()``."""
    return float(tag_size_m) / float(ground_width_m) * float(image_width_px)


@dataclass(frozen=True)
class TagSet:
    """A subset dictionary plus the row-index <-> real-id mapping.

    ``real_ids[i]`` is the true family id printed on the tag that OpenCV will
    report as index ``i``. Never index a detection into anything else.
    """

    dictionary: str
    real_ids: tuple[int, ...]
    max_correction_bits: int

    def to_real_id(self, row_index: int) -> int:
        """Detector row index -> printed tag id. Raises on an out-of-range index
        rather than returning something plausible-but-wrong."""
        if not 0 <= int(row_index) < len(self.real_ids):
            raise IndexError(
                f"detector returned row {row_index}, but the subset dictionary has "
                f"{len(self.real_ids)} rows {self.real_ids} — dictionary and marker "
                "map are out of sync")
        return self.real_ids[int(row_index)]

    def build_opencv_dictionary(self):
        """cv2.aruco.Dictionary containing only ``real_ids``, rows copied
        bit-identically from the canonical family codebook."""
        import cv2

        spec = family_spec(self.dictionary)
        full = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.opencv_dict))
        n_available = full.bytesList.shape[0]
        bad = [i for i in self.real_ids if not 0 <= i < n_available]
        if bad:
            raise ValueError(
                f"{self.dictionary} has ids 0..{n_available - 1}; marker map asks for {bad}")
        rows = np.ascontiguousarray(full.bytesList[list(self.real_ids), :, :])
        return cv2.aruco.Dictionary(rows, full.markerSize, int(self.max_correction_bits))

    def min_hamming_distance(self) -> int:
        """Smallest bit distance between any two chosen codewords. Bounds how
        much error correction is safe: correcting more than (d-1)//2 bits can
        turn one vehicle's tag into another's."""
        import cv2
        import itertools

        spec = family_spec(self.dictionary)
        full = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.opencv_dict))
        n_bits = spec.data_modules ** 2

        def bits(row) -> np.ndarray:
            return np.unpackbits(np.asarray(row[:, 0], dtype=np.uint8))[:n_bits]

        if len(self.real_ids) < 2:
            return n_bits
        return min(
            int(np.count_nonzero(bits(full.bytesList[a]) != bits(full.bytesList[b])))
            for a, b in itertools.combinations(self.real_ids, 2)
        )

    def safe_correction_bits(self) -> int:
        """Largest error correction that cannot alias one chosen tag onto another."""
        return max(0, (self.min_hamming_distance() - 1) // 2)
