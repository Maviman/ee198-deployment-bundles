"""Tests for the fast vision path: the shared Localizer, the camera config and
the pose-frame extras. Synthetic frames only (real tag images, real
perspective warp), so these run anywhere OpenCV does:

    cd portable_orin_perception && python -m pytest tests -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendored"))

import vision  # noqa: E402,F401
from hive_perception.core.camera_config import load_camera_config, write_local_override  # noqa: E402
from hive_perception.core.frame_builder import FrameBuilder, MarkerMap  # noqa: E402
from hive_perception.core.localizer import Localizer, corner_drift_px, expected_tag_px  # noqa: E402
from pc_controller.pose_stream import parse_pose_frame, parse_pose_frame_meta  # noqa: E402
from vision.synthetic import SceneRenderer, overhead_homography  # noqa: E402

SIZE = (1280, 720)
HALF = (0.915, 0.915)
CORNERS = {10: (-0.8, 0.8), 11: (0.8, 0.8), 12: (0.8, -0.8), 13: (-0.8, -0.8)}


def make(dictionary="DICT_4X4_50", pursuers=(1,), evader=0, **loc_kw):
    mm = MarkerMap(dictionary, evader, list(pursuers), [10, 11, 12, 13])
    H = overhead_homography(SIZE, HALF, fill=0.9)
    renderer = SceneRenderer(mm, H, SIZE, HALF, seed=1)
    loc = Localizer(mm, H, image_size=SIZE, arena_half_extent=HALF, **loc_kw)
    return mm, H, renderer, loc


def angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


@pytest.mark.parametrize("dictionary", ["DICT_4X4_50", "DICT_APRILTAG_36h11"])
def test_localizes_every_vehicle_with_identity_preserved(dictionary):
    # Non-contiguous printed ids: a row-index/printed-id mix-up would swap roles.
    mm, _H, r, loc = make(dictionary, pursuers=(3, 1), evader=2)
    truth = {3: (-0.4, 0.3, 0.5), 1: (0.5, -0.2, -2.0), 2: (0.1, 0.6, 3.0)}
    res = loc.process(r.render(truth), 0.0)
    assert set(res.found) == set(truth)
    for vid, (x, y, h) in truth.items():
        fx, fy, fh = res.found[vid]
        assert math.hypot(fx - x, fy - y) < 0.01, vid
        assert angdiff(fh, h) < math.radians(4), vid


def test_tracking_windows_take_over_and_rescue_on_the_same_frame():
    mm, _H, r, loc = make()
    pose = {1: (-0.3, 0.0, 0.0), 0: (0.4, 0.2, 1.0)}
    modes = [loc.process(r.render(pose), k / 30).mode for k in range(5)]
    assert modes[0] == "full" and all(m == "roi" for m in modes[1:])
    # A dropped frame later the pursuer is 0.4 m away: outside its tracking
    # window (~0.14 m), inside what a car can drive in 0.25 s. The SAME frame
    # must fall back to a full-frame search and still find it.
    moved = {1: (0.1, 0.0, 0.0), 0: (0.4, 0.2, 1.0)}
    res = loc.process(r.render(moved), 4 / 30 + 0.25)
    assert res.mode == "roi+full"
    assert 1 in res.found and abs(res.found[1][0] - 0.1) < 0.01


def test_jump_gate_turns_a_teleport_into_a_dropout_then_reacquires():
    mm, _H, r, loc = make()
    loc.process(r.render({1: (-0.5, 0.0, 0.0), 0: (0.5, 0.5, 0.0)}), 0.0)
    # 0.8 m in 1/30 s is ~24 m/s: not a car. Rejected, reported, not passed on.
    res = loc.process(r.render({1: (0.3, 0.0, 0.0), 0: (0.5, 0.5, 0.0)}), 1 / 30)
    assert 1 not in res.found and (1, "jump") in res.rejected
    # Past the jump window the new position is accepted (the car really moved).
    res = loc.process(r.render({1: (0.3, 0.0, 0.0), 0: (0.5, 0.5, 0.0)}), 1 / 30 + 0.6)
    assert 1 in res.found


def test_bounds_gate_rejects_a_tag_outside_the_arena():
    mm, _H, r, loc = make()
    res = loc.process(r.render({1: (1.3, 0.0, 0.0), 0: (0.0, 0.0, 0.0)}), 0.0)
    assert 1 not in res.found and (1, "bounds") in res.rejected
    assert 0 in res.found


def test_duplicate_id_without_a_track_is_ambiguous_and_rejected():
    from vision.synthetic import tag_corners_arena

    mm, H, r, loc = make()
    img = r.background.copy()
    # Two printed copies of tag 1 (a spare sheet on the floor) and no history
    # to say which is the car: identity is ambiguous, so neither is used.
    for x, y in ((-0.5, -0.5), (0.5, 0.5)):
        r._stamp(img, 1, tag_corners_arena(x, y, 0.0, mm.vehicle_tag_m))
    r._stamp(img, 0, tag_corners_arena(0.0, 0.0, 0.0, mm.vehicle_tag_m))
    res = loc.process(img, 0.0)
    assert sum(d.marker_id == 1 for d in res.detections) == 2
    assert 1 not in res.found and (1, "dup") in res.rejected
    assert 0 in res.found


def test_size_gate_rejects_a_tag_far_larger_than_a_vehicle_tag():
    mm, H, r, _loc = make()
    loc = Localizer(mm, H, image_size=SIZE, arena_half_extent=HALF)
    big = MarkerMap("DICT_4X4_50", 0, [1], [10, 11, 12, 13], vehicle_tag_m=0.20)
    r_big = SceneRenderer(big, H, SIZE, HALF, seed=2)
    res = loc.process(r_big.render({1: (0.0, 0.0, 0.0), 0: (0.5, 0.5, 0.0)}), 0.0)
    assert (1, "size") in res.rejected


def test_expected_tag_px_matches_what_is_detected():
    mm, H, r, loc = make()
    lo, hi = expected_tag_px(H, mm.vehicle_tag_m, HALF)
    res = loc.process(r.render({1: (0.0, 0.0, 0.0), 0: (0.5, 0.5, 0.0)}), 0.0)
    sides = [d.side_px for d in res.detections if d.marker_id in (0, 1)]
    assert all(0.9 * lo <= s <= 1.1 * hi for s in sides), (lo, hi, sides)


def test_corner_drift_measures_a_bumped_camera():
    mm, H, r, loc = make()
    res = loc.process(r.render({}, corners=CORNERS), 0.0)
    ref = {d.marker_id: d.corners.mean(axis=0) for d in res.detections}
    assert corner_drift_px(res.detections, ref) < 0.5
    shifted = {k: v + np.array([12.0, 0.0]) for k, v in ref.items()}
    assert 11.0 < corner_drift_px(res.detections, shifted) < 13.0


def test_camera_config_local_override_merges(tmp_path):
    cfg = tmp_path / "camera.yaml"
    cfg.write_text("device: /dev/video2\nwidth: 640\nheight: 480\nfps: 30\n", encoding="utf-8")
    assert load_camera_config(cfg).size == (640, 480)
    write_local_override(cfg, exposure=80)
    cam = load_camera_config(cfg)
    assert cam.exposure == 80 and cam.device == "/dev/video2"
    assert load_camera_config(cfg, local_overrides=False).exposure == "auto"
    cfg.write_text("widht: 640\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_config(cfg)


def test_shipped_camera_config_is_valid():
    cam = load_camera_config(ROOT / "config" / "camera.yaml", local_overrides=False)
    assert cam.fps % 10 == 0, "camera fps should be a multiple of the 10 Hz control tick"


def test_frame_extras_are_optional_and_cannot_override_contract_fields():
    fb = FrameBuilder(pursuer_ids=[1], evader_id=0)
    fb.update({1: (0.1, 0.2, 0.3), 0: (0.4, 0.5, 0.6)}, t=10.0)
    raw = fb.build_frame(10.01, extra={"seq": 7, "lat": 0.004, "t": 999.0})
    frame = json.loads(raw)
    assert frame["t"] == 10.0 and frame["seq"] == 7
    pursuers, evader, meta = parse_pose_frame_meta(raw, expected_pursuers=1)
    assert meta == {"seq": 7, "lat": 0.004} and pursuers[0].x == 0.1
    # the original parser (and the ROS bridge's extra-less frames) still work
    parse_pose_frame(raw, expected_pursuers=1)
    _p, _e, meta = parse_pose_frame_meta(fb.build_frame(10.01), expected_pursuers=1)
    assert meta == {"seq": None, "lat": None}


def test_vendored_parser_is_byte_identical_to_the_owner():
    owner = ROOT.parent / "portable_n1_controller" / "pc_controller" / "pose_stream.py"
    if not owner.exists():
        pytest.skip("controller bundle not alongside")
    assert owner.read_bytes() == (ROOT / "vendored" / "pc_controller" / "pose_stream.py").read_bytes()
