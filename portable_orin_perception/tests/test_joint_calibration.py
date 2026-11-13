"""Two corner cameras calibrated together (vision/joint_calibration.py, vision/lens.py,
`arena scan --camera A --camera B`).

The rig since 2026-11-11: a camera outside each of two opposite corners, each seeing
its own corner tag and the two side ones but never the far one, so neither has the
four points a homography needs. These pin that the tags both cameras see supply the
fourth, that the result is right across the WHOLE arena (not just at the tags), that
a wide lens's barrel distortion is measured from the boundary tape, and that the
cases it cannot solve are refused instead of guessed.

    cd portable_orin_perception && python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scan_arena  # noqa: E402
from hive_perception.core import arena_frame, marker_math  # noqa: E402
from hive_perception.core.camera_config import load_camera_config, write_local_override  # noqa: E402
from vision.joint_calibration import CameraView, TagView, fit_joint, mirrored_triples, project  # noqa: E402
from vision.lens import nominal_K, undistort  # noqa: E402
from vision.synthetic import corner_camera_homography, tag_corners_arena  # noqa: E402

HALF = (1.2192, 1.2192)
P = {10: (-1.104, 1.104), 11: (1.104, 1.104), 12: (1.104, -1.104), 13: (-1.104, -1.104)}
CAMS = {s: {k: v for k, v in kw.items() if k != "k1"} for s, kw in scan_arena.SYNTHETIC_CORNER_CAMS.items()}
TRUTH = {s: corner_camera_homography((1280, 720), HALF, kw["corner"],
                                     **{k: v for k, v in kw.items() if k != "corner"})
         for s, kw in CAMS.items()}
PARKED = {0: (-0.6, 0.3, 0.4, 0.07), 1: (0.5, -0.4, 2.0, 0.07), 2: (0.2, 0.75, 1.0, 0.07), 3: (-0.3, -0.8, -1.0, 0.07)}
GRID = [(x, y) for x in np.linspace(-1.1, 1.1, 7) for y in np.linspace(-1.1, 1.1, 7)]


def views(tags: dict, noise: float = 0.3, seed: int = 0, truth=TRUTH) -> list:
    """What each camera sees: every tag inside its frame and >= 12 px, with pixel noise."""
    rng = np.random.default_rng(seed)
    out = []
    for slot, H in truth.items():
        Hi = np.linalg.inv(H)
        corners, tv = {}, {}
        for tid, (x, y, hd, m) in tags.items():
            px, _ = project(Hi, tag_corners_arena(x, y, hd, m))
            px = px + rng.normal(0, noise, px.shape)
            side = float(np.mean(np.linalg.norm(px - np.roll(px, -1, 0), axis=1)))
            if px.min() < 0 or px[:, 0].max() > 1279 or px[:, 1].max() > 719 or side < 12:
                continue
            tv[tid] = TagView(px, side, m)
            if tid in P:
                corners[tid] = px.mean(axis=0)
        out.append(CameraView(slot, corners, tv))
    return out


def worst_error_m(H_fit, H_true) -> float:
    px, _ = project(np.linalg.inv(H_true), GRID)
    got, _ = project(H_fit, px)
    return float(np.max(np.linalg.norm(got - np.array(GRID), axis=1)))


def scene(extra=PARKED):
    tags = {k: (p[0], p[1], math.pi / 2, 0.084) for k, p in P.items()}
    tags.update(extra)
    return tags


def test_each_corner_camera_misses_its_far_corner():
    v = views(scene())
    assert sorted(v[0].corners_px) == [11, 12, 13] and sorted(v[1].corners_px) == [10, 11, 13]


def test_two_three_corner_cameras_are_right_across_the_whole_arena():
    fit = fit_joint(views(scene()), P, half=HALF, corner_ids=set(P))
    assert fit.ok, fit.error
    for slot, cf in fit.cameras.items():
        assert worst_error_m(cf.H, TRUTH[slot]) < 0.025, slot
        assert cf.uncertainty_m < 0.02
    # four shared tags: each one left out is still placed where both cameras agree
    assert all(s.loo_m is not None and s.loo_m < 0.03 for s in fit.shared)


def test_the_evader_and_one_car_are_enough():
    fit = fit_joint(views(scene({k: PARKED[k] for k in (0, 1)})), P, half=HALF, corner_ids=set(P))
    assert fit.ok, fit.error
    for slot, cf in fit.cameras.items():
        assert worst_error_m(cf.H, TRUTH[slot]) < 0.03, slot


def test_no_tag_in_view_of_both_cameras_is_refused():
    fit = fit_joint(views(scene({})), P, half=HALF, corner_ids=set(P))
    assert not fit.ok and "both cameras" in fit.error


def test_a_corner_no_camera_sees_is_refused():
    # both cameras miss tag 10: a projective freedom fixing 11, 12, 13 fits every shared tag
    v = views(scene())
    v[1].corners_px.pop(10)
    v[1].corners_px[12] = project(np.linalg.inv(TRUTH[1]), [P[12]])[0][0]
    fit = fit_joint(v, P, half=HALF, corner_ids=set(P))
    assert not fit.ok and "no camera sees corner tag 10" in fit.error


def test_a_camera_with_two_corners_is_refused():
    v = views(scene())
    v[1].corners_px.pop(10)
    fit = fit_joint(v, P, half=HALF, corner_ids=set(P))
    assert not fit.ok and "sees 2 corner tag" in fit.error


def test_two_swapped_corner_tags_are_a_mirror():
    v = views(scene())
    c = v[0].corners_px
    c[11], c[13] = c[13], c[11]
    assert mirrored_triples(c, P)
    fit = fit_joint(v, P, half=HALF, corner_ids=set(P))
    assert not fit.ok and "MIRRORED" in fit.error


def test_corner_camera_view_matches_the_real_x1():
    # tag 12 near (bottom centre), 13 / 11 near the top corners, 10 just past the top:
    # the X1 at 2026-11-12 18:19 (12 at 651,574 / 13 at 190,108 / 11 at 1108,124)
    px, _ = project(np.linalg.inv(TRUTH[0]), [P[k] for k in (10, 11, 12, 13)])
    assert px[0][1] < 0 and 500 < px[2][1] < 700 and px[1][1] < 250 and px[3][1] < 250


def test_lens_k1_is_recovered_from_straight_tape():
    import cv2

    from vision import lens

    K = nominal_K((1280, 720))
    # a pinhole render of the arena's tape, then barrel distortion k1 = -0.3
    img = np.full((720, 1280), 180, np.uint8)
    Hi = np.linalg.inv(TRUTH[0])
    edge, _ = project(Hi, [(-1.2, 1.2), (1.2, 1.2), (1.2, -1.2), (-1.2, -1.2)])
    cv2.polylines(img, [edge.astype(np.int32)], True, 60, 8)
    grid = np.stack(np.meshgrid(np.arange(1280.0), np.arange(720.0)), -1).reshape(-1, 2)
    src = undistort(grid, K, -0.3).reshape(720, 1280, 2).astype(np.float32)
    img = cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_LINEAR)
    cen, _ = project(Hi, [P[11], P[12], P[13]])
    # tags seen where the lens puts them
    norm = np.hstack([cen, np.ones((3, 1))]) @ np.linalg.inv(K).T
    dist, _ = cv2.projectPoints(norm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.array([-0.3, 0, 0, 0, 0.0]))
    c11, c12, c13 = dist.reshape(-1, 2)
    edges = [lens.tape_points(img.astype(np.float32), c12, c13, 40, 18, c11),
             lens.tape_points(img.astype(np.float32), c12, c11, 40, 18, c13)]
    assert all(e is not None for e in edges)
    k1, before, after, _ = lens.fit_k1(edges, (1280, 720))
    assert k1 == pytest.approx(-0.3, abs=0.03) and after < 0.5 * before


def test_camera_settings_follow_the_camera_not_the_slot(tmp_path):
    cfg = tmp_path / "camera.yaml"
    cfg.write_text("device: /dev/video0\nwidth: 1280\nheight: 720\nfps: 30\nexposure: auto\n")
    # 2026-11-12: camera.local.yaml still named the C270 (the old overhead camera) when the
    # X1 became camera 0; its exposure must stay with the C270, wherever that is listed
    (tmp_path / "camera.local.yaml").write_text("device: /dev/C270\nexposure: 120\ngain: 20\nfps: 30\n")
    x1 = load_camera_config(cfg, slot=0, device="/dev/X1")
    c270 = load_camera_config(cfg, slot=1, device="/dev/C270")
    assert (x1.device, x1.exposure, x1.gain) == ("/dev/X1", "auto", None)
    assert (c270.device, c270.exposure, c270.gain) == ("/dev/C270", 120, 20)
    # tuning the X1 in slot 0's file drops the C270's values from it
    write_local_override(cfg, slot=0, device="/dev/X1", exposure=30)
    assert load_camera_config(cfg, slot=0, device="/dev/X1").exposure == 30
    assert load_camera_config(cfg, slot=0, device="/dev/X1").gain is None
    # without a device (the one-camera arena) slot 0's file applies as before
    assert load_camera_config(cfg).exposure == 30


def test_two_corner_camera_scan_end_to_end(tmp_path, capsys):
    """The real `arena scan` on rendered frames: two corner cameras, the X1-like one with
    strong barrel distortion, parked tags. Both calibrations must be right everywhere."""
    rc = scan_arena.main(["--synthetic", "--camera", "a", "--camera", "b", "--out-dir", str(tmp_path),
                          "--frames", "20"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "cameras agree" in out and "cross-check" in out
    for slot, kw in scan_arena.SYNTHETIC_CORNER_CAMS.items():
        cal = yaml.safe_load((tmp_path / ("synthetic_homography.yaml" if slot == 0
                                          else f"synthetic_homography.{slot}.yaml")).read_text())
        info = yaml.safe_load((tmp_path / f"camera_info.{slot}.yaml").read_text())
        K = np.asarray(info["camera_matrix"]["data"]).reshape(3, 3)
        D = np.asarray(info["distortion_coefficients"]["data"])
        assert D[0] == pytest.approx(kw["k1"], abs=0.04)
        # ground truth: where the arena point really appears (pinhole + the true lens)
        Ktrue = nominal_K((1280, 720))
        px, _ = project(np.linalg.inv(TRUTH[slot]), GRID)
        norm = np.hstack([px, np.ones((len(px), 1))]) @ np.linalg.inv(Ktrue).T
        import cv2
        raw, _ = cv2.projectPoints(norm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Ktrue,
                                   np.array([kw["k1"], 0, 0, 0, 0.0]))
        raw = raw.reshape(-1, 2)
        inside = (raw[:, 0] >= 0) & (raw[:, 0] < 1280) & (raw[:, 1] >= 0) & (raw[:, 1] < 720)
        got = arena_frame.project_px_to_arena(np.asarray(cal["homography"]),
                                              marker_math.undistort_points(raw[inside], K, D))
        err = np.linalg.norm(got - np.array(GRID)[inside], axis=1)
        assert err.max() < 0.04, (slot, err.max())
