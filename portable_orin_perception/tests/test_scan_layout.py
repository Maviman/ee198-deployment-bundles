"""Tests for `arena scan`'s corner-layout matching and scale check.

A 4-point homography fits any id -> position assignment with a 0.00 cm residual,
so a corner tag taped in the wrong corner (it has happened three times) used to
give a twisted or mirrored arena frame without a word. These pin the fix: the
configured positions follow the corner each tag is SEEN in, and the printed tag
size exposes a config that describes a different-sized arena.

    cd portable_orin_perception && python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scan_arena  # noqa: E402
import vision.capture as capture  # noqa: E402
from hive_perception.core import arena_frame  # noqa: E402
from scan_arena import match_layout  # noqa: E402

H8 = 1.104
DOC = {10: (-H8, H8), 11: (H8, H8), 12: (H8, -H8), 13: (-H8, -H8)}     # the documented layout
# pixel centres as seen on 2026-10-23: 10 top-left, 11 top-right, 12 bottom-LEFT, 13 bottom-RIGHT
SEEN_OCT23 = {10: (209.4, 20.4), 11: (843.6, 31.7), 12: (97.3, 682.7), 13: (922.8, 687.2)}


def test_a_layout_that_matches_the_config_is_left_alone():
    px = {10: (200, 50), 11: (900, 60), 12: (910, 680), 13: (190, 670)}
    layout, why = match_layout(px, DOC)
    assert layout == DOC and why == ""


def test_swapped_bottom_tags_are_reassigned_by_where_the_camera_sees_them():
    layout, why = match_layout(SEEN_OCT23, DOC)
    assert layout[12] == (-H8, -H8) and layout[13] == (H8, -H8)
    assert layout[10] == DOC[10] and layout[11] == DOC[11]
    assert "tag 12 is bottom-left (config: bottom-right)" in why
    assert "tag 13 is bottom-right (config: bottom-left)" in why


def test_a_camera_turned_90_degrees_still_gives_image_up_as_plus_y():
    # documented layout, camera rotated a quarter turn: 13 now shows top-left, 10 top-right...
    px = {13: (200, 50), 10: (900, 60), 11: (910, 680), 12: (190, 670)}
    layout, why = match_layout(px, DOC)
    assert layout == {13: (-H8, H8), 10: (H8, H8), 11: (H8, -H8), 12: (-H8, -H8)}
    assert why


@pytest.mark.parametrize("px", [
    {10: (640, 40), 11: (1000, 360), 12: (640, 680), 13: (280, 360)},   # a diamond: camera at 45 deg
    {10: (200, 50), 11: (900, 60), 12: (910, 680), 13: (880, 650)},    # two tags in one corner
])
def test_a_layout_it_cannot_call_fails_instead_of_guessing(px):
    layout, why = match_layout(px, DOC)
    assert layout is None and why


def _scan(monkeypatch, tmp_path, capsys, place):
    """Run the real `arena scan --synthetic`, rendering the corner tags where ``place``
    (configured positions -> where they physically are) puts them."""
    real = capture.SyntheticSource

    def source(renderer, mm, *, corners=None, **kw):
        return real(renderer, mm, corners=place(dict(corners)), **kw)
    monkeypatch.setattr(capture, "SyntheticSource", source)
    rc = scan_arena.main(["--synthetic", "--out-dir", str(tmp_path), "--frames", "20"])
    out = capsys.readouterr().out
    cal = yaml.safe_load((tmp_path / "synthetic_homography.yaml").read_text())
    return rc, out, cal


def test_scan_calibrates_correctly_with_two_corner_tags_swapped(monkeypatch, tmp_path, capsys):
    def swap(c):
        c[12], c[13] = c[13], c[12]
        return c
    rc, out, cal = _scan(monkeypatch, tmp_path, capsys, swap)
    assert rc == 0, out
    assert "tag 12 is bottom-left (config: bottom-right)" in out
    truth = swap(dict(yaml.safe_load((ROOT / "config" / "arena_test_6ft.yaml").read_text())["calibration_points"]))
    H = np.asarray(cal["homography"])
    for cid, px in cal["corner_px"].items():
        got = arena_frame.project_px_to_arena(H, [px])[0]
        assert np.allclose(got, truth[int(cid)], atol=0.01), f"tag {cid} at {got}, really at {truth[int(cid)]}"


def test_scan_warns_when_the_arena_on_the_floor_is_not_the_configured_size(monkeypatch, tmp_path, capsys):
    # the config's positions, but the tags really sit 25% closer in (a smaller arena)
    rc, out, cal = _scan(monkeypatch, tmp_path, capsys,
                         lambda c: {k: (0.75 * x, 0.75 * y) for k, (x, y) in c.items()})
    line = next(l for l in out.splitlines() if "scale" in l)
    assert "[WARN]" in line and "off by x" in line, line
    assert float(line.rsplit("off by x", 1)[1]) == pytest.approx(0.75, abs=0.04), line
    assert cal["corner_tag_m_measured"] == pytest.approx(0.10 / 0.75, rel=0.05)


def test_scan_reports_a_consistent_scale_when_the_config_is_right(monkeypatch, tmp_path, capsys):
    rc, out, cal = _scan(monkeypatch, tmp_path, capsys, lambda c: c)
    assert rc == 0, out
    line = next(l for l in out.splitlines() if "scale" in l)
    assert "[ ok ]" in line, line
    assert "layout" not in out.split("[4/5]")[1].split("[5/5]")[0], "no layout warning when it matches"
