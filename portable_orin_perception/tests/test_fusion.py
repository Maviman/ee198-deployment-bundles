"""vision/fusion.py: one pose per vehicle from two corner cameras."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.fusion import PoseFuser  # noqa: E402


def test_one_camera_passes_through():
    f = PoseFuser([1, 0])
    out = f.update(0, {1: (0.5, -0.2, 1.0)}, {1: 30.0}, 10.0)
    assert set(out) == {1}
    x, y, h, t, share = out[1]
    assert (x, y, h, t) == (0.5, -0.2, 1.0, 10.0) and share == {0: 1.0}


def test_the_near_camera_dominates_by_tag_size_squared():
    f = PoseFuser([1, 0])
    f.update(0, {1: (0.0, 0.0, 0.0)}, {1: 50.0}, 10.0)       # 50 px: near
    out = f.update(1, {1: (0.10, 0.0, 0.0)}, {1: 15.0}, 10.02)  # 15 px: far, 10 cm off
    x, _y, _h, t, share = out[1]
    assert t == 10.02
    w_far = 15.0 ** 2 / (50.0 ** 2 + 15.0 ** 2)
    assert abs(x - 0.10 * w_far) < 1e-9
    assert abs(share[0] - (1 - w_far)) < 1e-9 and abs(share[1] - w_far) < 1e-9


def test_headings_are_averaged_on_the_circle():
    f = PoseFuser([1, 0])
    f.update(0, {1: (0.0, 0.0, math.pi - 0.1)}, {1: 20.0}, 1.0)
    out = f.update(1, {1: (0.0, 0.0, -math.pi + 0.1)}, {1: 20.0}, 1.01)
    h = out[1][2]
    assert abs(abs(h) - math.pi) < 1e-9, "the mean of +179 and -179 degrees is 180, not 0"


def test_stale_estimates_are_not_mixed_in():
    f = PoseFuser([1, 0], window_s=0.05)
    f.update(0, {1: (0.0, 0.0, 0.0)}, {1: 50.0}, 1.0)
    out = f.update(1, {1: (1.0, 0.0, 0.0)}, {1: 10.0}, 1.2)    # camera 0's view is 200 ms old
    assert out[1][0] == 1.0 and out[1][4] == {1: 1.0}
    out = f.update(0, {}, {}, 1.21)                             # camera 0 lost it: camera 1 still fresh
    assert out[1][0] == 1.0


def test_vehicles_unseen_by_every_camera_are_absent():
    f = PoseFuser([1, 0])
    assert f.update(0, {}, {}, 1.0) == {}
    f.update(0, {0: (0.1, 0.2, 0.3)}, {0: 20.0}, 1.0)
    f.forget(0)
    assert f.update(1, {}, {}, 1.0) == {}


def test_spread_reports_how_far_apart_the_cameras_put_one_car():
    f = PoseFuser([1], window_s=0.05)
    f.update(0, {1: (0.50, -0.40, 0.0)}, {1: 30.0}, t=1.000)
    assert 1 not in f.spread                      # one camera: nothing to compare
    f.update(1, {1: (0.53, -0.36, 0.0)}, {1: 15.0}, t=1.020)
    assert math.isclose(f.spread[1], 0.05, abs_tol=1e-6)
