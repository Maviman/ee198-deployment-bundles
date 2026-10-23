"""Unit tests of pc_controller.realtime.HoldZone: the stand-still zone around the evader.

    cd portable_n1_controller && python -m pytest tests/test_hold_zone.py -q

The end-to-end behaviour (a held car keeps receiving ZERO commands, not E-stops, and
drives again when the evader leaves) is in tests/test_realtime.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.realtime import HoldZone  # noqa: E402


def pose(x, y=0.0):
    return types.SimpleNamespace(x=x, y=y, heading=0.0)


EVADER = pose(0.0)


def test_outside_the_radius_the_policy_output_passes_through_untouched():
    z = HoldZone(0.6)
    act = np.array([0.8, -0.4])
    out = z.apply(act, [pose(1.0)], EVADER)
    assert out is act, "nothing held: the very same array, no copy"
    assert z.held == [False]


def test_inside_the_radius_the_pair_is_zeroed_and_the_policy_array_is_not_modified():
    z = HoldZone(0.6)
    act = np.array([0.8, -0.4])
    out = z.apply(act, [pose(0.5)], EVADER)
    assert list(out) == [0.0, 0.0]
    assert list(act) == [0.8, -0.4], "the policy's own output must stay as computed"
    assert z.held == [True]


def test_the_boundary_itself_is_inside():
    z = HoldZone(0.6)
    assert list(z.apply(np.array([1.0, 1.0]), [pose(0.6)], EVADER)) == [0.0, 0.0]


def test_hysteresis_a_held_car_resumes_only_beyond_radius_plus_margin():
    z = HoldZone(0.6, resume_margin=0.10)
    act = np.array([1.0, 0.5])
    assert z.apply(act, [pose(0.55)], EVADER).tolist() == [0.0, 0.0]          # enters
    assert z.apply(act, [pose(0.65)], EVADER).tolist() == [0.0, 0.0]          # just outside 0.6, still held
    assert z.apply(act, [pose(0.70)], EVADER).tolist() == [0.0, 0.0]          # exactly the resume edge
    assert z.apply(act, [pose(0.71)], EVADER).tolist() == [1.0, 0.5]          # beyond it: drives again
    assert z.apply(act, [pose(0.65)], EVADER).tolist() == [1.0, 0.5]          # and does not re-enter at 0.65
    assert z.apply(act, [pose(0.59)], EVADER).tolist() == [0.0, 0.0]          # re-enters inside 0.6


def test_each_car_is_judged_on_its_own_distance():
    z = HoldZone(0.6)
    act = np.array([0.9, 0.1, -0.7, 0.2])
    out = z.apply(act, [pose(2.0), pose(0.3)], EVADER)
    assert out.tolist() == [0.9, 0.1, 0.0, 0.0]
    assert z.held == [False, True]


def test_works_on_a_plain_list_too():
    z = HoldZone(0.6)
    act = [0.9, 0.1]
    assert z.apply(act, [pose(0.2)], EVADER) == [0.0, 0.0]
    assert act == [0.9, 0.1]


def test_reset_forgets_who_was_held():
    z = HoldZone(0.6)
    z.apply(np.array([1.0, 1.0]), [pose(0.3)], EVADER)
    assert z.held == [True]
    z.reset()
    assert z.held == []
    # 0.65 m is in the hysteresis band: after a reset nobody is held, so the car is NOT held there
    assert z.apply(np.array([1.0, 1.0]), [pose(0.65)], EVADER).tolist() == [1.0, 1.0]


def test_transitions_are_logged_once_each_not_every_tick():
    lines = []
    z = HoldZone(0.6)
    for x in (1.0, 0.5, 0.4, 0.3, 0.9):
        z.apply(np.array([1.0, 1.0]), [pose(x)], EVADER, lines.append)
    assert [("HOLD" in s, "RESUME" in s) for s in lines] == [(True, False), (False, True)]


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan")])
def test_a_non_positive_radius_is_refused(bad):
    with pytest.raises(ValueError):
        HoldZone(bad)


# --- where run_controller.py takes its default hold radius from -------------------------------------------

def _rc():
    import importlib
    return importlib.import_module("run_controller")


def test_default_radius_env_beats_file_and_file_is_the_fallback(tmp_path):
    rc = _rc()
    f = tmp_path / "hold_radius"
    f.write_text("0.5\n")
    assert rc.default_hold_radius({}, f) == (0.5, str(f))
    assert rc.default_hold_radius({"ARENA_HOLD_RADIUS": "0.7"}, f) == (0.7, "ARENA_HOLD_RADIUS")
    assert rc.default_hold_radius({}, tmp_path / "missing") == (None, "")


def test_default_radius_empty_file_means_off_and_garbage_names_its_source(tmp_path):
    rc = _rc()
    f = tmp_path / "hold_radius"
    f.write_text("\n")
    assert rc.default_hold_radius({}, f) == (None, "")
    f.write_text("about half a metre\n")
    with pytest.raises(ValueError, match="hold_radius must hold a number"):
        rc.default_hold_radius({}, f)
    with pytest.raises(ValueError, match="ARENA_HOLD_RADIUS"):
        rc.default_hold_radius({"ARENA_HOLD_RADIUS": "x"}, f)
