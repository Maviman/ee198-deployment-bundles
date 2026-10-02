"""The deployed model is locked in: what ships in deploy/arena.conf is n1_catch
driving one pursuer, and that choice is consistent end to end. Only the shipped
arena.conf is read, never a site's arena.local.conf (a deliberate switch made
with `arena fleet`).

Changing the deployed model on purpose means editing deploy/arena.conf AND
this test, in the same commit.
"""

from __future__ import annotations

import configparser
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # portable_n1_controller
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from pc_controller.loops import make_loop  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402

DEPLOYED_MODEL = "models/n1_catch"
DEPLOYED_PURSUERS = 1


def shipped_fleet() -> configparser.SectionProxy:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cp.read(REPO / "deploy" / "arena.conf", encoding="utf-8")
    return cp["fleet"]


def test_arena_conf_ships_n1_catch_for_one_pursuer():
    fleet = shipped_fleet()
    assert fleet["model"] == DEPLOYED_MODEL
    assert fleet["pursuers"].strip() == str(DEPLOYED_PURSUERS)


def test_the_deployed_model_is_what_the_config_says():
    manifest = json.loads((ROOT / DEPLOYED_MODEL / "policy.onnx.manifest.json").read_text(encoding="utf-8"))
    assert int(manifest["num_pursuers"]) == DEPLOYED_PURSUERS
    assert not manifest.get("use_car_cameras")
    assert isinstance(make_loop(ROOT / DEPLOYED_MODEL), PortableLoop)


def test_the_deployed_model_is_pinned_by_golden_vectors():
    reference = json.loads((ROOT / "reference_vectors.json").read_text(encoding="utf-8"))
    assert Path(DEPLOYED_MODEL).name in reference["models"], "selftest.py would not check the deployed model"


def test_the_vision_side_has_a_tag_for_each_deployed_car():
    text = (REPO / "portable_orin_perception" / "config" / "marker_map.yaml").read_text(encoding="utf-8")
    ids = re.search(r"^pursuer_ids:\s*\[([^\]]*)\]", text, re.M).group(1)
    assert len([i for i in ids.split(",") if i.strip()]) >= DEPLOYED_PURSUERS


def test_the_controller_and_sim_defaults_agree():
    assert f'default="{DEPLOYED_MODEL}"' in (ROOT / "run_controller.py").read_text(encoding="utf-8")
    assert f'default="{DEPLOYED_MODEL}"' in (REPO / "deploy" / "arena.py").read_text(encoding="utf-8")
