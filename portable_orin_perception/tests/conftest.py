"""Shared test setup for the perception bundle."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _scan_renders_the_documented_corner_tags(monkeypatch):
    """The synthetic `arena scan` tests render the arena from config/marker_map.yaml. Its
    corner_tag_m is a SITE fact (16 cm prints on the real arena since 2026-11-18) and must
    not change what these logic tests render: they are written for 10 cm corner tags
    (bigger ones cover the boundary tape the lens fit measures). Pin it here."""
    try:
        import scan_arena
    except Exception:          # a test module that does not use the scan
        return
    real = scan_arena.load_marker_map

    def documented(path):
        return dataclasses.replace(real(path), corner_tag_m=0.10)

    monkeypatch.setattr(scan_arena, "load_marker_map", documented)
