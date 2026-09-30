"""The ONE place the capture mode lives: config/camera.yaml.

HANDOFF.md found the capture resolution duplicated across five files that had
to agree by hand (launch file, calibrator, two debug tools, camera_info.yaml),
one of which already disagreed. Everything that opens the camera or interprets
its pixels now reads this file instead of carrying its own default:

    run_vision.py            the low-latency pipeline (the deployment path)
    scan_arena.py            calibration + arena check (`arena scan`)
    perception.launch.py     the ROS 2 pipeline's launch-argument defaults
    calibrate_arena.py       --width/--height defaults
    live/snapshot debug      --width/--height defaults

A homography is pixel geometry, so a calibration made at one resolution is
wrong at another. Consumers compare this file against the ``image_size``
recorded in config/arena_homography.yaml and refuse to run on a mismatch
rather than scaling every pose silently.

``config/camera.local.yaml`` (gitignored) is merged over this file when
present. `arena scan --tune-exposure` writes the exposure it measured for THIS
room there, so a site-specific lighting fix never becomes a commit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    # "auto", or a manual exposure time in the V4L2 unit (100 us): 80 = 8 ms.
    # Manual + short is what freezes a moving car; auto in a dim room drifts
    # toward 33 ms of blur per frame.
    exposure: str | int = "auto"
    # UVC "exposure priority": when true the camera may silently DROP its frame
    # rate (30 -> 15 -> 7.5 fps) to expose longer in dim light. That is hidden
    # latency, so it stays off unless a camera has no other way to see.
    dynamic_framerate: bool = False
    # Focus hunting blurs frames AND shifts the intrinsics under a calibration
    # made at a different focus. Lock it.
    autofocus: bool = False
    focus: int | None = None
    # Mains frequency of the room lights, for anti-flicker (50 or 60). None =
    # leave the camera's setting alone.
    power_line_hz: int | None = 60
    gain: int | None = None

    @property
    def size(self) -> tuple[int, int]:
        return int(self.width), int(self.height)

    @property
    def mode(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"

    def with_overrides(self, **kw) -> "CameraConfig":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


_FIELDS = {f.name for f in fields(CameraConfig)}


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(data) - _FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown camera setting(s) {unknown}; known: {sorted(_FIELDS)}")
    return data


def load_camera_config(path, *, local_overrides: bool = True) -> CameraConfig:
    """config/camera.yaml, with config/camera.local.yaml merged on top when present."""
    path = Path(path)
    data = _read(path) if path.exists() else {}
    local = path.with_name(path.stem + ".local" + path.suffix)
    if local_overrides and local.exists():
        data.update(_read(local))
    cfg = CameraConfig(**data)
    if cfg.exposure != "auto":
        cfg = replace(cfg, exposure=int(cfg.exposure))
    return replace(cfg, width=int(cfg.width), height=int(cfg.height), fps=int(cfg.fps))


def write_local_override(path, **values) -> Path:
    """Merge ``values`` into camera.local.yaml next to ``path``; returns its path."""
    path = Path(path)
    local = path.with_name(path.stem + ".local" + path.suffix)
    data = _read(local) if local.exists() else {}
    data.update(values)
    local.write_text(
        "# Site-specific camera overrides (gitignored). Written by `arena scan`;\n"
        "# merged over camera.yaml. Delete this file to return to the defaults.\n"
        + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return local
