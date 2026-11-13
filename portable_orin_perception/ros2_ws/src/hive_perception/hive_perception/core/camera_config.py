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


def local_override_path(path, slot: int | None = None) -> Path:
    """camera.local.yaml next to ``path``; camera.<slot>.local.yaml for camera slot >= 1
    (each corner camera has its own device, exposure and gain)."""
    path = Path(path)
    tag = "" if not slot else f".{int(slot)}"
    return path.with_name(path.stem + tag + ".local" + path.suffix)


# Settings that belong to ONE physical camera rather than to the site. A C270 tuned to
# exposure 120 / gain 20 as the overhead camera kept those values when the X1 took its
# slot (2026-11-12): they are looked up by the camera's device, not by its slot.
PER_CAMERA = ("device", "exposure", "gain", "focus")


def same_device(a, b) -> bool:
    """True when two device paths name the same camera (a by-id link and the /dev/videoN it
    points at count as the same)."""
    if not a or not b:
        return False
    if str(a) == str(b):
        return True
    import os
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def _local_files(path: Path) -> list[Path]:
    """camera.local.yaml, then camera.<k>.local.yaml in slot order."""
    shared = local_override_path(path)
    slots = sorted(path.parent.glob(f"{path.stem}.*.local{path.suffix}"),
                   key=lambda p: int(p.name.split(".")[1]) if p.name.split(".")[1].isdigit() else 1 << 30)
    return ([shared] if shared.exists() else []) + [p for p in slots if p.name.split(".")[1].isdigit()]


def load_camera_config(path, *, local_overrides: bool = True, slot: int | None = None,
                       device: str | None = None) -> CameraConfig:
    """config/camera.yaml, with config/camera.local.yaml merged on top when present, then
    config/camera.<slot>.local.yaml for a camera slot >= 1.

    Site settings (mode, anti-flicker, ...) merge in that order. The per-camera ones
    (PER_CAMERA) come from ONE file: with ``device`` given, the local file whose ``device``
    names that camera (its own slot's file first), so its exposure follows it to whatever
    slot it is in now and never lands on another camera; without one, or when no file
    names it, the slot's own file when that names no device (camera.local.yaml is slot 0's)."""
    path = Path(path)
    data = _read(path) if path.exists() else {}
    if local_overrides:
        own = local_override_path(path, slot)
        shared = local_override_path(path)
        for local in [shared] + ([own] if slot else []):
            if local.exists():
                data.update({k: v for k, v in _read(local).items() if k not in PER_CAMERA})
        files = {f: _read(f) for f in _local_files(path)}
        order = [own] + [f for f in files if f != own]
        pick = None
        if device is not None:
            pick = next((f for f in order if f in files and same_device(files[f].get("device"), device)), None)
        if pick is None and own in files and (device is None or not files[own].get("device")):
            pick = own
        if pick is not None:
            data.update({k: v for k, v in files[pick].items() if k in PER_CAMERA})
    if device is not None:
        data["device"] = device
    cfg = CameraConfig(**data)
    if cfg.exposure != "auto":
        cfg = replace(cfg, exposure=int(cfg.exposure))
    return replace(cfg, width=int(cfg.width), height=int(cfg.height), fps=int(cfg.fps))


def write_local_override(path, slot: int | None = None, **values) -> Path:
    """Merge ``values`` into camera.local.yaml next to ``path`` (camera.<slot>.local.yaml
    for a camera slot >= 1); returns its path. Writing a ``device`` other than the one
    the file names drops that other camera's exposure, gain and focus."""
    path = Path(path)
    local = local_override_path(path, slot)
    data = _read(local) if local.exists() else {}
    if values.get("device") and data.get("device") and not same_device(values["device"], data["device"]):
        data = {k: v for k, v in data.items() if k not in PER_CAMERA}
    data.update(values)
    local.write_text(
        "# Site-specific camera overrides (gitignored). Written by `arena scan`;\n"
        "# merged over camera.yaml. Delete this file to return to the defaults.\n"
        + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return local
