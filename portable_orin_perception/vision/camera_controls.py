"""UVC camera controls via v4l2-ctl: lock the settings that cost latency or
detections, and list the modes a camera actually offers.

Control names changed across kernels (exposure_auto -> auto_exposure,
exposure_absolute -> exposure_time_absolute, focus_auto ->
focus_automatic_continuous, exposure_auto_priority ->
exposure_dynamic_framerate), so every setting tries both names and reports
what it could not set instead of failing. Cameras differ; a missing control is
information, not an error.
"""

from __future__ import annotations

import re
import shutil
import subprocess

ALIASES = {
    "exposure_mode": ("auto_exposure", "exposure_auto"),
    "exposure_time": ("exposure_time_absolute", "exposure_absolute"),
    "dynamic_framerate": ("exposure_dynamic_framerate", "exposure_auto_priority"),
    "autofocus": ("focus_automatic_continuous", "focus_auto"),
    "focus": ("focus_absolute",),
    "power_line": ("power_line_frequency",),
    "gain": ("gain",),
}
# V4L2 exposure menu: 1 = manual, 3 = aperture priority (auto).
EXPOSURE_MANUAL, EXPOSURE_AUTO = 1, 3
POWER_LINE = {None: None, 0: 0, 50: 1, 60: 2}


def available() -> bool:
    return shutil.which("v4l2-ctl") is not None


def _run(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        p = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def list_controls(device: str) -> dict[str, dict]:
    """{name: {"value": int, "min": int, "max": int}} for the device's controls."""
    rc, out = _run(["-d", device, "--list-ctrls"])
    ctrls: dict[str, dict] = {}
    if rc != 0:
        return ctrls
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:(.*)", line)
        if not m:
            continue
        name, rest = m.group(1), m.group(3)
        fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
        ctrls[name] = {k: int(v) for k, v in fields.items()}
    return ctrls


def list_mjpeg_modes(device: str) -> list[tuple[int, int, float]]:
    """[(w, h, fps)] the camera offers as MJPEG, largest first."""
    rc, out = _run(["-d", device, "--list-formats-ext"])
    if rc != 0:
        return []
    modes, in_mjpg, size = [], False, None
    for line in out.splitlines():
        if re.search(r"'\w+'", line):
            in_mjpg = "'MJPG'" in line
            continue
        if not in_mjpg:
            continue
        m = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and size:
            modes.append((size[0], size[1], float(m.group(1))))
    return sorted(set(modes), key=lambda m: (-m[0] * m[1], -m[2]))


def _set(device: str, ctrls: dict, logical: str, value) -> str | None:
    """Set the first alias this camera has. Returns the name used, or None."""
    for name in ALIASES[logical]:
        if name in ctrls:
            rc, out = _run(["-d", device, f"--set-ctrl={name}={int(value)}"])
            return name if rc == 0 else None
    return None


def apply(cam, log=print) -> dict:
    """Apply CameraConfig ``cam``'s locks. Returns {setting: result} for the report."""
    report: dict[str, str] = {}
    if not available():
        report["v4l2-ctl"] = "not installed (sudo apt install v4l-utils); camera left on its defaults"
        log(report["v4l2-ctl"])
        return report
    dev = cam.device if str(cam.device).startswith("/dev/") else f"/dev/video{cam.device}"
    ctrls = list_controls(dev)
    if not ctrls:
        report["controls"] = f"could not read controls of {dev}"
        return report

    def do(logical, value, label):
        used = _set(dev, ctrls, logical, value)
        report[label] = f"{used}={value}" if used else "not supported by this camera"

    do("dynamic_framerate", 1 if cam.dynamic_framerate else 0, "dynamic_framerate")
    if cam.exposure == "auto":
        do("exposure_mode", EXPOSURE_AUTO, "exposure")
    else:
        do("exposure_mode", EXPOSURE_MANUAL, "exposure_mode")
        do("exposure_time", int(cam.exposure), "exposure")
    if not cam.autofocus:
        do("autofocus", 0, "autofocus")
        if cam.focus is not None:
            do("focus", int(cam.focus), "focus")
    else:
        do("autofocus", 1, "autofocus")
    if POWER_LINE.get(cam.power_line_hz) is not None:
        do("power_line", POWER_LINE[cam.power_line_hz], "power_line")
    if cam.gain is not None:
        do("gain", int(cam.gain), "gain")
    for k, v in report.items():
        log(f"camera {k}: {v}")
    return report


def set_exposure(device: str, value_100us: int) -> bool:
    ctrls = list_controls(device)
    return (_set(device, ctrls, "exposure_mode", EXPOSURE_MANUAL) is not None
            and _set(device, ctrls, "exposure_time", value_100us) is not None)
