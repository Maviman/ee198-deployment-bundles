"""Establish — by measurement, not assumption — which GPU paths this Jetson
actually offers for the perception pipeline, and what each one costs.

Run this ON THE ORIN before changing the pipeline. Everything the CUDA
migration plan depends on is an empirical fact about *this* board's JetPack,
OpenCV, GStreamer and (optionally) Isaac ROS install; several of those facts
have already burned this project once (HANDOFF.md: "this file previously said
[CUDA ArUco] probably would [work]. It was wrong.").

    python3 tools/probe_orin_gpu.py                # full probe + benchmarks
    python3 tools/probe_orin_gpu.py --no-bench     # inventory only, fast
    python3 tools/probe_orin_gpu.py --device /dev/video0

Nothing here writes to the pipeline or the config; it is read-only and safe to
run at any time. Sections that fail are reported and skipped, never fatal —
the point is a complete picture, not an early exit.

IMPORTANT: run it twice, once before `sudo jetson_clocks` and once after. The
board was profiled at 1.19 of 1.73 GHz under `schedutil` with ~60% run-to-run
variance, so any number taken on an unpinned board is soft.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Capture sizes to benchmark: the current pipeline resolution and the one the
# CPU-ceiling finding rejected (at 30 fps, which was the actual problem).
BENCH_SIZES = [(640, 480), (1280, 720)]
BENCH_ITERS = 60


def hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def run(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Run a command, returning stripped stdout or None if it isn't usable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() or None


# --------------------------------------------------------------------------
# 1. Platform, clocks, power
# --------------------------------------------------------------------------

def probe_platform() -> dict:
    hr("1. PLATFORM / CLOCKS")
    info: dict = {}

    l4t = Path("/etc/nv_tegra_release")
    info["l4t"] = l4t.read_text(encoding="utf-8").strip().splitlines()[0] if l4t.exists() else None
    print(f"L4T            : {info['l4t'] or 'not a Jetson (no /etc/nv_tegra_release)'}")

    osr = Path("/etc/os-release")
    if osr.exists():
        fields = dict(
            line.split("=", 1) for line in osr.read_text(encoding="utf-8").splitlines() if "=" in line
        )
        info["ubuntu"] = fields.get("VERSION_ID", "").strip('"')
    print(f"Ubuntu         : {info.get('ubuntu')}  -> ROS 2 "
          f"{ {'22.04': 'humble', '24.04': 'jazzy'}.get(info.get('ubuntu'), '?') }")
    print(f"Python / arch  : {platform.python_version()} / {platform.machine()}")

    model = Path("/proc/device-tree/model")
    if model.exists():
        info["model"] = model.read_text(encoding="utf-8").rstrip("\x00").strip()
        print(f"Board          : {info['model']}")

    # Power mode + governor + actual clock. This is the 1.45x sitting on the table.
    info["nvpmodel"] = run(["nvpmodel", "-q"])
    if info["nvpmodel"]:
        print(f"nvpmodel       : {' | '.join(info['nvpmodel'].splitlines())}")

    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["governor"] = gov.read_text(encoding="utf-8").strip() if gov.exists() else None
    cur = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    mx = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    if cur.exists() and mx.exists():
        cur_ghz = int(cur.read_text()) / 1e6
        max_ghz = int(mx.read_text()) / 1e6
        info["cpu_cur_ghz"], info["cpu_max_ghz"] = cur_ghz, max_ghz
        pct = 100.0 * cur_ghz / max_ghz if max_ghz else 0.0
        print(f"CPU clock      : {cur_ghz:.2f} / {max_ghz:.2f} GHz ({pct:.0f}%)  "
              f"governor={info['governor']}")
        if pct < 95.0:
            print("  >> NOT PINNED. Run `sudo jetson_clocks` and re-run this probe;")
            print("     every timing below is understating the board by "
                  f"~{max_ghz / cur_ghz:.2f}x.")
    try:
        import os
        info["cpu_count"] = os.cpu_count()
        print(f"CPU cores      : {info['cpu_count']}")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# 2. OpenCV: build flags, CUDA, tag families
# --------------------------------------------------------------------------

def probe_opencv() -> dict:
    hr("2. OPENCV")
    info: dict = {}
    try:
        import cv2
    except ImportError as exc:
        print(f"cv2 import failed: {exc}")
        return {"error": str(exc)}

    info["version"] = cv2.__version__
    info["path"] = getattr(cv2, "__file__", "?")
    print(f"version        : {cv2.__version__}")
    print(f"module         : {info['path']}")

    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        n = 0
    info["cuda_devices"] = n
    print(f"CUDA devices   : {n}  {'(CPU-only build)' if n == 0 else ''}")

    # The load-bearing negative result: no CUDA ArUco/AprilTag in ANY OpenCV build.
    info["cuda_aruco"] = bool(getattr(getattr(cv2, "cuda", None), "aruco", None))
    print(f"cv2.cuda.aruco : {info['cuda_aruco']} (expected False — no CUDA fiducials in OpenCV)")

    info["has_detector"] = hasattr(cv2.aruco, "ArucoDetector") if hasattr(cv2, "aruco") else False
    print(f"ArucoDetector  : {info['has_detector']} (need OpenCV >= 4.7)")

    fams = [d for d in dir(cv2.aruco) if d.startswith("DICT_APRILTAG")] if hasattr(cv2, "aruco") else []
    info["apriltag_dicts"] = fams
    print(f"AprilTag dicts : {fams or 'NONE — cannot generate/detect AprilTags on CPU'}")
    if "DICT_APRILTAG_36h11" in fams:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        print(f"  36h11 codebook: {d.bytesList.shape[0]} tags, {d.markerSize}x{d.markerSize} data bits")

    build = cv2.getBuildInformation()
    for key in ("NVIDIA CUDA", "cuDNN", "GStreamer", "libjpeg", "JPEG 2000"):
        line = next((ln.strip() for ln in build.splitlines() if ln.strip().startswith(key)), None)
        if line:
            print(f"  build: {line}")
            info.setdefault("build", {})[key] = line
    return info


# --------------------------------------------------------------------------
# 3. CUDA / VPI / NVJPEG / TensorRT / onnxruntime inventory
# --------------------------------------------------------------------------

def probe_cuda_stack() -> dict:
    hr("3. CUDA STACK")
    info: dict = {}

    nvcc = run(["nvcc", "--version"])
    info["nvcc"] = nvcc.splitlines()[-1] if nvcc else None
    print(f"nvcc           : {info['nvcc'] or 'not on PATH (usually /usr/local/cuda/bin)'}")

    try:
        import vpi  # type: ignore
        info["vpi"] = getattr(vpi, "__version__", "present")
        backends = [b for b in dir(vpi.Backend) if b.isupper()] if hasattr(vpi, "Backend") else []
        info["vpi_backends"] = backends
        print(f"VPI            : {info['vpi']}   backends: {', '.join(backends) or '?'}")
        print("  note: VPI has no AprilTag/ArUco detector — it is a decode/CV-op library here.")
    except ImportError:
        info["vpi"] = None
        print("VPI            : python bindings not importable "
              "(lib may still exist for C++/GStreamer use)")

    # Hardware JPEG: the NVJPG engine is reachable via GStreamer nvjpegdec and
    # via libnvjpeg. Presence of the .so is the cheap check.
    libs = []
    for pat in ("libnvjpeg*.so*", "libnvbufsurface*.so*"):
        libs += [str(p) for p in Path("/usr/lib/aarch64-linux-gnu").glob(pat)]
        libs += [str(p) for p in Path("/usr/local/cuda/lib64").glob(pat)]
    info["nvjpeg_libs"] = libs
    print(f"nvjpeg libs    : {len(libs)} found" + (f"  e.g. {libs[0]}" if libs else ""))

    trt = run(["dpkg-query", "-W", "-f=${Version}", "tensorrt"])
    info["tensorrt"] = trt
    print(f"TensorRT       : {trt or 'not installed via dpkg (may still be present)'}")

    try:
        import onnxruntime as ort  # type: ignore
        info["ort_version"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
        print(f"onnxruntime    : {ort.__version__}  providers={info['ort_providers']}")
        if "CUDAExecutionProvider" not in info["ort_providers"]:
            print("  note: CPU-only ORT. Policy inference is ~0.05 ms — this does NOT matter.")
    except ImportError:
        info["ort_version"] = None
        print("onnxruntime    : not installed")
    return info


# --------------------------------------------------------------------------
# 4. Isaac ROS / cuAprilTags
# --------------------------------------------------------------------------

def probe_isaac_ros() -> dict:
    hr("4. ISAAC ROS / cuAprilTags  (the GPU detector path)")
    info: dict = {}

    pkgs = run(["ros2", "pkg", "list"], timeout=30.0)
    if pkgs is None:
        print("ros2 not on PATH — source /opt/ros/<distro>/setup.bash and re-run "
              "for this section to mean anything.")
        info["ros2"] = False
        return info
    info["ros2"] = True
    isaac = sorted(p for p in pkgs.splitlines() if "isaac" in p or "nitros" in p)
    info["isaac_packages"] = isaac
    if isaac:
        print(f"Isaac ROS packages installed ({len(isaac)}):")
        for p in isaac:
            print(f"  {p}")
    else:
        print("No Isaac ROS packages installed.")
        print("  isaac_ros_apriltag is the ONLY genuine CUDA fiducial detector path.")
        print("  It is distributed per ROS distro + JetPack version — check the")
        print("  Isaac ROS release notes against the L4T/Ubuntu reported in section 1")
        print("  BEFORE planning around it. Do not assume Jazzy support.")

    for key in ("isaac_ros_apriltag", "isaac_ros_image_proc", "isaac_ros_nitros"):
        info[key] = key in (isaac or [])
        print(f"  {key:24s}: {'PRESENT' if info[key] else 'missing'}")

    # The cuAprilTags static lib ships inside isaac_ros_apriltag; find it directly
    # too, since it can exist without the ROS wrapper.
    found = []
    for root in ("/opt/nvidia", "/usr/lib", "/opt/ros"):
        p = Path(root)
        if p.exists():
            try:
                found += [str(f) for f in p.rglob("*cuapriltags*")][:5]
            except (OSError, PermissionError):
                pass
    info["cuapriltags_files"] = found
    print(f"cuAprilTags lib: {found[0] if found else 'not found on disk'}")
    return info


# --------------------------------------------------------------------------
# 5. Benchmarks: JPEG decode paths, and CPU detection cost per tag family
# --------------------------------------------------------------------------

def _synth_frame(w: int, h: int):
    """A frame with real texture (so JPEG size and detector candidate-rejection
    load are not trivially unrealistic) plus a few tags."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    img = rng.integers(90, 170, size=(h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)  # carpet-ish, not white noise
    for i in range(14):  # clutter quads for the detector to reject
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        cv2.rectangle(img, (x, y), (x + 40, y + 30), (40, 40, 40), -1)
    return img


def _place_tags(img, dict_name: str, ids, side_px: int):
    import cv2
    import numpy as np

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    h, w = img.shape[:2]
    for k, tid in enumerate(ids):
        tag = cv2.aruco.generateImageMarker(d, int(tid), side_px)
        tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        q = side_px // 4
        pad = np.full((side_px + 2 * q, side_px + 2 * q, 3), 255, np.uint8)
        pad[q:q + side_px, q:q + side_px] = tag
        s = pad.shape[0]
        x = 40 + (k % 3) * (s + 30)
        y = 40 + (k // 3) * (s + 30)
        if x + s < w and y + s < h:
            img[y:y + s, x:x + s] = pad
    return img


def bench_detection() -> dict:
    hr("5a. BENCHMARK — CPU detection cost, ArUco 4x4 vs AprilTag 36h11")
    import cv2
    import numpy as np

    info: dict = {}
    print("Same physical tag size, same frame. AprilTag 36h11 is 8x8 modules vs")
    print("ArUco 4x4_50's 6x6, so it needs ~1.33x the pixels per side to detect.\n")
    print(f"{'resolution':>12} {'family':>22} {'detect ms':>11} {'found':>6}")
    print("-" * 56)

    for (w, h) in BENCH_SIZES:
        base = _synth_frame(w, h)
        side = max(24, int(0.07 / 1.83 * w))  # 7 cm tag in a 1.83 m arena
        for dict_name in ("DICT_4X4_50", "DICT_APRILTAG_36h11"):
            if not hasattr(cv2.aruco, dict_name):
                continue
            img = _place_tags(base.copy(), dict_name, [0, 1, 2, 3], side)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
            det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
            for _ in range(5):
                det.detectMarkers(gray)
            t0 = time.perf_counter()
            for _ in range(BENCH_ITERS):
                corners, ids, _r = det.detectMarkers(gray)
            ms = (time.perf_counter() - t0) / BENCH_ITERS * 1e3
            n = 0 if ids is None else len(ids)
            info[f"{w}x{h}/{dict_name}"] = {"ms": ms, "found": n, "tag_px": side}
            print(f"{w}x{h:<7} {dict_name:>22} {ms:>10.2f} {n:>5}/4   (tag ~{side}px)")
    print("\n'found' matters as much as the timing: a family that cannot resolve the")
    print("tag at that pixel size is not a candidate regardless of how fast it is.")
    return info


def bench_jpeg_decode(device: str) -> dict:
    hr("5b. BENCHMARK — JPEG decode paths (CPU vs hardware NVJPG)")
    import cv2
    import numpy as np

    info: dict = {}

    # --- CPU baseline: cv2.imdecode, what usb_cam's mjpeg2rgb effectively does.
    for (w, h) in BENCH_SIZES:
        img = _synth_frame(w, h)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        for _ in range(5):
            cv2.imdecode(buf, cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ms = (time.perf_counter() - t0) / BENCH_ITERS * 1e3
        info[f"cpu_imdecode_{w}x{h}"] = ms
        print(f"CPU cv2.imdecode {w}x{h:<6}: {ms:6.2f} ms  ({len(buf) / 1024:.0f} KB jpeg)")

    # --- GStreamer element inventory (hardware decode lives here).
    print()
    gst_ok = "GStreamer" in cv2.getBuildInformation() and "YES" in next(
        (ln for ln in cv2.getBuildInformation().splitlines()
         if ln.strip().startswith("GStreamer")), "")
    info["opencv_gstreamer"] = gst_ok
    print(f"OpenCV built with GStreamer: {gst_ok}"
          + ("" if gst_ok else "  <-- hardware decode via cv2 unavailable; "
                              "rebuild or use a GStreamer-native node"))

    for el in ("nvjpegdec", "nvv4l2decoder", "nvvidconv", "jpegdec", "videoconvert"):
        out = run(["gst-inspect-1.0", el], timeout=15.0)
        info[f"gst_{el}"] = out is not None
        print(f"  gst element {el:<16}: {'present' if out else 'MISSING'}")

    # --- Live capture through candidate pipelines. Measured, not assumed.
    if device:
        print(f"\nLive capture trials on {device} (3 s each, {BENCH_SIZES[-1][0]}x{BENCH_SIZES[-1][1]}):")
        w, h = BENCH_SIZES[-1]
        candidates = {
            "v4l2 MJPEG -> nvjpegdec (HW)":
                f"v4l2src device={device} io-mode=2 ! image/jpeg,width={w},height={h},framerate=15/1 "
                f"! nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
            "v4l2 MJPEG -> jpegdec (CPU)":
                f"v4l2src device={device} io-mode=2 ! image/jpeg,width={w},height={h},framerate=15/1 "
                f"! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
        }
        for label, pipeline in candidates.items():
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    print(f"  {label:<32}: pipeline failed to open")
                    info[label] = None
                    continue
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < 3.0:
                    ok, _f = cap.read()
                    if not ok:
                        break
                    n += 1
                fps = n / (time.perf_counter() - t0)
                cap.release()
                info[label] = fps
                print(f"  {label:<32}: {fps:5.1f} fps ({n} frames)")
            except Exception as exc:  # noqa: BLE001 - probe must never abort
                info[label] = None
                print(f"  {label:<32}: error {exc}")
    else:
        print("\n(--device '' given; skipping live capture trials)")
    return info


def _get(d: dict, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def compare(path_a: str, path_b: str) -> None:
    """Print a side-by-side delta of two probe runs.

    The whole point of taking a 'before' and an 'after' is the delta, and eyeballing
    two 200-line transcripts is how a 1.45x speedup gets mistaken for noise.
    """
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    la = _get(a, "label") or Path(path_a).stem
    lb = _get(b, "label") or Path(path_b).stem

    hr(f"COMPARE  {la}  ->  {lb}")
    print(f"{'metric':38s} {la[:14]:>14} {lb[:14]:>14} {'change':>12}")
    print("-" * 82)

    def row(name, va, vb, lower_better=True):
        if va is None and vb is None:
            print(f"{name:38s} {'n/a':>14} {'n/a':>14} {'-':>12}")
            return
        # Booleans BEFORE the numeric path: float(False) is 0.0, which would
        # render a missing GStreamer element as a plausible-looking timing.
        if isinstance(va, bool) or isinstance(vb, bool):
            sa = "present" if va else "missing"
            sb = "present" if vb else "missing"
            print(f"{name:38s} {sa:>14} {sb:>14} "
                  f"{('same' if va == vb else 'CHANGED'):>12}")
            return
        try:
            fa, fb = float(va), float(vb)
        except (TypeError, ValueError):
            print(f"{name:38s} {str(va)[:14]:>14} {str(vb)[:14]:>14} "
                  f"{('same' if va == vb else 'CHANGED'):>12}")
            return
        if fa == 0 or fb == 0:
            delta = "-"
        else:
            improved = (fb < fa) if lower_better else (fb > fa)
            factor = (fa / fb) if lower_better else (fb / fa)
            if factor < 1.0:
                factor = 1.0 / factor
            delta = f"{factor:.2f}x {'better' if improved else 'worse'}"
        print(f"{name:38s} {fa:>14.2f} {fb:>14.2f} {delta:>12}")

    row("CPU clock GHz", _get(a, "platform", "cpu_cur_ghz"),
        _get(b, "platform", "cpu_cur_ghz"), lower_better=False)
    row("cv2.cuda devices", _get(a, "opencv", "cuda_devices"),
        _get(b, "opencv", "cuda_devices"), lower_better=False)

    for key in sorted(set(_get(a, "bench_detection", default={}))
                      | set(_get(b, "bench_detection", default={}))):
        row(f"detect {key}", _get(a, "bench_detection", key, "ms"),
            _get(b, "bench_detection", key, "ms"))

    for key in sorted(k for k in set(_get(a, "bench_jpeg", default={}))
                      | set(_get(b, "bench_jpeg", default={}))
                      if k.startswith("cpu_imdecode")):
        row(key, _get(a, "bench_jpeg", key), _get(b, "bench_jpeg", key))

    for name, path in (("Isaac ROS apriltag", ("isaac_ros", "isaac_ros_apriltag")),
                       ("gst nvjpegdec", ("bench_jpeg", "gst_nvjpegdec"))):
        row(name, _get(a, *path), _get(b, *path))

    ca, cb = _get(a, "platform", "cpu_cur_ghz"), _get(b, "platform", "cpu_cur_ghz")
    ma = _get(a, "platform", "cpu_max_ghz")
    if ca and cb and ma and ca / ma < 0.95 and cb / ma > 0.95:
        print(f"\nClocks went from {ca:.2f} to {cb:.2f} GHz — expect roughly "
              f"{cb/ca:.2f}x on every CPU-bound number above.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video0",
                    help="camera for live decode trials ('' to skip)")
    ap.add_argument("--no-bench", action="store_true", help="inventory only, no timing")
    ap.add_argument("--json", help="also write the full result dict here")
    ap.add_argument("--label", default=None,
                    help="name this run (e.g. 'before-clocks'); appears in --compare")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                    help="print a delta table for two earlier runs and exit")
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return

    print(__doc__.split("\n\n")[0])
    result = {
        "label": args.label or "",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": probe_platform(),
        "opencv": probe_opencv(),
        "cuda_stack": probe_cuda_stack(),
        "isaac_ros": probe_isaac_ros(),
    }
    if not args.no_bench:
        try:
            result["bench_detection"] = bench_detection()
        except Exception as exc:  # noqa: BLE001
            print(f"detection benchmark failed: {exc}")
        try:
            result["bench_jpeg"] = bench_jpeg_decode(args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"jpeg benchmark failed: {exc}")

    hr("VERDICT")
    plat = result["platform"]
    if plat.get("cpu_cur_ghz") and plat.get("cpu_max_ghz"):
        pinned = plat["cpu_cur_ghz"] / plat["cpu_max_ghz"] > 0.95
        print(f"clocks pinned         : {'YES' if pinned else 'NO  <-- do this first'}")
    isaac = result["isaac_ros"]
    print(f"GPU detector available: "
          f"{'YES (isaac_ros_apriltag)' if isaac.get('isaac_ros_apriltag') else 'NO — install Isaac ROS, or stay on CPU AprilTag'}")
    jp = result.get("bench_jpeg", {})
    print(f"HW JPEG decode usable : "
          f"{'YES (nvjpegdec)' if jp.get('gst_nvjpegdec') else 'NO — nvjpegdec element missing'}")
    print(f"CPU AprilTag fallback : "
          f"{'YES' if 'DICT_APRILTAG_36h11' in result['opencv'].get('apriltag_dicts', []) else 'NO'}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
