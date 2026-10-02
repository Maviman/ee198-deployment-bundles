#!/usr/bin/env bash
# Pre-merge / post-pull gate: prove this checkout is safe to run on THIS Jetson
# before pointing it at a real arena and real cars.
#
#   cd portable_orin_perception && ./preflight.sh
#
# Read-only and hardware-free -- it does not open the camera, does not build,
# does not touch config. Every check answers one question that has a wrong
# answer capable of driving a car badly, and prints WHY it matters when it
# fails. Exit code 0 means the checkout is consistent with the hardware you
# actually have; non-zero means fix it before a driving session.
#
# What it deliberately does NOT check (needs hardware or a build):
#   colcon build, the camera, GPU/Isaac ROS availability, the ESP32 link.
# Those are tools/probe_orin_gpu.py, tools/capture_run.py and
# portable_n1_controller/tools/link_test.py respectively.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PASS=0; FAIL=0; WARN=0
ok()   { echo "  [ OK ]  $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL]  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN]  $*"; WARN=$((WARN+1)); }
hdr()  { echo; echo "== $* =="; }

hdr "1. Python + OpenCV can do what the detector needs"
python3 - <<'PY'
import sys
try:
    import cv2, numpy, yaml
except ImportError as e:
    print(f"  [FAIL]  missing dependency: {e}"); sys.exit(1)
print(f"  [ OK ]  python {sys.version.split()[0]}, opencv {cv2.__version__}, numpy {numpy.__version__}")
if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
    print("  [FAIL]  cv2.aruco.ArucoDetector missing (need OpenCV >= 4.7, contrib build)")
    print("          fix: pip3 install --user opencv-contrib-python")
    sys.exit(1)
print("  [ OK ]  cv2.aruco.ArucoDetector present")
PY
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

hdr "2. Configured tag family is available, and the subset API works"
python3 - <<'PY'
import sys, pathlib, yaml
sys.path.insert(0, "ros2_ws/src/hive_perception")
import cv2
from hive_perception.core.frame_builder import load_marker_map
from hive_perception.core import tag_family

mm = load_marker_map("config/marker_map.yaml")
if not hasattr(cv2.aruco, mm.dictionary):
    print(f"  [FAIL]  this OpenCV has no {mm.dictionary}")
    print("          fix: pip3 install --user opencv-contrib-python")
    sys.exit(1)
print(f"  [ OK ]  family {mm.dictionary} available")
try:
    ts = mm.tag_set(); d = ts.build_opencv_dictionary()
except Exception as e:
    print(f"  [FAIL]  subset dictionary construction failed: {e}")
    print("          the detector cannot run without it -- likely an OpenCV too old")
    print("          for the cv2.aruco.Dictionary(bytesList, markerSize, maxcorr) ctor")
    sys.exit(1)
print(f"  [ OK ]  subset dictionary: {d.bytesList.shape[0]} codes {list(ts.real_ids)}, "
      f"correction {ts.max_correction_bits} bits")
if ts.max_correction_bits > ts.safe_correction_bits():
    print(f"  [FAIL]  correction {ts.max_correction_bits} exceeds the safe ceiling "
          f"{ts.safe_correction_bits()} for these ids -- one vehicle's tag could be "
          "read as another's")
    sys.exit(1)
print(f"  [ OK ]  error correction within the safe ceiling ({ts.safe_correction_bits()})")
PY
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

hdr "3. Printed sheets match the configured family"
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "ros2_ws/src/hive_perception")
from hive_perception.core.frame_builder import load_marker_map
mm = load_marker_map("config/marker_map.yaml")
out = pathlib.Path("markers")
if not out.exists():
    print("  [WARN]  markers/ missing -- run: python3 tools/generate_tags.py"); sys.exit(2)
have = {p.stem for p in out.glob("*.pdf")}
want_ids = set(mm.print_ids)
have_ids = set()
for stem in have:
    try: have_ids.add(int(stem.split("_")[1]))
    except (IndexError, ValueError): pass
missing = want_ids - have_ids
extra = have_ids - want_ids
if missing:
    print(f"  [FAIL]  no sheet for id(s) {sorted(missing)} -- run tools/generate_tags.py")
    sys.exit(1)
print(f"  [ OK ]  sheets present for ids {sorted(want_ids)}")
if extra:
    print(f"  [WARN]  extra sheets for id(s) {sorted(extra)} not in the marker map -- "
          "right id, possibly wrong family; delete or add them")
    sys.exit(2)
PY
rc=$?; [ $rc -eq 0 ] && PASS=$((PASS+1)) || { [ $rc -eq 2 ] && WARN=$((WARN+1)) || FAIL=$((FAIL+1)); }

hdr "4. Arena calibration exists and matches the capture resolution"
python3 - <<'PY'
import sys, re, pathlib, yaml
sys.path.insert(0, "ros2_ws/src/hive_perception")
h = pathlib.Path("config/arena_homography.yaml")
if not h.exists():
    print("  [WARN]  no config/arena_homography.yaml -- run ./calibrate_arena.sh")
    print("          (run_perception.sh refuses to start without it)")
    sys.exit(2)
cal = yaml.safe_load(h.read_text(encoding="utf-8"))
cal_size = cal.get("image_size")
from hive_perception.core.camera_config import load_camera_config
cam = load_camera_config("config/camera.yaml")
live = [cam.width, cam.height]
print(f"  [ OK ]  calibration present (max residual "
      f"{cal.get('max_residual_m', '?')} m, calibrated at {cal_size})")
if cal_size and list(cal_size) != live:
    print(f"  [FAIL]  calibrated at {cal_size} but config/camera.yaml captures {live}.")
    print("          A homography is pixel geometry -- using it at another resolution")
    print("          scales EVERY pose silently. Re-run `arena scan` (or")
    print(f"          ./calibrate_arena.sh) at the capture size, or set width/height in")
    print(f"          config/camera.yaml back to {cal_size[0]}x{cal_size[1]}.")
    sys.exit(1)
res = float(cal.get("max_residual_m") or 0)
if res > 0.02:
    print(f"  [WARN]  max residual {res:.4f} m > 0.02 m -- usable, but re-measure the "
          "worst corner")
    sys.exit(2)
PY
rc=$?; [ $rc -eq 0 ] && PASS=$((PASS+1)) || { [ $rc -eq 2 ] && WARN=$((WARN+1)) || FAIL=$((FAIL+1)); }

hdr "5. Camera intrinsics agree with the capture resolution"
python3 - <<'PY'
import sys, re, pathlib, yaml
ci = pathlib.Path("config/camera_info.yaml")
if not ci.exists():
    print("  [WARN]  no config/camera_info.yaml"); sys.exit(2)
d = yaml.safe_load(ci.read_text(encoding="utf-8"))
sys.path.insert(0, "ros2_ws/src/hive_perception")
from hive_perception.core.camera_config import load_camera_config
cam = load_camera_config("config/camera.yaml")
live = (cam.width, cam.height)
declared = (int(d.get("image_width", 0)), int(d.get("image_height", 0)))
k = d.get("camera_matrix", {}).get("data", [])
placeholder = bool(k) and float(k[0]) == 1000.0 and float(k[4]) == 1000.0
if declared != live:
    lvl = "WARN" if placeholder else "FAIL"
    print(f"  [{lvl}]  camera_info says {declared} but config/camera.yaml captures {live}.")
    print("          Intrinsics are in pixels and do not carry across resolutions.")
    if placeholder:
        print("          Harmless for now ONLY because these are placeholder values")
        print("          (fx=fy=1000). Fix before the first real checkerboard run.")
        sys.exit(2)
    sys.exit(1)
print(f"  [ OK ]  camera_info resolution {declared} matches capture")
if placeholder:
    print("  [WARN]  intrinsics are PLACEHOLDERS (fx=fy=1000) -- undistortion is a no-op;")
    print("          fine for bring-up, not for any accuracy number you report")
    sys.exit(2)
PY
rc=$?; [ $rc -eq 0 ] && PASS=$((PASS+1)) || { [ $rc -eq 2 ] && WARN=$((WARN+1)) || FAIL=$((FAIL+1)); }

hdr "6. Fleet wiring is self-consistent"
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "ros2_ws/src/hive_perception")
from hive_perception.core.frame_builder import load_marker_map
mm = load_marker_map("config/marker_map.yaml")
n = len(mm.pursuer_ids)
print(f"  [ OK ]  fleet of {n} pursuer tag(s): marker ids {mm.pursuer_ids}, evader {mm.evader_id}")
print(f"          -> `arena` tracks the first num_pursuers of them (the model's count; see `arena fleet`)")
print(f"          -> by hand: run_vision.py --pursuers <the model's num_pursuers>;")
print(f"             the ROS pipeline: expected_pursuers:={n} (it tracks every listed tag)")
for i, mid in enumerate(mm.pursuer_ids):
    print(f"          -> marker id {mid} = pursuer slot {i} = ESP32 CAR_INDEX {i} "
          f"= --esp address #{i + 1}")
if n > 1:
    print("  [WARN]  N>1: each ESP32 needs a DISTINCT CAR_INDEX set over serial")
    print("          ('index <n>'), and --esp addresses must be in that same order.")
    sys.exit(2)
PY
rc=$?; [ $rc -eq 0 ] && PASS=$((PASS+1)) || { [ $rc -eq 2 ] && WARN=$((WARN+1)) || FAIL=$((FAIL+1)); }

hdr "7. Full selftest (detection -> pose -> controller wire format)"
if python3 selftest.py >/tmp/preflight_selftest.log 2>&1; then
    ok "selftest passed"
else
    bad "selftest FAILED -- see /tmp/preflight_selftest.log"
    tail -15 /tmp/preflight_selftest.log | sed 's/^/          /'
fi

hdr "8. Workspace build is present and current"
if [ ! -f ros2_ws/install/setup.bash ]; then
    warn "ros2_ws not built yet -- run: cd ros2_ws && colcon build --symlink-install"
else
    newest_src=$(find ros2_ws/src -name '*.py' -newer ros2_ws/install/setup.bash 2>/dev/null | head -3)
    if [ -n "$newest_src" ]; then
        warn "source newer than the last build -- rebuild before running:"
        echo "$newest_src" | sed 's/^/            /'
        echo "            cd ros2_ws && colcon build --symlink-install"
    else
        ok "ros2_ws build is newer than its sources"
    fi
fi

echo
echo "=================================================="
printf "  passed %d   warnings %d   FAILURES %d\n" "$PASS" "$WARN" "$FAIL"
echo "=================================================="
if [ "$FAIL" -gt 0 ]; then
    echo "NOT READY -- fix the [FAIL] items above before a driving session."
    exit 1
fi
if [ "$WARN" -gt 0 ]; then
    echo "READY, with warnings. Read them: most are 'you have not done the"
    echo "physical step yet' (calibration, printing, per-car CAR_INDEX)."
    exit 0
fi
echo "READY."
