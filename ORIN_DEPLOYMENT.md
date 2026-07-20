# Orin all-in-one deployment (perception + controller on the Jetson)

The two bundles were built to talk over UDP between machines, but nothing
requires two machines. This runbook brings up BOTH on one Jetson Orin for the
immediate N=1 pursuer vs N=1 evader test:

```
webcam ──► usb_cam ► aruco_detector ► pose_bridge     (portable_orin_perception)
                                          │ UDP 127.0.0.1:9870
                                          ▼
              run_controller.py --source udp          (portable_n1_controller)
                                          │ UDP <esp-ip>:8888  (WiFi)
                                          ▼
                                   ESP32-S3 on the car
```

The controller needs only Python 3.10+ / numpy / gymnasium / onnxruntime — all
have aarch64 wheels; the ONNX policies are ~290 KB and run at 10 Hz on CPU with
ease. The Orin and the car's ESP32 must be on the same WiFi network.

Each phase gates the next. Do them in order; each bundle's own README stays the
authority for its internals (wire formats, dead-man semantics, constraints).

## Phase 0 — dev PC: make origin current

The Orin gets code by `git clone`/`git pull` only — never edit on the Orin.
Before every Orin session: commit here, push, pull there.

## Phase 1 — Orin, zero hardware (no ROS, no camera, no car)

Prove the whole AI side runs on this machine before installing anything heavy.

```bash
git clone https://github.com/Maviman/ee198-deployment-bundles.git
cd ee198-deployment-bundles

# Perception math selftest (plain python, no ROS needed)
cd portable_orin_perception
pip3 install -r requirements.txt        # 24.04: add --break-system-packages, or use a venv
python3 selftest.py                     # must print SELFTEST PASSED

# Controller in its own venv (keeps it independent of ROS/system python)
cd ../portable_n1_controller
sudo apt-get install -y python3-venv
python3 -m venv ~/.venvs/n1ctl
~/.venvs/n1ctl/bin/pip install -r requirements.txt
~/.venvs/n1ctl/bin/python selftest.py   # must print SELFTEST PASSED
```

`selftest.py` replays golden vectors through the vendored adapter + ONNX
policies (ACT_ATOL already allows for cross-machine onnxruntime variation).
A FAIL on the Orin means a broken install or numeric drift — stop and fix,
never loosen tolerances.

Then close the full loop with zero hardware, three terminals, all in
`portable_n1_controller/` (controller uses the venv python; the two tools are
stdlib+numpy so either python works):

```bash
python3 tools/mock_esp.py                                                  # T1
~/.venvs/n1ctl/bin/python run_controller.py --model models/n1_catch \
    --source udp --pose-port 9870 --esp 127.0.0.1:8888                     # T2
python3 ../portable_orin_perception/tools/fake_perception.py \
    --duration 6 --stall 1.2 --resume 3                                    # T3
```

T3 stands in for the entire perception stack (same FrameBuilder, same wire
bytes). Watch: poses in → commands out at ~10 Hz in mock_esp, then during the
stall an E-stop within 0.3 s, then re-arm on resume. This is the same
acceptance behavior as bring-up rung 6, moved onto the Orin.

## Phase 2 — ROS 2 install

```bash
cd portable_orin_perception
./setup_orin.sh          # detects 22.04→Humble / 24.04→Jazzy, installs, builds
```

Sanity per the bundle README rung 2: `demo_nodes_cpp talker` +
`demo_nodes_py listener` after sourcing `/opt/ros/<distro>/setup.bash`.

## Phase 3 — camera, markers, calibration

Follow `portable_orin_perception/README.md` rungs 3–4 exactly (print sheets at
100%, TOP edge toward the nose, intrinsics per `CALIBRATION.md`, then corner
markers + `./calibrate_arena.sh --device /dev/video0`, residuals ≤ 2 cm).
Remember: any camera move invalidates the homography — recalibrate.

## Phase 4 — live all-on-Orin session (the demo runbook)

```bash
# Terminal A — perception, pointed at this same machine
cd portable_orin_perception
./run_perception.sh 127.0.0.1 expected_pursuers:=1

# Terminal B — the AI
cd ../portable_n1_controller
~/.venvs/n1ctl/bin/python run_controller.py --model models/n1_catch \
    --source udp --pose-port 9870 --esp <esp-ip>:8888
```

Acceptance checks before wheels touch the ground, in order:

1. **Poses track reality.** Hand-push the tagged cars; watch x/y/heading in
   `python3 ../portable_orin_perception/tools/pose_frame_monitor.py` (works on
   loopback too — run it instead of the controller first).
2. **Dead-man.** Cover the lens mid-run: E-stop within ~0.3 s, re-arm on
   uncover. Silence IS the E-stop signal.
3. **Latency.** Everything on one box means one clock — the chrony rung is
   free, but the ≤ 0.1 s camera→policy→radio budget still stands. Read
   capture→arrival off pose_frame_monitor (target ≤ 50 ms) and keep the ESP
   RTT numbers from `tools/link_test.py` in budget.
4. **First driving session** stays wheels-off, per
   `portable_n1_controller/README.md` rung 2 (capped `THROTTLE_MAX_US`,
   failsafe pull-test), with real WiFi credentials in `wifi_credentials.h`
   (gitignored — set on the flashing machine, never committed).

## Phase 5 — installing / updating a policy (the AI install loop)

Models are code-signed in spirit: every model dir ships with its manifest,
metrics, and golden vectors, and `selftest.py` is the installer's receipt.
The loop, every time a new/retrained policy graduates from the training repo:

```
in AI Training:   train → evaluate → SIL-check
                  python -m controller_runtime.export_onnx --run-dir <run-dir>
copy              policy.onnx + manifest + metrics.json
                  → AI_Deployment/portable_n1_controller/models/<name>/
regenerate        python tools/make_reference_vectors.py
verify locally    python selftest.py            # on the dev PC
commit BOTH       models/<name>/ + reference_vectors.json in one commit; push
on the Orin       git pull && ~/.venvs/n1ctl/bin/python selftest.py
run it            run_controller.py --model models/<name> ...
```

**Arena gate (the big caveat):** the frozen `n1_catch` / `n1_pin` were trained
in the 14 m × 10 m arena and the observation scaling bakes that in. They will
run fine for pipeline/workflow validation on the Orin, but a real capture demo
in a small ArUco-bounded field needs the arena-scale retrain first (Track A3 in
the training repo's `docs/ROLLOUT_PLAN.md`), then this Phase-5 loop to install
the result. Perception itself is arena-agnostic once calibrated.
