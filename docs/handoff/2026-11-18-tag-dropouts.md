# Handoff: the car's runs end when a tag drops out (2026-11-18)

Written at the end of the 2026-11-18 session, for the next session to pick up.
Branch `claude/hold-radius`; both Orins run it (`~/arena-live` on the control Orin).

## The problem, in the operator's words

> "The car is pretty good at controlling its speed now. The problem is when the AI
> doesn't detect the car's tag, it fully loses vision of the car."

Every armed run today caught the evader and then ended with the dashboard's
`pose feed lost`. Earlier the operator also noted that when the car passes the
boundary, nothing brings it back.

## The discussion

**Operator's ideas**
1. Train the AI to account for leaving the arena, and send instructions ahead of time
   so the car can autopilot back in and out of the arena.
2. Expand the arena boundary a bit.

**Claude's first answer** (before the operator's correction below):
- Blamed speed first (binary drive is full power, measured run-on 0.65-0.98 m).
  **The operator says speed is fine now; treat that point as withdrawn.**
- Still relevant: past the tape the system is blind on purpose. The localizer rejects
  any pose more than 0.15 m outside (`bounds_margin_m`, `core/localizer.py:145,284`)
  and the hub watchdog disarms at the tape (`deploy/hub.py:493`). So nothing could
  steer a car back even if it tried.
- Do NOT enlarge the taped square: every model's inputs are normalised to the 8 ft
  arena (`arena_half_m` in each manifest). Instead, a **software buffer zone**: track
  0.3-0.4 m past the tape (the models were trained with `out_margin_m: 0.3`), and when
  the car crosses the tape pause the policy and let a simple scripted controller drive
  it back toward the centre, handing back ~0.2 m inside. Move the watchdog's disarm
  line to the buffer's edge. The 3-car runtime already has an out-of-bounds recovery
  (`pc_controller/runtimes/rc16_hive_lib/rc16/hive/recovery.py`); mirror it for N=1.
- Training recovery into the AI: later, with episodes that continue (penalty, not
  termination) after leaving, using the car's measured dynamics.

## What the telemetry says (run log `~/.arena/runs/20261118-173058/telemetry.jsonl`)

| run (armed at) | result | why it ended |
|---|---|---|
| 18:37:04 | caught: 1.1 m in < 3 s, held at 0.29 m | **evader** tag dropped (car parked beside it), stream cut, watchdog ended run |
| 18:37:50 | caught at 0.42 m, held 36 s, re-engaged when the evader was moved | **car** lost after reversing hard toward the top edge, last seen (0.06, 0.91); both cameras lost it |
| 18:39:26 | caught at 0.15 m in 2 s | **evader** tag dropped (car beside it) |
| 18:43:57 | started at the edge facing out | drove over the top edge; watchdog "car left the arena" (correct) |

Facts behind the table:
- The car was seen by ONE camera at a time (X1 = camera 0 on its half, C270 = camera 1
  on the other); the evader likewise. One camera missing one vehicle breaks the stream.
- A pose frame needs EVERY vehicle fresh: `FrameBuilder.build_frame` returns None when
  any vehicle is older than `hold_max_age_s` (0.25 s, `run_vision.py:56`,
  `vision/service.py:230`). The controller then E-stops after `stall_s` 0.3 s
  (`pc_controller/realtime.py:185`), and the hub's `run_watchdog` (`deploy/hub.py:469`,
  reason at :497) ENDS the run instead of letting it resume.
- No detections were rejected by the gates (bounds/jump/size/dup counters did not move):
  these were missed detections, not rejections.
- The X1 sees the 7 cm car tag at only ~14 px at its far side (detector needs ~21 px).

## Recommended fixes (Claude's, not yet built; operator was asked and the session ended)

1. **Hold the evader longer than the car.** Per-role hold: car 0.25 s (must stay
   fresh), evader ~1.0 s (slow or parked; a slightly old evader pose is harmless).
   Would have kept runs 1 and 3 alive. Where: `FrameBuilder` in
   `portable_orin_perception/ros2_ws/src/hive_perception/hive_perception/core/frame_builder.py`
   (+ a `--evader-hold-max-age-s` in `run_vision.py`, passed by `arena up`).
2. **Pause instead of ending the run.** On pose loss keep the car E-stopped, and if it
   is seen again within ~2 s resume automatically; end the run only if it stays lost.
   Where: `run_watchdog` in `deploy/hub.py` (today it disarms on any stall), and
   tests in `portable_n1_controller/tests/test_hub_controls.py`
   (`test_watchdog_ends_the_run_when_the_pose_feed_is_lost_rather_than_resuming`
   encodes today's rule and must change with it).
3. **Re-find a lost car faster**: after a loss, full-frame search on both cameras at
   once, nearest-first around the last pose plus its velocity (`core/localizer.py`).
4. **Physical:** bigger car tags (10-12 cm instead of 7 cm), or re-aim so the top edge
   is covered by a camera that sees it large.
5. Then the buffer zone + scripted recovery above, and AI recovery training last.

Also open from earlier (see the Nov 17 audit notes): roof-tag parallax between the two
cameras is not corrected (the car's fused pose can jump where both cameras see it).

## State at the end of the day

- Model on the controller: `models/rc16_n1_two_camera` (N=1). New models merged and
  verified: `rc16_n1_md_elite`, `rc16_hive_h2`, `rc16_hive_b` (3-car ones need torch,
  installed on the control Orin; ~38 ms per tick).
- Car: ESP32 on Bluetooth (`esp = ble:hive-car0`, board on `radio ble`), firmware
  `2026.11-servo-ramp-uart-ble-wifinvs`, servo steering GPIO19, binary drive GPIO2/GPIO1.
- Calibration: two corner cameras fitted on the shared corner tags 11/13 alone (new 16 cm
  black-square corner tags, `corner_tag_m: 0.16`): +-5.0 cm. Floor tags (evader, spares
  2 and 3) on the guide strip between corners 11 and 13 + CALIBRATE bring it to ~1 cm.
- START: an outward-facing car near the edge is now advisory (amber), not a block.
- Dashboard has FLASH CAR and MANUAL DRIVE (W A S D) for bench tests.
- GitHub: this branch is pushed; a PR to `main` was never opened (no `gh` on the
  Orin, and the app browser is signed out). Compare page:
  https://github.com/Maviman/ee198-deployment-bundles/compare/main...claude/hold-radius
