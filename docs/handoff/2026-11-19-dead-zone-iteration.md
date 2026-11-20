# Dead zone: how to iterate on it (for the next session, on the Orins)

Companion to [2026-11-19-dead-zone.md](2026-11-19-dead-zone.md), which says what the feature
does. This file is for changing it: where each behaviour lives, how to see what happened on a
run, which number to turn for which symptom, what must not be broken, and what is known to be
weak. Written on the dev PC; **nothing here has run on the real cars yet.**

## 1. The path of one pose, with the function that owns each step

| step | file and function | what it decides |
|---|---|---|
| a tag is detected | `portable_orin_perception/.../core/localizer.py` `Localizer._gate` | reject a pose more than `bounds_margin_m` (0.45 m) past the tape, a jump over 3.5 m/s, a wrong-size tag |
| optional | `core/arena_frame.py` `camera_position`, `lift_to_height` | roof-tag height correction (off unless `vehicle_tag_height_m` is set) |
| two cameras become one pose | `vision/fusion.py` `PoseFuser.update` | weight by tag size |
| the frame is built | `core/frame_builder.py` `FrameBuilder.build_frame` | held (age <= 0.25 s car / 1.0 s evader: pose plus its own `t`), lost (null pose, `lost_s`, `last`), `now`, never-seen = silence |
| it is sent | `vision/service.py` `VisionService._process` | adds `seq`, `lat`, `cam`, `arena` |
| it is parsed | `portable_n1_controller/pc_controller/pose_stream.py` `_vehicle`, `parse_pose_frame_meta` | `meta["seen"]`, `["lost_s"]`, `["now"]`, `["arena"]`; a lost vehicle comes back at its LAST SEEN pose |
| is each pose fresh? | `pc_controller/deadzone.py` `fresh_seen` | a car pose older than `FRESH_S` (0.12 s) against `now` is unseen |
| the model runs | `pc_controller/loops.py` `HiveLoop.tick(seen=)` | speeds are not updated from unseen poses; `seen` goes into a runtime that sets `HANDLES_UNSEEN` |
| hold zone | `pc_controller/realtime.py` `HoldZone.apply` | zero for a car near the evader |
| **the dead zone** | `pc_controller/deadzone.py` `DeadZone.apply` | the last word on what is sent. Three modes, below |
| the run is supervised | `deploy/hub.py` `Hub.run_watchdog` | ends the run on `gave_up`, the time cap, or a true pose stall |

`DeadZone.mode` (logged at controller start as a `DEAD ZONE:` line, and in telemetry):

- **`controller`**: one-car models (`rc16_n1_md_elite`, `rc16_n1_two_camera`). `EdgeRecovery.step`
  does everything: `_guard_cmd` (edge brake), `_seen_cmd` (drive back in), `_blind_cmd` (brake,
  then replay).
- **`runtime`**: the 3-car models (`rc16_hive_h2`, `rc16_hive_b`). The trained code in
  `pc_controller/runtimes/rc16_hive_lib/rc16/hive/recovery.py` and `shield.py` does it; `DeadZone`
  only zeroes a car unseen 0.5 s that the runtime is not recovering, applies the evader hold after
  3 s, and gives up after 12 s of recovery.
- **`stop-only`**: older models whose manifest has no car numbers (`n1_catch`, `c38_8ft_buttons`),
  or `--recovery off`. An unseen car gets zero; the dashboard still ends the run at the tape.

## 2. See what happened on a run

- **Controller log** (`arena logs`): `car N OUT OF VIEW`, `PAST THE TAPE`, `BACK`, `LOST`,
  `EVADER OUT OF VIEW`, and at start-up the `DEAD ZONE:` line that says which mode is on.
- **Run log**: `python3 portable_n1_controller/tools/deadzone_log.py` reads the newest
  `~/.arena/runs/*/telemetry.jsonl` and prints, per car, every change of state with the last
  seen pose, the command sent and how far past the tape; `--ticks` prints every tick of an
  episode. **Start here when a car does not come back.**
- **Telemetry fields** (controller records): `seen` (pursuers, then evader), `deadzone.mode`,
  `.unseen_s`, `.recovering`, `.blind`, `.gave_up`, `.edge_brake`, `.evader_hold`,
  `.evader_unseen_s`, `.give_up_s`. `poses` of an unseen car is where it was last seen.
- **Dashboard**: an unseen car is drawn faded at its last seen pose with `OUT OF VIEW: BACKING UP`,
  `DRIVING BACK IN` or `LOST`.

Reading an episode: the recovery started if the state went to `BLIND RECOVERY`; the command column
shows what was sent (a brake, then about -0.19 or pulses of 0.40); if the commands are right and
the car did not move, it is the drive (section 3, first two rows), not the logic.

## 3. Symptom to knob

Constants are at the top of `pc_controller/deadzone.py` unless another place is named.

| on the rig | most likely cause | change |
|---|---|---|
| blind recovery "runs" but the car does not move | firmware `drive binary` ignores throttles under 1/3 | `binary` in `~/.arena/recovery` |
| same, on a `drive pwm` car | the motor does not turn at 0.19 x `limit` | `min=<smallest throttle that moves it>` in `~/.arena/recovery` (find it with MANUAL DRIVE) |
| the car backs up too fast when blind | pulses too strong (`binary`), or `min=` too high | `BINARY_PULSE`; or lower `DRIVE_IN_MPS` (it sets the pulse share and the PWM throttle) |
| it comes back toward view but stops short | the slow replay covers less ground than the car did | `REPLAY_EXTRA_S` (1.0), `HISTORY_S` (2.0), `REPLAY_HOLD_MAX` (4) |
| `LOST` before it is back | budget too short for how far it got | first make it leave slower (next row); then `GIVE_UP_S` (4.5) |
| it goes far past the tape before anything brakes | the real car brakes weaker or later than planned | `GUARD_LATENCY_S` (0.30) up, `BRAKE_PLAN_MPS2` (4.0) down. Measure: stopping distance from a known speed |
| slow and hesitant near every edge | edge brake too cautious | the same two the other way; `noguard` in `~/.arena/recovery` turns it off |
| recovery starts while the car is clearly inside, on the far side of a camera | roof-tag height: the pose reads up to 0.2 m outward | set `vehicle_tag_height_m` in `portable_orin_perception/config/marker_map.yaml`; or raise `ENTER_OUTSIDE_M` (0.05) |
| it is handed back to the model too close to the tape and leaves again | `REJOIN_INSIDE_M` (0.20) | raise it |
| brakes or reverses on an ordinary missed detection in the middle | `BLIND_AFTER_S` (0.30) or `FRESH_S` (0.12) too tight for this camera | raise `BLIND_AFTER_S` first |
| cars freeze although the evader is on the floor | the evader's tag is hidden over 1 s | `run_vision.py --evader-hold-max-age-s`; the dashboard says `EVADER OUT OF VIEW` |
| a car just past the tape is not tracked at all | the cameras do not see that far, or the bounds gate | `run_vision.py --bounds-margin-m`; vision telemetry `rejected.bounds` |
| 3-car model: a recovering car stalls about 0.2 m inside the tape | it is seen too rarely to finish (known, 5 below) | camera coverage of that spot; it is given up after `RECOVERY_MAX_S` (12 s) |

Do not tune the constants inside `rc16_hive_lib/`: the 3-car models were trained with them
(section 5).

## 4. Try a change without the cars

```
cd portable_n1_controller
python3 -m pytest tests/test_deadzone.py -q          # 55 tests, a few seconds
python3 tools/deadzone_sweep.py                      # blind recovery alone: 192 runs of a simple car model
python3 tools/deadzone_sweep.py --guard              # with the edge brake
python3 tools/deadzone_sweep.py --trace --throttle 1.0 --steer 0.2 --yaw 45 --delay 1
```

Reference results at this commit, to compare against after a change:

- `deadzone_sweep.py`: 108 of 192 runs leave the cameras' view, 1 does not come back (full
  throttle, 0.2 s delay), furthest 1.71 m out.
- `deadzone_sweep.py --guard`: 41 leave view, 0 do not come back, furthest 0.20 m out.

The simulated car is `SimCar` in `tests/test_deadzone.py` (0.35 s speed lag, 4.8 m/s^2 brake,
0.8 m/s^2 coast, instant steering). When the real numbers are measured, put them there: the
sweep then means something for distances, which today it does not.

`arena sim` (ARENA.md, "Rehearsal with no hardware") runs the whole pipeline with rendered tags;
its synthetic camera sees everything, so it does not exercise a lost car without a change to
`vision/synthetic.py`.

## 5. What must stay true

1. **Silence means stop.** Perception sends nothing when a camera or the process is dead, and
   before every vehicle has been seen once. Do not make `build_frame` invent a frame.
2. **A lost vehicle never carries a pose** in `x`/`y`/`heading`. An older controller must fail to
   parse it (and stall). Do not "helpfully" fill in the last pose there.
3. **The two `pose_stream.py` files are byte-identical**
   (`portable_n1_controller/pc_controller/` and `portable_orin_perception/vendored/pc_controller/`).
   Edit the controller's, copy it over; `selftest.py` in the perception bundle checks.
4. **`rc16_hive_v1.py` and `rc16_hive_lib/` are copies** of the AI Training repo's
   `rc16/deploy/hive_runtime.py` and `rc16/hive/*.py`. A change belongs there first (and a
   change to `recovery.py`, `shield.py` or `rules.py` changes what the models were trained
   with: retrain or do not touch). `python3 selftest.py` replays each model's golden vectors
   and must stay at error 0.
5. **Only commands that reached the car are remembered for a replay.** `DeadZone.apply(...,
   sending=armed and manual is None)`; arming resets the memory. The replay retraces only
   ticks in which the car was measured moving, and never the recovery's own commands.
6. **A blind car is never sent a throttle in the direction it was last moving.** The tests
   assert this; keep them.
7. **The model's own commands are changed only by the edge brake, the evader hold and a
   recovery.** `min=` and `binary` shape recovery throttles only.
8. The firmware is not part of this. Its 300 ms dead-man is the last stop.

## 6. Known weak points (from the 2026-11-19 review; not fixed)

A second set of reviewers went through the first version. What they confirmed and is fixed is
in the commit after `07cb5e4`. What is still open:

- **Never run on Linux or the cars.** `tests/test_realtime.py` and `tests/test_serial_link.py`
  need ptys. Arming now calls `loop.reset()`; if a test there counted on state surviving an arm,
  it will say so.
- **Roof-tag height is uncorrected by default.** On the far side of a camera a car at the tape
  can read 0.1 to 0.2 m past it: recovery and the edge brake then act early. Set
  `vehicle_tag_height_m` once the camera positions in the vision log look right.
- **The evader's pose can be 1.0 s old and still count as seen** (on purpose: a parked car hides
  its tag). A driven evader that leaves view is chased at its frozen pose for that second; the
  3-car runtime then goes on for 2 s more before the cars stand still (`EVADER_BLIND_S`).
- **3-car models, a car lost in the middle of the arena** (not dead-reckoned past the tape): the
  runtime keeps driving it on its frozen pose for 0.5 s (`BLIND_STALE_S`), then it is sent zero
  and coasts; there is no brake and no replay for that case. One-car models do brake and replay.
- **3-car models, a recovering car seen only now and then** never counts 3 s blind and never gets
  0.3 m inside; it creeps or stalls until `RECOVERY_MAX_S` gives it up.
- **The blind replay has no edge check.** It retraces remembered ground, plus at most 1 s
  (0.5 m) more in a straight line, never longer than the memory itself. With a memory that
  starts at the opposite tape that extra can reach the other edge.
- **`stop-only` models are zeroed on any detection gap over 0.12 s** (they were stalled after
  0.25 + 0.3 s before). Safer, but jerkier on a camera that drops frames.
- **The blind brake is planned for 10 Hz** (`BLIND_BRAKE_MAX_S` 0.8 s is in seconds now, the
  rest of the recovery counts in ticks through `dt`). Do not raise `--tick-hz` without rereading
  `EdgeRecovery`.
- **Speed after a gap.** `_Speed` reads 0 for one tick when a car reappears after 0.35 s;
  `HiveLoop.SignedSpeed` instead averages over the gap (as in training). Either can be wrong by
  the car's whole speed for that tick.

## 7. Not started

- Training the one-car model with the dead zone in the loop (episodes that go on through a
  recovery instead of ending when the car leaves). AI Training repo, `rc16/train/chase_env.py`.
- A camera-coverage model in `arena sim` (so a lost car can be rehearsed end to end).
- Measuring the real car: stopping distance from 1, 2 and 2.7 m/s, coast distance, the smallest
  throttle that moves it. Those four numbers replace most of the guesses in section 3.
