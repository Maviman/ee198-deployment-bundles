# c38_8ft_buttons: the commander hive for the Orin (runtime `hive_commander_v1`)

Built by `isaac_hive/scripts/build_commander_handoff.py` in AI Training. Three pursuers, one shared commander.

## Files
- `policy.onnx`: `observations` (1, 124) float32 -> `role_logits` (1, 3, 5), `settings` (1, 3, 2),
  `residual` (1, 3, 2). The observation scaler is baked in; settings and residual are post-tanh. Opset 17.
- `policy.onnx.manifest.json`: every number the port reads (arena, car, playbook constants, setting ranges), the
  observation layout, notes and known transfer gaps.
- `hive_runtime.py`: numpy + onnxruntime only. `CommanderRuntime(model_dir, seed=None)`, `reset(roles=None)`,
  `step(pursuers (3, 4), evader (4,), dt=0.1) -> (3, 2)`, `roles`. Rows are `[x, y, yaw, v_signed]` in arena metres
  (origin at the centre, yaw CCW from +x); commands are `[throttle, steer]` in [-1, 1], +steer = left.
- `golden_vectors.json`: 188 ticks over 4 episodes from fixed initial roles: inputs, observation, raw
  network outputs, roles and commands. Tolerance 1e-5.
- `metrics.json` (the run's eval numbers) and `validation.json` (everything below).

## What was checked
- **ONNX against torch:** max abs 6.7e-06 on observations recorded from the real env.
- **The port's arithmetic against the real training env**, on 3771 recorded Isaac ticks. Geometry
  is float64 and the network float32:
  - network outputs within 6.7e-06;
  - commands within 1e-4 on all but 0 ticks;
  - observations within 1e-4 on all but 0;
  - closed-loop role decisions identical on 96.7% of ticks;
  - 5 of 64 episodes split off at a near-tie (lane plateau or slot
    permutation) and then follow their own path. GPU against CPU float32 shows the same class of flip.
- **Golden replay by `hive_runtime.py`:** max abs 0.0e+00 and
  0 role mismatches. The same under numpy 2's NEP 50 promotion (emulated).

## Notes
- THROTTLE was the ESC STICK in training (open loop, no speed loop): +1 full forward, back = brake while rolling / reverse from a stop. Through the remote's buttons it becomes FWD/BACK presses (mod, limit 0.6).
- Velocities are inputs (v_signed): the env gave true physics velocities. Yaw rate is derived inside, from the previous step's yaw over dt.
- Car order: the commander is permutation-equivariant across pursuers; outputs follow input order.
- Trained with NO latency and NO pose noise.

## Known transfer gaps: read before running on cars
- Arena: this manifest is for a 2.438 m square; the model trained with PHYSICAL WALLS (the real arena has a tape line). Role lengths follow the arena size (setting_scale 0.4064); car size, speed and turning are absolute.
- Car: sim driftking (contact 0.44 m, wheelbase 0.259 m, max steer 0.524 rad, speed scale 2.0 m/s). The real car's size is not settled.
- Evader: a simulated planner at difficulty d = 1, not a human driver.
