# c37_commit3_ft_g997: the commander hive for the Orin (runtime `hive_commander_v1`)

Built by `isaac_hive/scripts/build_commander_handoff.py` in AI Training. Three pursuers, one shared commander.

## Files
- `policy.onnx`: `observations` (1, 124) float32 -> `role_logits` (1, 3, 5), `settings` (1, 3, 2),
  `residual` (1, 3, 2). The observation scaler is baked in; settings and residual are post-tanh. Opset 17.
- `policy.onnx.manifest.json`: every number the port reads (arena, car, playbook constants, setting ranges), the
  observation layout, notes and known transfer gaps.
- `hive_runtime.py`: numpy + onnxruntime only. `CommanderRuntime(model_dir, seed=None)`, `reset(roles=None)`,
  `step(pursuers (3, 4), evader (4,), dt=0.1) -> (3, 2)`, `roles`. Rows are `[x, y, yaw, v_signed]` in arena metres
  (origin at the centre, yaw CCW from +x); commands are `[throttle, steer]` in [-1, 1], +steer = left.
- `golden_vectors.json`: 359 ticks over 4 episodes from fixed initial roles: inputs, observation, raw
  network outputs, roles and commands. Tolerance 1e-5.
- `metrics.json` (the run's eval numbers) and `validation.json` (everything below).

## What was checked
- **ONNX against torch:** max abs 8.6e-06 on observations recorded from the real env.
- **The port's arithmetic against the real training env**, on 9006 recorded Isaac ticks. Geometry
  is float64 and the network float32:
  - network outputs within 9.5e-06;
  - commands within 1e-4 on all but 3 ticks;
  - observations within 1e-4 on all but 26;
  - closed-loop role decisions identical on 99.1% of ticks;
  - 3 of 64 episodes split off at a near-tie (lane plateau or slot
    permutation) and then follow their own path. GPU against CPU float32 shows the same class of flip.
- **Golden replay by `hive_runtime.py`:** max abs 0.0e+00 and
  0 role mismatches. The same under numpy 2's NEP 50 promotion (emulated).

## Notes
- THROTTLE was a SPEED COMMAND in training: target speed = throttle x max_speed_mps, held by the sim car's speed loop (slew 3 m/s^2). A real ESC driven open loop is not that; expect a mismatch.
- Velocities are inputs (v_signed): the env gave true physics velocities. Yaw rate is derived inside, from the previous step's yaw over dt.
- Car order: the commander is permutation-equivariant across pursuers; outputs follow input order.
- Trained with NO latency and NO pose noise.

## Known transfer gaps: read before running on cars
- Arena: trained on a 6 m square with PHYSICAL WALLS; the real arena is 1.83 m with a tape line. The observation normalises positions by the half extents, but the playbooks use absolute distances (capture radius, standoffs 0.6-2.5 m, CLOSE ring radius) and so do car size, speed and turn radius. Expect poor driving at 1.83 m until a retrain.
- Car: sim driftking (contact 0.44 m, wheelbase 0.259 m, max steer 0.524 rad, speed scale 2.0 m/s). The real car's size is not settled.
- Evader: a simulated planner at difficulty d = 1, not a human driver.
