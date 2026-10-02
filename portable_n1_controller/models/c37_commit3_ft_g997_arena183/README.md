# c37_commit3_ft_g997_arena183: c37_commit3_ft_g997 scaled to a 1.83 m arena (runtime `hive_commander_v1`)

The same `policy.onnx` as `c37_commit3_ft_g997`, byte for byte. The manifest multiplies every length and speed by 0.3050, with time unchanged, and sets `length_scale` so `hive_runtime.py` scales the playbooks' hard-coded lengths too.

- Arena half extent 0.915 m; speed scale 0.610 m/s; CLOSE ring 0.152 m.
- **Equivalence, checked:** on the source goldens with positions and speeds scaled, this runtime reproduces the source's observations, network outputs and commands within 0.0e+00, with 0 role mismatches.
- **Backward compatible:** this `hive_runtime.py` replays the source's goldens within 0.0e+00 (length_scale defaults to 1).

## Known transfer gaps
- Not retrained: the network is the 6 m model's. Scaling keeps the geometry consistent (targets stay inside the tape), but the real car's turning circle, actuation (on/off buttons or ESC) and grip do not scale.
- Contact distances scale with the trained DriftKing (0.44 m x s = 0.134 m), not with the real car, whose size is not settled.
- Evader: a simulated planner at difficulty d = 1, not a human driver.
