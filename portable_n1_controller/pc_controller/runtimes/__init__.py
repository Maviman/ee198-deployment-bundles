"""Vendored model runtimes, selected by a model manifest's "runtime" key (see pc_controller/loops.py).

hive_commander_v1.py  the hive role commander (3 pursuers). An UNMODIFIED copy of
                      artifacts/deploy/c38_8ft_buttons/hive_runtime.py from the AI Training repo
                      (commit 7809a0e; source of truth: isaac_hive/deploy/hive_runtime.py there),
                      taken 2026-10-05, sha256 a8338d19c8b97851... It serves all three hive models:
                      models/c38_8ft_buttons (8 ft; its manifest ships the float32 heading table
                      training used, headings_f32), models/c37_commit3_ft_g997 (6 m) and
                      models/c37_commit3_ft_g997_arena183 (1.83 m, length_scale 0.305), whose
                      manifests have no table and keep the legacy one. Re-copy it whole when the
                      training side changes it; never edit it here. Each model's golden_vectors.json,
                      replayed by selftest.py, pins it.
"""
