"""Vendored model runtimes, selected by a model manifest's "runtime" key (see pc_controller/loops.py).

hive_commander_v1.py  the hive role commander (3 pursuers). An UNMODIFIED copy of
                      artifacts/deploy/c37_commit3_ft_g997_arena183/hive_runtime.py from the AI
                      Training repo (commit 68ad517; source of truth: isaac_hive/deploy/hive_runtime.py
                      there), taken 2026-10-02, sha256 434f7f6e28e2cd80... It serves both
                      models/c37_commit3_ft_g997 (6 m) and models/c37_commit3_ft_g997_arena183
                      (1.83 m, manifest length_scale 0.305). Re-copy it whole when the training side
                      changes it; never edit it here. Each model's golden_vectors.json, replayed by
                      selftest.py, pins it.
"""
