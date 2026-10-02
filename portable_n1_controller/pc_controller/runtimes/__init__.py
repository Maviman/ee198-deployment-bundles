"""Vendored model runtimes, selected by a model manifest's "runtime" key (see pc_controller/loops.py).

hive_commander_v1.py  the hive role commander (3 pursuers). An UNMODIFIED copy of
                      artifacts/deploy/c37_commit3_ft_g997/hive_runtime.py from the AI Training
                      repo (source of truth: isaac_hive/deploy/hive_runtime.py there), taken
                      2026-10-02, sha256 46993ee1a2ac1643... Re-copy it whole when the
                      training side changes it; never edit it here. The model's
                      golden_vectors.json, replayed by selftest.py, pins it.
"""
