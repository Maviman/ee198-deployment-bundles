"""rc16: a from-scratch 1:16 RC-car pursuit-evasion stack (CAD -> Isaac Sim -> training -> evaluation).

Self-contained on purpose: nothing here imports isaac_hive, isaac_lab_hive or single_pursuer.
Every physical number lives in rc16.specs with its source; everything else derives from it.
"""
