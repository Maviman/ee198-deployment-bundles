"""Single-pursuer foundation.

A deliberately small environment + single-agent PPO setup for the *current* project
scope: prove that one trained pursuer can capture one evader under a central
controller, before scaling back up to the three-car hive.

The three-car MAPPO stack (``foundation_demo``/``step2_training``) is intentionally
left frozen as the future scope; nothing here imports or modifies it. The only
shared dependency is the neutral ``vehicle_dynamics`` module (the validated
longitudinal model), which belongs to neither stack.
"""

from .env import SinglePursuerEnv

__all__ = ["SinglePursuerEnv"]
