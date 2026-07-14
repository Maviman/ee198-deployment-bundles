"""Central controller runtime: perception -> observation -> policy -> commands.

This package is the deployment-side glue the master roadmap calls "Layer 3" — it
turns a stream of vehicle poses (from the team's overhead ArUco perception, per
``docs/perception_interface_contract.md``) into the exact observation vector a
trained ``single_pursuer`` policy expects, runs inference, and hands back normalized
commands per pursuer. It has NO hardware dependency and is fully testable against
the 2D simulator (see ``controller_runtime.sil_harness``) before any car exists.

Modules:
  pose_types        -- the raw perception input type and a small pose-history buffer
                        that estimates velocity by finite difference.
  observation_adapter -- builds the policy's observation vector from raw poses by
                        DELEGATING to SinglePursuerEnv._observation() (never
                        reimplements the feature layout, to make drift impossible).
  action_history_buffer -- runtime twin of single_pursuer.action_history's wrapper,
                        for policies trained with action-history hardening.
  inference         -- loads a checkpoint (auto-detecting N/cams/action-history from
                        its metrics.json, same pattern as playback.py/robustness_eval.py)
                        and runs the deterministic policy.
  latency           -- per-hop timestamp tracking (the ≤0.1s budget from Phase 2).
  runtime_loop       -- ties the above into one tick() call.
  sil_harness        -- Software-In-the-Loop: drives the loop from a SinglePursuerEnv
                        instead of hardware, and checks capture-rate parity with
                        single_pursuer.train.evaluate().
"""
