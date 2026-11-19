# rc16_hive_h2: three pursuers, playbook commander, MotorDayz MD Elite, 8 x 8 ft arena

Model A (h2): the third car backs up the catch; a win needs at least two cars closed in. Sim win 50 % vs 36 % for the best single play.

Runtime `rc16_hive_v1` (`pc_controller/runtimes/rc16_hive_v1.py` + `rc16_hive_lib/`). **Needs torch (CPU) on the controller** besides onnxruntime: `pip install torch`. Every 0.5 s the commander picks one of 14 plays (CHASE_FLANK, AMBUSH, HERD, FEINT, NET, SQUEEZE, REGROUP); every 0.1 s the play gives each car a target and the pawn drivers, distance limiter and out-of-bounds recovery turn that into throttle and steering. The loop shows the running play as each car's role.

## Win rule it was trained for

checkmate: every escape arc of the evader (18 arcs of 0.40 m) is blocked within 0.15 m by a pursuer body or by the tape (leaving the square is not a legal move), at least 2 pursuers have their box within that distance of the evader's box, held 1.0 s while the evader tries to move. The runtime does not judge wins; this is what it was trained for.

## Evaluation (simulation, 256 episodes, evader difficulty 0.5, camera emulator, half with 0.1 s delay)

| run | cars that must close in | win (95 %) | foul | escape | time to win (s) | rams/ep | wins by cars near [0,1,2,3] |
|---|---|---|---|---|---|---|---|
| hive_h2_commander | 2 | 0.504 (0.443 to 0.565) | 0.000 | 0.000 | 11.6 | 0.12 | [0, 0, 86, 43] |
| hive_h2_commander_nomem | 2 | 0.480 (0.420 to 0.542) | 0.000 | 0.000 | 11.6 | 0.14 | [0, 0, 73, 50] |

## Before the first real run

- Same car set-up as `rc16_n1_md_elite`: perception arena 8 ft, origin at the centre; firmware `limit` so a full command reaches 2.7 m/s (start at 0.37); the car must accept reverse while rolling (that is the brake); servo end points at 30 deg.
- Pursuer order: the loop's p0, p1, p2 are interchangeable (the plays match cars to slots), but all three must be tracked. The evader is any car or a human-driven one.
- Start with `deploy_throttle_limit` 0.5 in the manifest and the cars wheels-off, then on the floor.
- Pose stalls: the loop's dead-man stop applies.

ONNX verified against torch to 1.1e-05. Golden replay: 630 env ticks, commands matched to 3.8e-06, same play on every tick.
