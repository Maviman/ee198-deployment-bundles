"""Offline check of the dead-zone recovery (pc_controller/deadzone.py): no cars, no cameras.

    python3 tools/deadzone_sweep.py                 # the sweep: how often a car that leaves view comes back
    python3 tools/deadzone_sweep.py --trace --throttle 1.0 --steer 0.2 --yaw 45 --delay 1
    python3 tools/deadzone_sweep.py --guard         # the same with the edge brake on

A simple model of the car (tests/test_deadzone.py SimCar: bicycle steering, speed following the
throttle with a 0.35 s lag, 4.8 m/s^2 brake on reverse throttle, coast at zero) is driven by a
"model" that holds one throttle and one steering value, under cameras that see only
`--view-band` metres past the tape. Use it after changing a constant in deadzone.py: run the
sweep before and after and compare the two summary lines. --trace prints one run tick by tick.

It checks BEHAVIOUR (does the car come back, how far out does it get in this model). The real
car brakes, coasts and creeps differently: set SimCar's numbers from a measurement before
reading distances off it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_deadzone as T  # noqa: E402
from pc_controller import deadzone as DZ  # noqa: E402


def one(throttle, steer, yaw_deg, delay, *, guard, binary, min_throttle, seconds=10.0, start=(0.3, -0.2)):
    car = T.SimCar(start[0], start[1], math.radians(yaw_deg), binary_drive=binary)
    dead = DZ.DeadZone(1, spec=T.SPEC, guard=guard, binary_drive=binary, min_throttle=min_throttle)
    log, msgs = T.run(car, lambda _t: [throttle, steer], seconds=seconds, dead=dead, delay_ticks=delay)
    lost = [i for i, r in enumerate(log) if not r["in_view"]]
    if not lost:
        return log, msgs, None
    back = next((i for i in range(lost[0] + 1, len(log)) if log[i]["in_view"]), None)
    seg = log[lost[0]:back]
    return log, msgs, {"out_m": max(r["outside"] for r in seg), "blind_s": 0.1 * len(seg),
                       "back": back is not None and not any(r["gave_up"] for r in seg)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guard", action="store_true", help="edge brake on (default: off, the blind recovery alone)")
    ap.add_argument("--binary", action="store_true", help="a `drive binary` car, recovery sent as pulses")
    ap.add_argument("--min-throttle", type=float, default=0.0)
    ap.add_argument("--view-band", type=float, default=T.VIEW_BAND_M, help="m past the tape the cameras still see")
    ap.add_argument("--trace", action="store_true", help="print ONE run tick by tick (uses the four values below)")
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--steer", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0, help="start heading, degrees")
    ap.add_argument("--delay", type=int, default=1, help="command delay, control ticks")
    a = ap.parse_args()
    T.VIEW_BAND_M = a.view_band
    kw = dict(guard=a.guard, binary=a.binary, min_throttle=a.min_throttle)
    if a.trace:
        log, msgs, res = one(a.throttle, a.steer, a.yaw, a.delay, **kw)
        for r in log:
            print(f"{r['t']:5.1f}  x={r['x']:+.2f} y={r['y']:+.2f} v={r['v']:+.2f}  out={r['outside']:.2f}  "
                  f"{'seen ' if r['seen'] else 'UNSEEN'} {'REC' if r['recovering'] else '   '} "
                  f"{'BLIND' if r['blind'] else '     '} {'GAVE-UP' if r['gave_up'] else ''}  "
                  f"cmd=({r['cmd'][0]:+.2f}, {r['cmd'][1]:+.2f})")
        print("\n".join(msgs))
        print(res or "never left the cameras' view")
        return
    n = bad = 0
    worst_out = worst_blind = 0.0
    for thr in (0.3, 0.6, 1.0, -0.6):
        for steer in (-0.6, -0.2, 0.2, 0.6):
            for yaw in (0, 45, 90, 200):
                for delay in (0, 1, 2):
                    _log, _msgs, res = one(thr, steer, yaw, delay, **kw)
                    if res is None:
                        continue
                    n += 1
                    worst_out, worst_blind = max(worst_out, res["out_m"]), max(worst_blind, res["blind_s"])
                    if not res["back"]:
                        bad += 1
                        print(f"NOT RECOVERED  throttle {thr:+.1f} steer {steer:+.1f} yaw {yaw} delay {delay}: "
                              f"{res['out_m']:.2f} m out")
    print(f"{n} runs left the cameras' view (of 192); {bad} did not come back; furthest out {worst_out:.2f} m; "
          f"longest out of view {worst_blind:.1f} s   [guard {'on' if a.guard else 'off'}, view band {a.view_band} m]")


if __name__ == "__main__":
    main()
