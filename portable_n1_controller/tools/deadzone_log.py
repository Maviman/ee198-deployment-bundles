"""What the dead zone did during a run, from the hub's run log.

    python3 tools/deadzone_log.py                      # the newest run in ~/.arena/runs
    python3 tools/deadzone_log.py ~/.arena/runs/20261119-173058/telemetry.jsonl
    python3 tools/deadzone_log.py --ticks              # every control tick of each episode, not only the changes

For each car it prints every change of state (seen / unseen, edge brake, past the tape, blind,
lost) with the time, where the car was (last seen pose), the command that was sent and how far
past the tape that pose is; then one summary line per episode: how long, how far out, how it
ended. This is the first thing to look at when a car does not come back: it shows whether the
recovery started, what it sent, and whether the car answered.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def newest_log() -> Path | None:
    runs = sorted((Path.home() / ".arena" / "runs").glob("*/telemetry.jsonl"))
    return runs[-1] if runs else None


def state_of(dz: dict, seen, i: int) -> str:
    def at(key):
        v = dz.get(key) or []
        return bool(v[i]) if i < len(v) else False
    if at("gave_up"):
        return "LOST (given up)"
    if at("blind"):
        return "BLIND RECOVERY"
    if at("recovering"):
        return "DRIVING BACK IN"
    if at("edge_brake"):
        return "edge brake"
    if seen is not None and i < len(seen) and not seen[i]:
        return "unseen"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--ticks", action="store_true")
    a = ap.parse_args()
    path = Path(a.log).expanduser() if a.log else newest_log()
    if path is None or not path.exists():
        raise SystemExit("no run log found (give the path of a telemetry.jsonl)")
    print(f"{path}")
    half, last_state, episode, n_ctl, armed_ticks = (1.2192, 1.2192), {}, {}, 0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("src") == "vision" and m.get("arena"):
                half = tuple(m["arena"][:2])
                continue
            if m.get("src") != "control":
                continue
            n_ctl += 1
            dz, poses = m.get("deadzone"), m.get("poses")
            if dz is None:
                continue
            armed_ticks += bool(m.get("armed"))
            stamp = time.strftime("%H:%M:%S", time.localtime(m.get("wall", 0))) + f".{int((m.get('wall', 0) % 1) * 10)}"
            for i, q in enumerate((poses or {}).get("pursuers") or []):
                st = state_of(dz, m.get("seen"), i)
                out = math.hypot(max(abs(q[0]) - half[0], 0.0), max(abs(q[1]) - half[1], 0.0))
                cmd = (m.get("cmd") or [0.0] * (2 * i + 2))[2 * i:2 * i + 2]
                where = (f"last seen ({q[0]:+.2f}, {q[1]:+.2f}) {out:.2f} m past the tape, "
                         f"unseen {(dz.get('unseen_s') or [0] * (i + 1))[i]:.1f} s, "
                         f"cmd ({cmd[0]:+.2f}, {cmd[1]:+.2f}) {'ARMED' if m.get('armed') else 'disarmed'}")
                changed = st != last_state.get(i, "ok")
                if changed or (a.ticks and st != "ok"):
                    print(f"  {stamp} car {i}: {st:16s} {where}")
                ep = episode.get(i)
                if st != "ok" and ep is None:
                    episode[i] = {"t0": m.get("wall", 0), "out": out, "blind": 0}
                elif st != "ok":
                    ep["out"] = max(ep["out"], out)
                    ep["blind"] += st == "BLIND RECOVERY"
                elif ep is not None:
                    print(f"  {stamp} car {i}: back to the model after {m.get('wall', 0) - ep['t0']:.1f} s; furthest "
                          f"SEEN {ep['out']:.2f} m past the tape; blind on {ep['blind']} logged ticks")
                    episode[i] = None
                last_state[i] = st
            if dz.get("evader_hold") != last_state.get("ev", False):
                last_state["ev"] = dz.get("evader_hold")
                print(f"  {stamp} evader: {'OUT OF VIEW, cars standing still' if dz.get('evader_hold') else 'back'}")
    if n_ctl and not last_state:
        print("  the controller in this log does not report a dead zone (older code), or no pose ever arrived")
    print(f"{n_ctl} controller records, {armed_ticks} of them armed; taped half-size {half[0]:.3f} x {half[1]:.3f} m")


if __name__ == "__main__":
    main()
