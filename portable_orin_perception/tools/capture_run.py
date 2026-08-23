"""Record ONE live perception run as a machine-comparable JSON file.

`probe_orin_gpu.py` measures the board in isolation. This measures the pipeline
actually running: per-node CPU, published topic rates, and — the number that
decides pass/fail — how many pose frames the bridge had to suppress because a
vehicle went stale.

    ./tools/capture_run.py --label camA        --seconds 60 -- 127.0.0.1
    ./tools/capture_run.py --label camB-gst    --seconds 60 -- 127.0.0.1 camera_backend:=gst
    ./tools/capture_run.py --label 720p15      --seconds 120 -- 127.0.0.1 \
        image_width:=1280 image_height:=720 framerate:=15

Everything after `--` is passed straight to run_perception.sh. Output lands in
--outdir (default ~/tuning) as <label>.json plus <label>.log, and
`--compare A.json B.json` prints the delta between two runs.

Why per-node CPU is read from /proc rather than `ps -o %cpu`: ps reports
utilisation averaged over the process's whole LIFETIME, so a loop sampling it
measures start-up transients forever and never shows steady state. Reading
utime+stime deltas out of /proc/<pid>/stat over a fixed interval gives true
instantaneous CPU, which is what a budget conversation needs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
WATCHED_NODES = ["aruco_detector", "pose_bridge", "usb_cam_node_exe", "gst_camera"]
TOPICS = ["/hive/vehicle_poses", "/image_raw"]


def find_pid(pattern: str) -> int | None:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = [int(p) for p in out.stdout.split() if p.isdigit()]
    # pgrep -f matches our own command line too; drop ourselves.
    pids = [p for p in pids if p != os.getpid()]
    return pids[0] if pids else None


def read_cpu_jiffies(pid: int) -> int | None:
    """utime + stime for a pid, in clock ticks. None if the process is gone."""
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (OSError, IndexError):
        return None
    # After the "comm) " split, field 0 is state; utime/stime are fields 11/12.
    try:
        return int(parts[11]) + int(parts[12])
    except (IndexError, ValueError):
        return None


def read_rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def sample_cpu(pids: dict[str, int], seconds: float) -> dict[str, dict]:
    """True instantaneous CPU% per node over `seconds`, plus peak RSS."""
    start = {n: read_cpu_jiffies(p) for n, p in pids.items()}
    t0 = time.monotonic()
    peak_rss = {n: (read_rss_mb(p) or 0.0) for n, p in pids.items()}
    # Sample RSS along the way; CPU only needs endpoints.
    while time.monotonic() - t0 < seconds:
        time.sleep(min(2.0, seconds / 10.0))
        for n, p in pids.items():
            r = read_rss_mb(p)
            if r and r > peak_rss[n]:
                peak_rss[n] = r
    elapsed = time.monotonic() - t0
    out = {}
    for name, pid in pids.items():
        end = read_cpu_jiffies(pid)
        if start[name] is None or end is None:
            out[name] = {"cpu_percent": None, "note": "process exited during sampling"}
            continue
        cpu_s = (end - start[name]) / CLK_TCK
        out[name] = {
            "cpu_percent": round(100.0 * cpu_s / elapsed, 1),
            "peak_rss_mb": round(peak_rss[name], 1),
            "pid": pid,
        }
    return out


def topic_rate(topic: str, window_s: float = 10.0) -> float | None:
    """Average publish rate from `ros2 topic hz`, or None if nothing published."""
    try:
        proc = subprocess.run(
            ["ros2", "topic", "hz", topic, "--window", "50"],
            capture_output=True, text=True, timeout=window_s)
        text = proc.stdout
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
    except (OSError, FileNotFoundError):
        return None
    rates = [float(m) for m in re.findall(r"average rate:\s*([0-9.]+)", text)]
    return round(sum(rates) / len(rates), 2) if rates else None


def parse_log(path: Path) -> dict:
    """Pull the facts that decide pass/fail out of the pipeline's own log."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    decoder = None
    m = re.search(r"decoding with (\w+)", text)
    if m:
        decoder = m.group(1)
    return {
        "suppressed_frames": len(re.findall(r"frame suppressed", text)),
        "frames_sent": max((int(x) for x in re.findall(r"(\d+) frames sent", text)), default=0),
        "decoder": decoder,
        "hardware_jpeg": ("HARDWARE NVJPG" in text) if decoder else None,
        "errors": sorted(set(re.findall(r"\[ERROR\].*", text)))[:10],
        "camera_read_failures": len(re.findall(r"camera read failed", text)),
    }


def run(args) -> int:
    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{args.label}.log"
    json_path = outdir / f"{args.label}.json"

    bundle = Path(__file__).resolve().parent.parent
    script = bundle / "run_perception.sh"
    if not script.exists():
        sys.exit(f"{script} not found — run this from inside the bundle")
    if not args.launch_args:
        sys.exit("no launch args: pass them after `--`, e.g.  -- 127.0.0.1 camera_backend:=gst")

    cmd = ["bash", str(script), *args.launch_args]
    print(f"[capture] {' '.join(cmd)}")
    print(f"[capture] log -> {log_path}")

    with open(log_path, "w", encoding="utf-8") as logf:
        # New session so we can kill the whole process group: run_perception.sh
        # spawns ros2 launch, which spawns the nodes.
        proc = subprocess.Popen(cmd, cwd=str(bundle), stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print(f"[capture] warming up {args.warmup}s...")
            time.sleep(args.warmup)
            if proc.poll() is not None:
                print(f"[capture] pipeline exited early (rc={proc.returncode}) — see {log_path}")
                return _write(json_path, args, parse_log(log_path), {}, {}, early_exit=True)

            pids = {n: p for n in WATCHED_NODES if (p := find_pid(n)) is not None}
            print(f"[capture] watching: {pids or 'NO NODES FOUND'}")

            rates = {t: topic_rate(t) for t in TOPICS}
            print(f"[capture] topic rates: {rates}")

            print(f"[capture] sampling CPU for {args.seconds}s...")
            cpu = sample_cpu(pids, args.seconds) if pids else {}
        finally:
            print("[capture] stopping pipeline...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(1.0)

    return _write(json_path, args, parse_log(log_path), cpu, rates)


def _write(json_path, args, log_facts, cpu, rates, early_exit=False) -> int:
    record = {
        "label": args.label,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "launch_args": args.launch_args,
        "seconds": args.seconds,
        "early_exit": early_exit,
        "log": log_facts,
        "cpu": cpu,
        "topic_rates_hz": rates,
    }
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n[capture] wrote {json_path}")

    print(f"\n{'=' * 60}\n{args.label}\n{'=' * 60}")
    for node, stats in cpu.items():
        pct = stats.get("cpu_percent")
        print(f"  {node:22s} {pct if pct is not None else '?':>6}% CPU"
              f"   RSS {stats.get('peak_rss_mb', '?')} MB")
    for topic, hz in rates.items():
        print(f"  {topic:22s} {hz if hz is not None else 'no data':>6} Hz")
    sup = log_facts.get("suppressed_frames", 0)
    verdict = "PASS" if sup == 0 else "FAIL — pipeline is over budget or dropping tags"
    print(f"  {'suppressed frames':22s} {sup:>6}   {verdict}")
    if log_facts.get("decoder"):
        hw = log_facts.get("hardware_jpeg")
        print(f"  {'jpeg decoder':22s} {log_facts['decoder']:>6}"
              f"   {'(HARDWARE)' if hw else '(CPU fallback)'}")
    if log_facts.get("errors"):
        print("  errors:")
        for e in log_facts["errors"]:
            print(f"    {e[:100]}")
    return 0 if sup == 0 and not early_exit else 1


def compare(a_path: str, b_path: str) -> None:
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    la, lb = a.get("label", "A"), b.get("label", "B")
    print(f"\n{'=' * 76}\nCOMPARE  {la}  ->  {lb}\n{'=' * 76}")
    print(f"{'metric':30s} {la[:16]:>16} {lb[:16]:>16} {'change':>10}")
    print("-" * 76)

    def row(name, va, vb, lower_better=True):
        fa = va if isinstance(va, (int, float)) else None
        fb = vb if isinstance(vb, (int, float)) else None
        if fa is None or fb is None:
            print(f"{name:30s} {str(va):>16} {str(vb):>16} {'-':>10}")
            return
        # Zero is not a missing value. For suppressed frames — the metric that
        # decides pass/fail — going to 0 is the whole point, and a ratio can't
        # express it, so say so in words instead of printing a dash.
        if fa == 0 or fb == 0:
            if fa == fb:
                note = "same"
            elif (fb == 0) == lower_better:
                note = "FIXED" if lower_better else "LOST"
            else:
                note = "REGRESSED" if lower_better else "GAINED"
            print(f"{name:30s} {fa:>16.1f} {fb:>16.1f} {note:>10}")
            return
        improved = (fb < fa) if lower_better else (fb > fa)
        factor = (fa / fb) if lower_better else (fb / fa)
        if factor < 1.0:
            factor = 1.0 / factor
        print(f"{name:30s} {fa:>16.1f} {fb:>16.1f} "
              f"{f'{factor:.2f}x ' + ('better' if improved else 'worse'):>10}")

    for node in sorted(set(a.get("cpu", {})) | set(b.get("cpu", {}))):
        row(f"{node} %CPU", a.get("cpu", {}).get(node, {}).get("cpu_percent"),
            b.get("cpu", {}).get(node, {}).get("cpu_percent"))
    for topic in sorted(set(a.get("topic_rates_hz", {})) | set(b.get("topic_rates_hz", {}))):
        row(f"{topic} Hz", a.get("topic_rates_hz", {}).get(topic),
            b.get("topic_rates_hz", {}).get(topic), lower_better=False)
    row("suppressed frames", a.get("log", {}).get("suppressed_frames"),
        b.get("log", {}).get("suppressed_frames"))
    print(f"{'jpeg decoder':30s} {str(a.get('log', {}).get('decoder')):>16} "
          f"{str(b.get('log', {}).get('decoder')):>16}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="name for this run; becomes <label>.json / <label>.log")
    ap.add_argument("--seconds", type=float, default=60.0, help="CPU sampling window")
    ap.add_argument("--warmup", type=float, default=12.0,
                    help="settle time before measuring (nodes need to start and the "
                         "camera to stabilise)")
    ap.add_argument("--outdir", default="~/tuning")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("launch_args", nargs="*",
                    help="args for run_perception.sh, after a `--` separator")
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.label:
        ap.error("--label is required (or use --compare)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
