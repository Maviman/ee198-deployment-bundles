#!/usr/bin/env python3
"""arena: one command per step of an arena session.

  first time     ./arena init --vision <user>@<host> --control <user>@<host>
                 ./arena setup          install deps on both Orins
                 ./arena tune           pin clocks, WiFi power-save off
  every session  ./arena sync           both Orins onto this checkout's commit, selftests
                 ./arena scan           calibrate + check the arena from the camera
                 ./arena cars           which cars are on the WiFi, and their CAR_INDEX
                 ./arena up             start vision + controller + dashboard (cars DISARMED)
                 ./arena go             arm: the policy drives the cars
                 ./arena monitor        live terminal dashboard (browser: the URL `up` prints)
                 ./arena halt           E-stop every car (services keep running; `go` re-arms)
                 ./arena stop           E-stop + stop everything
  cars           ./arena flash --ota    build the firmware and push it to every car over WiFi
                 ./arena flash --usb    ... or to one car on a USB cable
  anywhere       ./arena sim            the whole stack on this machine with a simulated
                                        arena and cars, for rehearsal (no hardware)
  diagnosis      ./arena status | logs <vision|controller|hub> [-f]

Runs from the dev PC, from either Orin, or from Windows (arena.cmd). Commands
for a remote Orin go over SSH; commands for the machine you are on run
directly. Topology: deploy/arena.conf + deploy/arena.local.conf (from `init`).
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy"
CONF, LOCAL_CONF = DEPLOY / "arena.conf", DEPLOY / "arena.local.conf"
IS_WIN = os.name == "nt"

# ------------------------------------------------------------------ output
if IS_WIN:
    os.system("")   # enables ANSI escape processing in the Windows console
try:
    sys.stdout.reconfigure(errors="replace")      # never die on a console that lacks a glyph
except AttributeError:
    pass
_TTY = sys.stdout.isatty()
_UTF = (sys.stdout.encoding or "").lower().replace("-", "").startswith("utf")
BLOCK, DOT, DASH = ("█", "·", "–") if _UTF else ("#", ".", "-")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def ok(s): print(_c("32", "  ok   ") + s)
def warn(s): print(_c("33", "  warn ") + s)
def bad(s): print(_c("31", "  FAIL ") + s)
def step(s): print(_c("1", f"\n== {s}"))
def die(s, code=1):
    print(_c("31", f"arena: {s}"), file=sys.stderr)
    raise SystemExit(code)


# ------------------------------------------------------------------ config
@dataclass
class Role:
    name: str
    host: str
    user: str
    repo: str
    python: str
    tools_python: str = "python3"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def local(self) -> bool:
        h = self.host.lower().removesuffix(".local")
        return h in ("local", "localhost", "127.0.0.1") or h in _my_names()

    @property
    def placeholder(self) -> bool:
        return self.host in ("orin-vision.local", "orin-control.local") and not self.local


_NAMES: set[str] | None = None


def _my_names() -> set[str]:
    """This machine's hostname and IPv4 addresses, so a role configured by IP
    is still recognised as 'here' (and never SSHes to itself)."""
    global _NAMES
    if _NAMES is None:
        me = socket.gethostname().lower()
        names = {me, me.removesuffix(".local")}
        try:
            names.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        try:
            out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
            names.update(out.split())
        except (OSError, subprocess.SubprocessError):
            pass
        _NAMES = names
    return _NAMES


class Config:
    def __init__(self) -> None:
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
        cp.read([CONF, LOCAL_CONF], encoding="utf-8")
        self.cp = cp
        c = cp["control"]
        self.vision = Role("vision", cp["vision"]["host"], cp["vision"].get("user", ""),
                           cp["vision"]["repo"], cp["vision"]["python"])
        self.control = Role("control", c["host"], c.get("user", ""), c["repo"], c["python"],
                            c.get("tools_python", "python3"))
        self.model = cp["fleet"]["model"]
        self.esp = cp["fleet"]["esp"]
        n = cp["net"]
        self.pose_port, self.tel_port = n.getint("pose_port"), n.getint("telemetry_port")
        self.ctl_port, self.preview_port = n.getint("control_port"), n.getint("preview_port")
        self.dash_port = n.getint("dashboard_port")
        self.control_addr_from_vision = n.get("control_addr_from_vision", "auto")
        self.vision_addr_from_control = n.get("vision_addr_from_control", "auto")

    @property
    def single(self) -> bool:
        return self.vision.host == self.control.host or (self.vision.local and self.control.local)

    def roles(self) -> list[Role]:
        return [self.control] if self.single else [self.vision, self.control]

    def require_topology(self) -> None:
        if self.vision.placeholder or self.control.placeholder:
            die("the Orins are not configured yet. Run:\n"
                "  ./arena init --vision <user>@<vision-orin> --control <user>@<control-orin>\n"
                "  ./arena init --single <user>@<orin>          (one Orin doing both)")


# ------------------------------------------------------------------ remote execution
SVC_LIB = r'''
A="$HOME/.arena"; mkdir -p "$A/run" "$A/logs"
# A pid file survives a reboot, and after one its number can belong to anything.
# Only trust it if that process really is `supervise.sh <name>`.
svc_pid() {
  local p; p=$(cat "$A/run/$1.pid" 2>/dev/null) || return 1
  tr '\0' ' ' 2>/dev/null < "/proc/$p/cmdline" | grep -q "supervise.sh $1 " && echo "$p"
}
svc_start() {  # name dir cmd...
  local name=$1 dir=$2; shift 2
  if pid=$(svc_pid "$name"); then echo "$name already running (pid $pid)"; return 0; fi
  [ -f "$A/logs/$name.log" ] && mv -f "$A/logs/$name.log" "$A/logs/$name.prev.log"
  setsid nohup bash deploy/supervise.sh "$name" "$dir" "$@" > "$A/logs/$name.log" 2>&1 < /dev/null &
  echo $! > "$A/run/$name.pid"; echo "$name started (pid $!)"
}
svc_stop() {
  local name=$1 pid
  pid=$(svc_pid "$name") || { rm -f "$A/run/$name.pid"; return 0; }
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null
  rm -f "$A/run/$name.pid"; echo "$name stopped"
}
'''


def run_on(role: Role, script: str, *, tty: bool = False, capture: bool = False,
           check: bool = False, timeout: float | None = None, lib: bool = False) -> subprocess.CompletedProcess:
    """Run a bash script inside ``role``'s repo checkout: locally, or over SSH."""
    body = (SVC_LIB if lib else "") + f"cd {role.repo} || exit 97\n" + script
    if role.local:
        cmd = ["bash", "-c", body]
    else:
        cmd = ["ssh", "-o", "ConnectTimeout=6", "-o", "ServerAliveInterval=5"]
        cmd += ["-t"] if tty else ["-o", "BatchMode=yes"]
        cmd += [role.target, "bash -c " + shlex.quote(body)]
    try:
        p = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout,
                           stdin=None if tty else subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        die(f"{role.name} ({role.target}): timed out")
    except FileNotFoundError as exc:
        die(f"cannot run {cmd[0]}: {exc}")
    if p.returncode == 97:
        die(f"{role.name} ({role.target}): no checkout at {role.repo}. `arena setup` clones it.")
    if p.returncode == 255 and not role.local:
        die(f"cannot SSH to {role.name} ({role.target}). Check it is on, and that key login works: "
            f"ssh {role.target} true   (set up keys with: ssh-copy-id {role.target})")
    if check and p.returncode != 0:
        die(f"{role.name}: command failed (exit {p.returncode})")
    return p


def local_head() -> tuple[str, str, bool]:
    def g(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "--short", "HEAD"), g("rev-parse", "--abbrev-ref", "HEAD"), bool(g("status", "--porcelain"))


def hub_url(cfg: Config) -> str:
    host = "127.0.0.1" if cfg.control.local else cfg.control.host
    return f"http://{host}:{cfg.dash_port}"


def fetch_state(url: str, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ commands
def cmd_init(args, _cfg) -> None:
    def split(spec):
        user, _, host = spec.rpartition("@")
        return user, host
    cp = configparser.ConfigParser()
    if LOCAL_CONF.exists():
        cp.read(LOCAL_CONF, encoding="utf-8")
    vis, ctl = (args.single, args.single) if args.single else (args.vision, args.control)
    if not (vis and ctl):
        die("give --vision and --control, or --single")
    for section, spec in (("vision", vis), ("control", ctl)):
        user, host = split(spec)
        if not cp.has_section(section):
            cp.add_section(section)
        cp[section]["host"] = host
        cp[section]["user"] = user
        if args.repo:
            cp[section]["repo"] = args.repo
    for key, value in (("control_addr_from_vision", args.link_ip),
                       ("vision_addr_from_control", args.vision_link_ip)):
        if value:
            if not cp.has_section("net"):
                cp.add_section("net")
            cp["net"][key] = value
    with open(LOCAL_CONF, "w", encoding="utf-8") as fh:
        fh.write("# Site overrides for deploy/arena.conf (gitignored). Written by `arena init`.\n")
        cp.write(fh)
    print(f"wrote {LOCAL_CONF.relative_to(REPO)}")
    cfg = Config()
    for r in cfg.roles():
        print(f"  {r.name:8s} {r.target}  repo {r.repo}  {'(this machine)' if r.local else ''}")
    print("\nNext: make sure key-based SSH works to each Orin (ssh-copy-id <user>@<host>), then:\n"
          "  ./arena setup     (first time)     ./arena status")


def cmd_status(args, cfg: Config) -> None:
    cfg.require_topology()
    head, branch, dirty = local_head()
    print(f"this checkout: {branch} @ {head}{' (+ uncommitted changes)' if dirty else ''}")
    probe = r'''
echo "host=$(hostname)"
echo "git=$(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null) $( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty)"
for n in vision controller hub; do if pid=$(svc_pid $n); then echo "svc_$n=running"; else echo "svc_$n=stopped"; fi; done
c=/sys/devices/system/cpu/cpu0/cpufreq
[ -r $c/scaling_cur_freq ] && echo "clock=$(( $(cat $c/scaling_cur_freq)/1000 ))/$(( $(cat $c/cpuinfo_max_freq)/1000 )) MHz min=$(( $(cat $c/scaling_min_freq)/1000 ))"
t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1); [ -n "$t" ] && echo "temp=$((t/1000)) C"
echo "cams=$(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
[ -f portable_orin_perception/config/arena_homography.yaml ] && echo "calib=$(grep -m1 calibrated_at portable_orin_perception/config/arena_homography.yaml | cut -d"'" -f2) $(grep -m1 -A1 image_size portable_orin_perception/config/arena_homography.yaml | tail -1 | tr -d ' -')"
w=$(iw dev 2>/dev/null | awk '/Interface/{print $2; exit}'); [ -n "$w" ] && echo "wifi_ps=$w $(iw dev $w get power_save 2>/dev/null | awk '{print $3}')"
'''
    for r in cfg.roles():
        step(f"{r.name}: {r.target}{' (this machine)' if r.local else ''}")
        p = run_on(r, probe, capture=True, lib=True, timeout=20)
        info = dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)
        rev = info.get("git", "?")
        (ok if rev.split()[0] == head else warn)(f"code {rev}" + ("" if rev.split()[0] == head else
                                                                  f"  (this checkout is {head}: `arena sync`)"))
        svcs = {k[4:]: v for k, v in info.items() if k.startswith("svc_")}
        print("        services: " + "  ".join(f"{k} {v}" for k, v in svcs.items()))
        if "clock" in info:
            m = re.match(r"(\d+)/(\d+) MHz min=(\d+)", info["clock"])
            pinned = m and int(m.group(3)) >= 0.97 * int(m.group(2))
            (ok if pinned else warn)(f"cpu {info['clock'].split(' min')[0]}"
                                     + ("  pinned" if pinned else "  NOT pinned (`arena tune`)"))
        if "temp" in info:
            print(f"        temp {info['temp']}")
        if r is cfg.vision or cfg.single:
            (ok if info.get("cams", "").strip() else bad)(f"cameras: {info.get('cams', '').strip() or 'none'}")
            (ok if "calib" in info else warn)(f"calibration: {info.get('calib', 'none -- `arena scan`')}")
        if "wifi_ps" in info:
            ps = info["wifi_ps"].split()
            (ok if ps[-1] == "off" else warn)(f"wifi {ps[0]} power-save {ps[-1]}"
                                              + ("" if ps[-1] == "off" else "  (adds 100 ms+ radio naps: `arena tune`)"))
    state = fetch_state(hub_url(cfg))
    step("session")
    if state is None:
        print(f"        hub not reachable at {hub_url(cfg)} (not running, or not reachable from here)")
    else:
        c = state.get("control") or {}
        print(f"        controller {c.get('state', '?')} {'ARMED' if c.get('armed') else 'disarmed'}; "
              f"{len(state.get('alerts', []))} alert(s); dashboard {hub_url(cfg)}")
        for a in state.get("alerts", []):
            (bad if a["level"] == "error" else warn)(a["text"])


def cmd_sync(args, cfg: Config) -> None:
    cfg.require_topology()
    head, branch, dirty = local_head()
    if dirty:
        warn("this checkout has uncommitted changes -- the Orins only get what is committed and pushed")
    ahead = subprocess.run(["git", "rev-list", "--count", f"origin/{branch}..HEAD"], cwd=REPO,
                           capture_output=True, text=True)
    if ahead.returncode != 0 or ahead.stdout.strip() not in ("0", ""):
        if args.push:
            subprocess.run(["git", "push", "-u", "origin", branch], cwd=REPO, check=True)
        else:
            die(f"{branch} @ {head} is not on origin yet. Push it (git push), or `arena sync --push`.")
    script = f'''
running=""; for n in vision controller hub; do svc_pid $n >/dev/null && running="$running $n"; done
# The Orins never carry commits of their own, so mirror origin exactly (this also
# follows a rebased / force-pushed branch). Tracked edits made on the Orin make
# the checkout refuse rather than being thrown away; untracked and ignored files
# (the arena calibration) are untouched.
git fetch --quiet origin && git checkout --quiet -B {shlex.quote(branch)} origin/{shlex.quote(branch)} || exit 3
echo "now at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
[ -n "$running" ] && echo "NOTE: still running old code:$running -- \\`arena stop\\` then \\`arena up\\`"
exit 0
'''
    failed = False
    for r in cfg.roles():
        step(f"{r.name}: {r.target}")
        p = run_on(r, script, lib=True, capture=True, timeout=120)
        print("        " + p.stdout.strip().replace("\n", "\n        "))
        if p.returncode == 3:
            bad(f"could not check out origin/{branch} on {r.name} (tracked files edited on the Orin? "
                f"never edit there):\n{p.stderr.strip()}")
            failed = True
            continue
        tests = []
        if r is cfg.vision or cfg.single:
            tests.append(("perception selftest", f"cd portable_orin_perception && {r.python if r is cfg.vision else cfg.vision.python} selftest.py"))
        if r is cfg.control:
            tests.append(("controller selftest", f"cd portable_n1_controller && {cfg.control.python} selftest.py"))
        for label, t in tests:
            p = run_on(r, t, capture=True, timeout=300)
            (ok if p.returncode == 0 else bad)(label + ("" if p.returncode == 0 else
                                                        ":\n" + (p.stdout + p.stderr)[-800:]))
            failed = failed or p.returncode != 0
    if failed:
        raise SystemExit(1)


def cmd_setup(args, cfg: Config) -> None:
    cfg.require_topology()
    origin = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    for r in cfg.roles():
        step(f"{r.name}: {r.target} (you may be asked for the sudo password)")
        clone = (f"[ -d {r.repo}/.git ] || git clone {shlex.quote(origin)} {r.repo}\n")
        pre = ["ssh", "-t", r.target, "bash -c " + shlex.quote(clone)] if not r.local else ["bash", "-c", clone]
        subprocess.run(pre)
        pip = ("python3 -m pip install --user -r portable_orin_perception/requirements.txt "
               "|| python3 -m pip install --user --break-system-packages -r portable_orin_perception/requirements.txt")
        script = "set -e\nsudo apt-get install -y v4l-utils python3-venv python3-pip iw\n"
        if r is cfg.vision or cfg.single:
            script += pip + "\n"
        if r is cfg.control:
            script += (f"[ -x {cfg.control.python} ] || python3 -m venv {Path(cfg.control.python).parent.parent.as_posix()}\n"
                       f"{cfg.control.python} -m pip install -r portable_n1_controller/requirements.txt\n")
        run_on(r, script, tty=True)
    print("\nNext: `arena tune`, then `arena sync` and `arena scan`.")


def cmd_tune(args, cfg: Config) -> None:
    cfg.require_topology()
    script = r'''
echo "power model: $(sudo nvpmodel -q 2>/dev/null | head -1)"
sudo jetson_clocks && echo "clocks pinned: $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)/1000 )) MHz"
w=$(iw dev 2>/dev/null | awk '/Interface/{print $2; exit}')
if [ -n "$w" ]; then sudo iw dev "$w" set power_save off && echo "wifi $w power-save: $(iw dev $w get power_save | awk '{print $3}')"; fi
echo "(both reset at reboot: re-run \`arena tune\` after a power cycle)"
'''
    for r in cfg.roles():
        step(f"{r.name}: {r.target}")
        run_on(r, script, tty=True)


def cmd_scan(args, cfg: Config) -> None:
    cfg.require_topology()
    v = cfg.vision
    extra = " ".join(a for a, on in (("--check", args.check), ("--tune-exposure", args.tune_exposure)) if on)
    # Taking the camera away stalls a running controller into E-stop, but it stays
    # ARMED, and would drive again the moment poses resume. Disarm it first; a
    # control Orin that is off or unreachable has nothing armed to worry about.
    try:
        _ctl(cfg, "disarm")
    except SystemExit:
        warn(f"could not reach the control Orin ({cfg.control.target}) to disarm; scanning anyway")
    step(f"scanning the arena from {v.target}")
    p = run_on(v, f'''
if pid=$(svc_pid vision); then echo "(stopping the vision service: the camera can only be opened once)"; svc_stop vision >/dev/null; fi
cd portable_orin_perception && {v.python} -u scan_arena.py {extra}
''', lib=True, tty=not v.local)
    if not v.local:
        dest = REPO / "arena_scan.png"
        scp = subprocess.run(["scp", "-q", f"{v.target}:.arena/scan/latest.png", str(dest)])
        if scp.returncode == 0:
            print(f"\nsnapshot copied to {dest}")
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    print("\nNext: `arena up`." if not args.check else "")


CARS_PY = r'''
import json, sys
sys.path.insert(0, "portable_n1_controller")
from pc_controller.esp_link import discover_cars
found = discover_cars(timeout_s=1.5)
print(json.dumps({str(k): v for k, v in found.items()}))
'''

INDEX_PY = r'''
import json, socket, sys
ip, idx = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2)
s.sendto(json.dumps({"cfg": {"index": idx}}).encode(), (ip, 8888))
try: print(s.recvfrom(1024)[0].decode())
except OSError as e: print(json.dumps({"error": str(e)}))
'''


def discover_on_control(cfg: Config) -> dict:
    c = cfg.control
    p = run_on(c, f"{c.python} -c {shlex.quote(CARS_PY)}", capture=True, timeout=30)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        die(f"car discovery failed on {c.name}:\n{p.stdout}{p.stderr}")


def cmd_cars(args, cfg: Config) -> None:
    cfg.require_topology()
    if args.index:
        ip, n = args.index
        step(f"setting CAR_INDEX {n} on {ip} (only accepted while that car is stopped)")
        p = run_on(cfg.control, f"{cfg.control.tools_python} -c {shlex.quote(INDEX_PY)} {shlex.quote(ip)} {int(n)}",
                   capture=True, timeout=15)
        print("        " + p.stdout.strip())
    step(f"cars answering on {cfg.control.target}'s network")
    found = discover_on_control(cfg)
    dupes = found.pop("-1", None)
    if not found:
        bad("no car answered. Powered on? Same WiFi as the control Orin? Firmware flashed?")
    for idx, info in sorted(found.items(), key=lambda kv: int(kv[0])):
        print(f"  car {idx}  {info['ip']:15s}  fw {info.get('fw', '?'):22s}  "
              f"{'stopped' if info.get('failsafe', True) else 'DRIVING'}  rssi {info.get('rssi', '?')}  "
              f"{info.get('mac', '')}")
    if dupes:
        bad("duplicate CAR_INDEX: " + ", ".join(f"{d['ip']}=index {d['car']}" for d in dupes)
            + ". Fix with: arena cars --index <ip> <n>")
    try:
        manifest = json.loads((REPO / "portable_n1_controller" / cfg.model / "policy.onnx.manifest.json").read_text())
        n = int(manifest["num_pursuers"])
        missing = [i for i in range(n) if str(i) not in found]
        (bad if missing else ok)(f"model {cfg.model} drives {n} car(s)"
                                 + (f"; missing index {missing}" if missing else "; all present"))
    except (OSError, ValueError, KeyError):
        pass


def _find_pio() -> str | None:
    for cand in (shutil.which("pio"), shutil.which("platformio"),
                 str(Path.home() / ".platformio/penv/Scripts/pio.exe"),
                 str(Path.home() / ".platformio/penv/bin/pio")):
        if cand and Path(cand).exists():
            return cand
    return None


def cmd_flash(args, cfg: Config) -> None:
    pio = _find_pio()
    if not pio:
        die("PlatformIO not found on this machine (pip install platformio, or the VS Code extension)")
    proj = REPO / "portable_n1_controller" / "esp32"
    creds = proj / "esp32_receiver" / "wifi_credentials.h"
    if not creds.exists():
        die(f"{creds.relative_to(REPO)} missing: copy wifi_credentials.h.example and fill in the WiFi")
    step("building firmware")
    if subprocess.run([pio, "run", "-d", str(proj)]).returncode != 0:
        die("build failed")
    if args.build_only:
        return
    if args.usb is not None:
        step("flashing over USB (use the port labelled UART)")
        cmd = [pio, "run", "-d", str(proj), "-t", "upload"] + (["--upload-port", args.usb] if args.usb else [])
        raise SystemExit(subprocess.run(cmd).returncode)
    m = re.search(r'^\s*#define\s+OTA_PASSWORD\s+"([^"]*)"', creds.read_text(encoding="utf-8"), re.M)
    if not m:
        die("OTA is off: add  #define OTA_PASSWORD \"...\"  to wifi_credentials.h, flash each car once "
            "over USB (`arena flash --usb`), then use --ota")
    if args.ip:
        targets = {str(i): {"ip": ip, "failsafe": True, "fw": "?"} for i, ip in enumerate(args.ip)}
    else:
        sys.path.insert(0, str(REPO / "portable_n1_controller"))
        from pc_controller.esp_link import discover_cars  # noqa: E402
        targets = {str(k): v for k, v in discover_cars(timeout_s=1.5).items() if k != -1}
        if not targets:
            die("no car answered discovery from this machine (must be on the cars' WiFi); pass --ip")
    if args.car is not None:
        targets = {k: v for k, v in targets.items() if int(k) == args.car}
    env = dict(os.environ, PLATFORMIO_UPLOAD_FLAGS=f"--auth={m.group(1)}")
    failed = []
    for idx, info in sorted(targets.items()):
        step(f"car {idx} @ {info['ip']} (fw {info.get('fw')})")
        if not info.get("failsafe", True):
            bad("car is being driven -- `arena halt` first. Skipped.")
            failed.append(idx)
            continue
        rc = subprocess.run([pio, "run", "-d", str(proj), "-e", "ota", "-t", "upload",
                             "--upload-port", info["ip"]], env=env).returncode
        (ok if rc == 0 else bad)(f"car {idx} {'updated' if rc == 0 else 'FAILED'}")
        if rc:
            failed.append(idx)
    if failed:
        raise SystemExit(1)


def _control_addr_from_vision(cfg: Config) -> str:
    if cfg.single:
        return "127.0.0.1"
    if cfg.control_addr_from_vision != "auto":
        return cfg.control_addr_from_vision
    p = run_on(cfg.vision, f"getent ahostsv4 {shlex.quote(cfg.control.host)} | awk '{{print $1; exit}}'",
               capture=True, timeout=15)
    addr = p.stdout.strip()
    if not addr:
        die(f"the vision Orin cannot resolve {cfg.control.host}. Set control_addr_from_vision in "
            "deploy/arena.local.conf to the control Orin's IP (on the wired link if you have one).")
    return addr


def cmd_up(args, cfg: Config) -> None:
    cfg.require_topology()
    head, _branch, _dirty = local_head()
    v, c = cfg.vision, cfg.control
    model = args.model or cfg.model
    esp = args.esp or cfg.esp
    ctrl_addr = _control_addr_from_vision(cfg)
    vis_addr = "127.0.0.1" if cfg.single else (
        v.host if cfg.vision_addr_from_control == "auto" else cfg.vision_addr_from_control)

    step(f"control: {c.target} -- hub + controller ({model}, cars {esp}) DISARMED")
    run_on(c, f'''
svc_start hub . {c.tools_python} -u deploy/hub.py --port {cfg.dash_port} --telemetry-port {cfg.tel_port} \
    --control-port {cfg.ctl_port} --preview http://{vis_addr}:{cfg.preview_port}
svc_start controller portable_n1_controller {c.python} -u run_controller.py --model {shlex.quote(model)} \
    --source udp --pose-port {cfg.pose_port} --esp {shlex.quote(esp)} \
    --telemetry 127.0.0.1:{cfg.tel_port} --control-port {cfg.ctl_port} --start-disarmed
''', lib=True, check=True, timeout=30)
    # A controller that was already running keeps its arm state across `arena up`
    # (svc_start leaves it alone). `up` promises DISARMED, so make it true before
    # any pose can flow. A controller that is still starting is disarmed anyway.
    _ctl(cfg, "disarm")
    step(f"vision: {v.target} -- camera -> poses -> {ctrl_addr}:{cfg.pose_port}")
    run_on(v, f'''
svc_start vision portable_orin_perception {v.python} -u run_vision.py \
    --pose-target {ctrl_addr}:{cfg.pose_port} --telemetry {ctrl_addr}:{cfg.tel_port} \
    --preview-port {cfg.preview_port}
''', lib=True, check=True, timeout=30)

    step("waiting for the pipeline to come up")
    url = hub_url(cfg)
    deadline, state, healthy = time.time() + args.wait, None, False
    while time.time() < deadline:
        state = fetch_state(url)
        if state is None and not c.local:
            # The hub may not be reachable from this machine; ask the control Orin itself.
            py = (f"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen("
                  f"'http://127.0.0.1:{cfg.dash_port}/api/state', timeout=2).read().decode())")
            p = run_on(c, f"{c.tools_python} -c {shlex.quote(py)}", capture=True, timeout=10)
            try:
                state = json.loads(p.stdout)
            except ValueError:
                state = None
        vis, ctl = (state or {}).get("vision") or {}, (state or {}).get("control") or {}
        if ctl.get("armed"):
            _ctl(cfg, "disarm")         # belt and braces: never report an armed fleet as disarmed
            continue
        if vis.get("fps", 0) > 0 and ctl.get("state") == "RUNNING":
            healthy = True
            break
        time.sleep(1.0)
    if not healthy:
        bad("pipeline not healthy yet. What each service says:")
        for role, name in ((c, "controller"), (v, "vision"), (c, "hub")):
            p = run_on(role, f"tail -n 12 ~/.arena/logs/{name}.log", capture=True, timeout=10)
            print(_c("2", f"--- {name} ({role.target}) ---\n") + p.stdout)
        print("Fix, then `arena stop` and `arena up` again. Alerts: `arena status`.")
        raise SystemExit(1)
    lat = ((ctl.get("lat_ms") or {}).get("total") or {})
    ok(f"camera {vis.get('mode')} at {vis.get('fps')} fps; cars: "
       + ", ".join(f"{x['idx']}@{x['addr']}" for x in ctl.get("cars", [])))
    if lat:
        ok(f"camera -> command p50 {lat.get('p50')} ms, p95 {lat.get('p95')} ms")
    for a in (state or {}).get("alerts", []):
        (bad if a["level"] == "error" else warn)(a["text"])
    print(f"\n  dashboard  {url}\n  cars are DISARMED (E-stop packets). Watch the poses, then:  ./arena go")


def _ctl(cfg: Config, command: str) -> dict:
    c = cfg.control
    py = (f"import json,socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(2); "
          f"s.sendto(json.dumps({{'cmd':'{command}'}}).encode(),('127.0.0.1',{cfg.ctl_port})); "
          f"print(s.recvfrom(1024)[0].decode())")
    p = run_on(c, f"{c.tools_python} -c {shlex.quote(py)}", capture=True, timeout=15)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "controller not answering (is `arena up` running?)"}


def cmd_go(args, cfg: Config) -> None:
    cfg.require_topology()
    r = _ctl(cfg, "arm")
    if "error" in r:
        die(r["error"])
    ok("ARMED -- the policy is driving. Stop: ./arena halt  (or the dashboard's STOP)")
    if r.get("state") != "RUNNING":
        warn(f"controller state is {r.get('state')}: cars move once pose frames flow")


def cmd_halt(args, cfg: Config) -> None:
    cfg.require_topology()
    r = _ctl(cfg, "disarm")
    if "error" in r:
        warn(r["error"] + " -- if nothing is running, the cars are already in failsafe")
    else:
        ok("DISARMED -- every car got E-stop and stays neutral. Re-arm: ./arena go")


def cmd_stop(args, cfg: Config) -> None:
    cfg.require_topology()
    _ctl(cfg, "disarm")
    for r in ([cfg.control] if cfg.single else [cfg.control, cfg.vision]):
        names = ["controller", "hub", "vision"] if cfg.single else (
            ["controller", "hub"] if r is cfg.control else ["vision"])
        p = run_on(r, "\n".join(f"svc_stop {n}" for n in names), lib=True, capture=True, timeout=30)
        for line in p.stdout.strip().splitlines():
            ok(f"{r.name}: {line}")
    ok("stopped; cars sent E-stop and are in failsafe")


def cmd_logs(args, cfg: Config) -> None:
    cfg.require_topology()
    role = cfg.vision if args.name == "vision" else cfg.control
    run_on(role, f"tail -n {args.lines} {'-f ' if args.follow else ''}~/.arena/logs/{args.name}.log",
           tty=args.follow)


# ------------------------------------------------------------------ terminal monitor
def _bar(value: float | None, scale: float, width: int = 30, color: str = "36") -> str:
    if value is None:
        return " " * width
    n = max(0, min(width, int(round(value / scale * width))))
    return _c(color, BLOCK * n) + DOT * (width - n)


def render_monitor(s: dict) -> str:
    v, c = s.get("vision") or {}, s.get("control") or {}
    age = s.get("age") or {}
    lines = []
    state = c.get("state", "OFFLINE") if age.get("control", 99) < 2 else "OFFLINE"
    armed = c.get("armed")
    badge = (_c("41;97", " STALLED ") if state == "STALLED" else _c("42;30", " ARMED ") if armed and state == "RUNNING"
             else _c("43;30", " DISARMED ") if state != "OFFLINE" else _c("100;97", " OFFLINE "))
    lat = (c.get("lat_ms") or {})
    tot = lat.get("total") or {}
    per = c.get("period_ms") or {}
    lines.append(f"{_c('1', 'ARENA')}  {badge}  {c.get('model', '')}   "
                 f"camera->command {tot.get('p50', DASH)} / {tot.get('p95', DASH)} ms (p50/p95)   "
                 f"tick {1000 / per['p50']:.1f} Hz" if per.get("p50") else f"{_c('1', 'ARENA')}  {badge}")
    lines.append("")
    vl = (lat.get("vision") or {}).get("p50")
    cl = (lat.get("control") or {}).get("p50")
    rtts = [x["rtt_ms"] for x in c.get("cars", []) if x.get("rtt_ms") is not None]
    rl = sum(rtts) / len(rtts) / 2 if rtts else None
    lines.append("latency p50       0 ms " + " " * 18 + "100 ms budget")
    for label, val, col in (("camera->pose sent", vl, "34"), ("controller", cl, "35"), ("radio (1/2 RTT)", rl, "32")):
        lines.append(f"  {label:18s}{_bar(val, 100.0, 40, col)} {'' if val is None else f'{val:6.1f} ms'}")
    lines.append("")
    m = v.get("ms") or {}
    h = v.get("host") or {}
    if v:
        lines.append(f"vision  {v.get('mode')}  {v.get('fps')} fps  decode {(m.get('decode') or {}).get('p50', DASH)} ms  "
                     f"detect {(m.get('detect') or {}).get('p50', DASH)} ms  tracking {v.get('roi_pct', DASH)}%  "
                     f"sent {v.get('sent')}  suppressed {v.get('suppressed')}")
        lines.append("        " + "   ".join(
            f"{x['role']}: {x['vis']:.0f}% seen" + (f" ({x['x']:+.2f},{x['y']:+.2f})" if x.get("x") is not None else "")
            for x in v.get("vehicles", [])))
        if h:
            clocks = (f"{h['cpu_ghz']}/{h['cpu_max_ghz']} GHz "
                      f"{'pinned' if h.get('clocks_pinned') else _c('33', 'NOT pinned')}  ") if "cpu_ghz" in h else ""
            temp = f"{h['temp_c']} C  " if "temp_c" in h else ""
            lines.append(f"        host {clocks}{temp}vision process {h.get('proc_cpu_pct', '?')}% cpu")
    else:
        lines.append(_c("33", "vision: no telemetry"))
    lines.append("")
    cmd = c.get("cmd") or []
    lines.append("cars")
    for car in c.get("cars", []):
        i = car["idx"]
        thr = cmd[2 * i] if len(cmd) > 2 * i else None
        st = cmd[2 * i + 1] if len(cmd) > 2 * i + 1 else None
        alive = car.get("ack_age_s") is not None and car["ack_age_s"] < 1
        lines.append(f"  {i} {car['addr']:21s} thr {'' if thr is None else f'{thr:+.2f}':>6s} "
                     f"steer {'' if st is None else f'{st:+.2f}':>6s}  rtt {car.get('rtt_ms') or DASH:>6} ms  "
                     f"rssi {car.get('rssi') or DASH:>4}  " + (_c("32", "ok") if alive else _c("31", "NO ACK")))
    if c.get("dist"):
        lines.append(f"  distance to evader: {', '.join(f'{d:.2f} m' for d in c['dist'])} "
                     f"(the policy's capture radius: {c.get('capture_radius', 0):.2f} m)")
    lines.append("")
    alerts = s.get("alerts") or []
    if not alerts:
        lines.append(_c("32", "all nominal"))
    for a in alerts[:8]:
        lines.append((_c("31", "ERROR ") if a["level"] == "error" else _c("33", "warn  ")) + a["text"])
    lines.append("")
    lines.append(_c("2", "ctrl-c to leave (does not stop anything)   arena go | halt | stop"))
    return "\n".join(lines)


def cmd_monitor(args, cfg: Config) -> None:
    url = args.url or hub_url(cfg)
    if not args.url:
        cfg.require_topology()
    if fetch_state(url) is None and not args.url and not cfg.control.local:
        # Hub not reachable from here (different network): watch from the control Orin itself.
        print(f"{url} not reachable from this machine; running the monitor on {cfg.control.target}")
        run_on(cfg.control, f"{cfg.control.tools_python} deploy/arena.py monitor --url http://127.0.0.1:{cfg.dash_port}",
               tty=True)
        return
    try:
        while True:
            s = fetch_state(url)
            out = render_monitor(s) if s else _c("31", f"hub not answering at {url} -- `arena up`?")
            sys.stdout.write("\033[H\033[2J" + out + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


# ------------------------------------------------------------------ local simulation
def cmd_sim(args, cfg: Config) -> None:
    """Everything on this machine: sim world (mock cars + physics), synthetic camera, the
    real vision service, the real controller and the hub."""
    py = sys.executable
    manifest = json.loads((REPO / "portable_n1_controller" / args.model / "policy.onnx.manifest.json").read_text())
    n = int(manifest["num_pursuers"])
    base = 18888
    esp = ",".join(f"127.0.0.1:{base + i}" for i in range(n))
    log_dir = Path.home() / ".arena" / "sim"
    log_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("world", REPO / "portable_n1_controller",
         [py, "-u", "tools/sim_world.py", "--cars", str(n), "--base-port", str(base), "--world-port", "19880"]),
        ("hub", REPO, [py, "-u", "deploy/hub.py", "--port", str(args.port), "--telemetry-port", "19871",
                       "--control-port", "19872", "--preview", "http://127.0.0.1:18090"]),
        ("controller", REPO / "portable_n1_controller",
         [py, "-u", "run_controller.py", "--model", args.model, "--source", "udp", "--pose-port", "19870",
          "--esp", esp, "--telemetry", "127.0.0.1:19871", "--control-port", "19872", "--start-disarmed"]),
        ("vision", REPO / "portable_orin_perception",
         [py, "-u", "run_vision.py", "--synthetic", "--world-port", "19880", "--pose-target", "127.0.0.1:19870",
          "--telemetry", "127.0.0.1:19871", "--preview-port", "18090"]),
    ]
    procs = []
    try:
        for name, cwd, cmd in specs:
            fh = open(log_dir / f"{name}.log", "w", encoding="utf-8")
            procs.append((name, subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT), fh))
        url = f"http://127.0.0.1:{args.port}"
        print(f"simulated arena up ({n} car(s), model {args.model}); logs in {log_dir}")
        print(f"  dashboard  {url}")
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)
        time.sleep(3.0)
        if not args.disarmed:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            s.sendto(b'{"cmd":"arm"}', ("127.0.0.1", 19872))
            try:
                s.recvfrom(1024)
                print("  armed: the policy is driving the simulated car")
            except OSError:
                print("  controller did not answer the arm command; see controller.log")
        end = time.time() + args.seconds if args.seconds else None
        while end is None or time.time() < end:
            for name, p, _fh in procs:
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited ({p.returncode}); see {log_dir / (name + '.log')}")
            if args.monitor:
                st = fetch_state(url)
                if st:
                    sys.stdout.write("\033[H\033[2J" + render_monitor(st) + "\n")
                    sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(_c("31", str(exc)))
    finally:
        for _name, p, fh in reversed(procs):
            p.terminate()
        for _name, p, fh in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
        print("simulation stopped")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="arena", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser("init", help="write deploy/arena.local.conf (which Orin is which)")
    p.add_argument("--vision", help="user@host of the Orin with the camera")
    p.add_argument("--control", help="user@host of the Orin on the cars' WiFi")
    p.add_argument("--single", help="user@host of one Orin doing both jobs")
    p.add_argument("--repo", help="checkout path on the Orins (default ~/ee198-deployment-bundles)")
    p.add_argument("--link-ip", help="control Orin's address on the wired link (poses + telemetry go here)")
    p.add_argument("--vision-link-ip", help="vision Orin's address on the wired link (camera preview)")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("status", help="both Orins: code version, services, clocks, camera, calibration, alerts") \
        .set_defaults(fn=cmd_status)
    p = sub.add_parser("sync", help="put both Orins on this checkout's commit and run the selftests")
    p.add_argument("--push", action="store_true", help="git push this branch first")
    p.set_defaults(fn=cmd_sync)
    sub.add_parser("setup", help="one-time install on both Orins (asks for sudo)").set_defaults(fn=cmd_setup)
    sub.add_parser("tune", help="pin clocks + WiFi power-save off on both Orins (asks for sudo)") \
        .set_defaults(fn=cmd_tune)

    p = sub.add_parser("scan", help="calibrate + check the arena from the overhead camera")
    p.add_argument("--check", action="store_true", help="verify the saved calibration only (camera moved?)")
    p.add_argument("--tune-exposure", action="store_true", help="find the shortest usable exposure")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("cars", help="find the cars on the WiFi; --index sets a car's CAR_INDEX")
    p.add_argument("--index", nargs=2, metavar=("IP", "N"))
    p.set_defaults(fn=cmd_cars)

    p = sub.add_parser("flash", help="build the ESP32 firmware and flash the cars")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--usb", nargs="?", const="", default=None, metavar="PORT", help="one car on USB")
    g.add_argument("--ota", action="store_true", help="every car over WiFi (default)")
    g.add_argument("--build-only", action="store_true")
    p.add_argument("--car", type=int, help="only this CAR_INDEX (with --ota)")
    p.add_argument("--ip", nargs="+", help="car IPs instead of discovery (with --ota)")
    p.set_defaults(fn=cmd_flash)

    p = sub.add_parser("up", help="start vision + controller + dashboard, cars DISARMED")
    p.add_argument("--model", help="override [fleet] model, e.g. models/n1_pin")
    p.add_argument("--esp", help="override [fleet] esp (auto, or ip:port list)")
    p.add_argument("--wait", type=float, default=25.0, help="seconds to wait for a healthy pipeline")
    p.set_defaults(fn=cmd_up)
    sub.add_parser("go", help="ARM: the policy drives the cars").set_defaults(fn=cmd_go)
    sub.add_parser("halt", help="DISARM: E-stop every car, keep services running").set_defaults(fn=cmd_halt)
    sub.add_parser("stop", help="E-stop and stop all services").set_defaults(fn=cmd_stop)

    p = sub.add_parser("monitor", help="live terminal dashboard")
    p.add_argument("--url", help="hub URL (default: from the config)")
    p.add_argument("--interval", type=float, default=0.25)
    p.set_defaults(fn=cmd_monitor)
    p = sub.add_parser("logs", help="show a service's log")
    p.add_argument("name", choices=["vision", "controller", "hub"])
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=60)
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("sim", help="rehearse the whole stack on this machine, no hardware")
    p.add_argument("--model", default="models/n1_catch")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until ctrl-c)")
    p.add_argument("--port", type=int, default=8080, help="dashboard port")
    p.add_argument("--disarmed", action="store_true", help="do not arm automatically")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--monitor", action="store_true", help="also show the terminal monitor here")
    p.set_defaults(fn=cmd_sim)

    args = ap.parse_args(argv)
    cfg = Config() if CONF.exists() else die(f"{CONF} missing")
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
