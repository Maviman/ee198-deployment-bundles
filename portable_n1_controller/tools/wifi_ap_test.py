"""WiFi link test with the Orin as the access point: the car joins the Orin's own
2.4 GHz network, no router, no campus WiFi.

    python tools/wifi_ap_test.py --provision /dev/ttyUSB0 --index 0   # once per car, over USB
    python tools/wifi_ap_test.py                                       # one car, untethered
    python tools/wifi_ap_test.py --cars 3                              # the fleet: indices 0, 1, 2

What a run does:
  1. Starts a hotspot on the Orin's WiFi adapter: SSID hive-arena, WPA2, 2.4 GHz
     (the ESP32-S3 has no 5 GHz), Orin at 10.43.0.1/24 handing out DHCP. Not
     10.42.0.0/24: that is the arena-link cable between the two Orins.
  2. Waits for every car (--cars N: car indices 0..N-1) to join, by broadcasting
     the firmware's {"probe": 1} on the hotspot subnet (a car also asks DHCP for
     the name hive-car<index>). A missing index, two cars answering to one index,
     or an index outside 0..N-1 fails the run before anything is commanded: that
     is the "wrong car moved" mix-up, caught on the bench.
  3. Measures: tools/link_test.py over UDP with the whole fleet at once (the same
     joint packet the controller sends; ACK, RTT, RSSI per car), ICMP ping to each
     car, and the Orin's view of each car (`iw station dump`: signal). PASS needs
     every car to pass.
  4. ALWAYS puts the adapter back on the network it was on before (SJSU_guest in
     the lab): on success, on failure, on Ctrl-C, on SIGTERM. A detached
     watchdog restores it too if this script is killed outright (--restore-after).

THE ORIN LOSES ITS INTERNET FOR THE WHOLE RUN. Its single RTL8822CE radio cannot
be an access point and a client at the same time ("interface combinations are not
supported"), so the hotspot replaces the campus link. Everything reached over
that link (SSH, a remote session) drops until step 4. A second USB WiFi adapter
would remove this: run the hotspot on it with --iface.

--provision PORT sends the car `wifi <ssid> <password>` and `radio wifi` on its
USB console (both are saved in NVS and reboot the board), then exits; with
--index N it first sets the car index (`index N`, NVS). One car on USB at a time:
provision car 0 with --index 0, unplug, car 1 with --index 1, and so on. After
that a car needs only a battery: USB is for flashing and the console.

The hotspot password is generated on first use and kept in
~/.config/hive-arena/wifi.psk (mode 600), never in git. --console PORT also logs
the car's console during the run (bench diagnosis only; opening the port does not
reset the board).
"""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc_controller.esp_link import discover_cars

TOOLS = Path(__file__).resolve().parent
PSK_FILE = Path.home() / ".config/hive-arena/wifi.psk"
PROFILE = "hive-arena"


def sh(*cmd: str, check: bool = True, timeout: float = 60) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {(p.stderr or p.stdout).strip()}")
    return p.stdout


def log(msg: str) -> None:
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except OSError:              # the terminal went away (window closed): keep going regardless
        pass


def hotspot_psk() -> str:
    if PSK_FILE.is_file():
        return PSK_FILE.read_text().strip()
    PSK_FILE.parent.mkdir(parents=True, exist_ok=True)
    psk = secrets.token_urlsafe(12)            # 16 chars, no spaces (the console splits on the last one)
    PSK_FILE.write_text(psk + "\n")
    PSK_FILE.chmod(0o600)
    return psk


def wifi_iface() -> str:
    for line in sh("nmcli", "-t", "-f", "DEVICE,TYPE", "dev").splitlines():
        dev, _, kind = line.rpartition(":")
        if kind == "wifi":
            return dev
    raise SystemExit("no WiFi adapter found (nmcli dev)")


def active_connection(iface: str) -> tuple[str, str]:
    """(uuid, name) of the connection active on `iface`, or ("", "") for none. Restored
    by UUID: nmcli's terse output escapes ':' and '\\' in names, a UUID has neither."""
    for line in sh("nmcli", "-t", "-f", "UUID,DEVICE,NAME", "con", "show", "--active").splitlines():
        uuid, dev, name = line.split(":", 2)
        if dev == iface:
            return uuid, name.replace("\\:", ":").replace("\\\\", "\\")
    return "", ""


WATCHDOG_PID = Path(os.environ.get("XDG_RUNTIME_DIR") or PSK_FILE.parent) / "hive-arena-watchdog.pid"


def pending_watchdog() -> int | None:
    """PID of a live watchdog from this or an earlier run, else None."""
    try:
        pid = int(WATCHDOG_PID.read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if PROFILE.encode() in cmd else None     # not a reused PID


def ensure_profile(args, iface: str, psk: str) -> None:
    """(Re)write the hotspot profile every run, so the settings here are the truth."""
    settings = [
        "connection.interface-name", iface,
        "connection.autoconnect", "no",
        "802-11-wireless.mode", "ap",
        "802-11-wireless.ssid", args.ssid,
        "802-11-wireless.band", "bg",
        "802-11-wireless.channel", str(args.channel),
        "802-11-wireless.powersave", "2",           # off: the AP must not nap either
        "wifi-sec.key-mgmt", "wpa-psk",
        "wifi-sec.proto", "rsn",
        "wifi-sec.pairwise", "ccmp",
        "wifi-sec.group", "ccmp",
        "wifi-sec.psk", psk,
        "ipv4.method", "shared",
        "ipv4.addresses", args.address,
        "ipv6.method", "disabled",
    ]
    exists = sh("nmcli", "-g", "connection.id", "con", "show", PROFILE, check=False).strip()
    if exists:
        sh("nmcli", "con", "modify", PROFILE, *settings)
    else:
        sh("nmcli", "con", "add", "type", "wifi", "con-name", PROFILE, "ssid", args.ssid, *settings)


class Restorer:
    """Puts the adapter back. Runs once, from whichever path gets there first.

    The watchdog is a shell started with `setsid -f`: setsid forks and exits, so the
    shell is re-parented away from this script, in its own session and process
    group. A kill of this script's whole process tree (Ctrl-C in a terminal, a tool
    stopping the task, SIGKILL) cannot reach it. It records its PID for the normal
    path to cancel it with."""

    def __init__(self, iface: str, uuid: str, name: str, after_s: int) -> None:
        self.iface, self.uuid, self.name, self.done = iface, uuid, name, False
        back = f"nmcli con up uuid {uuid}" if uuid else f"nmcli dev connect {shlex.quote(iface)}"
        WATCHDOG_PID.parent.mkdir(parents=True, exist_ok=True)
        WATCHDOG_PID.unlink(missing_ok=True)
        script = (f"echo $$ > {shlex.quote(str(WATCHDOG_PID))}; sleep {int(after_s)}; "
                  f"nmcli con down id {PROFILE}; {back}; rm -f {shlex.quote(str(WATCHDOG_PID))}")
        subprocess.run(["setsid", "-f", "sh", "-c", script], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
        for _ in range(100):
            self.watchdog = pending_watchdog()
            if self.watchdog:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("the restore watchdog did not start: not touching the network")

    def restore(self) -> None:
        if self.done:
            return
        self.done = True
        log(f"restoring {self.iface} -> {self.name or '(autoconnect)'}")
        sh("nmcli", "con", "down", "id", PROFILE, check=False)
        for attempt in range(3):
            cmd = (["nmcli", "--wait", "45", "con", "up", "uuid", self.uuid] if self.uuid
                   else ["nmcli", "--wait", "45", "dev", "connect", self.iface])
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if p.returncode == 0:
                log(f"back on {self.name or 'the usual network'}")
                break
            log(f"restore attempt {attempt + 1} failed: {p.stderr.strip()}")
            time.sleep(3)
        else:
            log(f"could not restore {self.iface}: the watchdog (pid {self.watchdog}) tries again "
                "when it fires")
            return
        try:
            os.killpg(self.watchdog, signal.SIGTERM)
        except ProcessLookupError:
            pass
        WATCHDOG_PID.unlink(missing_ok=True)


class ConsoleLog(threading.Thread):
    """The car's USB console, timestamped, for the report (bench runs only)."""

    def __init__(self, port: str) -> None:
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.2)
        self.lines: list[str] = []
        self.stop = False

    def run(self) -> None:
        buf = b""
        while not self.stop:
            try:
                buf += self.ser.read(512)
            except Exception as exc:                 # unplugged mid-run: keep what we have
                self.lines.append(f"(console closed: {exc})")
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode(errors="replace").rstrip()
                if text and not text.startswith('{"ack"'):
                    self.lines.append(f"{time.strftime('%H:%M:%S')} {text}")


def provision(args) -> None:
    import serial
    psk = hotspot_psk()
    with serial.Serial(args.provision, 115200, timeout=0.2) as ser:
        def say(cmd: str, wait_for: str, timeout: float = 12.0) -> str:
            ser.reset_input_buffer()
            ser.write(cmd.encode() + b"\n")
            out, deadline = "", time.monotonic() + timeout
            while time.monotonic() < deadline:
                out += ser.read(1024).decode(errors="replace")
                if wait_for in out:
                    break
            return out
        out = say("wifi", "wifi network")
        if "wifi network" not in out:
            raise SystemExit("the board has no `wifi` command: flash current firmware first "
                             "(./arena flash, or pio run -t upload)")
        if args.index is not None:
            out = say(f"index {args.index}", "car index")
            if f"car index set to {args.index}" not in out:
                raise SystemExit(f"the board refused index {args.index}:\n" + out)
            log(f"car index set to {args.index} (saved)")
        out = say(f"wifi {args.ssid} {psk}", "=== CAR INDEX")
        if f"wifi network set to '{args.ssid}'" not in out:
            raise SystemExit("the board refused the network:\n" + out.replace(psk, "<psk>"))
        log(f"car saved network '{args.ssid}' and rebooted")
        out = say("radio wifi", "=== CAR INDEX")
        if "radio set to wifi" in out:
            log("car switched to radio wifi and rebooted")
        else:
            out += say("radio", "radio")
            if "radio wifi" not in out:
                raise SystemExit("could not set radio wifi:\n" + out)
            log("car already on radio wifi")
        out = say("wifi", "wifi network")
        log(out.strip().splitlines()[-1] if out.strip() else "(no reply)")


def find_cars(prefix: str, n: int, deadline: float, discover=discover_cars,
              clock=time.monotonic) -> tuple[dict[int, dict], list[str]]:
    """Probe the hotspot subnet (`prefix` like "10.43.0.") until car indices 0..n-1
    have all answered or `deadline` passes. Returns ({index: probe reply}, problems).
    Problems are fatal: an index answered from two addresses (two cars share it),
    an index outside 0..n-1, or an index that never answered."""
    found: dict[int, dict] = {}
    dupes: dict[int, set[str]] = {}
    extra: dict[int, str] = {}
    while True:
        replies = discover(timeout_s=1.0, extra_targets=[prefix + "255"])
        flat = [(r["car"], r) for r in replies.pop(-1, [])]          # dupes within one round
        flat += [(idx, r) for idx, r in replies.items()]
        for idx, reply in flat:
            idx = int(idx)
            if not reply["ip"].startswith(prefix):
                continue                                             # another interface's car
            if not 0 <= idx < n:
                extra[idx] = reply["ip"]
            elif idx in found and found[idx]["ip"] != reply["ip"]:
                dupes.setdefault(idx, {found[idx]["ip"]}).add(reply["ip"])
            else:
                found.setdefault(idx, reply)
        if dupes or extra or len(found) == n or clock() >= deadline:
            break
    problems = [f"car index {i} answered from {', '.join(sorted(ips))}: two cars share it "
                f"(set one with `index <n>` on its console)" for i, ips in sorted(dupes.items())]
    problems += [f"car index {i} at {ip} is outside 0..{n - 1} (--cars {n}): wrong index, "
                 f"or pass a larger --cars" for i, ip in sorted(extra.items())]
    problems += [f"car index {i} never answered" for i in range(n) if i not in found]
    return found, problems


def link_verdicts(stdout: str) -> list[str]:
    """The per-car verdict lines of link_test.py, in car-index order."""
    return [l.split(":", 1)[1].strip() for l in stdout.splitlines() if l.startswith("verdict   :")]


def station_signals(dump: str) -> dict[str, str]:
    """`iw station dump` -> {mac (lower case): signal}."""
    out, mac = {}, None
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("Station "):
            mac = line.split()[1].lower()
        elif mac and line.startswith("signal:"):
            out[mac] = line.split(":", 1)[1].strip()
    return out


def station_dump(iface: str) -> str:
    return sh("iw", "dev", iface, "station", "dump", check=False)


def run_test(args, iface: str, report: list[str]) -> bool:
    t0 = time.monotonic()
    p = subprocess.run(["nmcli", "--wait", "30", "con", "up", "id", PROFILE],
                       capture_output=True, text=True, timeout=45)
    if p.returncode != 0:
        report.append(f"HOTSPOT FAILED: {p.stderr.strip()}")
        return False
    report.append(f"hotspot up on {iface} in {time.monotonic() - t0:.1f} s: '{args.ssid}' "
                  f"channel {args.channel}, Orin {args.address}")
    log(report[-1])

    t1 = time.monotonic()
    prefix = args.address.split("/")[0].rsplit(".", 1)[0] + "."
    log(f"waiting up to {args.join_timeout:.0f} s for car index(es) 0..{args.cars - 1}")
    cars, problems = find_cars(prefix, args.cars, t1 + args.join_timeout)
    for i, car in sorted(cars.items()):
        report.append(f"car {i} joined after {time.monotonic() - t1:.1f} s (or sooner): ip {car['ip']}, "
                      f"mac {car.get('mac')}, fw {car.get('fw')}, radio {car.get('radio')}, "
                      f"rssi {car.get('rssi')} dBm")
        log(report[-1])
    if problems:
        report.extend("PROBLEM: " + p for p in problems)
        report.append("Missing car: powered, on `radio wifi`, provisioned with this SSID "
                      "(--provision), antenna on, in range?")
        report.append("stations associated: " + (station_dump(iface).strip() or "none"))
        return False
    ips = [cars[i]["ip"] for i in range(args.cars)]

    log(f"link test: {args.count} command packets at 10 Hz over UDP to {args.cars} car(s)")
    lt = subprocess.run([sys.executable, str(TOOLS / "link_test.py"),
                         "--esp", ",".join(f"{ip}:8888" for ip in ips),
                         "--count", str(args.count), "--quiet"],
                        capture_output=True, text=True, timeout=args.count + 60)
    report.append("--- link_test.py (UDP command packets, zero throttle) ---")
    summary = lt.stdout.split("===== link test summary =====", 1)[-1]
    report.extend(l for l in summary.strip().splitlines() if l.strip())
    if lt.returncode != 0:
        report.append(lt.stderr.strip()[-800:])

    report.append("--- ping (ICMP, the radio alone), 100 x 20 ms per car ---")
    for i, ip in enumerate(ips):
        pg = subprocess.run(["ping", "-c", "100", "-i", "0.02", "-W", "1", ip],
                            capture_output=True, text=True, timeout=60)
        tail = [l for l in pg.stdout.strip().splitlines()[-2:] if l]
        report.append(f"car {i}: " + "  |  ".join(tail))

    report.append("--- the Orin's view of each car (iw station dump) ---")
    signals = station_signals(station_dump(iface))
    for i in range(args.cars):
        mac = str(cars[i].get("mac", "")).lower()
        report.append(f"car {i} ({mac}): signal {signals.get(mac, 'not listed')}")

    verdicts = link_verdicts(lt.stdout)
    return len(verdicts) == args.cars and all(v.startswith("PASS") for v in verdicts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssid", default="hive-arena")
    ap.add_argument("--channel", type=int, default=6, help="2.4 GHz channel 1-11")
    ap.add_argument("--address", default="10.43.0.1/24", help="the Orin on the hotspot (not 10.42.0.x)")
    ap.add_argument("--iface", help="WiFi adapter for the hotspot (default: the first one)")
    ap.add_argument("--cars", type=int, default=1,
                    help="how many cars must join: car indices 0..N-1 (default 1)")
    ap.add_argument("--count", type=int, default=300, help="link_test packets (10 Hz)")
    ap.add_argument("--join-timeout", type=float, default=60.0, help="seconds to wait for the car")
    ap.add_argument("--restore-after", type=int, default=240,
                    help="watchdog: seconds after which the old network comes back no matter what")
    ap.add_argument("--console", metavar="PORT", help="log the car's USB console during the run")
    ap.add_argument("--provision", metavar="PORT", help="set the car's network over USB, then exit")
    ap.add_argument("--index", type=int, help="with --provision: also set the car index (0..7)")
    ap.add_argument("--report", default=str(Path.home() / "wifi_ap_test.log"))
    args = ap.parse_args()
    if not 1 <= args.channel <= 11:
        raise SystemExit("--channel must be 1..11 (2.4 GHz; the ESP32-S3 has no 5 GHz)")
    if args.address.startswith("10.42.0."):
        raise SystemExit("10.42.0.0/24 is the arena-link cable between the Orins: pick another subnet")
    if not 1 <= args.cars <= 8:
        raise SystemExit("--cars must be 1..8 (the firmware's MAX_CARS)")
    if args.index is not None and not args.provision:
        raise SystemExit("--index only goes with --provision")
    if args.index is not None and not 0 <= args.index <= 7:
        raise SystemExit("--index must be 0..7")
    if args.provision:
        provision(args)
        return

    iface = args.iface or wifi_iface()
    uuid, previous = active_connection(iface)
    if previous == PROFILE:
        raise SystemExit(f"{iface} is already the hotspot; restore it first: nmcli con down {PROFILE}")
    old = pending_watchdog()
    if old:
        raise SystemExit(f"an earlier run's restore watchdog (pid {old}) is still pending and would "
                         f"switch the network mid-run: wait for it, or `kill -- -{old}`")
    ensure_profile(args, iface, hotspot_psk())
    min_watchdog = args.join_timeout + args.count / 10 + 90 + 5 * args.cars
    if args.restore_after < min_watchdog:
        raise SystemExit(f"--restore-after {args.restore_after} s is shorter than the run (~{min_watchdog:.0f} s)")
    console = ConsoleLog(args.console) if args.console else None
    if console:
        console.start()

    report = [f"wifi_ap_test {time.strftime('%Y-%m-%d %H:%M:%S')}: {iface} was on "
              f"'{previous or 'nothing'}', watchdog restores it after {args.restore_after} s"]
    log(report[0])

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    ok = False
    restorer = None
    try:
        restorer = Restorer(iface, uuid, previous, args.restore_after)
        log(f"the Orin goes OFF '{previous}' now")
        ok = run_test(args, iface, report)
    except BaseException as exc:                     # incl. Ctrl-C: report it, then restore
        report.append(f"ABORTED: {exc!r}")
    finally:
        # Nothing may interrupt the restore: a closed terminal (SIGHUP), a second
        # Ctrl-C or a SIGTERM now would leave the Orin on the hotspot.
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        if restorer:
            restorer.restore()
        if console:
            console.stop = True
            console.join(timeout=1)
            report.append("--- car console during the run ---")
            report.extend(console.lines[-60:])
        report.append(f"RESULT: {'PASS' if ok else 'FAIL'}")
        text = "\n".join(report) + "\n"
        Path(args.report).write_text(text)
        log(f"report saved to {args.report}")
        log("\n" + text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
