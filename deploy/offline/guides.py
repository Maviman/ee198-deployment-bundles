"""The START_HERE command sheets inside each offline package.

Each guide is data (sections of steps, terminal cards and tables), rendered
twice: START_HERE.html (opens offline in the Orin's browser, a copy button on
every command) and START_HERE.txt (the same text for a terminal: `less
START_HERE.txt`). One source, so the two never disagree.

Terminal cards carry a short label (machine + terminal name) because the
point of the sheet is "which window do I paste this into".
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

REPO = "~/ee198-deployment-bundles"
VENV_PY = "~/.venvs/n1ctl/bin/python"
MODELS = f"{REPO}/portable_n1_controller/models"
N1_MODEL = "models/n1_catch"                         # 1 car: the locked, tested default
N3_MODEL = "models/c38_8ft_buttons"                  # 3 cars: the hive commander for the 8 x 8 ft arena


# ------------------------------------------------------------------ model
@dataclass
class Term:
    """A terminal window: where it runs, its label, and what to paste into it."""
    machine: str                       # "VISION ORIN" / "AI ORIN" / "THIS ORIN"
    label: str                         # e.g. "Terminal 1 · ARENA"
    cmds: list[tuple[str, str]]        # (command, what it does)
    note: str = ""


@dataclass
class Steps:
    items: list[str]                   # HTML-safe text with <code> allowed


@dataclass
class Para:
    text: str                          # HTML-safe text with <code>/<b> allowed


@dataclass
class Table:
    head: tuple[str, ...]
    rows: list[tuple[str, ...]]


@dataclass
class Section:
    title: str
    blocks: list
    intro: str = ""


@dataclass
class Guide:
    key: str                           # vision / ai / emergency
    folder: str
    title: str
    lede: str
    sections: list[Section] = field(default_factory=list)


# ------------------------------------------------------------------ shared pieces
def _install(folder: str, machine: str, script: str) -> Section:
    return Section("1. Install (once, and again whenever you get a newer folder)", [
        Steps([
            f"Copy the whole <b>{folder}</b> folder from the USB stick into <b>Home</b> "
            "(Files app: drag it onto “Home”).",
            f"Open it, right-click an empty spot → <b>Open in Terminal</b>.",
            "Paste the command below. Press <b>Enter</b> to accept a default when it asks something.",
            "It ends with <b>Done.</b> and a summary. Then close that terminal and open a new one, "
            "so the <code>arena</code> command is found.",
        ]),
        Term(machine, "INSTALL (inside the folder)", [
            (f"bash {script}", "install / update; safe to run again"),
        ], note="No internet needed. It asks for your sudo password only for the optional "
                "system steps (Ethernet cable, a missing camera/WiFi tool)."),
    ])


def _models_section(folder: str, script: str) -> Section:
    return Section("Models: where they go and how to pick one", [
        Para(f"A model is a folder with three files: <code>policy.onnx</code>, "
             f"<code>policy.onnx.manifest.json</code> and <code>metrics.json</code>. "
             f"The AI Orin keeps them in:<br><code>{MODELS}/&lt;name&gt;/</code><br>"
             f"Five ship with the code: <code>n1_catch</code> (1 car, the locked default), "
             f"<code>n1_pin</code> (1 car), and the 3-car hive commander three times: "
             f"<code>c38_8ft_buttons</code> (the one to use: retrained for the 8 x 8 ft arena with the "
             f"button firmware in modulated mode), <code>c37_commit3_ft_g997_arena183</code> (older, "
             f"scaled to 1.83 m) and <code>c37_commit3_ft_g997</code> (older, 6 m). The hive network was "
             f"trained on a simulated car, so treat 3-car runs as experiments."),
        Steps([
            f"<b>Easiest:</b> put your model folder inside <b>{folder}/PUT_NEW_MODELS_HERE/</b> "
            f"before you run the installer. It copies it into place and test-runs it "
            f"(a model that cannot run is not installed).",
            "Give a retrained model a <b>new</b> folder name (e.g. <code>n1_catch_v2</code>); "
            "the names that ship with the code are refused.",
            "Already installed and only adding a model? Use the models-only command below.",
        ]),
        Term("AI ORIN", "INSTALL (inside the folder)", [
            (f"bash {script} --models-only", "just add the models from PUT_NEW_MODELS_HERE"),
        ]),
        Term("AI ORIN", "Terminal 1 · ARENA", [
            ("arena up --model models/n1_pin", "drive with a different model this session"),
            ("arena fleet", "which models are installed, and how many cars each drives"),
            (f"nano {REPO}/deploy/arena.local.conf", "change the DEFAULT: under [fleet] set  model = models/<name>"),
        ], note="The installer offers to make a new model the default. The car count must match the "
                "model (num_pursuers in its manifest); `arena cars` checks that."),
        Para("Where models come from: the AI Training repo exports them with "
             "<code>python -m controller_runtime.export_onnx --run-dir &lt;run&gt;</code>. "
             "Copy the three files into one folder named after the model."),
    ])


def _fleet_overview() -> Section:
    return Section("1 car or 3 cars: pick one", [
        Para("The arena runs <b>1 car with <code>n1_catch</code></b> unless you switch. Both use the "
             "same commands; only the <code>arena fleet</code> line at the start differs. "
             "<code>arena up</code> refuses to start if the car count and the model disagree."),
        Table(("", "1 car (N=1)", "3 cars (N=3)"), [
            ("model", f"<code>{N1_MODEL}</code>", f"<code>{N3_MODEL}</code> (the hive commander, 8 x 8 ft arena)"),
            ("on the floor", "1 pursuer + the evader", "3 pursuers + the evader"),
            ("pursuer roof tags", "tag 1 = CAR_INDEX 0", "tags 1, 2, 3 = CAR_INDEX 0, 1, 2"),
            ("evader roof tag", "tag 0", "tag 0"),
            ("status", "the locked, tested default",
             "experimental: trained on a simulated car (speed-controlled, no camera noise or delay)"),
            ("sim, catches in 60 s", "10 (9 with buttons)", "11 (4 with buttons)"),
        ]),
        Para("Each car's CAR_INDEX is set once and stays on the car. Check them with "
             "<code>arena cars</code>; change one (car stopped) with "
             "<code>arena cars --index &lt;car-ip&gt; &lt;n&gt;</code>. Tags on cars that are not "
             "driving are ignored."),
    ])


def _run_cards(machine: str, n: int, first: list[tuple[str, str]] | None = None) -> list:
    """A whole session for 1 or 3 cars, in order, in the one terminal you type in."""
    model = N1_MODEL if n == 1 else N3_MODEL
    cars = "car 0 answers" if n == 1 else "cars 0, 1 and 2 answer, each once"
    cmds = list(first or []) + [
        (f"arena fleet --pursuers {n} --model {model}",
         f"{n} car{'s' if n > 1 else ''} with {model.split('/')[1]}"
         + (" (already the default; needed only after a 3-car session)" if n == 1 else " (saved until you switch back)")),
        ("arena tune", "after EVERY power-on: full-speed clocks + WiFi power-save off (sudo)"),
        ("arena cars", f"check: {cars}"),
        ("arena status", "health check: code, camera, clocks, calibration, alerts"),
        ("arena scan", "calibrate from the corner tags (first time, or the camera moved)"),
        ("arena up", f"start everything for {n} car{'s' if n > 1 else ''}; cars stay DISARMED"),
        ("arena go", "ARM: the AI drives (push the cars by hand first and watch them on the dashboard)"),
        ("arena halt", "E-STOP every car (services keep running; `arena go` re-arms)"),
        ("arena stop", "E-stop and shut everything down"),
    ]
    if n > 1:
        cmds.append((f"arena fleet --pursuers 1 --model {N1_MODEL}", "afterwards: back to the 1-car default"))
    label = f"Terminal 1 · ARENA ({n} car{'s' if n > 1 else ''})"
    return [Term(machine, label, cmds)]


def _watch_cards(machine: str) -> list:
    return [
        Term(machine, "Terminal 2 · MONITOR (optional)", [
            ("arena monitor", "live numbers in the terminal; Ctrl-C to close"),
        ]),
        Term(machine, "BROWSER · DASHBOARD", [
            ("http://localhost:8080", "camera view, map, latency, per-car radio, and a STOP button"),
        ], note="Anyone watching can press STOP. Arming is only `arena go`, from the terminal. "
                "With 3 cars, each car is coloured by its current role."),
    ]


def _run_sections(machine: str, first: list[tuple[str, str]] | None = None) -> list:
    return [
        Section("Run with 1 car (N=1, n1_catch)", [
            Para("One pursuer with tag 1 (CAR_INDEX 0), and the evader with tag 0."),
            *_run_cards(machine, 1, first),
            *_watch_cards(machine),
        ]),
        Section("Run with 3 cars (N=3, the hive commander)", [
            Para("Three pursuers with tags 1, 2, 3 (CAR_INDEX 0, 1, 2), and the evader with tag 0. "
                 "First time only: give each car its index, as below. Treat 3-car runs as "
                 "experiments: the model was trained on a simulated car, not yours."),
            Term(machine, "Terminal 1 · ARENA (once per car)", [
                ("arena cars", "list the cars on the WiFi with their IP and CAR_INDEX"),
                ("arena cars --index <car-ip> 0", "the car with tag 1"),
                ("arena cars --index <car-ip> 1", "the car with tag 2"),
                ("arena cars --index <car-ip> 2", "the car with tag 3"),
            ], note="Each car must be stopped. The index is saved on the car."),
            *_run_cards(machine, 3, first),
            Para("The monitor and the dashboard work the same as for 1 car."),
        ]),
    ]


def _manual_cards(vision_machine: str, ai_machine: str, pose_to: str, preview_from: str) -> list:
    camera = (f"cd {REPO}/portable_orin_perception && python3 -u run_vision.py "
              f"--pose-target {pose_to}:9870 --telemetry {pose_to}:9871 --preview-port 8090")
    driver = (f"cd {REPO}/portable_n1_controller && {VENV_PY} -u run_controller.py "
              "--source udp --pose-port 9870 --esp auto --telemetry 127.0.0.1:9871 "
              "--control-port 9872 --start-disarmed")
    return [
        Term(vision_machine, "Terminal · CAMERA (paste ONE line)", [
            (f"{camera} --pursuers 1", "1 car: tracks tag 1 + the evader"),
            (f"{camera} --pursuers 3", "3 cars: tracks tags 1, 2, 3 + the evader"),
        ], note="Must match the AI DRIVER below: 1 car with 1 car, 3 cars with 3 cars. Ctrl-C to stop."),
        Term(ai_machine, "Terminal · DASHBOARD", [
            (f"cd {REPO} && python3 -u deploy/hub.py --port 8080 --telemetry-port 9871 "
             f"--control-port 9872 --preview http://{preview_from}:8090",
             "the dashboard at http://localhost:8080"),
        ]),
        Term(ai_machine, "Terminal · AI DRIVER (paste ONE line)", [
            (f"{driver} --model {N1_MODEL}", "1 car: n1_catch"),
            (f"{driver} --model {N3_MODEL}", "3 cars: the hive commander"),
        ], note="Finds the cars itself and starts DISARMED. Must match the CAMERA line."),
        Term(ai_machine, "Terminal · ARM / STOP", [
            ("arena go", "ARM"),
            ("arena halt", "E-STOP"),
        ], note="Stop everything: Ctrl-C in each terminal. The cars stop on their own "
                "within 0.3 s of losing commands."),
    ]


def _troubleshoot(extra: list[tuple[str, str]] | None = None) -> Section:
    rows = [
        ("<code>arena: command not found</code>",
         "Open a NEW terminal (or run <code>source ~/.bashrc</code>)."),
        ("The installer printed FAIL",
         "Read the line above it. The full log is in <code>~/.arena/install-*.log</code>. "
         "If it won't go away, use the <b>3_EMERGENCY_BOTH_ORINS</b> folder."),
        ("<code>arena up</code>: cannot SSH to vision",
         "Run <code>ssh-copy-id &lt;user&gt;@10.42.0.1</code> once on the AI Orin, or use the "
         "manual terminals below."),
        ("No cars found",
         "<code>arena cars</code>. Are the cars on? Is the AI Orin on the cars' WiFi?"),
        ("An alert on the dashboard",
         f"The alert table in <code>{REPO}/ARENA.md</code> says what each one means."),
        ("Which code version is installed?",
         f"<code>git -C {REPO} log -1 --oneline</code>"),
    ]
    return Section("If something is wrong", [Table(("symptom", "do this"), rows + (extra or []))])


# ------------------------------------------------------------------ the three guides
def vision_guide() -> Guide:
    folder = "1_VISION_ORIN"
    return Guide("vision", folder, "VISION ORIN",
                 "The Orin with the overhead camera plugged in. It turns every camera frame into "
                 "car positions and sends them down the Ethernet cable to the AI Orin. "
                 "It runs nothing else, so detection never waits on anything.",
                 [
                     _install(folder, "VISION ORIN", "install.sh"),
                     Section("2. Every session", [
                         Para("<b>Nothing to type here.</b> The AI Orin starts and stops this Orin's "
                              "camera service over the cable (<code>arena up</code> / "
                              "<code>arena stop</code>). Just check: powered on, camera plugged in, "
                              "Ethernet cable plugged in."),
                         Table(("", "1 car (N=1)", "3 cars (N=3)"), [
                             ("this Orin tracks", "tag 1 + the evader's tag 0",
                              "tags 1, 2, 3 + the evader's tag 0"),
                             ("chosen by", "the AI Orin (<code>arena fleet</code>)",
                              "the AI Orin (<code>arena fleet</code>)"),
                         ]),
                         Para("Other tags in view are ignored. If a car is not tracked, check its roof "
                              "tag is the right number, flat, and its top edge points at the car's nose."),
                         Term("VISION ORIN", "Terminal · CHECKS (only if something is off)", [
                             ("ls /dev/video*", "is the camera seen?"),
                             ("v4l2-ctl -d /dev/video0 --list-formats-ext | head -30",
                              "which modes it offers (needs MJPG 1280x720 @ 30)"),
                             ("ping -c 3 10.42.0.2", "is the cable to the AI Orin working?"),
                             (f"cd {REPO}/portable_orin_perception && bash preflight.sh",
                              "the full readiness check"),
                         ]),
                     ]),
                     Section("3. Manual mode (only if the AI Orin cannot start the camera)", [
                         Para("Use these if <code>arena up</code> on the AI Orin says it cannot SSH here. "
                              "Paste the 1-car OR the 3-car camera line, matching the model the AI Orin runs."),
                         *_manual_cards("VISION ORIN", "AI ORIN", "10.42.0.2", "10.42.0.1")[:1],
                         Term("VISION ORIN", "Terminal · CALIBRATE (CAMERA stopped first)", [
                             (f"cd {REPO}/portable_orin_perception && python3 scan_arena.py",
                              "calibrate from the corner tags (the camera opens once at a time)"),
                             ("sudo jetson_clocks", "full-speed clocks; after every power-on"),
                         ]),
                         Para("The AI Orin's half of manual mode is on its own sheet (2_AI_ORIN)."),
                     ]),
                     Section("Where things are on this Orin", [Table(("what", "where"), [
                         ("the code", f"<code>{REPO}</code>"),
                         ("arena calibration", f"<code>{REPO}/portable_orin_perception/config/arena_homography.yaml</code>"),
                         ("camera mode (1280x720 @ 30)", f"<code>{REPO}/portable_orin_perception/config/camera.yaml</code>"),
                         ("camera service log", "<code>~/.arena/logs/vision.log</code>"),
                         ("models", "not here: they live on the AI Orin"),
                     ])]),
                     _troubleshoot(),
                 ])


def ai_guide() -> Guide:
    folder = "2_AI_ORIN"
    return Guide("ai", folder, "AI ORIN",
                 "The Orin on the cars' WiFi. It runs the AI policy and the dashboard, and it is "
                 "where you type every command. It drives the vision Orin over the Ethernet cable.",
                 [
                     _install(folder, "AI ORIN", "install.sh"),
                     _fleet_overview(),
                     *_run_sections("AI ORIN"),
                     _models_section(folder, "install.sh"),
                     Section("Rehearse with no hardware", [
                         Term("AI ORIN", "Terminal 1 · ARENA", [
                             ("arena sim", "1 car (n1_catch): simulated arena + car; dashboard at "
                                           "http://localhost:8080; Ctrl-C to stop"),
                             (f"arena sim --model {N3_MODEL}", "3 cars: the hive commander"),
                             ("arena sim --buttons mod", "add this to either line: the sim cars are driven "
                                                         "through on/off remote buttons, like the real cars"),
                         ]),
                     ]),
                     Section("Manual mode (if `arena up` cannot reach the vision Orin)",
                             _manual_cards("VISION ORIN", "AI ORIN", "10.42.0.2", "10.42.0.1"),
                             intro="One window per service, in this order. Same result as "
                                   "<code>arena up</code>, without SSH."),
                     Section("One-time setup the installer may have left for later", [
                         Term("AI ORIN", "Terminal 1 · ARENA", [
                             ("ssh-copy-id <user>@10.42.0.1",
                              "let `arena` start the camera on the vision Orin (asks its password once)"),
                             ("arena init --vision <user>@10.42.0.1 --control $USER@localhost "
                              "--link-ip 10.42.0.2 --vision-link-ip 10.42.0.1",
                              "re-point `arena` at the vision Orin"),
                             ("arena cars", "list the cars on the WiFi and their CAR_INDEX"),
                             ("arena cars --index <car-ip> 1", "set a car's index (car stopped)"),
                         ], note="Replace <user> with the username on the vision Orin."),
                     ]),
                     _troubleshoot(),
                 ])


def emergency_guide() -> Guide:
    folder = "3_EMERGENCY_BOTH_ORINS"
    bundle = f"~/{folder}/files/code.bundle"
    return Guide("emergency", folder, "EMERGENCY · EITHER ORIN",
                 "Use this when an installer failed, when you only need the newest code (a "
                 "<code>git pull</code> without internet), or when one Orin is dead and the other "
                 "has to do everything. It works on either Orin and installs both jobs.",
                 [
                     Section("1. Pick the command", [
                         Steps([
                             f"Copy the whole <b>{folder}</b> folder into <b>Home</b>, open it, "
                             "right-click → <b>Open in Terminal</b>.",
                             "Paste ONE of these:",
                         ]),
                         Term("EITHER ORIN", "INSTALL (inside the folder)", [
                             ("bash emergency.sh --code-only",
                              "just the git pull: newest code, nothing else touched"),
                             ("bash emergency.sh",
                              "the git pull + everything both jobs need; asks what this Orin does"),
                         ], note="It asks: v = vision, a = AI, b = both (the other Orin is down), "
                                 "k = keep the current setup. Calibration, site config and models are kept."),
                     ]),
                     Section("2. One Orin doing both jobs (the other is down)", [
                         Para("Plug the camera into <b>this</b> Orin and put <b>this</b> Orin on the cars' "
                              "WiFi. If you answered <b>b</b>, it is already set up. Then run with 1 car "
                              "or 3 cars, below: each starts with the one-Orin setup line, which is "
                              "harmless to repeat."),
                         Para("Back to two Orins later (on the AI Orin): "
                              "<code>arena init --vision &lt;user&gt;@10.42.0.1 --control $USER@localhost "
                              "--link-ip 10.42.0.2 --vision-link-ip 10.42.0.1</code>"),
                     ]),
                     _fleet_overview(),
                     *_run_sections("THIS ORIN", [("arena init --single $USER@localhost",
                                                   "one Orin does both jobs (once)")]),
                     Section("3. The git pull by hand (if even the script fails)", [
                         Term("EITHER ORIN", "Terminal · GIT PULL", [
                             (f"cd {REPO} && git fetch {bundle} main && git checkout -B main FETCH_HEAD",
                              "update the code from this folder's bundle"),
                             (f"git clone -b main {bundle} {REPO}",
                              "no code on this Orin yet: make a fresh copy"),
                             (f"mkdir -p {REPO} && tar -xzf ~/{folder}/files/code.tar.gz -C {REPO} --strip-components=1",
                              "git itself is broken: plain files"),
                         ], note=f"These assume the folder is in Home (~/{folder})."),
                         Term("EITHER ORIN", "Terminal · GET EDITS BACK", [
                             (f"cd {REPO} && git stash list", "edits set aside during an update"),
                             (f"cd {REPO} && git stash apply", "put the newest set back"),
                         ]),
                     ]),
                     _models_section(folder, "emergency.sh"),
                     Section("Manual mode, one Orin (if `arena up` misbehaves)",
                             _manual_cards("THIS ORIN", "THIS ORIN", "127.0.0.1", "127.0.0.1")),
                     _troubleshoot(),
                 ])


GUIDES = {"vision": vision_guide, "ai": ai_guide, "emergency": emergency_guide}


# ------------------------------------------------------------------ text rendering
def _strip(s: str) -> str:
    import re
    s = s.replace("<br>", "\n    ")
    s = re.sub(r"</?(code|b|i)>", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def render_text(g: Guide, version: str) -> str:
    out = [f"EE198 · {g.title}", "=" * 72, _strip(g.lede), "", f"code: {version}", ""]
    for sec in g.sections:
        out += ["", sec.title.upper(), "-" * len(sec.title)]
        if sec.intro:
            out.append(_strip(sec.intro))
        for b in sec.blocks:
            if isinstance(b, Para):
                out += ["", _strip(b.text)]
            elif isinstance(b, Steps):
                out += [""] + [f"  {i}. {_strip(t)}" for i, t in enumerate(b.items, 1)]
            elif isinstance(b, Table):
                out.append("")
                for row in b.rows:
                    out.append(f"  * {_strip(row[0])}")
                    out += [f"      {_strip(h)}: {_strip(c)}" if h else f"      {_strip(c)}"
                            for h, c in zip(b.head[1:], row[1:])]
            elif isinstance(b, Term):
                out += ["", f"  [ {b.machine} · {b.label} ]"]
                for cmd, what in b.cmds:
                    out += [f"      {cmd}", f"          # {what}"]
                if b.note:
                    out.append(f"    note: {_strip(b.note)}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ HTML rendering
CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mute:#5d5d66;--line:#e2e0da;
--code-bg:#16181d;--code-ink:#e9eaee;--code-mute:#9aa0ad;--chip-ink:#fff;
--vision:#0f766e;--ai:#6d28d9;--either:#b45309;--this:#b45309;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#121316;--card:#1b1d22;--ink:#ececf0;
--mute:#a3a6b0;--line:#2c2f37;--code-bg:#0b0c0f;--accent:#7aa2ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Ubuntu,Roboto,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:28px 18px 64px}
header{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:8px}
.kicker{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute)}
h1{margin:4px 0 8px;font-size:32px;line-height:1.15}
h1 .dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:10px;vertical-align:middle;background:var(--role)}
.lede{color:var(--mute);margin:0;max-width:70ch}
.ver{margin-top:10px;font:12px ui-monospace,"DejaVu Sans Mono",Menlo,monospace;color:var(--mute)}
h2{font-size:19px;margin:34px 0 10px}
p{margin:10px 0;max-width:75ch}
ol{padding-left:22px;margin:8px 0}li{margin:4px 0}
code{font:13px ui-monospace,"DejaVu Sans Mono",Menlo,monospace;background:rgba(127,127,127,.14);padding:1px 5px;border-radius:4px}
.term{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:12px 0;overflow:hidden}
.term-h{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line)}
.chip{color:var(--chip-ink);font-size:11px;font-weight:700;letter-spacing:.08em;padding:3px 8px;border-radius:999px;background:var(--m)}
.tl{font-weight:600}
.cmds{background:var(--code-bg);color:var(--code-ink)}
.row{display:flex;align-items:flex-start;gap:10px;padding:9px 12px;border-top:1px solid rgba(255,255,255,.06)}
.row:first-child{border-top:0}
.cmdcol{flex:1;min-width:0}
.cmd{font:13.5px/1.5 ui-monospace,"DejaVu Sans Mono",Menlo,monospace;white-space:pre-wrap;word-break:break-all;margin:0}
.what{font-size:12.5px;color:var(--code-mute);margin-top:2px}
button.copy{flex:none;font:600 12px system-ui,sans-serif;border:1px solid rgba(255,255,255,.25);background:transparent;color:var(--code-ink);border-radius:6px;padding:4px 10px;cursor:pointer}
button.copy:hover{background:rgba(255,255,255,.1)}button.copy.done{border-color:#34d399;color:#34d399}
.tnote{padding:8px 12px;font-size:13px;color:var(--mute)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:10px 0}
th,td{text-align:left;padding:8px 12px;border-top:1px solid var(--line);vertical-align:top}
th{border-top:0;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
table.c2 td:first-child{width:36%}
table.c3 td:first-child{width:22%}
td code{overflow-wrap:anywhere;word-break:break-word}
.tw{overflow-x:auto}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:14px 0 0;font-size:13px}
nav a{color:var(--accent);text-decoration:none}nav a:hover{text-decoration:underline}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--m);border-radius:10px;padding:14px 16px}
.card h3{margin:0 0 6px;font-size:17px}.card p{margin:6px 0;font-size:14px;color:var(--mute)}
.card a{color:var(--accent);font-weight:600}
"""

JS = """
function copyText(t,b){
  const done=()=>{b.textContent='Copied';b.classList.add('done');setTimeout(()=>{b.textContent='Copy';b.classList.remove('done')},1400)};
  const fallback=()=>{const a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';
    document.body.appendChild(a);a.select();try{document.execCommand('copy');done()}catch(e){b.textContent='Select + Ctrl-C'}a.remove()};
  if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(t).then(done,fallback)}else{fallback()}
}
document.addEventListener('click',e=>{const b=e.target.closest('button.copy');if(!b)return;
  copyText(b.parentElement.querySelector('.cmd').textContent,b)});
"""

ROLE_COLOR = {"vision": "var(--vision)", "ai": "var(--ai)", "emergency": "var(--either)"}
MACHINE_COLOR = {"VISION ORIN": "var(--vision)", "AI ORIN": "var(--ai)",
                 "EITHER ORIN": "var(--either)", "THIS ORIN": "var(--this)"}


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _term_html(t: Term) -> str:
    color = MACHINE_COLOR.get(t.machine, "var(--accent)")
    rows = "".join(
        f'<div class="row"><div class="cmdcol"><pre class="cmd">{html.escape(c)}</pre>'
        f'<div class="what">{html.escape(w)}</div></div>'
        f'<button class="copy" type="button">Copy</button></div>'
        for c, w in t.cmds)
    note = f'<div class="tnote">{html.escape(t.note)}</div>' if t.note else ""
    return (f'<div class="term"><div class="term-h"><span class="chip" style="--m:{color}">'
            f'{html.escape(t.machine)}</span><span class="tl">{html.escape(t.label)}</span></div>'
            f'<div class="cmds">{rows}</div>{note}</div>')


def render_html(g: Guide, version: str) -> str:
    parts = [f"<title>EE198 {html.escape(g.title)}</title>",
             '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             f"<style>{CSS}</style>",
             f'<div class="wrap" style="--role:{ROLE_COLOR[g.key]}"><header>',
             f'<div class="kicker">EE198 offline install · folder {html.escape(g.folder)}</div>',
             f'<h1><span class="dot"></span>{html.escape(g.title)}</h1><p class="lede">{g.lede}</p>',
             f'<div class="ver">code {html.escape(version)}</div><nav>',
             *[f'<a href="#{_slug(s.title)}">{html.escape(s.title)}</a>' for s in g.sections],
             "</nav></header>"]
    for sec in g.sections:
        parts.append(f'<h2 id="{_slug(sec.title)}">{html.escape(sec.title)}</h2>')
        if sec.intro:
            parts.append(f"<p>{sec.intro}</p>")
        for b in sec.blocks:
            if isinstance(b, Para):
                parts.append(f"<p>{b.text}</p>")
            elif isinstance(b, Steps):
                parts.append("<ol>" + "".join(f"<li>{t}</li>" for t in b.items) + "</ol>")
            elif isinstance(b, Table):
                parts.append(f'<div class="tw"><table class="c{len(b.head)}"><tr>' + "".join(f"<th>{h}</th>" for h in b.head) + "</tr>"
                             + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in b.rows)
                             + "</table></div>")
            elif isinstance(b, Term):
                parts.append(_term_html(b))
    parts.append(f"</div><script>{JS}</script>")
    return "\n".join(parts) + "\n"


def render_index_html(version: str, sizes: dict[str, str]) -> str:
    cards = [
        ("1_VISION_ORIN", "var(--vision)", "Vision Orin",
         "The Orin with the overhead camera. Detects the cars, sends positions over the cable."),
        ("2_AI_ORIN", "var(--ai)", "AI Orin",
         "The Orin on the cars' WiFi. Runs the policy and dashboard; you type commands here. "
         "New models go in its PUT_NEW_MODELS_HERE folder."),
        ("3_EMERGENCY_BOTH_ORINS", "var(--either)", "Emergency (either Orin)",
         "An installer failed, you only need the newest code (offline git pull), or one Orin "
         "is down and the other must do everything."),
    ]
    body = "".join(
        f'<div class="card" style="--m:{c}"><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p>'
        f'<p><a href="{f}/START_HERE.html">{f}/START_HERE.html</a> · {html.escape(sizes.get(f, ""))}</p></div>'
        for f, c, t, d in cards)
    return "\n".join([
        "<title>EE198 Orin packages</title>",
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<style>{CSS}</style>",
        '<div class="wrap" style="--role:var(--accent)"><header>',
        '<div class="kicker">EE198 offline install</div>',
        '<h1><span class="dot"></span>Which folder goes on which Orin</h1>',
        '<p class="lede">Copy each folder to its Orin (into Home), open its START_HERE page, and '
        "paste the install command. No internet needed on the Orins.</p>",
        f'<div class="ver">code {html.escape(version)}</div></header>',
        f'<div class="cards">{body}</div>',
        "<p>Order the first time: install the <b>vision</b> Orin first, then the <b>AI</b> Orin "
        "(its installer copies an SSH key to the vision Orin, which must be on and cabled).</p>",
        "<h2>1 car or 3 cars</h2>",
        "<p>The arena runs <b>1 car with <code>n1_catch</code></b> unless you switch. On the AI Orin:</p>",
        _term_html(Term("AI ORIN", "Terminal 1 · ARENA", [
            (f"arena fleet --pursuers 3 --model {N3_MODEL}", "3 cars: the hive commander (experimental)"),
            (f"arena fleet --pursuers 1 --model {N1_MODEL}", "back to 1 car: the locked, tested default"),
        ])),
        "<p>The 2_AI_ORIN sheet has the whole session for each, step by step.</p>",
        f"</div><script>{JS}</script>"]) + "\n"


def render_index_text(version: str, sizes: dict[str, str]) -> str:
    return "\n".join([
        "EE198 ORIN PACKAGES -- which folder goes on which Orin",
        "=" * 60,
        f"code: {version}",
        "",
        f"1_VISION_ORIN           the Orin with the overhead camera        ({sizes.get('1_VISION_ORIN', '')})",
        f"2_AI_ORIN               the Orin on the cars' WiFi; you type     ({sizes.get('2_AI_ORIN', '')})",
        "                        commands here; new models go in its",
        "                        PUT_NEW_MODELS_HERE folder",
        f"3_EMERGENCY_BOTH_ORINS  either Orin: an installer failed, you    ({sizes.get('3_EMERGENCY_BOTH_ORINS', '')})",
        "                        only need the newest code (offline git",
        "                        pull), or one Orin must do everything",
        "",
        "Each folder: copy it into Home on the Orin, open START_HERE.html (or",
        "START_HERE.txt), right-click inside the folder -> Open in Terminal, and run:",
        "    bash install.sh          (emergency folder: bash emergency.sh)",
        "",
        "First time: install the VISION Orin first, then the AI Orin.",
        "",
        "1 CAR OR 3 CARS (on the AI Orin; the default is 1 car with n1_catch):",
        f"    arena fleet --pursuers 3 --model {N3_MODEL}",
        f"    arena fleet --pursuers 1 --model {N1_MODEL}",
        "The 2_AI_ORIN sheet has the whole session for each, step by step.",
        "",
    ])
