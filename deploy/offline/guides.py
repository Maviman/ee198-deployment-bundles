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
    head: tuple[str, str]
    rows: list[tuple[str, str]]


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
             f"Two ship with the code: <code>n1_catch</code> (the default) and <code>n1_pin</code>."),
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


def _session_cards(machine: str = "AI ORIN") -> list:
    return [
        Term(machine, "Terminal 1 · ARENA (the one you type in)", [
            ("arena tune", "after EVERY power-on: full-speed clocks + WiFi power-save off (sudo)"),
            ("arena status", "health check: code, camera, clocks, calibration, alerts"),
            ("arena fleet", "how many pursuers drive, with which model, and which tag is which car"),
            ("arena scan", "calibrate from the corner tags (first time, or the camera moved)"),
            ("arena up", "start everything; cars stay DISARMED"),
            ("arena go", "ARM: the AI drives the cars"),
            ("arena halt", "E-STOP every car (services keep running; `arena go` re-arms)"),
            ("arena stop", "E-stop and shut everything down"),
        ]),
        Term(machine, "Terminal 2 · MONITOR (optional)", [
            ("arena monitor", "live numbers in the terminal; Ctrl-C to close"),
        ]),
        Term(machine, "BROWSER · DASHBOARD", [
            ("http://localhost:8080", "camera view, map, latency, per-car radio, and a STOP button"),
        ], note="Anyone watching can press STOP. Arming is only `arena go`, from the terminal."),
    ]


def _manual_cards(vision_machine: str, ai_machine: str, pose_to: str, preview_from: str) -> list:
    return [
        Term(vision_machine, "Terminal · CAMERA", [
            (f"cd {REPO}/portable_orin_perception && python3 -u run_vision.py "
             f"--pose-target {pose_to}:9870 --telemetry {pose_to}:9871 --preview-port 8090 --pursuers 1",
             "camera -> car positions -> the AI Orin. Ctrl-C to stop"),
        ], note="--pursuers must equal the AI DRIVER model's car count: 1 for n1_catch, 3 for a 3-car model."),
        Term(ai_machine, "Terminal · DASHBOARD", [
            (f"cd {REPO} && python3 -u deploy/hub.py --port 8080 --telemetry-port 9871 "
             f"--control-port 9872 --preview http://{preview_from}:8090",
             "the dashboard at http://localhost:8080"),
        ]),
        Term(ai_machine, "Terminal · AI DRIVER", [
            (f"cd {REPO}/portable_n1_controller && {VENV_PY} -u run_controller.py "
             "--model models/n1_catch --source udp --pose-port 9870 --esp auto "
             "--telemetry 127.0.0.1:9871 --control-port 9872 --start-disarmed",
             "the policy; finds the cars itself; starts DISARMED"),
        ], note="For 3 cars: a 3-car --model here, and --pursuers 3 on the CAMERA terminal."),
        Term(ai_machine, "Terminal · ARM / STOP", [
            ("arena go", "ARM"),
            ("arena halt", "E-STOP"),
        ], note="Stop everything: Ctrl-C in each terminal. The cars stop on their own "
                "within 0.3 s of losing commands."),
    ]


def _fleet_section(machine: str = "AI ORIN") -> Section:
    return Section("How many pursuers (1 car or 3)", [
        Para("Each pursuer has a roof tag and a CAR_INDEX (set once per car): "
             "<b>P1 = tag 1 = CAR_INDEX 0</b>, <b>P2 = tag 2 = CAR_INDEX 1</b>, "
             "<b>P3 = tag 3 = CAR_INDEX 2</b>. The model decides how many cars it can drive "
             "(the <code>n1_*</code> models drive 1). The vision Orin tracks only that many; "
             "any other tagged car is ignored."),
        Term(machine, "Terminal 1 · ARENA", [
            ("arena fleet", "show: number of pursuers, model, and P1/P2/P3 -> tag -> CAR_INDEX"),
            ("arena fleet --pursuers 3 --model models/<3-car model>",
             "drive 3 cars from now on (refused unless the model is a 3-car model)"),
            ("arena fleet --pursuers auto --model models/n1_catch", "back to 1 car"),
            ("arena up --model models/<name>", "one session only: the car count follows the model"),
            ("arena cars", "check every car answers, with CAR_INDEX 0, 1, 2"),
            ("arena cars --index <car-ip> 2", "give a car its CAR_INDEX (only while it is stopped)"),
        ], note="arena up refuses to start when the car count and the model disagree, and says "
                "which installed model would fit."),
    ])


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
                         Para("Use these if <code>arena up</code> on the AI Orin says it cannot SSH here."),
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
                     Section("2. Every session", [
                         Para("Power on both Orins, cable plugged in, camera on the vision Orin. "
                              "Then everything happens from one terminal on <b>this</b> Orin:"),
                         *_session_cards("AI ORIN"),
                         Para("Order: <code>tune</code> → <code>status</code> → "
                              "<code>scan</code> (when needed) → <code>up</code> → push the "
                              "cars by hand and watch them move on the dashboard → <code>go</code>."),
                     ]),
                     _fleet_section("AI ORIN"),
                     _models_section(folder, "install.sh"),
                     Section("Rehearse with no hardware", [
                         Term("AI ORIN", "Terminal 1 · ARENA", [
                             ("arena sim", "the whole stack with a simulated arena + cars; "
                                           "dashboard at http://localhost:8080; Ctrl-C to stop"),
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
                              "WiFi. If you answered <b>b</b>, it is already set up; otherwise run the "
                              "first line once."),
                         Term("THIS ORIN", "Terminal 1 · ARENA", [
                             ("arena init --single $USER@localhost", "one Orin does both jobs (once)"),
                             ("arena tune", "after every power-on (sudo)"),
                             ("arena fleet", "how many pursuers, which model (see the next sheet section)"),
                             ("arena scan", "calibrate (first time, or the camera moved)"),
                             ("arena up", "start everything; cars DISARMED"),
                             ("arena go", "ARM"),
                             ("arena halt", "E-STOP"),
                             ("arena stop", "shut down"),
                         ]),
                         Term("THIS ORIN", "BROWSER · DASHBOARD", [
                             ("http://localhost:8080", "camera, map, latency, STOP button"),
                         ]),
                         Para("Back to two Orins later (on the AI Orin): "
                              "<code>arena init --vision &lt;user&gt;@10.42.0.1 --control $USER@localhost "
                              "--link-ip 10.42.0.2 --vision-link-ip 10.42.0.1</code>"),
                     ]),
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
                     _fleet_section("THIS ORIN"),
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
                for a, c in b.rows:
                    out += [f"  * {_strip(a)}", f"      {_strip(c)}"]
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
td:first-child{width:36%}
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
                parts.append('<div class="tw"><table><tr>' + "".join(f"<th>{h}</th>" for h in b.head) + "</tr>"
                             + "".join(f"<tr><td>{a}</td><td>{c}</td></tr>" for a, c in b.rows) + "</table></div>")
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
        "</div>"]) + "\n"


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
    ])
