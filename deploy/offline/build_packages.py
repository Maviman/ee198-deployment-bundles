#!/usr/bin/env python3
"""Build the three offline install folders for the Orins.

Run on a PC WITH internet (Windows, macOS or Linux); the Orins need none:

    python deploy/offline/build_packages.py                       # code from `main`
    python deploy/offline/build_packages.py --model D:/runs/n1_catch_v2   # pre-load a model
    python deploy/offline/build_packages.py --ref my-branch --tar

Writes, under dist/orin-packages/ (gitignored):

    START_HERE.html / READ_ME_FIRST.txt   which folder goes on which Orin
    1_VISION_ORIN/           install.sh + its START_HERE command sheet
    2_AI_ORIN/               install.sh + PUT_NEW_MODELS_HERE/
    3_EMERGENCY_BOTH_ORINS/  emergency.sh: the offline `git pull`, both jobs, either Orin

Each folder carries the code as a git bundle (the commit of --ref, committed
work only), the Python wheels for the Orins' ARM CPUs (Python 3.10 and 3.12),
and the two optional Ubuntu tools. Downloads are cached in dist/.cache.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guides  # noqa: E402

PYTHONS = ["3.10", "3.12"]           # Ubuntu 22.04 (JetPack 6) and 24.04
GLIBC_MAX = 35                       # Ubuntu 22.04's glibc; 24.04 has 2.39
PLATFORMS = ["manylinux2014_aarch64"] + [f"manylinux_2_{n}_aarch64" for n in range(17, GLIBC_MAX + 1)]
UBUNTU = ["jammy", "noble"]          # 22.04, 24.04
UBUNTU_MIRROR = "http://ports.ubuntu.com/ubuntu-ports"
DEB_ROOTS = {"vision": ["v4l-utils"], "ai": ["iw"], "emergency": ["v4l-utils", "iw"]}
MARKER = ".ee198-package"

PACKAGES = {
    # key: (folder, entry script source, entry script name, wheel sets, models folder?)
    "vision": ("1_VISION_ORIN", "install_vision.sh", "install.sh", ["vision"], False),
    "ai": ("2_AI_ORIN", "install_ai.sh", "install.sh", ["control"], True),
    "emergency": ("3_EMERGENCY_BOTH_ORINS", "emergency.sh", "emergency.sh", ["vision", "control"], True),
}

MODELS_README = """\
PUT NEW MODELS HERE
===================

One folder per model, named after the model, holding exactly these files:

    <name>/policy.onnx
    <name>/policy.onnx.manifest.json
    <name>/metrics.json

e.g.  PUT_NEW_MODELS_HERE/n1_catch_v2/policy.onnx  ...

Then run the installer (it copies each model to
~/ee198-deployment-bundles/portable_n1_controller/models/<name>/, test-runs
it, and offers to make it the default):

    bash install.sh --models-only        (AI folder)
    bash emergency.sh                    (emergency folder)

Rules:
  * Use a NEW name for a retrained model (n1_catch_v2, not n1_catch). The
    names that ship with the code are refused, so nothing silently changes.
  * Folder names: letters, digits, . _ - only.
  * The model must drive as many cars as you have (num_pursuers in the
    manifest). `arena cars` checks this.
  * The cars carry no cameras: a model trained with car cameras
    ("use_car_cameras": true in its manifest) is refused.

Use it once:    arena up --model models/<name>
Make it the default (the car count follows the model):
                arena fleet --model models/<name>
A 3-car model:  arena fleet --pursuers 3 --model models/<name>
List models and how many cars each drives:  arena fleet

Models come from the AI Training repo:
    python -m controller_runtime.export_onnx --run-dir <run>
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    print(f"build_packages: {msg}", file=sys.stderr)
    raise SystemExit(1)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ------------------------------------------------------------------ git
class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path

    def git(self, *args: str, binary: bool = False) -> str | bytes:
        p = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=self.path,
                           capture_output=True, text=not binary)
        if p.returncode != 0:
            err = p.stderr if not binary else p.stderr.decode(errors="replace")
            die(f"git {' '.join(args)} failed:\n{err}")
        return p.stdout if binary else p.stdout.strip()

    def show(self, commit: str, path: str) -> str:
        return self.git("show", f"{commit}:{path}")


def origin_ssh_url(url: str) -> str:
    """The Orins pull over SSH (the repo is private; ORIN_DEPLOYMENT.md phase 0.5)."""
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(\.git)?/?$", url)
    return f"git@github.com:{m.group(1)}/{m.group(2)}.git" if m else url


# ------------------------------------------------------------------ wheels
def requirement_sets(repo: Repo, commit: str) -> dict[str, str]:
    vision = repo.show(commit, "portable_orin_perception/requirements.txt")
    control = repo.show(commit, "portable_n1_controller/requirements.txt")
    # The controller's environment also gets OpenCV + yaml, so `arena sim` (every
    # service in one Python) can rehearse the whole stack on an Orin.
    control += "\n# added by the offline package: lets `arena sim` run on the Orin\nopencv-python>=4.7\npyyaml\n"
    return {"vision": vision, "control": control}


def constraints_for(pyver: str) -> str:
    """constraints.txt with its markers decided for the TARGET Python. pip download
    judges markers by the Python running it (this PC), not by --python-version."""
    try:
        from packaging.markers import Marker
    except ImportError:
        from pip._vendor.packaging.markers import Marker
    env = {"python_version": pyver, "python_full_version": pyver + ".0", "sys_platform": "linux",
           "platform_system": "Linux", "platform_machine": "aarch64", "os_name": "posix",
           "implementation_name": "cpython", "platform_python_implementation": "CPython"}
    keep = []
    for line in (HERE / "constraints.txt").read_text(encoding="utf-8").splitlines():
        req = line.split("#", 1)[0].strip()
        if not req:
            continue
        spec, _, marker = req.partition(";")
        if not marker.strip() or Marker(marker.strip()).evaluate(env):
            keep.append(spec.strip())
    return "\n".join(keep) + "\n"


def download_wheels(name: str, reqs: str, pyver: str, cache: Path) -> Path:
    constraints = constraints_for(pyver)
    key = hashlib.sha256("\n".join([reqs, constraints, pyver, *PLATFORMS]).encode()).hexdigest()[:12]
    dest = cache / "wheels" / f"{name}-cp{pyver.replace('.', '')}-{key}"
    if (dest / ".complete").exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    req_file, con_file = dest / "requirements.in", dest / "constraints.in"
    req_file.write_text(reqs + "\npip\n", encoding="utf-8")
    con_file.write_text(constraints, encoding="utf-8")
    abi = "cp" + pyver.replace(".", "")
    cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--disable-pip-version-check",
           "--dest", str(dest), "--only-binary=:all:", "--implementation", "cp",
           "--python-version", pyver, "--abi", abi,
           *[a for p in PLATFORMS for a in ("--platform", p)],
           "-r", str(req_file), "-c", str(con_file)]
    log(f"  downloading {name} wheels for Python {pyver} (aarch64)...")
    if subprocess.run(cmd).returncode != 0:
        die(f"pip download failed for {name} / Python {pyver}. Is this PC online?")
    req_file.unlink()
    con_file.unlink()
    (dest / ".complete").write_text(time.strftime("%F %T"), encoding="utf-8")
    return dest


# ------------------------------------------------------------------ Ubuntu .debs
def _order(c: str) -> int:
    if c == "~":
        return -1
    if c.isdigit() or not c:
        return 0
    if c.isalpha():
        return ord(c)
    return ord(c) + 256


def _verrevcmp(a: str, b: str) -> int:
    i = j = 0
    while i < len(a) or j < len(b):
        while (i < len(a) and not a[i].isdigit()) or (j < len(b) and not b[j].isdigit()):
            ac = _order(a[i]) if i < len(a) else 0
            bc = _order(b[j]) if j < len(b) else 0
            if ac != bc:
                return ac - bc
            i += 1
            j += 1
        while i < len(a) and a[i] == "0":
            i += 1
        while j < len(b) and b[j] == "0":
            j += 1
        first_diff = 0
        while i < len(a) and a[i].isdigit() and j < len(b) and b[j].isdigit():
            if not first_diff:
                first_diff = ord(a[i]) - ord(b[j])
            i += 1
            j += 1
        if i < len(a) and a[i].isdigit():
            return 1
        if j < len(b) and b[j].isdigit():
            return -1
        if first_diff:
            return first_diff
    return 0


def debver_cmp(a: str, b: str) -> int:
    """dpkg's version ordering (epoch:upstream-revision)."""
    def split(v: str) -> tuple[int, str, str]:
        epoch, _, rest = v.rpartition(":") if ":" in v else ("0", "", v)
        up, _, rev = rest.rpartition("-") if "-" in rest else (rest, "", "")
        return int(epoch or 0), up, rev
    ea, ua, ra = split(a)
    eb, ub, rb = split(b)
    if ea != eb:
        return ea - eb
    return _verrevcmp(ua, ub) or _verrevcmp(ra, rb)


def _fetch(url: str, dest: Path, max_age_s: float | None = None) -> Path:
    if dest.exists() and (max_age_s is None or time.time() - dest.stat().st_mtime < max_age_s):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.replace(dest)
    return dest


def ubuntu_index(release: str, cache: Path) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for pocket in (release, f"{release}-updates", f"{release}-security"):
        for comp in ("main", "universe"):
            gz = _fetch(f"{UBUNTU_MIRROR}/dists/{pocket}/{comp}/binary-arm64/Packages.gz",
                        cache / "debs" / "index" / f"{pocket}-{comp}.gz", max_age_s=24 * 3600)
            for stanza in gzip.decompress(gz.read_bytes()).decode("utf-8", "replace").split("\n\n"):
                fields = dict(re.findall(r"^([A-Za-z0-9-]+): (.*)$", stanza, re.M))
                if "Package" in fields:
                    idx.setdefault(fields["Package"], []).append(fields)
    return idx


def resolve_debs(root: str, idx: dict[str, list[dict]]) -> list[dict]:
    """The newest `root` plus every dependency pinned to an exact version (=),
    which is a sibling package from the same source that apt cannot mix."""
    import functools
    out: dict[str, dict] = {}

    def newest(name: str) -> dict:
        cands = idx.get(name) or die(f"Ubuntu package {name} not found")
        return max(cands, key=functools.cmp_to_key(lambda a, b: debver_cmp(a["Version"], b["Version"])))

    def add(st: dict) -> None:
        if st["Package"] in out:
            return
        out[st["Package"]] = st
        for alt in filter(None, (d.strip() for d in st.get("Depends", "").split(","))):
            first = alt.split("|")[0].strip()
            m = re.match(r"([a-z0-9.+-]+)(?::\w+)?\s*(?:\((=)\s*([^)]+)\))?", first)
            if m and m.group(2) == "=":
                exact = [s for s in idx.get(m.group(1), []) if s["Version"] == m.group(3).strip()]
                if not exact:
                    die(f"{st['Package']} needs {m.group(1)} = {m.group(3)}, not in the index")
                add(exact[0])
    add(newest(root))
    return list(out.values())


def download_debs(cache: Path) -> dict[str, dict[str, list[Path]]]:
    """{release: {root package: [deb paths]}}"""
    got: dict[str, dict[str, list[Path]]] = {}
    roots = sorted({r for rs in DEB_ROOTS.values() for r in rs})
    for rel in UBUNTU:
        log(f"  Ubuntu {rel} arm64: {', '.join(roots)}")
        idx = ubuntu_index(rel, cache)
        got[rel] = {}
        for root in roots:
            paths = []
            for st in resolve_debs(root, idx):
                dest = cache / "debs" / "pool" / Path(st["Filename"]).name
                _fetch(f"{UBUNTU_MIRROR}/{st['Filename']}", dest)
                if hashlib.sha256(dest.read_bytes()).hexdigest() != st.get("SHA256"):
                    dest.unlink()
                    die(f"checksum mismatch for {dest.name}")
                paths.append(dest)
            got[rel][root] = paths
            log(f"    {root}: " + ", ".join(p.name for p in paths))
    return got


# ------------------------------------------------------------------ assembly
def check_model(src: Path) -> None:
    missing = [f for f in ("policy.onnx", "policy.onnx.manifest.json", "metrics.json") if not (src / f).is_file()]
    if missing:
        die(f"--model {src}: missing {', '.join(missing)}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", src.name):
        die(f"--model {src}: folder name may only use letters, digits, . _ -")
    json.loads((src / "policy.onnx.manifest.json").read_text(encoding="utf-8"))


def write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def clean_dir(path: Path) -> None:
    if path.exists():
        if not (path / MARKER).exists():
            die(f"{path} exists and was not made by this script; move it away first")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / MARKER).write_text("built by deploy/offline/build_packages.py\n", encoding="utf-8")


def make_tar(folder: Path) -> Path:
    out = folder.with_suffix(".tar.gz")

    def fix(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 1000
        ti.uname = ti.gname = ""
        if ti.isdir() or ti.name.endswith(".sh"):
            ti.mode = 0o755
        else:
            ti.mode = 0o644
        return ti
    with tarfile.open(out, "w:gz") as tf:
        tf.add(folder, arcname=folder.name, filter=fix)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(HERE.parent.parent), help="the AI_Deployment checkout")
    ap.add_argument("--ref", default="main", help="branch to package (committed work only)")
    ap.add_argument("--out", default=None, help="output folder (default <repo>/dist/orin-packages)")
    ap.add_argument("--cache", default=None, help="download cache (default <repo>/dist/.cache)")
    ap.add_argument("--model", action="append", default=[], metavar="DIR",
                    help="a model folder to pre-load into PUT_NEW_MODELS_HERE (repeatable)")
    ap.add_argument("--skip-debs", action="store_true", help="leave out the optional Ubuntu tools")
    ap.add_argument("--tar", action="store_true", help="also write one .tar.gz per folder")
    args = ap.parse_args()

    repo = Repo(Path(args.repo).resolve())
    out = Path(args.out).resolve() if args.out else repo.path / "dist" / "orin-packages"
    cache = Path(args.cache).resolve() if args.cache else repo.path / "dist" / ".cache"
    models = [Path(m).resolve() for m in args.model]
    for m in models:
        check_model(m)

    # ---- the code
    branch = args.ref
    if not repo.git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"):
        die(f"--ref must be a local branch name ({branch!r} is not)")
    commit = repo.git("rev-parse", f"refs/heads/{branch}")
    subject = repo.git("log", "-1", "--format=%s", commit)
    cdate = repo.git("log", "-1", "--format=%cd", "--date=short", commit)
    version = f"{branch} @ {commit[:7]} ({cdate}) {subject}"
    log(f"packaging {version}")
    if repo.git("status", "--porcelain", "--untracked-files=no"):
        log("  note: this checkout has uncommitted changes; they are NOT in the packages (commit first)")
    upstream = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"],
                              cwd=repo.path, capture_output=True, text=True).stdout.strip()
    if upstream and upstream != commit:
        log(f"  note: {branch} differs from origin/{branch} (push it so GitHub has the same code)")
    origin = origin_ssh_url(repo.git("remote", "get-url", "origin"))

    stage = cache / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    bundle = stage / "code.bundle"
    repo.git("bundle", "create", str(bundle), f"refs/heads/{branch}")
    heads = repo.git("bundle", "list-heads", str(bundle))
    if commit not in heads:
        die(f"bundle check failed: {heads}")
    archive = stage / "code.tar.gz"
    repo.git("archive", "--format=tar.gz", "--prefix=ee198-deployment-bundles/", "-o", str(archive), commit)

    # ---- Python wheels and Ubuntu tools
    log("Python packages (cached after the first build):")
    reqs = requirement_sets(repo, commit)
    wheels = {(s, py): download_wheels(s, reqs[s], py, cache) for s in reqs for py in PYTHONS}
    debs = {} if args.skip_debs else (log("Ubuntu tools:") or download_debs(cache))

    # ---- the folders
    out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, str] = {}
    built_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for key, (folder, entry_src, entry_name, sets, has_models) in PACKAGES.items():
        pkg = out / folder
        clean_dir(pkg)
        files = pkg / "files"
        files.mkdir()
        shutil.copy2(bundle, files / "code.bundle")
        shutil.copy2(archive, files / "code.tar.gz")
        write_lf(files / "COMMIT", commit + "\n")
        write_lf(files / "BRANCH", branch + "\n")
        write_lf(files / "ORIGIN_URL", origin + "\n")
        write_lf(files / "common.sh", (HERE / "common.sh").read_text(encoding="utf-8"))
        write_lf(pkg / entry_name, (HERE / entry_src).read_text(encoding="utf-8"))
        for s in ("vision", "control"):
            if s in sets:
                write_lf(files / f"requirements-{s}.txt", reqs[s])
        lines = [version, f"built {built_at} on {platform.node()}", ""]
        for py in PYTHONS:
            wdir = files / "wheels" / ("cp" + py.replace(".", ""))
            wdir.mkdir(parents=True)
            for s in sets:
                for w in wheels[(s, py)].glob("*.whl"):
                    if not (wdir / w.name).exists():
                        shutil.copy2(w, wdir / w.name)
            lines.append(f"Python {py}: " + ", ".join(sorted(w.name.split("-")[0] + " " + w.name.split("-")[1]
                                                               for w in wdir.glob("*.whl"))))
        for rel, by_root in debs.items():
            for root in DEB_ROOTS[key]:
                ddir = files / "debs" / rel / root
                ddir.mkdir(parents=True)
                for d in by_root[root]:
                    shutil.copy2(d, ddir / d.name)
                lines.append(f"Ubuntu {rel}: " + ", ".join(d.name for d in by_root[root]))
        write_lf(files / "VERSION.txt", "\n".join(lines) + "\n")
        if has_models:
            mdir = pkg / "PUT_NEW_MODELS_HERE"
            mdir.mkdir()
            write_lf(mdir / "README.txt", MODELS_README)
            for m in models:
                shutil.copytree(m, mdir / m.name)
        g = guides.GUIDES[key]()
        write_lf(pkg / "START_HERE.html", guides.render_html(g, version))
        write_lf(pkg / "START_HERE.txt", guides.render_text(g, version))
        sizes[folder] = human(dir_size(pkg))
        if args.tar:
            t = make_tar(pkg)
            log(f"  {t.name}: {human(t.stat().st_size)}")

    write_lf(out / "START_HERE.html", guides.render_index_html(version, sizes))
    write_lf(out / "READ_ME_FIRST.txt", guides.render_index_text(version, sizes))
    shutil.rmtree(stage)

    log(f"\nbuilt {out}")
    for folder, size in sizes.items():
        log(f"  {folder:24s} {size}")
    if models:
        log(f"  models pre-loaded: {', '.join(m.name for m in models)}")
    log("\nNext: copy the whole folder to a USB stick, then each subfolder to its Orin.")


if __name__ == "__main__":
    main()
