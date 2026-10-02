# Offline install folders for the Orins

The Orins have no internet at school, and `arena setup` / `arena sync` both
need it (GitHub for the code, PyPI for the Python packages). These scripts
build three folders on a PC that does have internet. You carry them over on a
USB stick, and they install everything with no network at all.

```bash
python deploy/offline/build_packages.py          # from `main`; ~30 s after the first build
```

Output, in `dist/orin-packages/` (gitignored):

| folder | goes on | command on the Orin |
|---|---|---|
| `1_VISION_ORIN/` | the Orin with the camera | `bash install.sh` |
| `2_AI_ORIN/` | the Orin on the cars' WiFi | `bash install.sh` |
| `3_EMERGENCY_BOTH_ORINS/` | either Orin | `bash emergency.sh` (or `--code-only`) |

Each folder has a `START_HERE.html` (opens offline in the Orin's browser, with
a copy button on every command) and a `START_HERE.txt` with the same text.
The top-level `START_HERE.html` says which folder goes where.

**Rebuild after every code change you want on the Orins.** The folders carry
one commit. Commit first: uncommitted work is not included.

## What is inside a folder

| path | what | why |
|---|---|---|
| `files/code.bundle` | the branch as a git bundle | updating is a real `git pull`, from a file |
| `files/code.tar.gz` | the same commit as plain files | fallback when an Orin has no git |
| `files/wheels/cp310`, `cp312` | Python wheels for aarch64 | Ubuntu 22.04 (JetPack 6) and 24.04 |
| `files/debs/<jammy,noble>/` | `v4l-utils`, `iw` + their exact-version siblings | only installed if missing |
| `files/common.sh` | the installer logic | shared by all three folders |
| `PUT_NEW_MODELS_HERE/` | drop model folders here (AI + emergency) | copied in and test-run by the installer |

Options: `--model DIR` pre-loads a model into `PUT_NEW_MODELS_HERE` (repeat for
several), `--ref BRANCH` packages another branch, `--tar` also writes one
`.tar.gz` per folder, `--skip-debs` leaves out the Ubuntu tools.

## What the installers do

All three are idempotent: a re-run changes only what is missing or out of date.

- **Code** goes to `~/ee198-deployment-bundles`. An existing checkout gets
  `git fetch <bundle>` + `git checkout -B main <commit>`, like `arena sync`.
  Edits made on the Orin ride along when the new code did not touch that file.
  Otherwise they are stashed, the code is updated, and the stash is re-applied;
  if that clashes, the new code wins and the edits stay in `git stash`.
  Ignored files (calibration, `arena.local.conf`, `camera.local.yaml`) and your
  models are never touched. An Orin that already has newer code keeps it
  unless you say otherwise. `origin` stays pointed at GitHub, so a later
  online `git pull` works.
- **Python**: the vision service uses the system `python3` with a `--user`
  install (as `arena setup` does). The controller uses `~/.venvs/n1ctl`,
  created with `--without-pip` so it needs no `python3-venv` package; pip
  itself runs from the bundled wheel. **Only missing packages are installed.**
  A package that already works is never upgraded or downgraded, so ROS keeps
  the system numpy it was built against. The controller env also gets OpenCV,
  which lets `arena sim` run on an Orin.
- **Versions** are pinned in `constraints.txt` to what has run here: OpenCV
  4.10.0.84 (the first Orin's), onnxruntime 1.27.0 (1.23.2 on Python 3.10),
  gymnasium 1.3.0. The newest OpenCV is 5.x and untested on this hardware.
- **Optional, sudo**: the Orin-to-Orin cable (`nmcli` profile `arena-link`,
  10.42.0.1 vision / 10.42.0.2 AI, never the default route), and `v4l-utils` /
  `iw` from the bundled `.deb`s if missing. apt fetches everything before it
  installs, so offline it stops before changing anything if a dependency is
  also missing.
- **AI Orin**: runs `arena init` for the two-Orin layout, makes an SSH key, and
  offers `ssh-copy-id` to the vision Orin (or prints it for later).
- **Checks**: each selftest, plus `preflight.sh` on the vision side. A
  `~/.arena/install-<time>.log` keeps the full output.
- Puts an `arena` command in `~/.local/bin` (works from any directory) and a
  copy of the command sheet on the Desktop.

## Not included

- **ROS 2.** The arena fast path does not use it; installing it needs apt + internet.
- **PlatformIO / ESP32 flashing.** Flash the cars from the dev PC (`arena flash`).
- **A Python other than 3.10 or 3.12.** Add it to `PYTHONS` in `build_packages.py`.

## Tested

The tests run on the dev PC (Windows, Git Bash, real git, the real repo history).
None of this has run on an Orin yet.

- **Offline git update** (39 checks). Fresh install; the old PR #1 checkout
  with a calibration, an edited `camera_info.yaml` and a custom model; re-run;
  an edit merged back; a clashing edit kept in the stash; an Orin with newer
  code; a non-git folder; no git at all; an untracked file where the new code
  has one (moved aside); a stale git lock (fails, touches nothing).
- **Steps** (24 checks). pip from the wheel into a `--without-pip` venv;
  "already satisfied" left alone; models good, incomplete, corrupt and
  name-clashing; `arena init` for two Orins and one; the `arena` wrapper.
- `shellcheck -S warning` is clean.

Not testable here, so watch these on the first run: `python3 -m venv
--without-pip` on the Orin's Python, the `--user` install on 24.04, `nmcli`,
and `apt-get` with the local `.deb`s.
