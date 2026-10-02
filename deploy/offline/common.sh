# common.sh -- the shared half of the three offline Orin installers.
# Sourced by install.sh / emergency.sh at the top of each package; never run.
#
# Nothing here needs the internet:
#   code            files/code.bundle  -- a git bundle, so updating is a `git pull`
#                                         from a file (files/code.tar.gz if git is missing)
#   Python packages files/wheels/cpXY/ -- installed with the bundled pip wheel
#   system tools    files/debs/<ubuntu>/ -- v4l-utils (camera controls) and iw (WiFi
#                                         power-save), only if missing
#
# Every step is idempotent: re-running a package, or a newer one, only changes
# what is missing or out of date. Steps report failure and carry on, and the
# summary at the end says what needs attention.

set -uo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
F="$PKG/files"
REPO="$HOME/ee198-deployment-bundles"
VENV="$HOME/.venvs/n1ctl"
BRANCH="$(cat "$F/BRANCH")"
COMMIT="$(cat "$F/COMMIT")"
ORIGIN_URL="$(cat "$F/ORIGIN_URL")"
ME="$(id -un)"
ASSUME_YES=0
NO_SUDO=0
RESULTS=()
FAILED=0

if [ -t 1 ]; then
    BOLD=$'\e[1m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RST=$'\e[0m'
else
    BOLD=""; GRN=""; YEL=""; RED=""; DIM=""; RST=""
fi

step() { printf '\n%s== %s%s\n' "$BOLD" "$*" "$RST"; }
ok()   { printf '  %sok%s    %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$YEL" "$RST" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n' "$RED" "$RST" "$*"; }
note() { printf '        %s\n' "$*"; }
die()  { printf '\n%sSTOPPED: %s%s\n' "$RED" "$*" "$RST"; exit 1; }
result() {  # result ok|warn|FAIL "what"
    RESULTS+=("$1|$2")
    [ "$1" = FAIL ] && FAILED=$((FAILED + 1))
    return 0
}

# ask "question" Y|N -> 0 for yes. Non-interactive or --yes: the default.
ask() {
    local q=$1 def=$2 hint reply
    if [ "$def" = Y ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
        printf '  %s %s %s(%s)%s\n' "$q" "$hint" "$DIM" "$def" "$RST"
        [ "$def" = Y ]; return
    fi
    read -r -p "  $q $hint " reply
    reply=${reply:-$def}
    case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# ask_value "question" default -> prints the answer
ask_value() {
    local q=$1 def=$2 reply
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then echo "$def"; return; fi
    read -r -p "  $q [$def] " reply
    echo "${reply:-$def}"
}

# ask_choice "question" default -> the first letter of the answer, lowercase
ask_choice() {
    local q=$1 def=$2 reply
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then echo "$def"; return; fi
    read -r -p "  $q " reply
    reply=${reply:-$def}
    echo "${reply:0:1}" | tr '[:upper:]' '[:lower:]'
}

have_sudo() {  # ask for the password once, lazily; --no-sudo skips every sudo step
    [ "$NO_SUDO" = 1 ] && return 1
    sudo -v 2>/dev/null || { warn "sudo refused -- skipping this step"; return 1; }
}

# ------------------------------------------------------------------ start / end
ee_begin() {  # ee_begin "TITLE" "$@"
    local title=$1; shift
    for a in "$@"; do
        case "$a" in
            --yes|-y) ASSUME_YES=1 ;;
            --no-sudo) NO_SUDO=1 ;;
        esac
    done
    [ "$(id -u)" -ne 0 ] || die "run this as your normal user, not with sudo: bash ${0##*/}
         (it asks for the sudo password itself when a step needs it)"
    [ "$(uname -s)" = Linux ] || die "this installer is for the Jetson Orins (Linux)."

    mkdir -p "$HOME/.arena"
    LOG="$HOME/.arena/install-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1

    printf '%s%s%s\n' "$BOLD" "================================================================" "$RST"
    printf '%s  EE198 offline install: %s%s\n' "$BOLD" "$title" "$RST"
    printf '  package code: %s\n' "$(head -n 1 "$F/VERSION.txt")"
    printf '  this Orin:    %s (%s), %s\n' "$(hostname)" "$(uname -m)" \
        "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown Linux}")"
    printf '  log:          %s\n' "$LOG"
    printf '%s%s%s\n' "$BOLD" "================================================================" "$RST"

    [ "$(uname -m)" = aarch64 ] || warn "this machine is $(uname -m), not an Orin (aarch64): the bundled Python packages will not install here"
    command -v python3 >/dev/null || die "python3 is missing. Every JetPack image has it -- is this an Orin?"
    PYTAG="$(python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
    WHEELS="$F/wheels/$PYTAG"
    PIPWHL=""
    if [ -d "$WHEELS" ]; then
        PIPWHL="$(ls "$WHEELS"/pip-*.whl 2>/dev/null | head -n 1)"
    else
        warn "no bundled Python packages for $(python3 --version 2>&1) (this package has: $(ls "$F/wheels" 2>/dev/null | tr '\n' ' '))"
        note "rebuild the package with that Python added (see deploy/offline/README.md)"
    fi
}

ee_end() {
    step "Summary"
    local r lvl what
    for r in "${RESULTS[@]}"; do
        lvl=${r%%|*}; what=${r#*|}
        case "$lvl" in
            ok) ok "$what" ;;
            warn) warn "$what" ;;
            *) bad "$what" ;;
        esac
    done
    echo
    if [ "$FAILED" -gt 0 ]; then
        printf '%s%d step(s) FAILED.%s Scroll up for the reason, or open the log:\n  %s\n' "$RED" "$FAILED" "$RST" "$LOG"
        printf 'Still stuck? The 3_EMERGENCY_BOTH_ORINS folder installs everything for both jobs.\n'
    else
        printf '%sDone.%s' "$GRN" "$RST"
        [ "${ARENA_CMD:-0}" = 1 ] && printf ' Open a NEW terminal so the `arena` command is found.'
        printf '\n'
    fi
    [ -n "${GUIDE_COPY:-}" ] && printf 'Command sheet: %s\n' "$GUIDE_COPY"
    sleep 0.3   # let tee flush before the shell prompt returns
    [ "$FAILED" -eq 0 ]
}

# ------------------------------------------------------------------ code ("git pull" from a file)
_git() { git -c user.name="ee198-offline" -c user.email="offline@localhost" -c advice.detachedHead=false "$@"; }

_copy_back_local_files() {  # old_dir: carry calibration + site config into the fresh checkout
    local old=$1 f d
    for f in portable_orin_perception/config/arena_homography.yaml \
             portable_orin_perception/config/camera.local.yaml \
             deploy/arena.local.conf; do
        if [ -f "$old/$f" ] && [ ! -e "$REPO/$f" ]; then
            mkdir -p "$(dirname "$REPO/$f")" && cp -p "$old/$f" "$REPO/$f" && note "kept your $f"
        fi
    done
    for d in "$old"/portable_n1_controller/models/*/; do
        [ -d "$d" ] || continue
        if [ ! -e "$REPO/portable_n1_controller/models/$(basename "$d")" ]; then
            cp -rp "$d" "$REPO/portable_n1_controller/models/" && note "kept your model $(basename "$d")"
        fi
    done
}

_update_from_tarball() {
    step "Code -> $REPO (plain files: git is not installed)"
    mkdir -p "$REPO"
    if tar -xzf "$F/code.tar.gz" -C "$REPO" --strip-components=1; then
        echo "$COMMIT" > "$REPO/.offline_commit"
        chmod +x "$REPO/arena" "$REPO"/deploy/*.sh "$REPO"/portable_orin_perception/*.sh 2>/dev/null
        ok "code at ${COMMIT:0:7} (files copied over the old ones; calibration and models untouched)"
        warn "git is missing, so \`arena status\` cannot show the code version"
        result ok "code updated to ${COMMIT:0:7} (no git)"
    else
        bad "could not unpack files/code.tar.gz"
        result FAIL "code update"
    fi
}

_clone_fresh() {
    _git clone --quiet --branch "$BRANCH" "$F/code.bundle" "$REPO" || { bad "git clone from the bundle failed"; return 1; }
    # Point origin back at GitHub, so an Orin that is ever online again can `git pull`.
    git -C "$REPO" remote set-url origin "$ORIGIN_URL"
}

ee_update_code() {
    if ! command -v git >/dev/null; then _update_from_tarball; return; fi
    local short=${COMMIT:0:7} out head tracking
    step "Code -> $REPO (branch $BRANCH @ $short)"

    if [ ! -e "$REPO" ]; then
        _clone_fresh || { result FAIL "code install"; return 1; }
        ok "installed at $short"; result ok "code installed at $short"; return 0
    fi
    if [ ! -d "$REPO/.git" ]; then
        local aside
        aside="$REPO.before-$(date +%Y%m%d-%H%M%S)"
        warn "$REPO exists but is not a git checkout; moving it to $aside"
        mv "$REPO" "$aside" || { bad "could not move it"; result FAIL "code update"; return 1; }
        _clone_fresh || { result FAIL "code install"; return 1; }
        _copy_back_local_files "$aside"
        ok "fresh checkout at $short"
        result ok "code installed fresh at $short (old folder kept as $aside)"
        return 0
    fi

    # The `git pull`, from a file: fetch the bundle, then move the branch to it.
    if ! out=$(_git -C "$REPO" fetch --quiet "$F/code.bundle" "$BRANCH" 2>&1); then
        bad "git could not read the bundle: $out"; result FAIL "code update"; return 1
    fi
    head="$(git -C "$REPO" rev-parse -q --verify HEAD 2>/dev/null)"
    if [ "$head" = "$COMMIT" ]; then
        _git -C "$REPO" checkout --quiet -B "$BRANCH" "$COMMIT" 2>/dev/null
        ok "already at $short"; result ok "code already current ($short)"; return 0
    fi
    if [ -n "$head" ] && git -C "$REPO" merge-base --is-ancestor "$COMMIT" "$head" 2>/dev/null; then
        warn "this Orin has NEWER code ($(git -C "$REPO" log -1 --format='%h %s' "$head")) than the package ($short)"
        if ! ask "Go back to the package's older code anyway?" N; then
            result warn "kept this Orin's newer code ${head:0:7} (package has $short)"; return 0
        fi
    fi

    # A file you added on this Orin where the new code now has one (e.g. a model
    # folder whose name was later committed) blocks the checkout: move it aside.
    local f stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    while IFS= read -r f; do
        [ -n "$f" ] && [ -e "$REPO/$f" ] || continue
        mv "$REPO/$f" "$REPO/$f.local-$stamp" && warn "the new code has its own $f: yours is kept as $f.local-$stamp"
    done < <(LC_ALL=C comm -12 <(git -C "$REPO" ls-files -o --exclude-standard | LC_ALL=C sort) \
                               <(git -C "$REPO" ls-tree -r --name-only "$COMMIT" | LC_ALL=C sort))

    # A tracked file edited on this Orin (e.g. camera_info.yaml after an intrinsics
    # calibration) rides along when the new code did not touch it. If it did:
    # set the edits aside, update, and put them back.
    if out=$(_git -C "$REPO" checkout --quiet -B "$BRANCH" "$COMMIT" 2>&1); then
        ok "updated ${head:0:7} -> $short"
    else
        warn "edits made on this Orin are in the way:"
        git -C "$REPO" status --short --untracked-files=no | sed 's/^/          /'
        local before after
        before="$(git -C "$REPO" rev-parse -q --verify refs/stash 2>/dev/null)"
        _git -C "$REPO" stash push --quiet -m "before offline update to $short ($(date '+%F %T'))" >/dev/null 2>&1
        after="$(git -C "$REPO" rev-parse -q --verify refs/stash 2>/dev/null)"
        if [ "$before" = "$after" ]; then   # nothing was set aside: the block is something else
            bad "git refused the update: $out"; result FAIL "code update"; return 1
        fi
        if ! out=$(_git -C "$REPO" checkout --quiet -B "$BRANCH" "$COMMIT" 2>&1); then
            _git -C "$REPO" stash pop --quiet   # the stash this run just made, nothing older
            bad "update failed even with the edits set aside: $out"; result FAIL "code update"; return 1
        fi
        if _git -C "$REPO" stash apply --quiet >/dev/null 2>&1; then
            _git -C "$REPO" stash drop --quiet
            ok "updated ${head:0:7} -> $short and put your edits back"
        else
            git -C "$REPO" reset --hard --quiet
            warn "your edits clash with the new code, so the new code won. They are saved:"
            note "see them:     cd $REPO && git stash show -p"
            note "restore them: cd $REPO && git stash apply"
            result warn "your edits are saved in \`git stash\` (they clash with the new code)"
        fi
    fi
    # The bookkeeping a real `git fetch origin` does, so `git status` reads "up to
    # date with origin/main" and an online `git pull` later starts from here.
    tracking="$(git -C "$REPO" rev-parse -q --verify "refs/remotes/origin/$BRANCH" 2>/dev/null)"
    if [ -z "$tracking" ] || git -C "$REPO" merge-base --is-ancestor "$tracking" "$COMMIT" 2>/dev/null; then
        git -C "$REPO" update-ref "refs/remotes/origin/$BRANCH" "$COMMIT"
    fi
    git -C "$REPO" branch --quiet --set-upstream-to="origin/$BRANCH" "$BRANCH" 2>/dev/null
    result ok "code updated to $short"
}

# ------------------------------------------------------------------ Python packages
_pip() {  # _pip <python> <pip args...>: the bundled pip, the bundled wheels, no network
    "$1" "$PIPWHL/pip" install --no-index --find-links "$WHEELS" \
        --disable-pip-version-check --no-warn-script-location --progress-bar off "${@:2}"
}

ee_vision_python() {
    step "Python packages for the vision service (system python3, --user)"
    [ -n "$PIPWHL" ] || { bad "no bundled packages for $PYTAG"; result FAIL "vision Python packages"; return 1; }
    local extra=()
    # Ubuntu 24.04 marks the system Python "externally managed"; a --user install is
    # what `arena setup` always did, and there it needs this flag.
    if python3 -c 'import os,sys,sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")))'; then
        extra+=(--break-system-packages)
    fi
    # Only what is missing gets installed: a package that already works is never
    # upgraded or downgraded (the system numpy that ROS was built against stays).
    { _pip python3 --user "${extra[@]}" -r "$F/requirements-vision.txt"; } 2>&1 | grep -v "already satisfied" || true
    if python3 - <<'PY'
import sys
try:
    import cv2, numpy, yaml
except ImportError as e:
    print(f"  missing: {e}"); sys.exit(1)
ok = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
print(f"        python {sys.version.split()[0]}, opencv {cv2.__version__} ({cv2.__file__}), numpy {numpy.__version__}")
sys.exit(0 if ok else 2)
PY
    then
        ok "OpenCV with the tag detector, numpy, yaml"
        result ok "vision Python packages"
    else
        bad "python3 cannot import a working OpenCV/numpy/yaml (see above)"
        result FAIL "vision Python packages"
    fi
}

ee_control_venv() {
    step "AI controller environment: $VENV"
    [ -n "$PIPWHL" ] || { bad "no bundled packages for $PYTAG"; result FAIL "AI controller environment"; return 1; }
    if [ -x "$VENV/bin/python" ]; then
        local vtag
        vtag="$("$VENV/bin/python" -c 'import sys; print("cp%d%d" % sys.version_info[:2])' 2>/dev/null)"
        if [ "$vtag" != "$PYTAG" ]; then
            warn "the existing environment is ${vtag:-broken}, the system is $PYTAG: rebuilding it"
            mv "$VENV" "$VENV.old-$(date +%Y%m%d-%H%M%S)"
        else
            ok "existing environment kept ($vtag); adding only what is missing"
        fi
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        mkdir -p "$(dirname "$VENV")"
        # --without-pip: needs no python3-venv/ensurepip, which a stock Jetson image
        # may lack and an offline apt cannot add. pip comes from the bundled wheel.
        python3 -m venv --without-pip "$VENV" || { bad "python3 -m venv failed"; result FAIL "AI controller environment"; return 1; }
        ok "created"
    fi
    # opencv + yaml are here too, so `arena sim` (every service in one Python) can
    # rehearse the whole stack on this Orin.
    { _pip "$VENV/bin/python" pip -r "$F/requirements-control.txt"; } 2>&1 | grep -v "already satisfied" || true
    if "$VENV/bin/python" -c "import numpy, onnxruntime, gymnasium, cv2, yaml; print('        onnxruntime', onnxruntime.__version__, '| numpy', numpy.__version__, '| gymnasium', gymnasium.__version__, '| opencv', cv2.__version__)"; then
        ok "onnxruntime, numpy, gymnasium, opencv, yaml"
        result ok "AI controller environment"
    else
        bad "the environment cannot import its packages (see above)"
        result FAIL "AI controller environment"
    fi
}

# ------------------------------------------------------------------ models
MODEL_CHECK_PY='
import sys
import numpy as np
sys.path.insert(0, ".")
from controller_runtime.pose_types import VehiclePose
from pc_controller.loops import make_loop     # the loop its manifest asks for
loop = make_loop(sys.argv[1])
p = loop.policy
for k in range(3):   # three frames through the real adapter -> policy path
    t = 0.1 * k
    a = loop.tick(pursuer_poses=[VehiclePose(-1.0 + 0.4 * i + 0.05 * k, -1.0, 0.3, t) for i in range(p.num_pursuers)],
                  evader_pose=VehiclePose(1.0, 1.0, 3.0, t))
assert a.shape == (p.action_dim,) and np.all(np.isfinite(a)), a
print(f"{p.num_pursuers} car(s), {p.input_dim} inputs -> {p.action_dim} outputs")
'
NEW_MODELS=()

ee_models() {  # copy PUT_NEW_MODELS_HERE/<name>/ into the code's models folder, checked
    local src="$PKG/PUT_NEW_MODELS_HERE" dst="$REPO/portable_n1_controller/models" d name info
    step "Models (from PUT_NEW_MODELS_HERE)"
    if [ ! -d "$dst" ]; then
        bad "the code is not installed on this Orin yet: run the full installer first"
        result FAIL "models (no code installed yet)"; return 1
    fi
    local found=0
    for d in "$src"/*/; do
        [ -d "$d" ] || continue
        name="$(basename "$d")"
        found=1
        if [ ! -f "$d/policy.onnx" ] || [ ! -f "$d/policy.onnx.manifest.json" ] || [ ! -f "$d/metrics.json" ]; then
            bad "$name: needs policy.onnx, policy.onnx.manifest.json and metrics.json (see the README in that folder)"
            result FAIL "model $name"; continue
        fi
        case "$name" in *[!A-Za-z0-9._-]*)
            bad "$name: use only letters, digits, . _ - in the folder name"; result FAIL "model $name"; continue ;;
        esac
        if git -C "$REPO" ls-files --error-unmatch "portable_n1_controller/models/$name/policy.onnx" >/dev/null 2>&1; then
            bad "$name: a model with that name ships with the code. Rename your folder (e.g. ${name}_v2)"
            result FAIL "model $name (name taken)"; continue
        fi
        rm -rf "$dst/.$name.new" && cp -r "$d" "$dst/.$name.new"
        if [ ! -x "$VENV/bin/python" ]; then
            warn "$name: copied, but not tested (no AI controller environment on this Orin)"
            rm -rf "${dst:?}/$name" && mv "$dst/.$name.new" "$dst/$name"
            NEW_MODELS+=("$name"); result warn "model $name installed, untested"; continue
        fi
        if info=$(cd "$REPO/portable_n1_controller" && "$VENV/bin/python" -c "$MODEL_CHECK_PY" "$dst/.$name.new" 2>&1); then
            rm -rf "${dst:?}/$name" && mv "$dst/.$name.new" "$dst/$name"
            ok "$name: $(echo "$info" | tail -n 1)  ->  models/$name"
            NEW_MODELS+=("$name"); result ok "model $name installed"
        else
            rm -rf "$dst/.$name.new"
            bad "$name: the controller cannot run it:"
            echo "$info" | grep -v '^ \|^Traceback' | tail -n 2 | sed 's/^/          /'
            result FAIL "model $name (does not run; not installed)"
        fi
    done
    [ "$found" = 1 ] || ok "none to add (the folder is empty); installed models: $(ls "$dst" | tr '\n' ' ')"
}

ee_choose_default_model() {  # offer to make a just-installed model the one `arena up` uses
    [ "${#NEW_MODELS[@]}" -gt 0 ] || return 0
    local m
    for m in "${NEW_MODELS[@]}"; do
        if ask "Make models/$m the model \`arena up\` drives with? (the default stays n1_catch)" N; then
            # `arena fleet` checks and saves it, with the car count following the
            # model (the shipped config pins 1 car, for n1_catch).
            if (cd "$REPO" && python3 deploy/arena.py fleet --pursuers auto --model "models/$m" >/dev/null); then
                ok "default model is now models/$m (back: arena fleet --pursuers 1 --model models/n1_catch)"
                result ok "default model models/$m"
            else
                bad "arena fleet refused models/$m"
                result FAIL "default model models/$m"
            fi
            return 0
        fi
    done
}

# ------------------------------------------------------------------ system tools (optional)
ee_tool() {  # ee_tool <command> <package> <why>
    local cmd=$1 pkg=$2 why=$3 codename dir
    step "System tool: $pkg ($why)"
    if command -v "$cmd" >/dev/null || [ -x "/usr/sbin/$cmd" ] || [ -x "/sbin/$cmd" ]; then
        ok "$cmd present"; result ok "$pkg present"; return 0
    fi
    codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
    dir="$F/debs/$codename/$pkg"
    if ! ls "$dir"/*.deb >/dev/null 2>&1; then
        warn "$cmd missing and this package has no $pkg for Ubuntu '$codename'"
        result warn "$pkg missing ($why)"; return 0
    fi
    if ! ask "$cmd is missing. Install $pkg from this package (needs the sudo password)?" Y || ! have_sudo; then
        result warn "$pkg not installed ($why)"; return 0
    fi
    # apt fetches everything before dpkg runs, so if a dependency is missing too it
    # fails on the (offline) download and changes nothing. Not --no-download: that
    # also drops the local .debs named here. Recommends would need a download.
    # A copy in /tmp keeps apt's unprivileged _apt user able to read them.
    local tmp
    tmp="$(mktemp -d)" && cp "$dir"/*.deb "$tmp"/ && chmod 755 "$tmp" && chmod 644 "$tmp"/*.deb
    if sudo apt-get install -y --no-install-recommends -o Acquire::Retries=0 \
            -o DPkg::Lock::Timeout=60 "$tmp"/*.deb; then
        ok "$pkg installed"; result ok "$pkg installed"
    else
        warn "apt could not install $pkg offline (see above); the system is unchanged"
        result warn "$pkg not installed ($why)"
    fi
    rm -rf "$tmp"
}

# ------------------------------------------------------------------ network
ee_wired_link() {  # ee_wired_link <this Orin's address>
    local ip=$1 dev out
    step "Orin-to-Orin Ethernet cable: this Orin = $ip"
    command -v nmcli >/dev/null || { warn "nmcli missing: set $ip/24 on the Ethernet port by hand"; result warn "cable link not set up"; return 0; }
    dev="$(nmcli -t -f DEVICE,TYPE device status 2>/dev/null | awk -F: '$2=="ethernet"{print $1; exit}')"
    [ -n "$dev" ] || { warn "no Ethernet port found"; result warn "cable link not set up (no Ethernet port)"; return 0; }
    have_sudo || { result warn "cable link not set up (no sudo)"; return 0; }
    # never-default: the cable never becomes the route to the internet/WiFi.
    local props=(connection.interface-name "$dev" ipv4.method manual ipv4.addresses "$ip/24"
                 ipv4.never-default yes ipv6.method ignore connection.autoconnect yes
                 connection.autoconnect-priority 50)
    if nmcli -t -f NAME connection show 2>/dev/null | grep -qx arena-link; then
        sudo nmcli connection modify arena-link "${props[@]}"
    else
        sudo nmcli connection add type ethernet con-name arena-link "${props[@]}" >/dev/null
    fi || { bad "nmcli could not save the link"; result FAIL "cable link"; return 1; }
    if out=$(sudo nmcli connection up arena-link 2>&1); then
        ok "$dev is up as $ip"
        result ok "cable link: $dev = $ip"
    elif [ "$(cat "/sys/class/net/$dev/carrier" 2>/dev/null)" != 1 ]; then
        ok "saved: $dev becomes $ip as soon as the cable is plugged in"
        result ok "cable link saved: $dev = $ip (plug the cable in)"
    else
        warn "saved, but the cable is in and the link would not start: $out"
        result warn "cable link saved but not up ($dev)"
    fi
}

ee_ssh_server() {  # the AI Orin starts the camera service here over SSH
    step "SSH server (the AI Orin uses it to start the camera service)"
    # Ubuntu 24.04 listens on ssh.socket and starts ssh.service on the first connection.
    if systemctl is-active --quiet ssh 2>/dev/null || systemctl is-active --quiet sshd 2>/dev/null \
            || systemctl is-active --quiet ssh.socket 2>/dev/null; then
        ok "running"; result ok "SSH server running"; return 0
    fi
    if systemctl list-unit-files 2>/dev/null | grep -q '^ssh\.service'; then
        if have_sudo && sudo systemctl enable --now ssh; then ok "started"; result ok "SSH server started"; return 0; fi
    fi
    warn "no SSH server: \`arena up\` on the AI Orin cannot start the camera here."
    note "Use the manual CAMERA terminal from the command sheet instead."
    result warn "no SSH server (use the manual CAMERA terminal)"
}

ee_configure_two_orin() {  # on the AI Orin: which Orin is the vision Orin, and SSH to it
    step "Telling \`arena\` where the vision Orin is"
    local vuser vip myip target
    vuser="$(ask_value "Username on the VISION Orin:" "$ME")"
    vip="$(ask_value "VISION Orin address (cable default):" "10.42.0.1")"
    myip="$(ask_value "THIS Orin's address the vision Orin sends poses to:" "10.42.0.2")"
    target="$vuser@$vip"
    if (cd "$REPO" && python3 deploy/arena.py init --vision "$target" --control "$ME@localhost" \
            --link-ip "$myip" --vision-link-ip "$vip" >/dev/null); then
        ok "deploy/arena.local.conf: vision = $target, poses -> $myip"
        result ok "arena configured (vision = $target)"
    else
        bad "arena init failed"; result FAIL "arena configuration"; return 1
    fi

    step "SSH key from this Orin to the vision Orin (so \`arena up\` can start the camera there)"
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"   # ssh-keygen -f does not create it
    [ -f "$HOME/.ssh/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -C "ee198-ai-orin"
    if ssh -o BatchMode=yes -o ConnectTimeout=4 "$target" true 2>/dev/null; then
        ok "already works"; result ok "SSH to the vision Orin"; return 0
    fi
    # A reflashed vision Orin (or another board on that address) has a new host key,
    # and ssh refuses it, even for ssh-copy-id. Offer to forget the old one.
    if ssh -o BatchMode=yes -o ConnectTimeout=4 "$target" true 2>&1 | grep -q "IDENTIFICATION HAS CHANGED"; then
        warn "the vision Orin at $vip has a new SSH identity (reflashed?), so SSH refuses it"
        if ask "Forget the old identity of $vip? (only do this if you know it was reflashed)" Y; then
            ssh-keygen -R "$vip" >/dev/null 2>&1 && ok "forgot it"
        else
            note "Later: ssh-keygen -R $vip   then   ssh-copy-id $target"
        fi
    fi
    if ! timeout 4 bash -c "exec 3<>/dev/tcp/$vip/22" 2>/dev/null; then
        warn "the vision Orin ($vip) is not answering right now (off, not installed yet, or cable out)."
        note "When it is up, run this once (asks the vision Orin's password):"
        note "  ssh-copy-id -o StrictHostKeyChecking=accept-new $target"
        result warn "SSH key not copied yet: ssh-copy-id $target"
        return 0
    fi
    if ask "Copy the key now? (asks the VISION Orin's password once)" Y; then
        ssh-copy-id -o StrictHostKeyChecking=accept-new -o ConnectTimeout=6 "$target" </dev/tty
    fi
    if ssh -o BatchMode=yes -o ConnectTimeout=4 "$target" true 2>/dev/null; then
        ok "SSH works"; result ok "SSH to the vision Orin"
    else
        warn "SSH key login still fails. Later: ssh-copy-id $target"
        result warn "SSH key not copied: ssh-copy-id $target"
    fi
}

ee_single_orin() {  # this Orin does both jobs
    if (cd "$REPO" && python3 deploy/arena.py init --single "$ME@localhost" >/dev/null); then
        ok "arena set to ONE Orin (this one) doing both jobs"
        result ok "arena configured: single Orin"
    else
        bad "arena init --single failed"; result FAIL "arena configuration"
    fi
}

# ------------------------------------------------------------------ the `arena` command, checks, guide
ee_arena_command() {
    step "The \`arena\` command"
    mkdir -p "$HOME/.local/bin"
    cat > "$HOME/.local/bin/arena" <<EOF
#!/usr/bin/env bash
# \`arena\` from any directory. Installed by the EE198 offline package.
# \`arena sim\` runs every service in one Python, so it uses the AI controller's
# environment (which also has OpenCV); everything else uses the system python3.
if [ "\${1:-}" = sim ] && [ -x "$VENV/bin/python" ]; then
    exec "$VENV/bin/python" "$REPO/deploy/arena.py" "\$@"
fi
exec "$REPO/arena" "\$@"
EOF
    chmod +x "$HOME/.local/bin/arena" "$REPO/arena" 2>/dev/null
    if ! grep -q '>>> ee198 arena >>>' "$HOME/.bashrc" 2>/dev/null; then
        cat >> "$HOME/.bashrc" <<'EOF'

# >>> ee198 arena >>>  (added by the EE198 offline installer)
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac
# <<< ee198 arena <<<
EOF
    fi
    ARENA_CMD=1
    if "$HOME/.local/bin/arena" --help >/dev/null 2>&1; then
        ok "\`arena\` works (open a new terminal to use it)"; result ok "\`arena\` command"
    else
        bad "\`arena --help\` failed"; result FAIL "\`arena\` command"
    fi
}

ee_selftest_vision() {
    step "Vision selftest"
    local out
    if out=$(cd "$REPO/portable_orin_perception" && python3 selftest.py 2>&1); then
        ok "$(echo "$out" | tail -n 1)"; result ok "vision selftest"
    else
        echo "$out" | tail -n 15 | sed 's/^/          /'
        bad "vision selftest FAILED"; result FAIL "vision selftest"
    fi
    out=$(cd "$REPO/portable_orin_perception" && bash preflight.sh 2>&1)
    echo "$out" | grep -E '\[(FAIL|WARN)\]' | sed 's/^ */          /'
    echo "$out" | grep -E 'passed .* warnings' | sed 's/^ */        /'
    if echo "$out" | grep -q 'NOT READY'; then
        warn "preflight found something to fix before driving (above)"
        result warn "preflight: fix the [FAIL] lines before driving"
    else
        ok "preflight: ready (warnings are steps you haven't done yet, like \`arena scan\`)"
    fi
}

ee_selftest_control() {
    step "AI controller selftest"
    local out
    if out=$(cd "$REPO/portable_n1_controller" && "$VENV/bin/python" selftest.py 2>&1); then
        ok "$(echo "$out" | tail -n 1)"; result ok "AI controller selftest"
    else
        echo "$out" | tail -n 15 | sed 's/^/          /'
        bad "AI controller selftest FAILED"; result FAIL "AI controller selftest"
    fi
}

ee_copy_guide() {  # the command sheet, somewhere a browser can open it
    local dest="$HOME"
    [ -d "$HOME/Desktop" ] && dest="$HOME/Desktop"
    GUIDE_COPY="$dest/$(basename "$PKG") - START HERE.html"
    cp "$PKG/START_HERE.html" "$GUIDE_COPY" 2>/dev/null || GUIDE_COPY=""
}
