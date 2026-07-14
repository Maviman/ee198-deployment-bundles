"""Refresh this distribution repo from the main AI Training project.

The main project is the SOURCE OF TRUTH; this repo is the deployment/backup
mirror that machines like the Jetson Orin clone. Run after any bundle change
in the main project:

    python sync_from_main.py
    git add -A && git status   # eyeball the diff summary
    git commit -m "sync bundles from main project"
    git push

What it does:
  - Copies portable_n1_controller/ and portable_orin_perception/ here,
    excluding caches and build output (__pycache__, esp32/.pio, colcon
    build/install/log, generated arena_homography.yaml, .vscode).
  - Excludes wifi_credentials.h (the real credentials — gitignored in the
    main project too; the sketch ships wifi_credentials.h.example instead).
    Any inline WIFI_* assignment that ever reappears in the sketch is
    rewritten to a placeholder, then the whole staged tree is scanned and
    the sync ABORTS if anything credential-shaped survived.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = Path(r"G:\Codex\EE198 Senior Prject\AI Training")
BUNDLES = ["portable_n1_controller", "portable_orin_perception"]
EXCLUDE_DIRS = {"__pycache__", ".pio", "build", "install", "log", ".vscode",
                ".pytest_cache"}
EXCLUDE_FILES = {"arena_homography.yaml", "wifi_credentials.h"}

SKETCH = Path("portable_n1_controller/esp32/esp32_receiver/esp32_receiver.ino")
CRED_LINES = {
    re.compile(r'(const\s+char\*\s+WIFI_SSID\s*=\s*)"[^"]*"'): r'\1"YOUR_WIFI_SSID"',
    re.compile(r'(const\s+char\*\s+WIFI_PASS\s*=\s*)"[^"]*"'): r'\1"YOUR_WIFI_PASSWORD"',
}
# Post-copy tripwire: any quoted WIFI_* assignment that is not the placeholder.
LEAK_PATTERN = re.compile(r'WIFI_(SSID|PASS)\s*=\s*"(?!YOUR_WIFI_)[^"]+"')


def ignore_filter(directory: str, names: list[str]) -> set[str]:
    ignored = {n for n in names if n in EXCLUDE_DIRS or n in EXCLUDE_FILES}
    return ignored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="path to the main AI Training project")
    args = parser.parse_args()
    source = Path(args.source)
    dest_root = Path(__file__).resolve().parent

    for bundle in BUNDLES:
        src = source / bundle
        if not src.is_dir():
            sys.exit(f"source bundle missing: {src}")
        dst = dest_root / bundle
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_filter)
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        print(f"copied {bundle}: {n_files} files")

    sketch = dest_root / SKETCH
    if sketch.exists():
        text = sketch.read_text(encoding="utf-8")
        total = 0
        for pattern, replacement in CRED_LINES.items():
            text, n = pattern.subn(replacement, text)
            total += n
        if total:
            sketch.write_text(text, encoding="utf-8")
            print(f"sanitized {total} inline credential line(s) in {SKETCH}")
        else:
            print("no inline credentials in sketch (wifi_credentials.h layout)")

    leaks = []
    for bundle in BUNDLES:
        for path in (dest_root / bundle).rglob("*"):
            if not path.is_file() or path.suffix in {".png", ".pdf", ".onnx"}:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if LEAK_PATTERN.search(content):
                leaks.append(path.relative_to(dest_root))
    if leaks:
        sys.exit(f"ABORT: credential-shaped content survived sanitization in: {leaks}\n"
                 "Fix the source or extend sync_from_main.py, then re-run.")
    print("credential scan clean")
    print("\nnow: git add -A && git status (review!) && git commit && git push")


if __name__ == "__main__":
    main()
