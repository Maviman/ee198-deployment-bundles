#!/usr/bin/env bash
# VISION ORIN installer (the Orin with the camera). No internet needed.
#
#   bash install.sh            run it from inside this folder
#   bash install.sh --yes      accept every default, ask nothing
#   bash install.sh --no-sudo  skip the steps that need the sudo password
#
# Safe to re-run: it only changes what is missing or out of date.
source "$(dirname "${BASH_SOURCE[0]}")/files/common.sh"
ee_begin "VISION ORIN (camera -> car positions)" "$@"

ee_update_code
ee_vision_python
ee_tool v4l2-ctl v4l-utils "locks the camera's exposure and focus"
if ask "Set up the Ethernet cable to the AI Orin (this Orin = 10.42.0.1)?" Y; then
    ee_wired_link 10.42.0.1
fi
ee_ssh_server
ee_arena_command
ee_selftest_vision
ee_copy_guide
ee_end
