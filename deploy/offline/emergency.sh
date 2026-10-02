#!/usr/bin/env bash
# EMERGENCY installer: works on EITHER Orin and installs BOTH jobs, so this one
# Orin can run everything alone if the other is down. No internet needed.
#
#   bash emergency.sh              the git pull + everything both jobs need
#   bash emergency.sh --code-only  ONLY the git pull (fastest; nothing else touched)
#   bash emergency.sh --yes        accept every default, ask nothing
#   bash emergency.sh --no-sudo    skip the steps that need the sudo password
#
# Safe to re-run. Your calibration, site config and models are kept.
source "$(dirname "${BASH_SOURCE[0]}")/files/common.sh"
ee_begin "EMERGENCY (either Orin, both jobs)" "$@"

ee_update_code
if [[ " $* " == *" --code-only "* ]]; then
    ee_end
    exit $?
fi

ee_vision_python
ee_control_venv
ee_models
ee_tool v4l2-ctl v4l-utils "locks the camera's exposure and focus"
ee_tool iw iw "turns WiFi power-save off, which otherwise adds 100 ms naps"
ee_arena_command
step "What does this Orin do?"
note "v = VISION  (camera plugged in)"
note "a = AI      (on the cars' WiFi; you type the arena commands here)"
note "b = BOTH    (the other Orin is down: this one does everything)"
note "k = keep    (leave the current setup alone)"
case "$(ask_choice "Choose v / a / b / k [k]:" k)" in
    v)  if ask "Set up the Ethernet cable to the AI Orin (this Orin = 10.42.0.1)?" Y; then ee_wired_link 10.42.0.1; fi
        ee_ssh_server ;;
    a)  if ask "Set up the Ethernet cable to the vision Orin (this Orin = 10.42.0.2)?" Y; then ee_wired_link 10.42.0.2; fi
        ee_configure_two_orin ;;
    b)  ee_single_orin ;;
    *)  ok "setup left as it was (deploy/arena.local.conf unchanged)" ;;
esac
ee_choose_default_model
ee_selftest_vision
ee_selftest_control
ee_copy_guide
ee_end
