#!/usr/bin/env bash
# AI ORIN installer (the Orin on the cars' WiFi: policy, dashboard, operator).
# No internet needed.
#
#   bash install.sh                run it from inside this folder
#   bash install.sh --models-only  just add the models in PUT_NEW_MODELS_HERE
#   bash install.sh --yes          accept every default, ask nothing
#   bash install.sh --no-sudo      skip the steps that need the sudo password
#
# Safe to re-run: it only changes what is missing or out of date.
source "$(dirname "${BASH_SOURCE[0]}")/files/common.sh"
ee_begin "AI ORIN (policy + dashboard + cars)" "$@"

if [[ " $* " == *" --models-only "* ]]; then
    ee_models
    ee_choose_default_model
    ee_end
    exit $?
fi

ee_update_code
ee_control_venv
ee_models
ee_tool iw iw "turns WiFi power-save off, which otherwise adds 100 ms naps"
if ask "Set up the Ethernet cable to the vision Orin (this Orin = 10.42.0.2)?" Y; then
    ee_wired_link 10.42.0.2
fi
ee_arena_command
ee_configure_two_orin
ee_choose_default_model
ee_selftest_control
ee_copy_guide
ee_end
