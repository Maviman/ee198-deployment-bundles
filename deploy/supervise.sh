#!/usr/bin/env bash
# supervise.sh NAME DIR CMD... -- run CMD in DIR, restart it if it crashes.
#
# A crashed vision service or controller leaves the cars in failsafe (silence
# stops them), so the right response is a fast restart, not an operator
# noticing. Gives up after 5 crashes inside 60 s: a crash loop is a bug or a
# config error (arena not scanned, no cars found), and restarting forever only
# hides it. `arena logs NAME` shows why.
#
# Started by `arena up` under setsid, so the whole process group (this script
# plus its child) is stopped together by `arena stop`.
name=$1; dir=$2; shift 2
cd "$dir" || { echo "[supervise] cannot cd to $dir"; exit 1; }
child=0
trap 'kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; echo "[supervise] $name stopped"; exit 0' TERM INT
crashes=()
while true; do
    echo "[supervise] $(date +%T) starting $name: $*"
    "$@" &
    child=$!
    wait "$child"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[supervise] $(date +%T) $name exited cleanly"
        exit 0
    fi
    now=$(date +%s)
    crashes+=("$now")
    recent=0
    for t in "${crashes[@]}"; do [ $((now - t)) -lt 60 ] && recent=$((recent + 1)); done
    if [ "$recent" -ge 5 ]; then
        echo "[supervise] $(date +%T) $name failed $recent times in 60 s -- giving up. Fix the error above, then \`arena up\`."
        exit 1
    fi
    echo "[supervise] $(date +%T) $name exited with code $rc -- restarting in 1 s"
    sleep 1
done
