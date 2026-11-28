#!/usr/bin/env bash
# Starts only missing W61 media/curve services on the D57 execution node.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
source scripts/autodl_d57_env.sh
export PYTHONPATH="$root/scripts:$root/scripts/w61:$PYTHONPATH"
queue=work/w61/learning/W61-E-M1-hybrid-v1/e61-hybrid-m1-01/render-queue
mkdir -p "$queue"
case "${1:-start}" in
    render) exec ionice -c2 -n7 nice -n 10 python -u scripts/w61/evaluation_media.py serve >>"$queue/render.log" 2>&1 ;;
    physics|publish) exec python -u scripts/w61/curve_media_dispatch.py "$1" >>"$queue/curve-$1.log" 2>&1 ;;
    start)
        for role in render physics publish; do
            if [ "$role" = render ]; then session=w61-e-media-render; else session="w61-e-curve-$role"; fi
            if ! tmux has-session -t "=$session" 2>/dev/null; then
                tmux new-session -d -s "$session" -c "$root" "bash scripts/w61/start_media_services.sh $role"
            fi
        done ;;
    *) exit 2 ;;
esac
