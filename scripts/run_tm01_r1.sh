#!/usr/bin/env bash
set -eu
source "$(dirname "$0")/tm01_env.sh"
cd "$TM01_ROOT"
device="${1:-gpu}"
run="${2:-r1-foothold-restored}"
config="${3:-experiments/tm01/r1-foothold-restored.json}"
for pair in "fig3 acados" "fig4 acados" "table1 acados" "fig3 naive" "fig3 mppi" "fig4 adaptive" "table1 naive" "table1 adaptive"; do
    read -r experiment method <<< "$pair"
    python -u scripts/run_tm01_r1.py --experiment "$experiment" --method "$method" --device "$device" --run "$run" --config "$config"
done
# Same adaptive implementation, with only the frequency choice restricted.
python -u scripts/run_tm01_r1.py --experiment table1 --method adaptive_fixed --device "$device" --run "$run" --config "$config"
python scripts/plot_tm01_r1.py --run "$run" --methods acados naive adaptive_fixed adaptive
python scripts/video_tm01_r1.py --run "$run" --config "$config" --methods acados naive adaptive_fixed adaptive
