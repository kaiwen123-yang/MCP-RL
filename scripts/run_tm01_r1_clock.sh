#!/usr/bin/env bash
# Complete the declared clock-correction cohort; unchanged groups retain their
# original r1-foothold-restored trajectory paths in the comparison figures.
set -eu
source "$(dirname "$0")/tm01_env.sh"
cd "$TM01_ROOT"
device="${1:-gpu}"
run="${2:?Pass a new output run identifier}"
context="${3:-shared-throughput}"
config="experiments/tm01/r1-foothold-clock-repaired.json"
for method in adaptive_fixed adaptive; do
    python -u scripts/run_tm01_r1.py --experiment table1 --method "$method" --device "$device" --run "$run" --config "$config" --timing-context "$context"
done
python -u scripts/run_tm01_r1.py --experiment fig4 --method adaptive --device "$device" --run "$run" --config "$config" --timing-context "$context"
python scripts/plot_tm01_r1.py --run "$run" --reference-run r1-foothold-restored --methods acados naive adaptive_fixed adaptive
python scripts/compare_tm01_runs.py --run "$run" --clock-comparison
python scripts/analyze_tm01_clock.py --runs "$run"
python scripts/analyze_tm01_failures.py --runs "$run"
python scripts/replay_tm01_footholds.py --runs "$run"
python scripts/analyze_tm01_failures.py --runs "$run" --reference-figures-only
python scripts/video_tm01_r1.py --run "$run" --config "$config" --experiment table1 --methods adaptive_fixed adaptive
python scripts/video_tm01_r1.py --run "$run" --config "$config" --experiment fig4 --methods adaptive
