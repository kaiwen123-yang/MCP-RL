#!/usr/bin/env bash
# Planned R1 closeout: 100 fixed-frequency episodes and 3 isolated Fig. 3 runs.
# Use an unused prefix; resume an interrupted cohort with the runner's seed range.
set -eu
source "$(dirname "$0")/tm01_env.sh"
cd "$TM01_ROOT"
stage="${1:?Pass frequency or timing}"
run="${2:?Pass a new output run prefix}"
case "$stage" in
    frequency)
        for frequency in 2p0 2p4; do
            config="experiments/tm01/r1-fixed-${frequency}.json"
            group="${run}-fixed-${frequency}"
            python -u scripts/run_tm01_r1.py --experiment table1 --method adaptive_fixed --device gpu --run "$group" --config "$config" --timing-context shared-throughput
            python scripts/video_tm01_r1.py --run "$group" --config "$config" --experiment table1 --methods adaptive_fixed
        done
        ;;
    timing)
        # Calling this stage declares that an exclusive resource window is available.
        # This flag records the condition; it neither reserves hardware nor stops jobs.
        config="experiments/tm01/r1-fig3-isolated.json"
        group="${run}-fig3-isolated"
        for method in acados naive mppi; do
            python -u scripts/run_tm01_r1.py --experiment fig3 --method "$method" --device gpu --run "$group" --config "$config" --timing-context isolated
        done
        # No reference-run: the timing figures must use these newly executed runs.
        python scripts/plot_tm01_r1.py --run "$group" --fig3-only
        python scripts/video_tm01_r1.py --run "$group" --config "$config" --experiment fig3
        ;;
    *)
        echo "Usage: bash scripts/run_tm01_r1_closeout.sh {frequency|timing} NEW_RUN_PREFIX" >&2
        exit 2
        ;;
esac
