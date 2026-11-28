#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
contract=experiments/terrain_memory/w61/gait/decoder-queries.json
fit=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/gait/orbit-decoder-v1
for x in heading-entry lateral-entry brake-recovery; do
    for method in fixed library cpg bridge 6111 6112 6113; do
        extras=()
        role=orbit
        case "$method" in
            fixed) role=fixed;;
            library|cpg) role="$method";extras=(--mixin w61_controlled_orbits:ControlledOrbit);;
            bridge) extras=(--mixin w61_support_bridge:SupportBridge);;
            *) extras=(--mixin w61_orbit_decoder:LearnedOrbit --fit "$fit/seed-$method.json");;
        esac
        python scripts/w61_gait_rollout.py --contract "$contract" --case "$x" --role "$role" "${extras[@]}" --out "work/w61/gait/decoder-v1-$x-$method"
    done
done
