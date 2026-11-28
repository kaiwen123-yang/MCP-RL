#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
for q in query0 query1; do
    for method in library cpg feedback material orbit; do
        python scripts/w61_gait_rollout.py --contract experiments/terrain_memory/w61/gait/orbit.json --case "$q" --role "$method" --mixin w61_controlled_orbits:ControlledOrbit --out "work/w61/gait/v1-$method-$q"
    done
done
