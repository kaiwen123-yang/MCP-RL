#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
fit=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/wbc/phase-design-v1/basis.json
contract=experiments/terrain_memory/w61/wbc/natural.json
for q in query0 query1; do
    for method in eso hybrid; do
        mix=w61_wbc_internal_model:WBCObserver
        if [ "$method" = hybrid ]; then mix=w61_wbc_internal_model:HybridWBC; fi
        python scripts/w61_gait_rollout.py --contract "$contract" --fit "$fit" --case "$q" --role "$method" --mixin "$mix" --out "work/w61/wbc/phase-v1-$method-$q"
    done
done
