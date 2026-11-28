#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
fit=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/wbc/phase-design-v1/basis.json
for scenario in load friction delay-bias combined; do
    for method in fixed eso decoupled hybrid; do
        mix=()
        case "$method" in
            eso) mix=(--mixin w61_wbc_internal_model:WBCObserver);;
            decoupled) mix=(--mixin w61_natural_wbc:NaturalWBCMixin);;
            hybrid) mix=(--mixin w61_wbc_internal_model:HybridWBC);;
        esac
        python scripts/w61_gait_rollout.py --contract experiments/terrain_memory/w61/wbc/perturbations.json \
           --fit "$fit" --case "$scenario" --role "$method" "${mix[@]}" --out "work/w61/wbc/perturb-v1-$scenario-$method"
    done
done
