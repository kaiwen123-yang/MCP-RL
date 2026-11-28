#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
root=$PWD;pool=/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py
fit=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/gait/style-cpg-v1
for x in heading-entry lateral-entry brake-recovery; do
    for seed in 6111 6112 6113; do
        python "$pool" --runtime "$root" --job "w61-b-style-$x-$seed" --slots 1 --cores 1 --ram-gib .7 --scratch-gib .15 -- \
          python scripts/w61_gait_rollout.py --contract experiments/terrain_memory/w61/gait/decoder-queries.json --case "$x" \
          --mixin w61_style_orbit:StyleOscillator --role cpg --fit "$fit/seed-$seed.json" \
          --out "work/w61/gait/style-v1-$x-$seed"
    done
done
