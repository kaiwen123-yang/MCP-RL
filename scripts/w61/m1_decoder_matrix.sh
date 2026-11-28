#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
repo=$PWD
out="$repo/work/w61/gait/b3-heatmap-v1"; pool=/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py
for method in fixed library cpg orbit 6111 6112 6113; do
    for case in A B; do
        condition=yaw_plus5
        [ "$case" = A ] || condition=yaw_minus5
        python "$pool" --runtime "$out/runtime" --job "w61-b-m1-16-$case-$method" --slots 1 --cores 1 --gpu-mib 1400 --ram-gib 3 --scratch-gib 1 -- \
          bash -c 'source scripts/autodl_d57_env.sh; exec python scripts/w61_b3_pair.py "$@"' worker \
          --source /root/autodl-tmp/MCP-RL-w59 --out "$out" --case "$case" --gait "$method" --wbc fixed \
          --condition "$condition" --series w61-b-m1-16-v1 --window-s 16
    done
done
