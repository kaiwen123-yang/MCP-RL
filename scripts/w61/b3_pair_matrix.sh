#!/usr/bin/env bash
set -eu
source scripts/autodl_d57_env.sh
owner=/root/autodl-tmp/MCP-RL-w61-gait
out="$owner/work/w61/gait/b3-pair-v1"
for case in A B; do
    for arms in 'orbit fixed' 'fixed hybrid' 'orbit hybrid'; do
        read -r gait wbc <<< "$arms"
        python scripts/w61_b3_pair.py --source /root/autodl-tmp/MCP-RL-w59 --out "$out" \
               --case "$case" --gait "$gait" --wbc "$wbc"
    done
done
