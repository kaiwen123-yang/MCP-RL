#!/usr/bin/env bash
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh academic >/dev/null
remaining=$((1793374023 - $(date +%s) - 30))
timeout --signal=TERM --kill-after=20s "${remaining}s" bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 0 --window-start-unix 1793359623 --deadline-unix 1793371623
code=$?
printf '%s\n' "$code" > outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PANN_PPO_S5-seed0/d57-window-launch.exit
