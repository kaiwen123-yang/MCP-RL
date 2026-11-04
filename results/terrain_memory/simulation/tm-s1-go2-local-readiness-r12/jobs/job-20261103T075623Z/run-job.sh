#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh clash
unset GH_TOKEN GITHUB_TOKEN
set +e
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 0 --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PANN_PPO_S5-seed0/resume-state-ep00144.pt --contract /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/execution-contract.json --background-job --job-start-unix 1793692583.50354 --deadline-unix 1793793383.50354 > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/train.log 2>&1
training_exit=$?
set -e
printf '%s\n' "$training_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/training-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/training-finished-at.txt
set +e
python -u scripts/run_s1_go2_r12.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/evaluation.log 2>&1
evaluation_exit=$?
set -e
printf '%s\n' "$evaluation_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/evaluation-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z/evaluation-finished-at.txt
python -u scripts/finish_s1_go2_r12_job.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261103T075623Z --hours 6.0 --training-exit-code "$training_exit"
