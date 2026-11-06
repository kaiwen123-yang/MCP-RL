#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh academic
unset GH_TOKEN GITHUB_TOKEN
set +e
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 1 --resume /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt --contract /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/execution-contract.json --background-job --job-start-unix 1793961028.2264671 --deadline-unix 1794061828.2264671 > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/train.log 2>&1
training_exit=$?
echo "$training_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/training-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/training-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/run_s1_go2_r12.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/evaluation.log 2>&1
evaluation_exit=$?
echo "$evaluation_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/evaluation-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/evaluation-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/finish_s1_go2_r12_job.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a --hours 6 --training-exit-code "$training_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a/finish.log 2>&1
