#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh academic
unset GH_TOKEN GITHUB_TOKEN
set +e
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 2 --resume /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PANN_PPO_S5-seed2/resume-state-ep00251.pt --contract /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/execution-contract.json --background-job --job-start-unix 1793849034.713909 --deadline-unix 1793949834.713909 > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/train.log 2>&1
training_exit=$?
echo "$training_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/training-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/training-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/run_s1_go2_r12.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/evaluation.log 2>&1
evaluation_exit=$?
echo "$evaluation_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/evaluation-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/evaluation-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/finish_s1_go2_r12_job.py --job-dir /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b --hours 6 --training-exit-code "$training_exit" > /root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261105T032354Z-w39-b/finish.log 2>&1
