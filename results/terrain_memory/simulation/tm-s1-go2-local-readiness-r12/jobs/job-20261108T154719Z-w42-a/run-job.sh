#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/MCP-RL-w42
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh academic
unset GH_TOKEN GITHUB_TOKEN
set +e
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 1 --resume /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w41-a/PANN_PPO_S5-seed1/resume-state-ep00387.pt --contract /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/execution-contract.json --background-job --job-start-unix 1794152839.26973 > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/train.log 2>&1
training_exit=$?
echo "$training_exit" > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/training-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/training-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/run_s1_go2_r12.py --job-dir /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/evaluation.log 2>&1
evaluation_exit=$?
echo "$evaluation_exit" > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/evaluation-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/evaluation-finished-at.txt
/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python -u scripts/finish_s1_go2_r12_job.py --job-dir /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a --training-exit-code "$training_exit" > /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261108T154719Z-w42-a/finish.log 2>&1
