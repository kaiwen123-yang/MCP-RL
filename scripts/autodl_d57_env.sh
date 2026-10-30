#!/usr/bin/env bash
# Source this file to enter the D57 project environment; it does not start a job.
MPC_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$MPC_RL_ROOT/work/envs/d57/bin/activate"
export PYTHONPATH="$MPC_RL_ROOT/src:$MPC_RL_ROOT/work/upstreams/mujoco_mpc/python"
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export MUJOCO_GL=egl
cd "$MPC_RL_ROOT"
