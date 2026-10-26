#!/usr/bin/env bash
# Source only in this project's experiment process.
TM01_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$TM01_ROOT/work/envs/pympc/bin:$PATH"
export ACADOS_SOURCE_DIR="$TM01_ROOT/work/terrain-memory/upstreams/Quadruped-PyMPC/quadruped_pympc/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$TM01_ROOT/work/envs/pympc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$TM01_ROOT/work/envs/pympc/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export CPATH="$TM01_ROOT/work/envs/pympc/include${CPATH:+:$CPATH}"

export PYTHONPATH="$TM01_ROOT/src:$TM01_ROOT/work/terrain-memory/upstreams/Quadruped-PyMPC${PYTHONPATH:+:$PYTHONPATH}"
