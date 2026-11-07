#!/usr/bin/env bash
# Activate the recorded project environment before invoking this wrapper.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
# Group/start and the frozen completion policy belong to this job; use the active Python.
exec python -u scripts/train_s1_go2_r12.py "$@"
