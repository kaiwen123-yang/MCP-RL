#!/usr/bin/env bash
cd /home/fanyuan/projects/kaiwen-learning/terrain-memory-r12
export PATH=/home/fanyuan/miniconda3/envs/kaiwen-learn/bin:$PATH
export PYTHONPATH=$PWD/src:/home/fanyuan/projects/kaiwen-learning/terrain-memory-s1-go2-wb/work/terrain-memory/upstreams/mujoco_mpc-go2/python
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
nice -n 10 python -u scripts/train_s1_go2_r12.py --group "$1" --seed 0 --window-start-unix 1793277066 --deadline-unix 1793289666
