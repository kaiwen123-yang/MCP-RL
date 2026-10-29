#!/usr/bin/env bash
cd /home/fanyuan/projects/kaiwen-learning/terrain-memory-r12
export PATH=/home/fanyuan/miniconda3/envs/kaiwen-learn/bin:$PATH
export PYTHONPATH=$PWD/src:/home/fanyuan/projects/kaiwen-learning/terrain-memory-s1-go2-wb/work/terrain-memory/upstreams/mujoco_mpc-go2/python
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 python -u scripts/run_s1_go2_r12.py --attempt w1-S5-r21-E30-02 --workers 5 --window-start-unix 1793270873 --deadline-unix 1793283473
code=$?
printf '%s\n' "$code" >work/w1-E30-02.exit
exit "$code"
