#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p work/w56-c
TIMEFORMAT='wall_s=%R user_s=%U system_s=%S'
time g++ -O3 -fopenmp -fPIC -shared -x c++ -std=c++17 scripts/s1_w56_column.cu -o work/w56-c/column_cpu.so
time /usr/local/cuda/bin/nvcc -O3 --std=c++17 -arch=sm_120 -shared -Xcompiler=-fPIC scripts/s1_w56_column.cu -o work/w56-c/column_gpu.so
