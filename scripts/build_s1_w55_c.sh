#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
TIMEFORMAT="wall=%3R user=%3U sys=%3S"; date -u +%s
mkdir -p work/w55-c
time g++ -O3 -fopenmp -fPIC -shared -x c++ -std=c++17 scripts/s1_w55_c_column.cu -o work/w55-c/column_cpu.so
time /usr/local/cuda/bin/nvcc -O3 --std=c++17 -arch=sm_120 -shared -Xcompiler=-fPIC scripts/s1_w55_c_column.cu -o work/w55-c/column_gpu.so
