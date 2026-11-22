#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p work/w57-m
g++ -O3 -fPIC -shared -x c++ -std=c++17 scripts/s1_w57_forecast_batch.cu -o work/w57-m/forecast_cpu.so
/usr/local/cuda/bin/nvcc -O3 --std=c++17 -arch=sm_120 -shared -Xcompiler=-fPIC scripts/s1_w57_forecast_batch.cu -o work/w57-m/forecast_gpu.so
