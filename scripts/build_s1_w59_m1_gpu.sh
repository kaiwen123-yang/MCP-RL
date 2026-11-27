#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=work/w59/m1-gpu-study
mkdir -p "$out"
if [[ "${1:-}" == --one-column-block ]]; then
    /usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_120 --fmad=false -DW59_ONE_COLUMN_BLOCK \
        -shared -Xcompiler=-fPIC scripts/s1_w59_m1_gpu.cu -o "$out/propagate-one-column.pending.so"
    mv -T "$out/propagate-one-column.pending.so" "$out/propagate-one-column.so"
    sha256sum "$out/propagate-one-column.so"
    exit 0
fi
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_120 --fmad=false \
    -shared -Xcompiler=-fPIC scripts/s1_w59_m1_gpu.cu -o "$out/propagate.pending.so"
mv -T "$out/propagate.pending.so" "$out/propagate.so"
g++ -O3 -std=c++17 -fPIC -shared -I scripts scripts/s1_w59_m1_correction.cc \
    -o "$out/correction.pending.so"
mv -T "$out/correction.pending.so" "$out/correction.so"
sha256sum "$out/propagate.so" "$out/correction.so"
