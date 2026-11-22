#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p work/w57-p-equal
g++ -O3 -std=c++17 -DW57_IDENTICAL_STATES -fPIC -shared scripts/s1_w57_candidate.cc -o work/w57-p-equal/candidate.so
