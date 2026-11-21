#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p work/w56-r60
g++ -O3 -std=c++17 -fPIC -shared scripts/s1_w60_columns.cc -o work/w56-r60/columns.so
