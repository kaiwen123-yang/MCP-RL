#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p work/w57-m
g++ -O3 -std=c++17 -fPIC -shared scripts/s1_w57_material_columns.cc -o work/w57-m/material.so
