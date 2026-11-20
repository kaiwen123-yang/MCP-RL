#!/bin/bash
# Intended for the same uninterrupted W55 fit identity. Preserve natural outputs.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
py=work/envs/d57/bin/python
compact=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/c
raw=outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55-c
while tmux has-session -t '=s1-w55-c-targeted-fit' 2>/dev/null; do
  if "$py" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["state"]!="candidate_fitted")' "$compact/fit.json" 2>/dev/null; then break; fi
  if [ "$(tmux display-message -p -t '=s1-w55-c-targeted-fit' '#{pane_dead}')" = 1 ]; then break; fi
  sleep 3
done
"$py" -c 'import json,sys; assert json.load(open(sys.argv[1]))["state"]=="candidate_fitted"' "$compact/fit.json"
candidate="$raw/W55-short-load-ideal-potentials.npz"
taskset -c 20-21 env JAX_PLATFORMS=cpu "$py" -u scripts/s1_w55_c_historical.py evaluate >work/w55-c/historical-evaluate.log 2>&1 &
historical_pid=$!
taskset -c 18 "$py" -u scripts/s1_w55_c_replay.py --device cpu --weights "$candidate" --tag candidate N44 F51 N48 F52 B46 F53 >work/w55-c/candidate-evaluate.log 2>&1
taskset -c 18 "$py" -u scripts/s1_w55_c_replay.py --device cpu --tag frozen B46 F53 >work/w55-c/holdout-frozen.log 2>&1
taskset -c 18-19 "$py" -u scripts/check_s1_w55_c_columns.py candidate "$candidate" N44 >work/w55-c/candidate-map-check.log 2>&1
taskset -c 19 "$py" -u scripts/s1_w55_c_gpu_comparison.py >work/w55-c/candidate-comparison.log 2>&1
wait "$historical_pid"
taskset -c 18 "$py" -u scripts/s1_w55_c_known.py
taskset -c 18-19 "$py" -u scripts/s1_w55_c_observer.py
taskset -c 18 "$py" -u scripts/summarize_s1_w55_c.py
