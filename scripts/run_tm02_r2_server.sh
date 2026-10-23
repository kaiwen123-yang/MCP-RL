#!/usr/bin/env bash
# Fixed full R2 budget. Run from the repository root on fy-5080.
set -e
R2_PYTHON=work/envs/tm02/bin/python
R2_RUN=${1:-tm02-rnn-20261023}
R2_CONFIG=${2:-experiments/tm02/r2-rnn.json}
env OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda "$R2_PYTHON" -u scripts/run_tm02_r2.py --config "$R2_CONFIG" --run "$R2_RUN" --phase train
env OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu "$R2_PYTHON" -u scripts/run_tm02_r2.py --config "$R2_CONFIG" --run "$R2_RUN" --phase evaluate
env OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu "$R2_PYTHON" -u scripts/run_tm02_r2.py --config "$R2_CONFIG" --run "$R2_RUN" --phase revisit
