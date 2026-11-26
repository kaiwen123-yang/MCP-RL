#!/usr/bin/env bash
# Two independently qualified cases use one shared slot per physical episode.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
runtime=$(realpath "$1")
arm=$2
which_case=$3
first_attempt=${4:-01}
rel=terrain_memory/simulation/tm-s1-go2-local-readiness-r12
cd "$runtime"
source scripts/autodl_d57_env.sh
export NUMEXPR_NUM_THREADS=1
for level in E30 E60; do
    attempt=01
    [[ "$level" == E30 ]] && attempt=$first_attempt
    name="w59-${arm,,}-${which_case,,}-original-${level,,}-a$attempt"
    result="$root/results/$rel/w59/episodes/$name.json"
    if [[ ! -f "$result" ]]; then
        python "$root/scripts/s1_w59_lease.py" --runtime "$runtime" --slots 1 --job "$name" -- \
            python -u scripts/run_s1_w59_research.py --case "$which_case" --arm "$arm" \
            --task original --level "$level" --attempt "$name"
    fi
    python - "$result" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit('No terminal episode receipt for ' + str(p))
r = json.loads(p.read_text())
print(p.stem, r['stage_status'], r['simulation_s'], flush=True)
sys.exit(0 if r['stage_status'] == 'passed' else 1)
PY
done
