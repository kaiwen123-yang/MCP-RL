#!/usr/bin/env bash
set -eu
root=$(git rev-parse --show-toplevel)
base=d584eb827c9253df9f888049579235f1e336e593
dver=b2557568df5324c6fe0997a51623f6b0f520e416
bver=$(git rev-parse HEAD)
dest="$root/work/w61/gait/b3-pair-v1/runtime"
shared=/root/autodl-tmp/MCP-RL-w59
mkdir -p "$(dirname "$dest")"
mkdir "$dest"
git archive "$base" scripts src experiments | tar -x -C "$dest"
for f in w61_b3_components.py w61_b3_pair.py w61_support_bridge.py w61_controlled_orbits.py; do
    git show "$bver:scripts/$f" >"$dest/scripts/$f"
done
for f in w61_natural_wbc.py w61_wbc_internal_model.py; do
    git show "$dver:scripts/$f" >"$dest/scripts/$f"
done
git show "$dver:results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/wbc/phase-design-v1/basis.json" >"$dest/phase-basis.json"
mkdir "$dest/work"
for f in envs assets upstreams r11-frozen w56-r60 w57-m terrain-memory; do
    ln -s "$shared/work/$f" "$dest/work/$f"
done
python - "$dest" "$base" "$bver" "$dver" <<'PY'
from pathlib import Path
import json,sys
root,base,b,d=sys.argv[1:]
(Path(root)/'pair-runtime.json').write_text(json.dumps(dict(base=base,gait=b,wbc=d,
    raw_dependencies='read-only links to existing W59 weights, model and native observer'),indent=2)+'\n')
PY
