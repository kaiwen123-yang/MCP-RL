#!/usr/bin/env bash
# Install an immutable executable tree on D57. Analysis/publication use their own auxiliary CPUs.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
source scripts/autodl_d57_env.sh
mkdir -p work/w58
exec 9>work/w58/launch.lock
flock -x 9
rev=$(git rev-parse HEAD)
record="$root/work/w58/runtime.json"
if test -f "$record"; then
  snapshot=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$record")
  expected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$record")
  test "$(git -C "$snapshot" rev-parse HEAD)" = "$expected"
else
  git fetch --no-tags origin main
  git merge-base --is-ancestor "$rev" origin/main
  snapshot="$root-runtime-${rev:0:12}"
  git worktree add --detach --no-checkout "$snapshot" "$rev"
  rel=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12
  (cd "$snapshot"
   git sparse-checkout set --no-cone '/scripts/' '/src/' '/experiments/' '/configs/' '/.gitignore' '/AGENTS.md' "/$rel/w56/f/F-*-r60/job.json"
   git checkout --detach "$rev"
   ln -s "$root/work" work
   ln -s "$root/outputs" outputs
   ln -s "$root/$rel/w58" "$rel/w58")
  python - "$record" "$snapshot" "$rev" <<'PY'
import json, os, sys
from pathlib import Path
cpu=sorted(os.sched_getaffinity(0))
rec=dict(path=sys.argv[2],commit=sys.argv[3],physics_cpu_ids=cpu[:18],auxiliary_cpu_ids=cpu[18:22],
    initial_slots=3,conditional_max_slots=6,explicit_native_workers=1,
    native_execution='P2 FP64 synchronous on its calling thread. No four-worker executor in this identity.',
    background=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
Path(sys.argv[1]).write_text(json.dumps(rec,indent=2)+'\n')
PY
fi
launch() {
  local session="$1" source_dir="$2" cpu="$3" program="$4" log="$5" network="$6"
  if tmux has-session -t "=$session" 2>/dev/null; then
    tmux list-panes -t "=$session" -F '#{session_name} #{pane_pid} dead=#{pane_dead}'
    return
  fi
  local prefix=
  if test "$network" != none; then prefix="source scripts/autodl_network.sh $network >/dev/null &&"; fi
  tmux new-session -d -s "$session" -c "$source_dir" "source scripts/autodl_d57_env.sh && $prefix export W58_TMUX=$session W58_PHYSICS_CPUS=$physics S1_RELEASE_STREAMS=${S1_RELEASE_STREAMS:-3} && exec taskset -c $cpu python -u scripts/$program >>'$root/work/w58/$log' 2>&1"
}
physics=$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["physics_cpu_ids"])))' "$record")
read -ra aux <<< "$(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["auxiliary_cpu_ids"])' "$record")"
launch w58-unified-dispatch "$root" "${aux[0]}" queue_s1_w58.py dispatch.log none
launch w58-github-integration "$root" "${aux[1]}" integrate_s1_w58.py integration.log academic
S1_RELEASE_STREAMS=3 launch w58-d57-delivery "$root" "${aux[2]},${aux[3]}" publish_s1_w58.py delivery.log direct

