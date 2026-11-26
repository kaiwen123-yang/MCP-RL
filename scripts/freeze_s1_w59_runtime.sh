#!/usr/bin/env bash
# The shared workspace holds originals and leases. This detached tree owns the execution source.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
rev=$(git rev-parse "${1:-HEAD}^{commit}")
rt="$root-runtime-${rev:0:12}"
rel=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12
if [[ ! -d "$rt" ]]; then
    git worktree add --detach --no-checkout "$rt" "$rev" >&2
    (
        cd "$rt"
        git sparse-checkout set --no-cone '/scripts/' '/src/' '/experiments/' '/configs/' '/.gitignore' '/AGENTS.md' "/$rel/w56/f/F-*-r60/job.json"
        git checkout --detach "$rev" >&2
        ln -s "$root/work" work
        ln -s "$root/outputs" outputs
        ln -s "$root/$rel/w59" "$rel/w59"
    )
fi
test "$(git -C "$rt" rev-parse HEAD)" = "$rev"
echo "$rt"
