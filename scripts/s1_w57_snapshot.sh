#!/usr/bin/env bash
# Freeze the executed D57 source while sharing only historical assets, raw paths and compact receipts.
set -eu
base=$(cd "$(dirname "$0")/.." && pwd)
rev=$(git -C "$base" rev-parse "$1^{commit}")
dest=$2
if [ -e "$dest" ]; then
    test "$(git -C "$dest" rev-parse HEAD)" = "$rev"
    test -e "$dest/work/envs/d57/bin/python"
    echo "$dest $rev"
    exit 0
fi
git -C "$base" worktree add --detach --no-checkout "$dest" "$rev"
cd "$dest"
git sparse-checkout set --no-cone '/scripts/' '/src/' '/experiments/' '/configs/' '/AGENTS.md' '/.gitignore' '/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f/F-A-r60/job.json' '/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f/F-B-r60/job.json'
git checkout --detach "$rev"
mkdir -p work/r11-frozen work/w56-r60 work/w57-m work/w57-p
for name in assets envs upstreams terrain-memory; do ln -s "$base/work/$name" "work/$name"; done
ln -s "$base/work/r11-frozen/S5-r41" work/r11-frozen/S5-r41
cp -L "$base/work/w56-r60/columns.so" work/w56-r60/
cp -L "$base/work/w57-m/material.so" work/w57-m/
if [ -e "$base/work/w57-p/candidate.so" ]; then cp -L "$base/work/w57-p/candidate.so" work/w57-p/; fi
ln -s "$base/outputs" outputs
r12=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12
ln -s "$base/$r12/w57" "$r12/w57"
echo "$dest $rev"
