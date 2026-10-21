#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
upstream="$root/work/terrain-memory/upstreams/Quadruped-PyMPC"
conda create -y --prefix "$root/work/envs/pympc" --override-channels -c conda-forge python=3.11 pip pinocchio=4.1.0 casadi=3.7.2 cmake=4.4.4
"$root/work/envs/pympc/bin/python" scripts/prepare_terrain_memory_sources.py --upstream-only
git -C "$upstream" submodule update --init quadruped_pympc/acados
git -C "$upstream/quadruped_pympc/acados" submodule update --init external/hpipm external/blasfeo
# Only project-scoped library/header paths, never shell startup files.
source scripts/tm01_env.sh
pip install -r experiments/tm01/requirements.txt -e "$upstream"
python scripts/patch_tm01_gym.py
cmake -S "$ACADOS_SOURCE_DIR" -B "$ACADOS_SOURCE_DIR/build" -DACADOS_WITH_SYSTEM_BLASFEO=ON -DCMAKE_PREFIX_PATH="$root/work/envs/pympc" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$ACADOS_SOURCE_DIR/build" --target install -j2
pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
# Official renderer expected by this exact acados submodule.
mkdir -p "$ACADOS_SOURCE_DIR/bin"
curl -fL https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -o "$ACADOS_SOURCE_DIR/bin/t_renderer"
chmod u+x "$ACADOS_SOURCE_DIR/bin/t_renderer"
git -C "$upstream" apply "$root/experiments/tm01/author.patch"
