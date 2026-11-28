#!/usr/bin/env bash
# Run on D57 from this gait worktree. Every asset is a finite original from a finished attempt.
set -euo pipefail
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh pc
root=$PWD
raw="$root/work/w61/gait/b3-pair-v1"
out=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/gait/bd-m1-e60-v1
tag=tm-s1-go2-local-readiness-r12-w61
repo=kaiwen123-yang/MCP-RL
export MUJOCO_GL=egl
mkdir -p "$raw/release"
for c in a b; do
    for arms in orbit-fixed fixed-hybrid orbit-hybrid; do
        id="w61-bd-v1-$c-$arms"
        while [ ! -f "$raw/episodes/$id.json" ]; do sleep 15; done
        if [ ! -f "$out/$id/result.json" ]; then
            python scripts/w61/bd_route_evidence.py --root "$raw" --out "$out" \
                --renderer "$raw/pinned_stride_evidence.py" \
                --protocol experiments/terrain_memory/w61/gait/reachable-targets-v1.json "$id"
        fi
        if [ -s "$out/$id/release.json" ]; then continue; fi
        parts="$raw/release/$id"
        mkdir -p "$parts"
        if [ ! -f "$parts/local.sha256" ]; then
            tar -C "$raw" -cf - "raw/$id" "jobs/$id" "episodes/$id.json" "episodes/$id-resolved.json" "streams/$id" \
                | gzip -1 | split -b 700M -d -a 2 - "$parts/$id-originals.tar.gz.part-"
            (cd "$parts" && sha256sum *part-* > local.sha256)
        fi
        cd "$parts"
        while read -r checksum asset; do
            meta=$(gh api "repos/$repo/releases/tags/$tag" --jq ".assets[] | select(.name==\"$asset\") | .digest")
            if [ -z "$meta" ]; then gh release upload "$tag" --repo "$repo" "$asset"; fi
            gh api "repos/$repo/releases/tags/$tag" --jq ".assets[] | select(.name==\"$asset\") | {name,id,url:.browser_download_url,size,digest,state}" >"$asset.remote.json"
            python - "$asset" "$checksum" "$asset.remote.json" <<'PY'
from pathlib import Path
import json,sys
asset,checksum,meta=sys.argv[1:]
r=json.loads(Path(meta).read_text())
assert (r['state'],r['size'],r['digest'])==('uploaded',Path(asset).stat().st_size,'sha256:'+checksum)
PY
        done <local.sha256
        cd "$root"
        python - "$parts" "$out/$id/release.json" <<'PY'
from pathlib import Path
import json,sys
d=Path(sys.argv[1]);parts=[json.loads(p.read_text()) for p in sorted(d.glob('*.remote.json'))]
Path(sys.argv[2]).write_text(json.dumps(dict(parts=parts,restore='concatenate parts by ascending suffix; unpack as tar.gz',includes='full video, native solver and physics telemetry, actor stream, resolved contract and physical model'),indent=2)+'\n')
PY
        git add --sparse "$out/$id"
        git commit -m "Deliver $id measured route and verified original assets"
    done
done
