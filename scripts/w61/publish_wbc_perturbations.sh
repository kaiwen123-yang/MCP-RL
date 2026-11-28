#!/usr/bin/env bash
set -euo pipefail
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh pc
root=$PWD;protocol=perturb-v1
out=results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/wbc/perturbations-v1
while [ ! -s work/w61/wbc/perturb-v1-combined-hybrid/result.json ]; do sleep 12; done
python scripts/w61/stride_evidence.py --lane wbc --output "$out" \
    work/w61/wbc/perturb-v1-{load,friction,delay-bias,combined}-{fixed,eso,decoupled,hybrid}
cd work/w61/wbc
tar -cf - perturb-v1-{load,friction,delay-bias,combined}-{fixed,eso,decoupled,hybrid} \
    | gzip -1 > w61-d-perturbation-originals-v1.tar.gz
asset=w61-d-perturbation-originals-v1.tar.gz
tag=tm-s1-go2-local-readiness-r12-w61; repo=kaiwen123-yang/MCP-RL
gh release upload "$tag" --repo "$repo" "$asset"
gh api "repos/$repo/releases/tags/$tag" --jq ".assets[] | select(.name==\"$asset\") | {name,id,url:.browser_download_url,size,digest,state}" > "$root/$out/release.json"
python - "$asset" "$root/$out/release.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
file,receipt=map(Path,sys.argv[1:]);r=json.loads(receipt.read_text())
with file.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert (r['state'],r['size'],r['digest'])==('uploaded',file.stat().st_size,'sha256:'+digest)
print(r)
PY
cd "$root"; git add --sparse "$out"; git commit -m 'Deliver sixteen actuator and environmental perturbation trials and their verified originals'
