#!/usr/bin/env python3
"""Compile the declared R10 formulation and preserve its own source/binary."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--form', choices=['G'], required=True)
    a = p.parse_args()
    source = ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
    relative = Path('mjpc/tasks/quadruped/quadruped.cc')
    prior = ROOT/'work/r10-source-before'/relative
    prior.parent.mkdir(parents=True, exist_ok=True)
    if not prior.exists():
        shutil.copy2(source/relative, prior)
    old = prior.read_text()
    new = old.replace('ground_height + kFootRadius + step[foot]',
        'ground_height + model->geom_size[3*foot_geom_id_[foot]] + step[foot]')
    new = new.replace('ResidualFn::kFootRadius + ground[foot]',
        'model->geom_size[3*residual_.foot_geom_id_[foot]] + ground[foot]')
    new = new.replace('ResidualFn::kFootRadius + step[foot] + ground[foot]',
        'model->geom_size[3*residual_.foot_geom_id_[foot]] + step[foot] + ground[foot]')
    (source/relative).write_text(new)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
        fromfile='a/'+str(relative), tofile='b/'+str(relative)))
    (ROOT/'experiments/terrain_memory/s1-go2-r10-G.patch').write_text(patch)
    begin = time.time()
    subprocess.run(['cmake','--build',str(source/'build-autodl'),
        '--target','agent_server','-j','12'], check=True)
    frozen = ROOT/'work/r10-frozen'/a.form
    frozen.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'mjpc', frozen/'mjpc')
    shutil.copy2(source/'SOURCE_COMMIT', frozen/'SOURCE_COMMIT')
    shutil.copy2(source/'build-autodl/bin/agent_server', frozen/'agent_server')
    (frozen/'lib').mkdir()
    shutil.copy2(source/'build-autodl/lib/libmujoco.so.3.2.3', frozen/'lib/libmujoco.so.3.2.3')
    identity = dict(formulation=a.form, compiled_unix_s=time.time(),build_wall_s=time.time()-begin,
        source_sha256=hashlib.sha256(new.encode()).hexdigest(),
        binary_sha256=hashlib.sha256((frozen/'agent_server').read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        geometry='Per-foot radius from compiled model; physical geometry unchanged')
    (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity),flush=True)


if __name__ == '__main__':
    main()
