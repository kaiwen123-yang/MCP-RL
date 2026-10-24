#!/usr/bin/env python3
"""Package only new S1 evidence on its execution host for direct Release upload."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',default='formal-v1')
    parser.add_argument('--release-tag',default='tm-s1-go2-contact-evidence-20261024')
    a=parser.parse_args()
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-contact.json').read_text())
    out=ROOT/c['outputs']['large_root']
    small=ROOT/c['outputs']['compact_root']
    indexpath=small/'index.json'
    index=json.loads(indexpath.read_text())
    if index['completed_episodes']!=85 or not index['fixed_matrix_complete']:
        raise ValueError('The authorized 85-episode matrix is not complete.')
    assets=[]
    cases=[(terrain,method) for terrain in [x['id'] for x in c['terrains']] for method in ['C0','C1','C2','C3']]+[('tm02_anchor','C4')]
    for terrain,method in cases:
        files=sorted((out/a.cohort).glob(terrain+'-'+method+'-*'))
        archive=out/(c['run_id']+'-'+terrain+'-'+method+'-trajectories.tar.gz')
        if not archive.exists():
            with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
                for path in files:
                    tar.add(path,arcname=str(path.relative_to(out)))
        assets.append(archive)
    development=out/(c['run_id']+'-development-and-model.tar.gz')
    if not development.exists():
        with tarfile.open(development,'w:gz',compresslevel=1) as tar:
            for folder in sorted(out.glob('development-*')):
                tar.add(folder,arcname=folder.name)
            for path in sorted((out/a.cohort).glob('*')):
                if not path.name.startswith(tuple(x['id']+'-' for x in c['terrains'])):
                    tar.add(path,arcname=str(path.relative_to(out)))
            for path in [ROOT/'scripts/run_terrain_memory_s1.py',*sorted((ROOT/'src/tm02').glob('s1*.py')),
                         ROOT/'experiments/terrain_memory/s1-go2-contact.json']:
                tar.add(path,arcname='source/'+str(path.relative_to(ROOT)))
    assets.append(development)
    assets.extend(sorted((out/'videos').glob('*.mp4')))
    url='https://github.com/kaiwen123-yang/MCP-RL/releases'
    index['publication']=dict(release_url=url+'/tag/'+a.release_tag,uploaded=False,
        user_computer_secondary_archive=False,weights_repackaged=False,
        assets=[dict(filename=p.name,execution_host_path=str(p),bytes=p.stat().st_size,sha256=sha(p),
                     url=url+'/download/'+a.release_tag+'/'+p.name) for p in assets])
    indexpath.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(index['publication'],indent=2),flush=True)


if __name__=='__main__':main()
