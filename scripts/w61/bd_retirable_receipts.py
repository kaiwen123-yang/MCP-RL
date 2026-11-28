"""Emit H-receiver candidates only for closed, verified, unused B/D upload staging."""
import argparse
import json
from pathlib import Path
import time

REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61')


def candidate(project,lane,study,run=None):
    work=project/'work/w61'/lane
    if run is None:
        return (project/REL/lane/study/'release.json',work,work/('h-retirable-'+study+'.json'))
    name,outset,stage=run
    return (project/REL/lane/outset/name/'release.json',work/stage/'release'/name,
            work/('h-retirable-'+name+'.json'))


def emit(receipt,directory,destination):
    if destination.exists():return True
    if not receipt.is_file():return False
    published=json.loads(receipt.read_text())
    members=published.get('parts',[published])
    result=[]
    for member in members:
        p=directory/member['name'];size=member['size'];digest=member['digest'].removeprefix('sha256:')
        if member['state']!='uploaded' or len(digest)!=64:
            raise ValueError(f'Incomplete remote verification: {receipt}')
        # The owning publisher matched its local hash and size before committing the
        # remote receipt. Only uploader staging can be a receiver candidate.
        if not p.exists():continue
        if p.stat().st_size!=size:
            raise ValueError(f'Staging bytes changed after verified publication: {p}')
        result.append(dict(path=str(p.resolve()),size=size,sha256=digest))
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(destination)
    print(destination,len(result),flush=True)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('gait',type=Path);p.add_argument('wbc',type=Path);a=p.parse_args()
    gait=a.gait.resolve();wbc=a.wbc.resolve()
    cases=[candidate(gait,'gait','rigid-orbits-v1'),candidate(gait,'gait','decoder-closed-v1'),
           candidate(wbc,'wbc','rigid-all-v1'),candidate(wbc,'wbc','perturbations-v1')]
    for c in ['a','b']:
        for arms in ['orbit-fixed','fixed-hybrid','orbit-hybrid']:
            n=f'w61-bd-v1-{c}-{arms}';cases.append(candidate(gait,'gait',None,(n,'bd-m1-e60-v1','b3-pair-v1')))
        for m in ['fixed','library','cpg','orbit','6111','6112','6113']:
            n=f'w61-b-m1-16-v1-{c}-{m}-fixed';cases.append(candidate(gait,'gait',None,(n,'m1-first-traverse-v1','b3-heatmap-v1')))
    n='w61-bd-reduced-v1-a-orbit-reduced';cases.append(candidate(gait,'gait',None,(n,'bd-reduced-v1','b3-reduced-v1')))
    while cases:
        cases=[c for c in cases if not emit(*c)]
        if cases:time.sleep(12)
