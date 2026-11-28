"""Freeze and register the measured CPU implementation after its real comparison gates."""
import hashlib,json,os,shutil,time
from pathlib import Path
from build import HERE,ROOT,WORK,digest
from experiment import OUT, save,read,now

POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/work/w61/pool')
MARKER=POOL/'e-cpu-execution.json'

def freeze():
    base=WORK/'build/identity.json';build=read(base);library=Path(build['binary'])
    driver=(HERE/'driver.py').read_bytes()
    entry=(HERE/'entry.py').read_text()
    anchor='ROOT=HERE.parents[2]'
    if entry.count(anchor)!=1:raise ValueError('Cannot bind the deployment bootstrap to its owning scientific checkout')
    entry=entry.replace(anchor,"ROOT=Path(os.environ['W61_EXACT_ROOT']).resolve()").encode()
    tag=hashlib.sha256(build['sha256'].encode()+driver+entry).hexdigest()[:16]
    folder=WORK/('runtime-'+tag)
    folder.mkdir(exist_ok=True)
    for name,blob in [('driver.py',driver),('entry.py',entry),('exact-material.so',library.read_bytes())]:
        p=folder/name
        if p.exists() and p.read_bytes()!=blob:raise ValueError('Frozen runtime changed '+name)
        if not p.exists():p.write_bytes(blob)
    ident=dict(build,binary=str(folder/'exact-material.so'),original_build=str(base),source_binary_sha256=build['sha256'])
    save(folder/'build.json',ident)
    return dict(name='cpu-exact-'+tag,root=str(ROOT),driver=str(folder/'driver.py'),driver_sha256=digest(folder/'driver.py'),
                entry_sha256=digest(folder/'entry.py'),build=str(folder/'build.json'),binary=ident['binary'],
                binary_sha256=build['sha256'],base_collector_sha256=digest(ROOT/'scripts/w61/learning/collect.py'))

def certification():
    gates={}
    for stage in ('entry','full'):
        path=OUT/(stage+'.json')
        if not path.is_file():return None
        row=read(path)
        if row.get('status')!='closed' or not row.get('qualifies_for_stage'):return None
        gates[stage]=dict(receipt=str(path),sha256=digest(path),speedup=row['wall_speedup'])
    return gates

def activate():
    cert=certification()
    if not cert:raise ValueError('Both complete paired comparisons must qualify before enabling production')
    runtime=freeze()
    native=OUT/'native-e30.json'
    enabled=native.is_file() and read(native).get('qualifies_for_frozen_episodes') is True
    old=read(MARKER) if MARKER.exists() else {}
    if old and old['runtime']!=runtime:raise ValueError('An existing material backend belongs to a different immutable runtime')
    status=dict(study='w61-e-cpu-exact-v1',registered_shanghai=old.get('registered_shanghai',now()),runtime=runtime,
        training_enabled=True,curve_enabled=True,native_enabled=enabled,training_evidence=cert,
        frozen_evaluation_evidence=dict(receipt=str(native),sha256=digest(native),speedup=read(native)['speedup']) if enabled else None,
        switch='Job registration at natural batch boundaries, old leases retain their original command. A frozen head uses one native backend for all its cases.')
    save(MARKER,status)
    save(OUT/'production.json',status)
    return enabled

def watch():
    while certification() is None:
        for stage in ('entry','full'):
            p=OUT/(stage+'.json')
            if p.exists() and read(p).get('qualifies_for_stage') is False:return
        time.sleep(5)
    activate()
    while not (OUT/'native-e30.json').is_file():time.sleep(5)
    activate()

if __name__=='__main__':watch()
