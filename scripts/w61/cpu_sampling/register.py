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
    save(OUT/'status.json',dict(stage='production',production_selected=True,shanghai=status['registered_shanghai'],
        validated_binary=runtime['binary_sha256'],training=True,curve=True,native=enabled,
        qualification_receipts=['entry.json','full.json','native-e30.json'],deployment_receipt='production.json'))
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

def first_production():
    from deliver import compact
    # The normal programme writes its own original receipts. Record the real mapped
    # library, forward physical clock and the actual distinct update of its first pair.
    status_path=OUT/'first-production.json'
    while True:
        jobs=[read(p) for p in (POOL/'jobs').glob('w61-e61-hybrid-m1-01-*-train-v*.json')]
        jobs=[j for j in jobs if j.get('material_execution') and j.get('started') and j['command'][-1]=='train']
        if jobs:break
        time.sleep(3)
    job=min(jobs,key=lambda x:x['started'])
    name=job['job'];method,seed,version=job['command'][-4:-1]
    head=method.lower()+'-s'+seed
    v=int(version);study='e61-hybrid-m1-01'
    run=ROOT/'work/w61/learning/W61-E-M1-hybrid-v1'/study/head
    ledger=OUT.parent/study/head
    runtime=job['material_execution'];streams={}
    if not status_path.exists():
        for case in ('A','B'):
            episode=run/f'train-{v:03d}-{case}'/f'{study}-{head}-train-v{v:03d}-{case}-e000'
            binding=episode/'cpu-implementation.json'
            live=episode/'live.json'
            while not (binding.is_file() and live.is_file() and read(live)['observed']>0):time.sleep(2)
            snap=read(live);identity=read(episode/'identity.json');impl=read(binding)
            if (impl['sha256'],impl['source_wrapper_sha256'])!=(runtime['binary_sha256'],runtime['entry_sha256']):
                raise ValueError('The production collector has a different material image')
            maps=Path('/proc')/str(snap['pid'])/'maps'
            mapped=(any(line.split()[-1]==runtime['binary'] for line in maps.read_text().splitlines()) if maps.exists() else None)
            if mapped is False:raise ValueError('Active formal collector does not map the registered CPU binary')
            streams[case]=dict(original=str(episode),actor_sha256=identity['actor_sha'],image_sha256=impl['sha256'],
                               live_process_mapped_image=mapped,pid=snap['pid'],physical_s=snap['physical_s'],
                               closed_events_seen=snap['observed'],role=identity['role'],physical_dt=identity['physical_dt'])
        if streams['A']['actor_sha256']!=streams['B']['actor_sha256']:raise ValueError('Two cases have different actors')
        save(status_path,dict(job=name,started=job['started'],lease_pid=job['pid'],slots=job['slots'],head=head,
                              checkpoint_before=str(run/f'policy-{v:03d}.pt'),checkpoint_before_sha256=digest(run/f'policy-{v:03d}.pt'),
                              shanghai=now(),streams=streams,version=v,runtime=runtime['name']))
        compact([status_path,OUT/'status.json'],'Record actual material binary mappings in both formal W61 policy collectors')
    job_path=POOL/'jobs'/(name+'.json')
    while (job:=read(job_path))['state']!='ended':time.sleep(4)
    result=ledger/f'train-{v:03d}.json'
    closed=dict(job=name,ended=job['ended'],started=job['started'],returncode=job['exit_code'],
                original_receipt=str(result),original_receipt_sha256=digest(result) if result.exists() else None)
    if result.is_file():
        record=read(result)
        closed.update(status=record['status'],runtime=record['runtime_revision'],observed=[c.get('observed') for c in record['cases']])
        if record['status']=='updated':
            ids=read(run/f'used-{v:03d}.json')['event_ids']
            if len(ids)!=len(set(ids)) or len(ids)!=256 or runtime['binary_sha256'][:12] not in record['runtime_revision']:
                raise ValueError('Formal updated-batch event or implementation identity differs')
            closed.update(distinct_real_events=len(set(ids)),head_used=record['used'],written_checkpoint_sha256=record['updated_checkpoint_sha256'])
    canonical=OUT/'first-production-update.json'
    save(canonical,closed);compact([canonical],'Record the first official exact CPU paired training outcome')


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--first-production']:first_production()
    else:watch()
