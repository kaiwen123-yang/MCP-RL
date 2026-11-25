"""Reconcile both H receipts against each D57 process-file action and current physics."""
import collections
import json
import os
from pathlib import Path
import sys
import subprocess

from d57 import BG, CTL, RAW, ROOT, SMALL, free, now, read, save, sha

def main():
    h=json.load(sys.stdin)
    if h['status']!='completed' or h['serial']!='6063F4DA' or not h['restore_member_verified']:
        raise ValueError('The final H verification is incomplete')
    counts=collections.defaultdict(lambda:[0,0]);mounts=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
    absent=collections.defaultdict(set);expected=set();ids=set();retained={}
    stages=['duplicates','process']
    if (CTL/'followup/index.json').is_file():stages.append('followup')
    for stage in stages:
        base=CTL/stage
        index=read(base/'index.json');ids.update(index['closed_attempts'])
        for group in index['packs']:
            key=group['id'];items=read(base/(key+'.json'));receipts=CTL/'receipts'
            proof=read(receipts/(key+'-H-proof.json'))
            if proof['objects']!=[[e['h'],e['n']] for e in items] or not proof['h_readback']:
                raise ValueError('Missing H object verification '+key)
            journal={r['path']:r for r in map(json.loads,(receipts/(key+'-actions.jsonl')).read_text().splitlines())}
            for e in items:
                for name in e['links']+e['drops']:
                    if name in expected:raise ValueError('Duplicate retirement target '+name)
                    expected.add(name);p=Path(name);a=journal[name]['action'];size=e['n']
                    if journal[name]['bytes']!=size:raise ValueError('Recorded size differs '+name)
                    if a in ('linked','already_linked'):
                        src=Path(e['source'])
                        if not p.samefile(src) or p.stat().st_size!=size:raise ValueError('Changed frozen read path '+name)
                    elif a in ('removed','already_absent'):
                        absent[p.parent].add(p.name)
                    elif a.startswith('retained'):
                        retained[name]=a
                        if not p.is_file():raise ValueError('Reported retained source missing '+name)
                    else:raise ValueError(a)
                    b=counts[a];b[0]+=1;b[1]+=size
                    m=mounts['personal' if name.startswith('/autodl-fs/') else 'scratch'][a];m[0]+=1;m[1]+=size
            if len(journal)!=sum(len(x['links'])+len(x['drops']) for x in items):raise ValueError('Unmatched D57 actions '+key)
    for p,names in absent.items():
        present=names.intersection(f.name for f in os.scandir(p))
        if present:raise ValueError('A retired name remains at '+str(p/next(iter(present))))
    if h['duplicate_paths_linked']!=counts['linked'][0] or h['process_files_removed']!=counts['removed'][0]:
        raise ValueError('Final H and D57 counts differ')
    live=read(SMALL/'pool.json');v=live['latest'];physics={};weights={}
    for a in live['active']:
        path=RAW/a['id']/'live.json'
        b=path.read_text() if path.exists() else ''
        if b.endswith('\n}\n'):
            info=json.loads(b)
            if Path('/proc',str(info['pid'])).is_dir():
                physics[a['id']]=dict(pid=info['pid'],physics_s=info['physical_time_s'],phase=info['phase'])
        path=SMALL/'attempts'/(a['id']+'.json')
        if path.exists():
            parent=read(path);args=parent.get('command',[])
            if '--checkpoint' in args:
                p=Path(args[args.index('--checkpoint')+1])
                if p not in weights:
                    weights[p]=sha(p)
                if weights[p]!=parent['checkpoint_sha256']:raise ValueError('Checkpoint mismatch '+str(p))
    for name in ids:
        if not (SMALL/'attempts'/(name+'.json')).is_file() or read(SMALL/'delivery'/(name+'.json'))['status']!='verified':
            raise ValueError('Missing immutable scientific or Release receipt '+name)
    scored=[];physics_chunks=0
    for name in sorted(ids):
        a=read(SMALL/'attempts'/(name+'.json'))
        if a['status']=='technical_interruption':continue
        result=read(SMALL/'episodes'/(name+'.json'))
        raw=RAW/name
        if (raw/'trajectory.npz').stat().st_size!=result['trajectory']['size']:
            raise ValueError('Trajectory membership changed '+name)
        for part in result['chunks']:
            if part['stream']=='physical':
                if (raw/Path(part['path']).name).stat().st_size!=part['size']:
                    raise ValueError('Missing recorded physics chunk '+name)
                physics_chunks+=1
        if a['w58']['split']=='development' and a['task_mode']=='short':
            metrics=read(SMALL/'mechanics'/(name+'.json'))
            if not metrics['full_window'] or len(metrics['targets'])!=12:
                raise ValueError('Incomplete frozen short scoring '+name)
            scored.append(name)
    condensed={k:h[k] for k in ('packs','new_payload_bytes','previously_archived_objects','catalog_path','catalog_sha256','new_zip_bytes','verified_catalog_paths','unique_content_objects','restore_member_verified','serial')}
    report=dict(event='20261125/w58/h-archive-process-complete',status='completed',verified_shanghai=now(),
        scopes=['Completed or technically closed W58 attempts with D57 Release receipts at the archive inventory cut',
                'Exact duplicate sealed W58 chunks share a local inode after media readback. Distinct scientific inputs retain a local copy.',
                'Only explicitly numbered solver and drift diagnostics, plus bulk of a delivered interrupted prefix, qualify for unlink.'],
        closed_attempts=sorted(ids),scored_short_episodes=scored,retained_physical_chunks_checked=physics_chunks,
        verified_action_paths=len(expected),actions=dict(counts),
        actions_by_volume={k:dict(v) for k,v in mounts.items()},retained_open_or_changed=retained,
        H=condensed,free_bytes=free(),active_physics=physics,active_checkpoints_verified=len(weights),
        pool_sample_shanghai=v['time_shanghai'],distinct_learning=read(SMALL/'index.json')['joint_distinct'],
        initial_source_revision='662299057a8d7d3732bba80dc3e899b1b3e4cf6c',
        source_revision=subprocess.check_output(['git','log','-1','--format=%H','--','scripts/storage_d57_H_20261125'],cwd=ROOT,text=True).strip(),**BG)
    target=SMALL/'storage/20261125-H-process-archive.json'
    save(target,report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
