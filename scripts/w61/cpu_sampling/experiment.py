"""Compare whole, on-policy Go2 collectors with the measured packet optimization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from build import ROOT,HERE,WORK,digest

REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/learning')
STUDY='w61-e-cpu-exact-v1'
OUT=ROOT/REL/STUDY
OG=ROOT/'scripts/w61/learning/collect.py'
START=ROOT/'work/w61/learning/W61-E-M1-hybrid-v1/e61-hybrid-m1-01'
LEASE=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')
ZONE=ZoneInfo('Asia/Shanghai')
def now():return datetime.now(ZONE).isoformat(timespec='seconds')
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.pending');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())
def spec():return read(ROOT/'experiments/terrain_memory/w61/learning/cpu-exact-v1.json')

def one_run(config,stage,arm,case,cpus,directory):
    info=config['stages'][stage];ck=START/'ppo-s6001'/f"policy-{info['version']:03d}.pt"
    if digest(ck)!=info['checkpoint_sha256']:raise ValueError('Registered frozen policy changed')
    out=directory/case
    entry=OG if arm=='original' else HERE/'entry.py'
    cmd=['taskset','-c',','.join(str(n) for n in cpus),sys.executable,'-u',str(entry),
         '--out',str(out),'--weights',str(ck),'--method','PPO','--seed','6001','--version',str(info['version']),
         '--episode','0','--case',case,'--role','train','--offset','0','--quota',str(info['quota']),
         '--duration',str(info['duration_s'])]
    with (directory/(case+'.log')).open('w') as log:
        rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    data=read(out/'result.json') if (out/'result.json').is_file() else {}
    return dict(case=case,rc=rc,command=cmd,actor=digest(ck),physical_s=data.get('physical_s'),
                observed=data.get('observed'),physical_steps=data.get('physical_steps'),reason=data.get('reason'),
                full_ms=data.get('control_full_ms_p50_p95_p99'),wall_s=data.get('wall_s'),cpu_h=data.get('cpu_h'))

def compare(a,b):
    def records(path):
        return [json.loads(l) for l in path.read_text().splitlines() if l]
    def normalize(values):
        return [{k:v for k,v in r.items() if k not in ('event_id','attempt')} for r in values]
    with np.load(a/'trajectory.npz') as pa,np.load(b/'trajectory.npz') as pb:
        x,y=pa['control'],pb['control']
        equal=x.shape==y.shape and x.tobytes()==y.tobytes()
        n=min(len(x),len(y))
        delta=np.max(np.abs(x[:n]-y[:n]),axis=0) if n else None
    ea,eb=records(a/'events.jsonl'),records(b/'events.jsonl')
    ra,rb=records(a/'requests.jsonl'),records(b/'requests.jsonl')
    # A request ID contains the diagnostic folder; compare its sequence suffix and all contents.
    return dict(trajectory_float32_bitwise=equal,frames=[len(x),len(y)],
                max_qpos_difference=None if delta is None else float(delta[2:21].max()),
                max_measured_GRF_difference_N=None if delta is None else float(delta[75:87].max()),
                exact_events_and_rewards=normalize(ea)==normalize(eb),exact_requests=normalize(ra)==normalize(rb),
                event_ids_ordered=[v['event_id'].rsplit('/',1)[1] for v in ea]==[v['event_id'].rsplit('/',1)[1] for v in eb],
                exact_swing_references=read(a/'swing.json')==read(b/'swing.json'),
                same_end=read(a/'result.json')['reason']==read(b/'result.json')['reason'])

def physical(stage):
    cfg=spec();seats=json.loads(os.environ['W61_PAIR_CPUS'])
    if len(seats)!=2:raise ValueError('Two explicitly leased collector partitions required')
    values={}
    for arm in cfg['stages'][stage]['order']:
        path=WORK/stage/arm
        previous=path/'pair.json'
        if previous.is_file():
            values[arm]=read(previous);continue
        path.mkdir(parents=True,exist_ok=False)
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as ex:
            future=[ex.submit(one_run,cfg,stage,arm,cpu,ids,path) for cpu,ids in zip(('A','B'),seats)]
            cases=[f.result() for f in future]
        wall=time.monotonic()-start
        data=dict(arm=arm,stage=stage,shanghai=now(),slots=os.environ['W61_SLOTS'],cpus=seats,cases=cases,
                  wall_s=wall,events_per_s=sum((r['observed'] or 0) for r in cases)/wall,
                  linux_process_per_case=True)
        save(previous,data);values[arm]=data
        print(stage,arm,round(wall,3),[v['observed'] for v in cases],flush=True)
        if any(c['rc'] for c in cases):
            save(OUT/(stage+'.json'),dict(status='technical_interruption',completed_arms=values))
            return 1
    congruence={c:compare(WORK/stage/'original'/c,WORK/stage/'exact'/c) for c in ('A','B')}
    speed=values['original']['wall_s']/values['exact']['wall_s']
    identical=all(all(v[k] for k in ('trajectory_float32_bitwise','exact_events_and_rewards','exact_requests','event_ids_ordered','exact_swing_references','same_end')) for v in congruence.values())
    completed=all(row['observed']==cfg['stages'][stage]['quota'] and row['reason']=='batch_cut'
                  for a in values.values() for row in a['cases']) if stage=='full' else all(row['reason'] in ('horizon','success') for a in values.values() for row in a['cases'])
    gates=identical and completed and speed>=cfg['minimum_end_to_end_speedup']
    receipt=dict(study=STUDY,status='closed',stage=stage,arms=values,parity=congruence,
        bitwise_and_event_match=identical,registered_work_closed=completed,wall_speedup=speed,
        qualifies_for_stage=gates,formal_used=0,source_commit=cfg['source_identity'],base_collector_sha256=digest(OG),
        runtime_build=read(WORK/'build/identity.json'))
    save(OUT/(stage+'.json'),receipt)
    print(stage,'end-to-end',speed,'identical',identical,'work',completed,flush=True)
    return 0

def matrix():
    cfg=spec()
    for stage in ('entry','full'):
        job=f'{STUDY}-{stage}'
        cmd=[sys.executable,str(LEASE),'--runtime',str(ROOT),'--job',job,'--slots','2','--cores','2','--ram-gib','4.2',
             '--gpu-mib','1450','--scratch-gib','.75','--',sys.executable,'-u',str(Path(__file__).resolve()),stage]
        if subprocess.call(cmd):return 1
        row=read(OUT/(stage+'.json'))
        if not row['qualifies_for_stage']:
            save(OUT/'status.json',dict(stage=stage,production_selected=False,reason='measured contrast does not qualify',
                                        shanghai=now(),retained='existing CPU'))
            return 0
    save(OUT/'status.json',dict(stage='qualified_full',production_selected=False,shanghai=now(),
                                validated_binary=read(WORK/'build/identity.json')['sha256'],
                                reason='two real fixed-policy contrasts closed; ready for the next natural batch registration'))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('matrix','entry','full'))
    mode=p.parse_args().mode
    sys.exit(matrix() if mode=='matrix' else physical(mode))
