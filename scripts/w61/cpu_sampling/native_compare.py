"""Compare the complete native E30 time series with the same original case and frozen actor."""
import json,os,subprocess,sys,time,zipfile
from pathlib import Path
from build import ROOT,WORK,HERE,digest
from experiment import OUT,START,LEASE,save,read
from driver import identity,bind_native

ID='w61-e-cpu-exact-v1-native-a-e30'
GOLD='w61-e-final-ppo-s6001-v000-a-e30-nominal'
RAW=WORK/'native-originals'
LEDGER=WORK/'native-ledger'
NATIVE_REL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/learning/e61-hybrid-m1-01/final-evaluation'
TIMING={'planner_s.npy','feedback_s.npy','sync_s.npy'}

def physical():
    import evaluation_physics as frozen
    current=frozen.native
    current.RUN=WORK/'native-root'
    target=current.RUN/'ppo-s6001'/'policy-000.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(START/'ppo-s6001/policy-000.pt')
    current.DATA=RAW
    current.LEDGER=LEDGER
    current.ident=lambda *args:ID
    frozen.INTENTS=WORK/'native-diagnostic-render-intent'
    bind_native(current,identity())
    result=frozen.physical_native('PPO',6001,0,'A','E30','nominal')
    print('native closed',result['stage_status'],result['physical_s'],flush=True)

def compare_npz(a,b,skip):
    import hashlib
    with zipfile.ZipFile(a) as l,zipfile.ZipFile(b) as r:
        all_names=set(l.namelist())|set(r.namelist())
        different=[]
        for key in sorted(all_names-skip):
            if key not in l.namelist() or key not in r.namelist():
                different.append(key);continue
            with l.open(key) as x,r.open(key) as y:
                if hashlib.file_digest(x,'sha256').digest()!=hashlib.file_digest(y,'sha256').digest():different.append(key)
    return len(all_names-skip),different

def grade():
    ref=START/'final-originals'/GOLD
    new=RAW/ID
    a=read(NATIVE_REL/'ppo-s6001'/(GOLD+'.json'));b=read(LEDGER/'ppo-s6001'/(ID+'.json'))
    files=['trajectory.npz','material-final.npz',*[p.name for p in sorted(ref.glob('physical-*.npz'))],*[p.name for p in sorted(ref.glob('material-[0-9]*.npz'))]]
    n=0;fail={}
    for f in files:
        if not (new/f).is_file():fail[f]=['missing file'];continue
        num,wrong=compare_npz(ref/f,new/f,TIMING if f=='trajectory.npz' else set());n+=num
        if wrong:fail[f]=wrong
    for pattern in ('physical-*.npz','material-[0-9]*.npz'):
        extra={p.name for p in new.glob(pattern)}-{p.name for p in ref.glob(pattern)}
        for f in extra:fail[f]=['additional file']
    events=ref/'material-events.jsonl';same_events=digest(events)==digest(new/events.name)
    speed=a['wall_s']/b['wall_s']
    parity=not fail and same_events and a['stage_status']==b['stage_status']=='passed' and a['simulation_s']==b['simulation_s']
    save(OUT/'native-e30.json',dict(status='closed',diagnostic_attempt=ID,reference_attempt=GOLD,source=str(ref),
        raw=str(new),unchanged_scientific_array_members=n,different_members=fail,material_events_byte_equal=same_events,
        original_physics_wall_s=a['wall_s'],optimized_physics_wall_s=b['wall_s'],speedup=speed,
        control_time_array_exclusions=sorted(TIMING),stage=[a['stage_status'],b['stage_status']],physical_s=b['simulation_s'],
        matched_native_physics=parity,qualifies_for_frozen_episodes=parity and speed>=1.05,reused_original_published_video=(GOLD if parity else None),
        material_execution=identity(),qualification_of_formal_head=None))
    print('native original/candidate wall:',a['wall_s'],b['wall_s'],'speedup',speed,'parity',parity,'differences',fail,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='physical':physical()
    elif sys.argv[1]=='grade':grade()
    else:
        while not (OUT/'entry.json').is_file():time.sleep(5)
        if read(OUT/'entry.json').get('qualifies_for_stage'):
            cmd=[sys.executable,str(LEASE),'--runtime',str(ROOT),'--job',ID,'--slots','1','--cores','1','--ram-gib','3.',
                 '--gpu-mib','800','--scratch-gib','1.5','--',sys.executable,'-u',str(Path(__file__).resolve()),'physical']
            subprocess.run(cmd,check=True);grade()
