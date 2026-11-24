"""Measured 2 -> 4 -> 6 slot admission for already-earned W57 frozen evaluations.

The original dispatcher owns scoring, its two inherited leases, the private fork and the
long gates. Later eligible jobs enter under the same dispatch mutex and canonical attempt
names. Two still-unclaimed predecessors form an adoption barrier for its two workers.
"""
from collections import deque
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/root/autodl-tmp/MCP-RL-w57')
sys.path.insert(0, str(ROOT/'scripts'))
from queue_s1_w57_evaluations import source_for, CAMPAIGN, R, WORK, BACKGROUND
from s1_w56_slot import QUEUE
from publish_s1_go2_w54 import write

RAW = ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
STORE = ROOT/'work/w57-raw-scratch'
LOG = WORK/'expanded-evaluations'
PUBLIC = R/'dispatch/evaluation-parallelism.json'
ARMS = ('B','P','R','PR','blind')
TZ = ZoneInfo('Asia/Shanghai')
MARK = struct.Struct('<Qdd40sQ')
GIB = 2**30
EXTRA_CPUS = (4,5,6,7,8,9,14,15)
def now():return datetime.now(TZ).isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text())
def key(arm,case,mode):
    return f'w57-m1-{arm.lower()}-{case.lower()}-{mode}'+('-e60' if mode=='original' else '')+'-a01'
def receipt(n):
    p=R/'attempts'/(n+'.json')
    return read(p) if p.is_file() else None
def worker_args():
    seen=set()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            for flag in (b'--attempt',b'--label'):
                if flag in a:seen.add(os.fsdecode(a[a.index(flag)+1]))
        except (OSError,IndexError):pass
    return seen
def marker(name):
    f=RAW/name/'r60-current-call.bin'
    if not f.is_file():return None
    b=f.read_bytes()
    if len(b)==MARK.size:
        lo,t,when,call,hi=MARK.unpack(b)
        if lo==hi:return t
    return None
def rows_live():
    result={}
    for p in (R/'attempts').glob('*.json'):
        row=read(p)
        if 'ended_unix_s' not in row and row['status'] in ('declared','running'): result[row['attempt']]=row
    return result
def cpu_cost(arm):return 5 if arm in ('P','PR') else 1
def memory_budget(row):
    # The two completed 196s cases peaked at 2.62 GiB; current P/PR recordings use the same arrays.
    return 3.25 if row['task_mode']=='recovered' else 2.85
def sample():
    cg=Path('/sys/fs/cgroup');t=time.monotonic()
    vals=lambda f:{k:int(v) for k,v in (s.split() for s in (cg/f).read_text().splitlines())}
    proc={};hz=os.sysconf('SC_CLK_TCK');pg=os.sysconf('SC_PAGE_SIZE')
    live=rows_live();pid_names={v.get('pid'):n for n,v in live.items()}
    for pid in Path('/proc').glob('[0-9]*'):
        try:
            cwd=os.readlink(pid/'cwd')
            if 'MCP-RL' not in cwd and '.mcp-rl' not in cwd:continue
            stat=(pid/'stat').read_text().rsplit(') ',1)[1].split()
            proc[int(pid.name)]=(int(stat[11])+int(stat[12]))/hz
            if int(pid.name) in pid_names:live[pid_names[int(pid.name)]]['rss_GiB']=int(stat[21])*pg/GIB
        except (OSError,ValueError,IndexError):pass
    m=vals('memory.stat');mr=(int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())+m['inactive_file'])/GIB
    q,p=(cg/'cpu.max').read_text().split()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    g=[int(v.strip()) for v in gpu.stdout.strip().split(',')] if gpu.returncode==0 else []
    pressure=lambda n:float((cg/(n+'.pressure')).read_text().splitlines()[0].split()[1].split('=')[1])
    return dict(t=t,unix=time.time(),live=live,clock={n:v for n in live if (v:=marker(n)) is not None},
                proc=proc,total_cpu=vals('cpu.stat')['usage_usec']/1e6,quota=int(q)/int(p),ram=mr,
                gpu_free=g[0] if g else 0,gpu_util=g[1] if g else None,io=pressure('io'),cpu_pressure=pressure('cpu'),
                mem_pressure=pressure('memory'),working_disk=shutil.disk_usage(STORE).free/GIB)
def historical_pair():
    path=RAW/'w57/resource-trace.jsonl'
    names=[key('P',c,'recovered') for c in 'AB']
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-3000000));f.readline()
        data=[json.loads(l) for l in f if l.endswith(b'\n')]
    end=data[-1]
    avail=[v for v in data if all(k in v['physics'] for k in names) and 180<=end['unix_s']-v['unix_s']<=360]
    if not avail:return {}
    b=avail[0];dt=end['unix_s']-b['unix_s']
    return {n:(end['physics'][n]['physical_s']-b['physics'][n]['physical_s'])/dt for n in names if n in end['physics']}
def sequence():
    """Only jobs that the original queue is guaranteed to visit may serve as a fence."""
    if not (CAMPAIGN/'final.json').is_file():return []
    seq=[]
    for stage in ('recovered','original'):
        for arm in ARMS:
            for case in 'AB':
                a=receipt(f'w57-m1-{arm.lower()}-{case.lower()}-original-e30-a01')
                s=receipt(key(arm,case,'short'))
                if not a or a['status']!='passed' or not s or not s.get('ended_unix_s') or s['status']=='technical_interruption':continue
                if stage=='original':
                    v=receipt(key(arm,case,'recovered'))
                    if not v or not v.get('ended_unix_s') or v['status']=='technical_interruption':continue
                seq.append((key(arm,case,stage),arm,case,stage))
    return seq

def record_event(message,**values):
    with (LOG/'events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),event=message,**values),separators=(',',':'))+'\n')
    print(now(),message,values,flush=True)

class Evaluations:
    def __init__(self):
        self.procs={};path=LOG/'decisions.json'
        self.state=read(path) if path.exists() else dict(initial_pair=historical_pair(),limit=4,
            four_validated=False,six_validated=False,decisions=[],dispatched={},tested_six=False)
        self.recent=deque(maxlen=31)
        self.previous=None
    def save(self):write(LOG/'decisions.json',self.state)
    def limits(self,v,prev):
        dt=v['t']-prev['t'];total=max(0.,v['total_cpu']-prev['total_cpu'])/dt
        own=sum(max(0.,s-prev['proc'][p]) for p,s in v['proc'].items() if p in prev['proc'])/dt
        phys=sum(max(0.,v['proc'][pid]-prev['proc'][pid])/dt for r in v['live'].values()
                 if (pid:=r.get('pid')) in v['proc'] and pid in prev['proc'])
        other=max(0.,total-own);aux=max(0.,own-phys)
        return dict(observed_total_cores=total,external_cores=other,aux_cores=aux,
                    cpu_budget=min(18.,max(0.,v['quota']-3.-other-aux)),
                    ram=v['ram'],gpu_free_MiB=v['gpu_free'],gpu_util=v['gpu_util'],
                    io_pressure=v['io'],cpu_pressure=v['cpu_pressure'],memory_pressure=v['mem_pressure'],
                    scratch_free_GiB=v['working_disk'])
    def window(self,live):
        count=len(live['live']);names=set(live['live'])
        history=[r for r in self.recent if r['names']==names and 115<=live['t']-r['t']<=210]
        if not history or not names<=set(live['clock']):return None
        base=history[0]
        if not names<=set(base['clock']):return None
        dt=live['t']-base['t']
        return {n:(live['clock'][n]-base['clock'][n])/dt for n in names}
    def evaluate_cap(self,live,lim):
        n=len(live['live']);rates=self.window(live)
        if not rates or n not in (4,6):return rates
        initial=self.state['initial_pair']
        common=set(initial)&set(rates)
        details=dict(at=now(),slot_count=n,rate_by_attempt=rates,aggregate=sum(rates.values()),entry_reference=initial,
                     matched_incumbents=sorted(common),**lim)
        if n==4 and not self.state['four_validated']:
            # Receipts cover >115 wall seconds of real physics on exactly these four PIDs.
            # Compare original P cases with their own already-measured pair and report each rate.
            original=sum(initial[k] for k in common)
            incumbent=sum(rates[k] for k in common)
            if (len(common)==2 and original>0 and all(r>0 for r in rates.values())
                    and incumbent>=.85*original and sum(rates.values())>=1.15*sum(initial.values())):
                self.state.update(four_validated=True,limit=6,four_reference=rates,trial_approved_at=now())
                self.state['decisions'].append(dict(kind='four_streams_advance_six_trial_authorized',**details));self.save()
                record_event('four-slot measured throughput qualified a six-slot trial',aggregate=details['aggregate'],incumbent=incumbent)
            elif len(common)==2 and original>0 and incumbent<.75*original:
                self.state.update(limit=2)
                self.state['decisions'].append(dict(kind='no_further_admissions_after_four_stream_degradation',**details));self.save()
        if n==6 and not self.state['six_validated'] and self.state['limit']==6:
            ref=self.state.get('four_reference',{});overlap=set(ref)&set(rates)
            if not overlap:return rates
            held=sum(rates[k] for k in overlap);old=sum(ref[k] for k in overlap)
            if all(x>0 for x in rates.values()) and sum(rates.values())>sum(ref.values()) and held>=.85*old:
                self.state['six_validated']=True;self.state['decisions'].append(dict(kind='six_streams_advancing',**details));self.save()
                record_event('six streams measured',aggregate=details['aggregate'])
            elif held<.75*old or sum(rates.values())<=sum(ref.values()):
                self.state['limit']=4;self.state['decisions'].append(dict(kind='stop_new_admissions_above_four',**details));self.save()
                record_event('six-slot trial withdrew future extra admissions')
        return rates
    def launch(self,slot,job,cores,mutex,seq):
        name,arm,case,mode=job
        # Two earlier *unclaimed* jobs cannot be bypassed by both legacy threads
        # while this very same dispatch mutex is held. A legacy worker already past
        # the mutex is excluded by its argv and/or job directory. Keep the mutex until
        # the immutable episode launcher publishes its actual runner PID.
        active_names=worker_args()
        if name in active_names or receipt(name) or (RAW/'jobs'/name).exists():return False
        elig=sequence()
        pending=[n for n,*_ in elig if not receipt(n) and not (RAW/'jobs'/n).exists() and n not in active_names]
        if name not in pending or pending.index(name)<2:return False
        predecessors=pending[:pending.index(name)][:2]
        if not all(not receipt(n) for n in predecessors):return False
        lease=(QUEUE/f'slot-{slot}.lock').open('a')
        try:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
            src=source_for(arm);parent=receipt(key(arm,case,'short')) if mode=='recovered' else receipt(f'w57-m1-{arm.lower()}-{case.lower()}-original-e30-a01')
            cmd=['taskset','-c',','.join(map(str,cores)),sys.executable,'-u','scripts/run_s1_w57.py','episode',
                 '--case',case,'--arm',arm,'--task',mode,'--material','M1','--level','E60','--attempt',name]
            if arm in ('PR','R','blind'):
                head='blind' if arm=='blind' else 'full'
                cmd+=['--timing-config',str(CAMPAIGN/head/'frozen-evaluation.json')]
            env=dict(os.environ,S1_W56_SLOT=str(slot),W57_TMUX='w57-frozen-wide',W57_RAW_STORE=str(STORE),NUMEXPR_NUM_THREADS='1',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     PYTHONPATH=str(src/'src')+':'+str(src/'work/upstreams/mujoco_mpc/python'))
            record_event('lease_granted',attempt=name,slot=slot,cores=cores,adoption_fence=predecessors,command=cmd)
            with (WORK/(name+'.log')).open('a') as log:
                proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lease.fileno(),))
            self.procs[name]=proc
            published=None
            while published is None and proc.poll() is None:
                published=receipt(name)
                if published is None:time.sleep(.1)
            if published is None:raise RuntimeError(f'{name} exited before recording its actual attempt; see {WORK/name}.log')
            fields=('source_commit','controller','checkpoint_sha256','normalization_sha256','timing_checkpoint_sha256','active_sources')
            if published.get('runner_pid')!=proc.pid or any(published.get(k)!=parent.get(k) for k in fields):
                raise RuntimeError('Different frozen execution identity; retain actual result and stop additional admission: '+name)
            self.state['dispatched'][name]=dict(slot=slot,cores=cores,pid=proc.pid,controller=published['controller'],
                     source=published['source_commit'],timing_sha256=published.get('timing_checkpoint_sha256'),
                     borrowed_qualification=False,admission_fence=predecessors,started=published['started_unix_s'])
            self.save()
            return True
        finally:lease.close()
    def admit(self,v,lim):
        with (QUEUE/'dispatch.lock').open('a') as mux:
            fcntl.flock(mux,fcntl.LOCK_EX)
            if any((Path('/proc')/f.stem.rsplit('-',1)[-1]).exists() for f in QUEUE.glob('*.waiting')):return
            live=rows_live();cap=self.state['limit'];existing=len(live)
            if existing>=cap or cap<3:return
            slots_used={int(x['slot']) for x in live.values() if str(x.get('slot')).isdigit()}
            used_cores=set().union(*(set(x.get('cpu_affinity',[])) for x in live.values())) if live else set()
            free_cores=[c for c in EXTRA_CPUS if c not in used_cores]
            idle=[i for i in range(2,cap) if i not in slots_used]
            if not idle:return
            seq=sequence()
            # Add the matched blind restores first; both use the already-trained frozen blind checkpoint.
            def priority(j):
                _,arm,case,mode=j
                return (0 if mode=='recovered' and arm=='blind' else 1 if mode=='recovered' else 2,ARMS.index(arm),case)
            for j in sorted(seq,key=priority):
                name,arm,case,mode=j
                if not idle or len(live)>=cap:break
                if name in live or name in self.state['dispatched'] or receipt(name):continue
                cores=cpu_cost(arm);reserve=memory_budget(dict(task_mode=mode))
                if (sum(cpu_cost(x['arm']) for x in live.values())+cores>lim['cpu_budget'] or len(free_cores)<cores):continue
                missing_growth=sum(max(0.,memory_budget(x)-v['live'].get(n,{}).get('rss_GiB',0.)) for n,x in live.items())
                if (lim['ram']<7.0+reserve+missing_growth or lim['gpu_free_MiB']<1800+850*max(1,len(live)-existing+1)
                        or lim['io_pressure']>12 or lim['cpu_pressure']>12 or lim['memory_pressure']>4
                        or lim['scratch_free_GiB']<12+6*(len(live)+1)):continue
                cpus=free_cores[:cores];slot=idle[0]
                if not self.launch(slot,j,cpus,mux,seq):continue
                live[name]=receipt(name);idle.remove(slot);free_cores=free_cores[cores:]
    def run(self):
        base=self.state['initial_pair'];record_event('entry',baseline_pair=base)
        if not base:raise ValueError('Original P A/B throughput window missing')
        self.save()
        while True:
            v=sample();previous=self.previous;self.previous=v
            for p in self.procs.values():p.poll()
            if previous is None:time.sleep(6);continue
            lim=self.limits(v,previous);rates=self.evaluate_cap(v,lim)
            self.recent.append(dict(t=v['t'],names=set(v['live']),clock=v['clock'].copy()))
            current={n:dict(pid=a.get('pid'),runner_pid=a.get('runner_pid'),slot=a.get('slot'),arm=a['arm'],cpu=a.get('cpu_affinity'),clock=v['clock'].get(n),
                            worker_budget=cpu_cost(a['arm']),rss_GiB=a.get('rss_GiB')) for n,a in v['live'].items()}
            entry=dict(at=now(),active=current,admission_limit=self.state['limit'],**lim)
            with (LOG/'resources.jsonl').open('a') as f:f.write(json.dumps(entry,separators=(',',':'))+'\n')
            ended=read(R/'dispatch/frozen-evaluations.json')['stage']=='queue_ended'
            # The queued E300/E600 are each per-case sequential and already fit the original two slots.
            original_done=all(receipt(key(a,c,'original')) and receipt(key(a,c,'original')).get('ended_unix_s') for a in ARMS for c in 'AB')
            extras_alive=any(n in self.state['dispatched'] for n in current)
            terminal=(ended or original_done) and not extras_alive
            write(PUBLIC,dict(schema='W57_additional_frozen_evaluation_leases_v1',
                status='ended' if terminal else 'running',source='scripts/queue_s1_w57_extra_evaluations.py',
                coordinator_pid=os.getpid(),legacy_primary='w57-frozen-evaluations',lease_directory=str(QUEUE),
                max_physics_slots=6,statistics=entry,progress_window_rates=rates,**self.state,**BACKGROUND))
            if terminal:return
            self.admit(v,lim)
            time.sleep(12)

if __name__=='__main__':
    LOG.mkdir(parents=True,exist_ok=True)
    if not (CAMPAIGN/'final.json').is_file() or read(R/'dispatch/private-short-A.json').get('status') not in ('ended','not_run'):
        raise RuntimeError('The W57 learners/tails and private evaluation fork must already be terminal')
    if read(R/'dispatch/frozen-evaluations.json')['stage']!='frozen_evaluations_running':raise RuntimeError('Original frozen queue is not active')
    with (LOG/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);Evaluations().run()
