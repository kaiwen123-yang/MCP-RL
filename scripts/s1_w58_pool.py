"""D57 shared leases, measured admission and completion records for W58 physical jobs."""
import argparse
from collections import deque
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inspect_s1_w58_predecessor import BG, REL, SH, proc_info, read
from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/w58'
POOL=WORK/'pool'
STATE=ROOT/'results'/REL/'w58'
RAW=ROOT/'outputs'/REL
CG=Path('/sys/fs/cgroup')
PERSONAL=Path('/autodl-fs/data')
PER_STREAM_CORES=3  # P2 uses the synchronous single-native-worker kernel.
SAMPLE=20


def gpu():
    def query(what,fields):
        p=subprocess.run(['nvidia-smi','--query-'+what+'='+fields,'--format=csv,noheader,nounits'],
                         capture_output=True,text=True)
        return [r.split(',') for r in p.stdout.strip().splitlines()] if p.returncode==0 else []
    board=query('gpu','utilization.gpu,memory.used,memory.total')
    used={}
    for pid,m in query('compute-apps','pid,used_memory'):
        if m.strip().isdigit():used[int(pid.strip())]=int(m)
    return (list(map(int,board[0])) if board else None), used


def pressure(resource):
    return float((CG/(resource+'.pressure')).read_text().splitlines()[0].split()[1].split('=')[1])


def snapshot():
    now=time.monotonic()
    cpu_file=dict(line.split() for line in (CG/'cpu.stat').read_text().splitlines())
    cpu=float(cpu_file['usage_usec'])/1e6
    q,p=(CG/'cpu.max').read_text().split()
    mem={k:int(v) for k,v in (line.split() for line in (CG/'memory.stat').read_text().splitlines())}
    available=int((CG/'memory.max').read_text())-int((CG/'memory.current').read_text())+mem['inactive_file']
    dev,apps=gpu()
    hz=os.sysconf('SC_CLK_TCK')
    project={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            cwd=os.readlink(proc/'cwd')
            if '/MCP-RL-w58' not in cwd:continue
            stat=(proc/'stat').read_text().rsplit(') ',1)[1].split()
            project[int(proc.name)]=dict(auxiliary=cwd!=str(ROOT) or int(proc.name)==os.getpid(),cpu=(int(stat[11])+int(stat[12]))/hz,
                                        rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),gpu_MiB=apps.get(int(proc.name),0))
        except (OSError,IndexError):
            continue
    host=[int(x) for x in Path('/proc/stat').open().readline().split()[1:9]]
    disks=[x.split() for x in Path('/proc/diskstats').read_text().splitlines()]
    queues={x[2]:int(x[11]) for x in disks if x[2]=='md0' or (x[2].startswith('nvme') and 'p' not in x[2])}
    return dict(t=now,unix_s=time.time(),cpu_seconds=cpu,cpu_quota=int(q)/int(p),project=project,
                available_ram_bytes=available,gpu=dev,io_pressure_pct=pressure('io'),cpu_pressure_pct=pressure('cpu'),
                io_wait_ticks=host[4],host_ticks=sum(host),disk_outstanding_io=queues,
                dirty_plus_writeback=mem['file_dirty']+mem['file_writeback'],
                scratch_free_inodes=os.statvfs(RAW.parent).f_favail,personal_free_inodes=os.statvfs(PERSONAL).f_favail,
                scratch_free_bytes=shutil.disk_usage(RAW.parent).free,
                personal_free_bytes=shutil.disk_usage(PERSONAL).free)


def load_delta(prev,cur):
    dt=cur['t']-prev['t']
    used=max(0.,cur['cpu_seconds']-prev['cpu_seconds'])/dt
    own=sum(max(0.,r['cpu']-prev['project'][p]['cpu']) for p,r in cur['project'].items() if p in prev['project'])/dt
    aux=sum(max(0.,r['cpu']-prev['project'][p]['cpu']) for p,r in cur['project'].items() if r['auxiliary'] and p in prev['project'])/dt
    other=max(0.,used-own)
    iowait=(cur['io_wait_ticks']-prev['io_wait_ticks'])/max(1,cur['host_ticks']-prev['host_ticks'])
    return dict(total_core_s_per_s=used,w58_core_s_per_s=own,auxiliary_core_s_per_s=aux,other_core_s_per_s=other,
                host_iowait_fraction=iowait,available_cpu_cores=max(0.,min(18.,cur['cpu_quota']-3.-other-aux)))

def space_for_original(mount, bytes_needed, file_count):
    stats=os.statvfs(mount)
    return stats.f_bavail*stats.f_frsize>=bytes_needed and stats.f_favail>=file_count


def recording_inodes(duration):
    # Actual W57 originals used 3212 entries for E60 and 2249 for P/E30.
    # Reserve at the higher observed cadence, plus job and rendering metadata.
    return 100+math.ceil(80*duration)


def allocate_original(attempt,duration):
    """Reserve enough space for the next real episode, including each natural collector tail."""
    root=POOL/'raw-reservations'
    root.mkdir(parents=True,exist_ok=True)
    path=root/(attempt+'.json')
    size=int((1.+duration*.03)*2**30)
    entries=recording_inodes(duration)
    preferred=os.environ.get('W58_RAW_STORE','')
    volumes=[('',RAW.parent),(str(PERSONAL/'MCP-RL/W58/originals'),PERSONAL)]
    volumes.sort(key=lambda v:v[0]!=preferred)
    owner=proc_info(os.getpid())
    while True:
        with (POOL/'raw-admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            used={key:0 for key,_ in volumes}
            inodes={key:0 for key,_ in volumes}
            for p in root.glob('*.json'):
                if p==path:continue
                row=read(p)
                if row['state']=='allocated':
                    live=proc_info(row['pid'])
                    if live and live['start_ticks']==row['start_ticks']:
                        used[row['store']]+=row['reserved_bytes']
                        inodes[row['store']]+=row.get('reserved_inodes',0)
            for store,mount in volumes:
                if space_for_original(mount,used[store]+size+5*2**30,inodes[store]+entries):
                    write(path,dict(state='allocated',attempt=attempt,store=store,reserved_bytes=size,reserved_inodes=entries,
                                    pid=owner['pid'],start_ticks=owner['start_ticks'],**BG))
                    if store:os.environ['W58_RAW_STORE']=store
                    else:os.environ.pop('W58_RAW_STORE',None)
                    return path
            if not path.exists():
                write(path,dict(state='waiting_storage',attempt=attempt,required_bytes=size,
                                required_inodes=entries,free_inodes={str(v):os.statvfs(v).f_favail for _,v in volumes},
                                free_bytes={str(v):shutil.disk_usage(v).free for _,v in volumes},**BG))
        time.sleep(20)



def worker(a):
    record=Path(a.record)
    data=read(record)
    info=proc_info(os.getpid())
    data.update(pid=info['pid'],start_ticks=info['start_ticks'],status='running',started_unix_s=time.time())
    write(record,data)
    fds=tuple(map(int,os.environ['S1_W58_LEASE_FDS'].split(',')))
    code=subprocess.call(data['command'],cwd=data['runtime'],pass_fds=fds)
    data.update(status='ended',exit_code=code,ended_unix_s=time.time())
    write(record,data)
    return code


class Pool:
    def __init__(self, runtime):
        self.runtime=Path(runtime)
        self.records=POOL/'jobs'
        self.records.mkdir(parents=True,exist_ok=True)
        self.children={}
        self.limit=3
        self.samples=deque(maxlen=48)
        self.reference={}
        self.qualified={}
        self.rejected=set()
        self.last=None
        self.metrics={}
        self.free_ids=[]
        self.last_tick=0.
        self.last_progress={}
        self.last_distinct={}

    def submit(self,job,slots,store):
        record=self.records/(job['id']+'.json')
        # The dispatch tmux is bound to the auxiliary pool. Worker affinities are drawn
        # from the explicit physics partition recorded at launch, within the cgroup.
        available=list(map(int,os.environ['W58_PHYSICS_CPUS'].split(',')))
        groups=[available[i*PER_STREAM_CORES:(i+1)*PER_STREAM_CORES] for i in slots]
        if any(len(g)!=PER_STREAM_CORES for g in groups):raise ValueError('Under-sized physics CPU partition')
        leases=[]
        with (POOL/'admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                for i in slots:
                    f=(POOL/f'slot-{i}.lock').open('a')
                    leases.append(f)
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                for f in leases:f.close()
                return False
            try:
                fds=tuple(f.fileno() for f in leases)
                env=dict(os.environ,S1_W58_LEASE_FDS=','.join(map(str,fds)),S1_W58_SLOT=','.join(map(str,slots)),
                    S1_W58_PAIR_CPUS=json.dumps(groups),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(self.runtime/'src')+':'+str(self.runtime/'work/upstreams/mujoco_mpc/python'))
                if store:env['W58_RAW_STORE']=store
                data=dict(**job,slots=slots,cpu_groups=groups,runtime=str(self.runtime),raw_store=store,
                          status='dispatching',dispatched_unix_s=time.time(),**BG)
                write(record,data)
                cores=','.join(str(c) for g in groups for c in g)
                with (POOL/(job['id']+'.log')).open('a') as log:
                    command=['taskset','-c',cores,sys.executable,'-u',str(self.runtime/'scripts/s1_w58_pool.py'),'--record',str(record)]
                    self.children[job['id']]=subprocess.Popen(command,env=env,cwd=self.runtime,stdout=log,
                                                              stderr=subprocess.STDOUT,pass_fds=fds)
            finally:
                for f in leases:f.close()
        return True

    def running(self):
        rows=[]
        for p in self.records.glob('*.json'):
            r=read(p)
            if r['status'] not in ('dispatching','running'):continue
            child=self.children.get(r['id'])
            if child is not None and child.poll() is None:
                rows.append(r)
            elif r.get('pid') and (info:=proc_info(r['pid'])) and info['start_ticks']==r['start_ticks']:
                rows.append(r)
            else:
                # The ledger still reserves the recorded slots if a worker disappeared.
                # Identify surviving physics/collectors before continuing this exact attempt.
                raise RuntimeError('W58 lease owner unavailable: '+str(p))
        return rows

    def physical_progress(self,active):
        clocks={}
        for r in active:
            if r['slots_needed']==1:names=[r['id']]
            else:
                names=[]
                run=WORK/'campaigns'/('seed-'+str(r['seed']))/r['head']
                for case in 'AB':
                    p=run/f'episodes-{case}.jsonl'
                    if p.exists() and (lines:=p.read_text().splitlines()):
                        names.append(json.loads(lines[-1])['attempt'])
            for name in names:
                p=RAW/name/'live.json'
                if p.is_file():clocks[name]=read(p)['physical_time_s']
        return clocks

    def tick(self,active):
        previous=self.last
        cur=snapshot()
        self.last=cur
        if previous is None:return
        load=load_delta(previous,cur)
        clocks=self.physical_progress(active)
        count=sum(len(r['slots']) for r in active)
        kinds={r['kind'] for r in active}
        dt=cur['t']-previous['t']
        progress={k:(v-self.last_progress[k])/dt for k,v in clocks.items() if k in self.last_progress}
        self.last_progress=clocks
        distinct={}
        for r in active:
            if r['slots_needed']==2:
                p=WORK/'campaigns'/('seed-'+str(r['seed']))/r['head']/'published.json'
                if p.exists():distinct[r['id']]=read(p).get('cumulative_distinct',0)
        now=dict(time_shanghai=datetime.now(SH).isoformat(),unix_s=cur['unix_s'],count=count,kind=next(iter(kinds)) if len(kinds)==1 else None,
                 dt_s=dt,physics_s_per_wall_s=progress,clock_s=clocks,distinct_committed=distinct,
                 committed_distinct_per_wall_hour={k:3600*(v-self.last_distinct[k])/dt for k,v in distinct.items() if k in self.last_distinct},**load,
                 gpu=cur['gpu'],project_gpu_MiB=sum(p['gpu_MiB'] for p in cur['project'].values()),
                 project_rss_bytes=sum(p['rss'] for p in cur['project'].values()),
                 ram_available_bytes=cur['available_ram_bytes'],io_pressure_pct=cur['io_pressure_pct'],
                 host_disk_outstanding_io=cur['disk_outstanding_io'],dirty_writeback_bytes=cur['dirty_plus_writeback'],
                 scratch_free_bytes=cur['scratch_free_bytes'],personal_free_bytes=cur['personal_free_bytes'],
                 scratch_free_inodes=cur['scratch_free_inodes'],personal_free_inodes=cur['personal_free_inodes'])
        self.last_distinct=distinct
        self.samples.append(now)
        self.metrics=now
        with (WORK/'throughput.jsonl').open('a') as f:f.write(json.dumps(now,separators=(',',':'))+'\n')
        # Compare like formal work, complete short windows, actual advancing streams.
        same=list(self.samples)[-4:]
        kind=now['kind']
        if kind and len(same)==4 and all(
                v['kind']==kind and v['count']==count and len(v['physics_s_per_wall_s'])==count
                and min(v['physics_s_per_wall_s'].values(),default=0)>0 for v in same):
            total=sum(sum(v['physics_s_per_wall_s'].values()) for v in same)/4
            ext=sum(v['other_core_s_per_s'] for v in same)/4
            clean=all(v['available_cpu_cores'] >= min(18,(count+1)*PER_STREAM_CORES) and
                      v['ram_available_bytes']>10*2**30 and v['io_pressure_pct']<5. and
                      v['host_iowait_fraction']<.05 and v['gpu'] and v['gpu'][0]<85 and
                      v['gpu'][2]-v['gpu'][1]>6000 for v in same)
            previous_cap={4:3,6:4}.get(count)
            if previous_cap and (kind,previous_cap) in self.reference:
                old,other=self.reference[kind,previous_cap]
                if (total<=old or total/count<.90*old/previous_cap or
                        ext+1<other and other>2):
                    self.rejected.add((kind,count))
                    self.limit=previous_cap
                else:
                    self.qualified[kind,count]=True
            if (kind,count) not in self.rejected:
                self.reference.setdefault((kind,count),(total,ext))
                if count in (3,4) and clean and (count==3 or self.qualified.get((kind,4))):
                    nxt=4 if count==3 else 6
                    if (kind,nxt) not in self.rejected:self.limit=nxt
        write(STATE/'pool.json',dict(limit=self.limit,initial_limit=3,conditional_max=6,active=active,
             latest=now,throughput_trace=str(WORK/'throughput.jsonl'),
             certified=[list(k) for k in self.qualified],rejected=[list(k) for k in sorted(self.rejected)],**BG))

    def batch(self,jobs,pulse):
        pending=list(jobs)
        while pending or self.running():
            active=self.running()
            if self.last is None or time.monotonic()-self.last['t']>=SAMPLE:self.tick(active)
            pulse()
            active=self.running()
            in_use={s for r in active for s in r['slots']}
            if not self.metrics:
                time.sleep(2)
                continue
            for job in pending[:]:
                if (self.records/(job['id']+'.json')).exists():
                    pending.remove(job)
                    continue
                cap=self.limit
                # Larger caps apply only to the measured task family, and are trialled
                # on real jobs from the same family as the stored baseline.
                if cap>3 and (job['kind'],3) not in self.reference:cap=3
                available=[i for i in range(cap) if i not in in_use]
                n=job['slots_needed']
                ms=self.metrics
                if (len(available)<n or (len(in_use)+n)*PER_STREAM_CORES>ms['available_cpu_cores']
                        or ms['ram_available_bytes']<(len(in_use)+n)*2.5*2**30
                        or ms['io_pressure_pct']>15 or not ms['gpu'] or ms['gpu'][2]-ms['gpu'][1]<n*1100):
                    continue
                store=None
                # Relocate only a future raw allocation, never existing originals.
                entries=recording_inodes(job['disk_GiB']/.03)
                for volume, mount in ((None,RAW.parent),(str(PERSONAL/'MCP-RL/W58/originals'),PERSONAL)):
                    same=[r for r in active if r.get('raw_store')==volume]
                    gib=5+job['disk_GiB']+sum(r['disk_GiB'] for r in same)
                    reserved=sum(recording_inodes(r['disk_GiB']/.03) for r in same)
                    if space_for_original(mount,gib*2**30,entries+reserved):
                        store=volume
                        break
                else:
                    continue
                chosen=available[:n]
                if not self.submit(job,chosen,store):break
                in_use.update(chosen)
                active=self.running()
                pending.remove(job)
            for p in tuple(self.children.values()):p.poll()
            if pending or self.running():time.sleep(3)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',type=Path,required=True)
    sys.exit(worker(p.parse_args()))

