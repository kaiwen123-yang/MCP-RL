"""Measured capacity for independent W58 original/recovered E60 evaluations."""
from bisect import bisect_right
from collections import defaultdict, deque
from pathlib import Path
import json

from inspect_s1_w58_predecessor import BG, REL, read
from publish_s1_go2_w54 import write


def load_history(path):
    if not path.exists():
        return []
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-1024*1024))
        if f.tell():
            f.readline()
        return [json.loads(line) for line in f if line.endswith(b'\n')]


def reference_work(points,t):
    if t>points[-1][0]:
        a,b=points[-7],points[-1]  # six frozen five-second intervals
    else:
        end=min(max(1,bisect_right(points,[t,float('inf')])),len(points)-1)
        a,b=points[end-1],points[end]
    return a[1]+(t-a[0])*(b[1]-a[1])/(b[0]-a[0])


def slope(times,values):
    mean=sum(times)/len(times)
    return sum((t-mean)*v for t,v in zip(times,values))/sum((t-mean)**2 for t in times)


def room(rows,width):
    return all(r['available_cpu_cores']>=3*width and r['ram_available_bytes']>max(10,2.5*width)*2**30
               and r['io_pressure_pct']<5 and r['host_iowait_fraction']<.05
               and r['gpu'] and r['gpu'][0]<85 and r['gpu'][2]-r['gpu'][1]>6000 for r in rows)


class EvaluationCapacity:
    def __init__(self,root,state,work,local=False):
        self.spec=read(root/'experiments/terrain_memory/s1-w58-evaluation-scaling.json')
        self.directory=state/'dispatch'
        self.root=root
        self.local=local
        ref='evaluation-scratch-work-reference.json' if local else 'evaluation-work-reference.json'
        self.reference_path=self.directory/ref
        self.profiles=read(self.reference_path)['profiles'] if self.reference_path.is_file() else None
        self.path=self.directory/('evaluation-scratch-capacity.json' if local else 'evaluation-capacity.json')
        self.work=work
        self.history=deque(load_history(work/'throughput.jsonl')[-48:],maxlen=48)
        self.state=(read(self.path) if self.path.exists() else
                    dict(admission_limit=3,retained=3,windows={},last_end={},bad={},blocked=[],events=[],**BG))

    def freeze_scratch_reference(self):
        # Same-volume measured shapes are fixed before considering a first local-only expansion.
        raw=self.root/'outputs'/REL
        profiles={}
        for task in ('original','recovered'):
            for case in 'AB':
                arm='b2' if task=='original' else 'p2-private'
                ending='original-e60' if task=='original' else 'development-recovered'
                name=f'w58-g0-m1-{arm}-{case.lower()}-{ending}-a01'
                path=raw/'jobs'/name/'physics.log'
                if not path.is_file() or (raw/name).resolve().parent!=raw.resolve():return False
                points=[]
                with path.open() as log:
                    for line in log:
                        if not line.startswith('{') or not line.endswith('\\n'):continue
                        row=json.loads(line)
                        if 'time_s' not in row or 'wall_s' not in row:continue
                        t=round(row['time_s'],6)
                        if task=='recovered' and (t>180 or row.get('phase') in ('terminal_stand','final_hold','complete')):break
                        points.append([t,row['wall_s']])
                if len(points)<7:return False
                if task=='original':
                    ep=self.directory.parent/'episodes'/(name+'.json')
                    if not ep.is_file():return False
                    result=read(ep)
                    if result['simulation_s']<60-1.e-6:return False
                    if points[-1][0]<60:
                        points.append([60.,result['physical_ended_unix_s']-result['physical_started_unix_s']])
                profiles[f'p2-{task}-E60/{case}']=dict(attempt=name,log=str(path),points=points)
        write(self.reference_path,dict(storage='scratch',units=['physics_s','reference_wall_s'],profiles=profiles,
              scope='G0/M1, same-volume B2 original and P2-private recovered case curves before all-local expansion',
              continuation='Use the last 30 s mean beyond the measured prefix; exclude terminal recovery.'))
        self.profiles=profiles
        return True

    def applies(self,active):
        return (any(r['kind']=='p2-recovered-E60' for r in active) and
                all(r['slots_needed']==1 and r['kind'] in self.spec['kinds'] for r in active))

    def cap(self,job,active,ordinary):
        if job['kind'] in self.spec['kinds'] and self.applies([*active,job]):
            limit=self.state['admission_limit']
            if limit>3 and self.history and not room([self.history[-1]],min(limit,len(active)+1)):
                return 3
            return limit
        return ordinary

    def window(self,active,now):
        names={r['id'] for r in active}
        width=len(names);cut=self.state['last_end'].get(str(width),0)
        rows=[]
        for sample in reversed(self.history):
            owners={n for g in sample['families'].values() for n in g['owners']}
            if owners!=names or sample['count']!=width or sample['unix_s']<cut:break
            if rows and rows[-1]['unix_s']-sample['unix_s']>50:break
            if set(sample['clock_s'])!=names:continue
            if min(sample['clock_s'].values())<self.spec['minimum_initial_physics_s']:break
            phases=sample.get('physics_phases',{})
            if any(r['kind']=='p2-recovered-E60' and phases.get(r['id']) in
                   ('terminal_stand','final_hold','complete') for r in active):break
            rows.append(sample)
            if now['unix_s']-sample['unix_s']>=self.spec['window_s']:break
        rows.reverse()
        if len(rows)<8 or rows[-1]['unix_s']-rows[0]['unix_s']<self.spec['window_s']:return None
        times=[r['unix_s']-rows[0]['unix_s'] for r in rows]
        streams={};by_class=defaultdict(list)
        for job in active:
            name=job['id'];phys=[r['clock_s'][name] for r in rows]
            if phys[-1]-phys[0]<self.spec['minimum_window_progress_s']:return None
            kind=job['kind'];profile=self.profiles[kind+'/'+job['data']['case']]
            reservation=self.work/'pool/raw-reservations'/(name+'.json')
            if not reservation.is_file():return None
            allocated=read(reservation)
            if allocated.get('state')!='allocated':return None
            medium='personal' if allocated['store'] else 'scratch'
            scaled=slope(times,[reference_work(profile['points'],t) for t in phys])
            by_class[kind+'/'+medium].append(scaled)
            streams[name]=dict(kind=kind,store=medium,first_s=phys[0],last_s=phys[-1],
                               physics_s_per_wall_s=slope(times,phys),reference_work_per_wall_s=scaled)
        m=dict(width=width,start=rows[0]['time_shanghai'],end=rows[-1]['time_shanghai'],
               end_unix=rows[-1]['unix_s'],wall_s=times[-1],observations=len(rows),streams=streams,
               physics_s_per_wall_s=sum(s['physics_s_per_wall_s'] for s in streams.values()),
               reference_work_per_wall_s=sum(s['reference_work_per_wall_s'] for s in streams.values()),
               classes={k:dict(n=len(v),mean=sum(v)/len(v)) for k,v in by_class.items()},
               other_cpu=sum(r['other_core_s_per_s'] for r in rows)/len(rows),
               min_available_cpu=min(r['available_cpu_cores'] for r in rows),
               min_available_ram_GiB=min(r['ram_available_bytes'] for r in rows)/2**30,
               max_io_pressure_pct=max(r['io_pressure_pct'] for r in rows),
               max_host_iowait=max(r['host_iowait_fraction'] for r in rows),
               min_gpu_free_MiB=min(r['gpu'][2]-r['gpu'][1] for r in rows if r['gpu']) if all(r['gpu'] for r in rows) else None)
        return m,rows

    def compare(self,base,new):
        limit=self.spec['minimum_per_stream_ratio']
        work='reference_work_per_wall_s';raw='physics_s_per_wall_s'
        common=base['classes'].keys() & new['classes'].keys()
        ratios={k:new['classes'][k]['mean']/base['classes'][k]['mean'] for k in common}
        shared=base['streams'].keys() & new['streams'].keys()
        incumbent=(sum(new['streams'][k][work] for k in shared)/
                   sum(base['streams'][k][work] for k in shared)) if shared else None
        same_mix=(base['classes'].keys()==new['classes'].keys() and all(
            a['n']*new['width']==new['classes'][k]['n']*base['width'] for k,a in base['classes'].items()))
        check_raw=set(base['streams']).issubset(new['streams']) or same_mix
        total=new[work]/base[work];avg=total*base['width']/new['width']
        ext_ok=not (new['other_cpu']+1<base['other_cpu'] and base['other_cpu']>2)
        ok=(bool(common) and total>1 and avg>=limit and min(ratios.values())>=limit and
            (incumbent is None or incumbent>=limit) and (not check_raw or new[raw]>base[raw]) and ext_ok)
        return dict(passed=ok,raw_total_ratio=new[raw]/base[raw],raw_comparable=check_raw,
                    total_work_ratio=total,per_stream_work_ratio=avg,common_class_ratios=ratios,
                    continuing_stream_ratio=incumbent,external_cpu_comparable=ext_ok)

    def observe(self,now,active):
        self.history.append(now)
        s=self.state;width=len(active)
        if not self.applies(active) or width not in self.spec['widths'] or width>s['admission_limit']:return
        if width<s['retained'] or width in s['blocked']:return
        if self.profiles is None and (not self.local or not self.freeze_scratch_reference()):return
        s['storage_stratum']='scratch' if self.local else 'with_personal'
        candidate=self.window(active,now)
        if candidate is None:return
        sample,rows=candidate
        if not room(rows,width):return
        s['last_end'][str(width)]=sample['end_unix']
        prev=3 if width==4 else 4
        if width>3 and str(prev) not in s['windows']:return
        cmp=self.compare(s['windows'][str(prev)],sample) if width>3 else None
        good=cmp is None or cmp['passed']
        action='measured'
        if good:
            s['bad'][str(width)]=0
            if width>=s['retained']:
                s['retained']=width
                if str(width) not in s['windows']:s['windows'][str(width)]=sample
            target={3:4,4:6}.get(width)
            if target and target not in s['blocked'] and room(rows,target) and s['admission_limit']<=width:
                s['windows'][str(width)]=sample
                s['admission_limit']=target
                action='open_trial_'+str(target)
            elif width>3:action='retain_'+str(width)
        else:
            s['bad'][str(width)]=s['bad'].get(str(width),0)+1
            action='below_reference'
            if s['bad'][str(width)]>=self.spec['negative_windows_before_retiring']:
                s['admission_limit']=prev;s['retained']=prev;s['blocked'].append(width)
                action='drain_to_'+str(prev)
        event=dict(at=sample['end'],action=action,measurement=sample,comparison=cmp)
        if action!='measured' or not s['events']:s['events'].append(event)
        s['latest']=event
        write(self.path,s)


class EvaluationCapacityByStore:
    def __init__(self,root,state,work):
        self.with_personal=EvaluationCapacity(root,state,work)
        self.scratch=EvaluationCapacity(root,state,work,local=True)
        self.work=work
        self.current=self.with_personal

    @property
    def state(self):
        return self.current.state

    @property
    def path(self):
        return self.current.path

    def applies(self,active):
        return self.with_personal.applies(active)

    def cohort(self,active,extra_store=None):
        if extra_store:return self.with_personal
        for job in active:
            path=self.work/'pool/raw-reservations'/(job['id']+'.json')
            record=read(path) if path.is_file() else {}
            volume=record.get('store') if record.get('state')=='allocated' else job.get('raw_store')
            if volume:return self.with_personal
        return self.scratch

    def cap(self,job,active,ordinary,store=None):
        if not self.applies([*active,job]):return ordinary
        group=self.cohort(active,store)
        return group.cap(job,active,ordinary)

    def observe(self,now,active):
        self.current=self.cohort(active)
        self.current.observe(now,active)
