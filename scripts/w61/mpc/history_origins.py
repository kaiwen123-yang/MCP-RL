"""GPU search at every registered, synchronous four-foot transfer overlapping the historical strip."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
FROZEN=ROOT.parent/'MCP-RL-w59'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.append(str(FROZEN/'scripts'))
from s1_w59_model_fields import STATE
from s1_w59_model_open_loop import origin_tables
from w61.mpc.problem import build_problem
from w61.mpc.solver import search

REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
O=FROZEN/REL/'w59/model'
WORK=FROZEN/'work/w59/model'
RUN='w61-c-world-origin-v1'
K0=1227272.7272727273


def main():
    out=ROOT/REL/'w61/mpc'/RUN;out.mkdir(parents=True,exist_ok=True)
    sources=json.loads((O/'historical-origins.json').read_text())['episodes']
    roots={r['episode']:Path(r['root']) for r in json.loads((O/'bank.json').read_text())['episodes']}
    rate=json.loads((O/'recovery.json').read_text())['parameters']['positive_recovery_rate_s_inv']
    rows=[];count=dict(registered=0,missing_synchronous_plan=0,not_four_contact=0,not_strip=0)
    start=time.perf_counter();cpu=time.process_time()
    for row in sources:
        ep=row['episode'];root=roots[ep];mpc_dir=root/'w55-f'
        with np.load(WORK/'historical-origins'/(ep+'.npz')) as d:
            states=d['state'];worlds=d['world'];indices=d['index']
        count['registered']+=len(states)
        series=mpc_dir/'mpc-solves.jsonl'
        plans={}
        if series.is_file():
            for line in series.read_text().splitlines():
                z=json.loads(line);plans[round(z['time_s']/.0002)]=z['file']
        selected=[]
        for j,x in enumerate(states):
            tick=int(indices[j])*100
            path=mpc_dir/plans[tick] if tick in plans else None
            if path is None or not path.is_file():
                count['missing_synchronous_plan']+=1;continue
            with np.load(path) as z:
                if not z['contact'][0].all():
                    count['not_four_contact']+=1;continue
                if not (z['feet'][0,:,0].max()>.8 and z['feet'][0,:,0].min()<2.2):
                    count['not_strip']+=1;continue
            selected.append((j,path))
        if not selected:continue
        with np.load(root/'trajectory.npz') as tr:
            feet=tr['feet'];force=tr['actual_grf'][:,:,2]
        ix=indices[[j for j,f in selected]]
        private=[]
        for leg in range(4):
            one=np.zeros_like(force);one[:,leg]=force[:,leg]
            private.append(origin_tables(feet,one,ix))
        for k,(j,plan) in enumerate(selected):
            with np.load(plan) as z:p=dict(z)
            p.update(torque_lower=-np.array([23.7,23.7,45.43]*4),torque_upper=np.array([23.7,23.7,45.43]*4))
            t=indices[j]*.02
            x=states[j];yaw=p['x0'][2]
            speed=float(np.dot(x[STATE['command']][:2],[np.cos(yaw),np.sin(yaw)]))
            raw=dict(episode=ep,index_20ms=int(indices[j]),physical_s=t,publication_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                     plan_file=str(plan),speed_m_s=speed,alternatives={})
            for arm in ('virgin','private','shared','posterior'):
                access=set()
                def lookup(leg,key):
                    idx=(key[0]-40)*60+key[1]+30
                    orig=worlds[j,idx]
                    a=private[leg][k,idx] if arm=='private' else orig
                    if orig[4] and orig[2]<0 and int(orig[3])!=leg:access.add((leg,key))
                    if not a[4] or (arm=='virgin' and a[2]<0):return 0.
                    return float(a[0]*np.exp(-rate*max(0.,t-a[1]))) if a[2]<0 else float(a[0])
                spring=K0*np.array([.7,1.,1.3]);belief=np.array([0.,1.,0.])
                if arm=='posterior':
                    tt=int(indices[j])*10
                    lo=max(0,tt-200)
                    pos=feet[lo:tt+1:10];f=force[lo:tt+1:10]
                    mask=(pos[...,0]>.8+.016)&(pos[...,0]<2.2-.016)&(f>1)
                    if mask.any():
                        depth=np.maximum(.022-pos[:,:,2],0)[mask]
                        discrepancy=(depth[:,None]*spring[None,:]-f[mask,None])/30.
                        l=np.log([.25,.5,.25])-.5*np.mean(discrepancy**2,axis=0)
                        belief=np.exp(l-l.max());belief/=belief.sum()
                problem=build_problem(p,speed,np.ones(4),belief,0.,memory=lookup,stiffness=spring,compliant=(True,True,True))
                sol=search(problem,'hybrid',61430+int(indices[j]),candidates=12,iterations=2)
                raw['alternatives'][arm]=dict(mode=sol['mode'],objective=sol['per_mode_cost'].tolist(),
                  f0=sol['force_reference'][0].tolist(),fz_sequence=sol['force_reference'][:,:,2].tolist(),
                  max_published_depth_m=float(problem['initial_delta'].max()),foreign_released_cell_foot_queries=len(access),
                  posterior=belief.tolist(),candidates=sol['evaluated_candidates'],synchronized_wall_s=sol['search_wall_s'])
            rows.append(raw)
            print(ep,t,raw['alternatives']['shared']['foreign_released_cell_foot_queries'],flush=True)
    record=dict(run=RUN,runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),counts=count,
         eligible=len(rows),rows=rows,wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,
         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,decay_rate_s_inv=rate,
         scope='Saved true four-contact states, same physical-time inputs. Four independent topology proposals against common scenarios. Archived causal depth/free-decay proxy, not the full PANN observer; candidate differences are forecasts, not newly observed gains.',
         new_plant_steps=0,deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
