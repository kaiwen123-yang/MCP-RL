"""Same archived event plans: one versus four independent material components."""
import argparse
import json
import time

import numpy as np

from qualify_s1_w57_planning import Saved,OUT,CFG,solve
from s1_w57_planning import Forecast,rank


def run(side,backend):
    saved=Saved(side)
    prefix='parallel-components' if backend=='original' else 'equal-components'
    output=OUT/f'{prefix}-{side}.jsonl'
    with output.open('x') as stream:
        for n,events in saved.groups():
            with np.load(saved.raw/f'mpc-{n:06d}.npz') as z:p=dict(z)
            ts=p['times'];t=float(ts[0]);saved.state(t);tick=round(t/.002)
            seen=saved.control(tick//500)['observed_contact'][tick%500]
            fc=Forecast(saved.obs.ground,ts,ts,p['foot_center_plan'][:-1,:,:2],p['contact'],saved.obs.published_bottom,seen)
            if not len(fc.keys):continue
            inputs={k:p[k] for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
            values={};timing={}
            for workers in ((4,1) if n%2 else (1,4)):
                fc.workers=workers;fc.backend=backend if workers==4 else 'original';start=time.perf_counter();cpu=time.process_time()
                values[workers]=rank(fc,inputs,p['inertia'],CFG,p['forces'],solve,.0002,'full')
                timing[str(workers)]=dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)
            a,b=values[1],values[4]
            bitwise=all(np.array_equal(aa[k],bb[k]) for aa,bb in zip(a['predictions'],b['predictions'])
                        for k in ('z','mean_z','states','inverse_residual'))
            bitwise &= np.array_equal(a['cost'],b['cost']) and np.array_equal(a['corrected'],b['corrected'])
            row=dict(attempt=saved.root.name,mpc=n,publication_s=t,components=[len(x) for x in fc.components],
                     backend4=backend,winner_equal=a['winner']==b['winner'],all_outputs_bitwise_equal=bool(bitwise),timing=timing)
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(n,row['all_outputs_bitwise_equal'],timing,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('side',choices=['a','b']);p.add_argument('--backend',choices=['original','equal_columns'],default='original')
    a=p.parse_args();run(a.side,a.backend)
