"""Compare the specialized mixed-boundary map to unmodified deployed JAX equations."""
import time
import numpy as np
import jax
from s1_w55_c_data import RAW,COMPACT,WEIGHTS,DT,ROOT,write
from tm02.s1_material import LearnedColumns


def check(tag,weights,name):
    jax.config.update('jax_enable_x64',True)
    t0=time.time()
    law=LearnedColumns(weights)
    with np.load(RAW/f'{name}-histories.npz') as data,np.load(RAW/f'{tag}-{name}-model.npz') as predictions:
        results=[]
        for clock,inputs,dt in [('fine',data['normal'],DT),('observer',data['normal'][data['coarse']],.002)]:
            model=predictions[clock]
            use=np.flatnonzero((model[:,:,4].min(axis=0)<-.1))
            events=[]
            for j in use:
                active=model[:,j,4]<-1e-7
                switches=np.flatnonzero(active[1:]!=active[:-1])+1
                events.extend([(int(np.argmin(model[:,j,4])),j),(int(np.argmax(np.abs(model[:,j,2]))),j)])
                events.extend((int(i),j) for i in switches[::max(1,len(switches)//4)])
            events=sorted(set((int(t),int(j)) for t,j in events if t>0))
            events=[events[k] for k in np.unique(np.linspace(0,len(events)-1,min(48,len(events)),dtype=int))]
            jj=np.array([j for i,j in events]);ii=np.array([i for i,j in events])
            prev=model[ii-1,jj,:4];normal=inputs[ii,predictions['unique'][jj]]
            with jax.experimental.enable_x64():
                dt0=dt;z=np.zeros(len(ii))
                free=tuple(np.asarray(x) for x in law.solve(prev[:,:2],prev[:,2:],z,dt0,z.astype(bool)))
                fixed=normal<=free[0][:,0]
                nr=np.where(fixed,normal,0.)
                cand=tuple(np.asarray(x) for x in law.solve(prev[:,:2],prev[:,2:],nr,dt0,fixed))
            active=fixed&(cand[2][:,0]<0)
            x=np.c_[np.where(active[:,None],cand[0],free[0]),np.where(active[:,None],cand[1],free[1])]
            pressure=np.where(active,cand[2][:,0],0.)
            err=np.max(np.abs(x-model[ii,jj,:4]),axis=0)
            perr=float(np.max(np.abs(pressure-model[ii,jj,4])))
            rr=float(np.max(np.maximum(cand[3],free[3])))
            if max(err)>1e-9 or perr>1e-6 or rr>1e-7:
                raise ValueError((clock,err,perr,rr))
            results.append(dict(clock=clock,transitions=len(ii),contact_queries=int(fixed.sum()),
                                e11_e22_q11_q22_max_abs=err.tolist(),max_stress_difference_MPa=perr,
                                max_deployed_Newton_residual_MPa=rr))
    write(COMPACT/f'{tag}-deployed-map-check.json',dict(start_unix=t0,end_unix=time.time(),checks=results,
        source='Unmodified tm02.s1_material.LearnedColumns.solve on RTX PRO 6000, previous independently calculated model state; node selection covers loading peaks and contact transitions.'))
    print(results,flush=True)


if __name__=='__main__':
    import sys
    from pathlib import Path
    check(sys.argv[1],Path(sys.argv[2]),sys.argv[3])
