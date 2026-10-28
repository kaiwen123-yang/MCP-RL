"""Compare contact approximations on recorded actual kinematics, without integration."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
small=ROOT/'results/terrain_memory/simulation/tm-s1-go2-material-continuation-r11'
K0=.025*.001/.05*1e6*(9*1500*1000/(3*1500+1000))
Ki=.025*.001/.05*1e6*(9*500*300/(3*500+300))
eta=9*400*200/(3*400+200)
rate0=((K0-Ki)/(.025*.001/.05*1e6))/eta
inputs=[
 ('R10-dev11',Path('/root/autodl-tmp/MCP-RL-r10-gnc-r3'),'tm-s1-go2-soft-landing-r10','dev11-GNC-r3-cap0215'),
 ('R11-S5-r1',ROOT,'tm-s1-go2-material-continuation-r11','dev01-S5-r1-E30'),
 ('R11-S5-r4',ROOT,'tm-s1-go2-material-continuation-r11','dev04-S5-r4-E30')]
summary=[];rows=[]
for label,root,run,name in inputs:
    out=root/'outputs/terrain_memory/simulation'/run/name
    result=json.loads((root/'results/terrain_memory/simulation'/run/(name+'.json')).read_text())
    a=np.load(out/'trajectory.npz');t=a['time']
    f=np.concatenate([np.load(out/n)['foot'] for n in result['material']['chunks']])
    pos=f[:,6:18].reshape(-1,4,3);vel=f[:,18:30].reshape(-1,4,3);force=f[:,30:42].reshape(-1,4,3)
    for ti in np.flatnonzero(a['planner_s']>0):
        origin=t[ti]
        if origin<2:continue
        anchor=a['support_reference_userdata'][ti,62:78].reshape(4,4)
        before=int(np.searchsorted(f[:,0],origin-.0002))
        if abs(f[before-1,0]-(origin-.0002))<abs(f[before,0]-(origin-.0002)):before-=1
        delta0=-pos[before,:,2];F0=force[before,:,2]
        overstress=(F0-Ki*delta0)/(.025*.001*1e6)
        rate=rate0/(.1+.9*np.exp(-(overstress/10)**2))
        for horizon in [.002,.01,.02,.05,.1,.2]:
            query=origin+horizon
            if query>f[-2,0]:continue
            j=int(np.searchsorted(f[:,0],query))
            if abs(f[j-1,0]-query)<abs(f[j,0]-query):j-=1
            elapsed=f[j,0]-f[before,0]
            decay=np.exp(-rate*elapsed)
            factor=-np.expm1(-rate*elapsed)/(rate*elapsed)
            stiffness=Ki+(K0-Ki)*factor
            intercept=decay*(F0-Ki*delta0)-(stiffness-Ki)*delta0
            for leg in range(4):
                if not f[j,2+leg]:continue
                overlap=np.prod(np.maximum(1-abs(pos[j,leg,:2]-anchor[:,1:3])/.03162277660168379,0),axis=1)*anchor[:,0]
                depth=-pos[j,leg,2]
                old=max(K0*(depth+np.dot(overlap,anchor[:,3])),0)
                newer=max(K0*depth+np.dot(overlap,stiffness*depth+intercept-K0*depth),0)
                # A conditional local comparison, not a new trajectory or a fit to true q/e.
                if overlap[leg]>.8 and F0[leg]>20 and force[j,leg,2]>20 and np.linalg.norm(vel[j,leg,:2])<.02:
                    rows.append(dict(source=label,origin_s=float(origin),horizon_s=horizon,leg=leg,
                        time_s=float(f[j,0]),observed_Fz_N=float(force[j,leg,2]),frozen_C_Fz_N=old,
                        local_relaxation_Fz_N=newer,observable_rate_s_inv=float(rate[leg]),overlap_sum=float(overlap.sum()),
                        actual_vz_m_s=float(vel[j,leg,2]),overlap=float(overlap[leg])))
    for horizon in [.002,.01,.02,.05,.1,.2]:
        selected=[x for x in rows if x['source']==label and x['horizon_s']==horizon]
        if not selected:continue
        errors=np.array([[x['frozen_C_Fz_N']-x['observed_Fz_N'],x['local_relaxation_Fz_N']-x['observed_Fz_N']] for x in selected])
        item=dict(source=label,horizon_s=horizon,matched_samples=len(selected),
            old_bias_N=float(errors[:,0].mean()),relax_bias_N=float(errors[:,1].mean()),
            old_RMSE_N=float(np.sqrt(np.mean(errors[:,0]**2))),relax_RMSE_N=float(np.sqrt(np.mean(errors[:,1]**2))))
        summary.append(item);print(json.dumps(item),flush=True)
with (small/'recorded-kinematics-contact-comparison.csv').open('w',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(ROOT/'work/r11-relaxation-evidence.json').write_text(json.dumps(dict(K0_N_m=K0,Kinf_N_m=Ki,rate0_s_inv=rate0,summary=summary,
 method='Recorded actual kinematics, current-force/position anchor; constant-average strain-rate local SLS approximation. Only observed forces/positions and original material constants; no true q/e. State-dependent rate uses observable excess over equilibrium force. This is conditional model-error analysis, not closed-loop validation.'),indent=2)+'\n')
