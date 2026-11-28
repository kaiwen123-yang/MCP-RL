"""Report B x D interactions by original cap actor, separating route outcome and shared-clock costs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np


def metrics(raw,limit):
    with np.load(raw/'trajectory.npz') as z:
        t=z['time'];q=z['qpos'];v=z['qvel'];ref=z['command_velocity'];tau=z['torque'];force=z['actual_grf'];rpy=z['rpy']
    end=min(limit,float(t[-1]))
    dt=np.clip(np.minimum(np.r_[t[1:],t[-1]],end)-t,0.,None)
    keep=t<=end+1.e-9
    work=tau*v[:,6:]
    return dict(duration_s=end,x_end_m=float(q[keep][-1,0]),
        velocity_ISE=float(np.sum((v[:,:2]-ref[:,:2])**2*dt[:,None])),
        work_abs_J=float(np.sum(np.abs(work)*dt[:,None])),work_positive_J=float(np.sum(np.maximum(work,0.)*dt[:,None])),
        path_m=float(np.linalg.norm(np.diff(q[keep,:2],axis=0),axis=1).sum()),
        maximum_abs_tilt_degrees=float(np.rad2deg(np.abs(rpy[keep,:2]).max())),
        GRFz_peak_500Hz_N=float(force[keep,:,2].max()))


def publish_case(args,case):
    rid='tm-s1-go2-local-readiness-r12'
    legacy='w59-b3-'+case+'-original-e60-a01'
    cells={'B0_D0':(legacy,args.frozen/'outputs/terrain_memory/simulation'/rid/legacy,
                     args.frozen/'results/terrain_memory/simulation'/rid/'w59/episodes'/(legacy+'.json'))}
    for key,arm in [('B1_D0','orbit-fixed'),('B0_D1','fixed-hybrid'),('B1_D1','orbit-hybrid')]:
        name='w61-bd-v1-'+case+'-'+arm
        cells[key]=(name,args.raw/'raw'/name,args.raw/'episodes'/(name+'.json'))
    for _,_,path in cells.values():
        while not path.exists():time.sleep(12)
    outcome={key:json.loads(file.read_text()) for key,(_,_,file) in cells.items()}
    prefix=min(r['simulation_s'] for r in outcome.values())
    landmarks={'first_10s_or_first_stop':min(10.,prefix),'all_arms_observed':prefix}
    measured={}
    for key,(name,path,file) in cells.items():
        r=outcome[key]
        measured[key]=dict(attempt=name,observed_s=r['simulation_s'],success=bool(r['success']),
            first_route_completion_s=r['first_completion_s'],status=r['stage_status'],termination=r['reason'],
            legal_roundtrips=r['legal_route']['completed_roundtrips'],full_60s=bool(r['simulation_s']>=60.-1.e-6),
            full_observed_metrics=metrics(path,r['simulation_s']),
            common_prefix={label:metrics(path,cut) for label,cut in landmarks.items()})
    def interaction(values):
        return values['B1_D1']-values['B1_D0']-values['B0_D1']+values['B0_D0']
    additive={}
    for label in landmarks:
        vals={key:r['common_prefix'][label] for key,r in measured.items()}
        additive[label]={k:interaction({name:x[k] for name,x in vals.items()}) for k in
            ('x_end_m','velocity_ISE','work_abs_J','work_positive_J','path_m','maximum_abs_tilt_degrees','GRFz_peak_500Hz_N')}
    result=dict(case=case.upper(),method='B0 frozen nominal B3 gait, B1 condition-dependent orbit and capture bridge; D0 frozen WBC, D1 partially decoupled task inertia plus periodic/nonperiodic correction.',
        original_60s_outcome_interaction=interaction({k:int(v['success']) for k,v in measured.items()}),
        shared_landmark_s=landmarks,additive_interactions=additive,measured=measured,
        scope='Three simulations plus the frozen B3 reference per cap actor; interaction is descriptive. Common-clock work uses measured joint torque and velocity. Full observed work has unequal exposure after a failure.')
    args.out.mkdir(parents=True,exist_ok=True);p=args.out/(case+'.json')
    p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    subprocess.run(['git','add','--sparse',str(p)],check=True)
    subprocess.run(['git','commit','-m',f'Record measured B by D route and shared-clock effects for case {case.upper()}'],check=True)
    print(case,result['original_60s_outcome_interaction'],landmarks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('frozen','raw','out'):p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args()
    for case in ('a','b'):publish_case(a,case)
