#!/usr/bin/env python3
"""R8 compact evidence and asset inventory from completed continuous records."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_s1_go2_wb_baseline import ROOT,dump,index_file,sha
CONFIG='experiments/terrain_memory/s1-go2-wb-support.json'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42})

def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(name,c):
    small=ROOT/c['outputs']['root'];out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
    r=json.loads((small/(name+'.json')).read_text());resolved=json.loads((small/(name+'-resolved.json')).read_text())
    with np.load(out/'trajectory.npz') as z:a={k:z[k] for k in z.files}
    t=a['time'];q=a['qpos'];v=a['qvel'];speed=np.linalg.norm(v[:,:3],axis=1);omega=np.linalg.norm(v[:,3:6],axis=1)
    limits=np.array([23.7,23.7,45.43]*4);saturation_seconds=np.zeros(12);sat_first=[None]*12
    max_request=np.zeros(12);max_actual=np.zeros(12);foot_impulse=np.zeros(4);nonfoot_impulse=np.zeros(3)
    total_time=0.;force_rows=[];prediction_rows=[]
    for chunk in r['chunks']:
        with np.load(chunk['path']) as z:
            if chunk['stream']=='physical':
                ph=z['physical'];dt=ph[:,1]-ph[:,0];total_time+=float(dt.sum())
                force=ph[:,30:42].reshape(-1,4,3);foot_impulse+=np.sum(force[:,:,2]*dt[:,None],axis=0)
                nonfoot_impulse+=np.sum(ph[:,45:48]*dt[:,None],axis=0)
                mask=abs(ph[:,49:61])>=limits-1e-6;saturation_seconds+=np.sum(mask*dt[:,None],axis=0)
                max_request=np.maximum(max_request,abs(ph[:,49:61]).max(axis=0));max_actual=np.maximum(max_actual,abs(ph[:,2:14]).max(axis=0))
                for j in range(12):
                    ix=np.flatnonzero(mask[:,j])
                    if len(ix) and sat_first[j] is None:sat_first[j]=float(ph[ix[0],0])
                # Time-weighted 0.1s bins retain extrema as well as means.
                bin_id=np.floor(ph[:,0]*10+1e-8).astype(int)
                for b in np.unique(bin_id):
                    ix=bin_id==b;weight=dt[ix]
                    force_rows.append(dict(start_s=float(ph[ix,0][0]),end_s=float(ph[ix,1][-1]),
                        foot_vertical_mean_N=float(np.sum(force[ix,:,2].sum(axis=1)*weight)/weight.sum()),
                        foot_vertical_min_N=float(force[ix,:,2].sum(axis=1).min()),foot_vertical_max_N=float(force[ix,:,2].sum(axis=1).max()),
                        all_contact_fraction=float(np.sum(ph[ix,26:30].all(axis=1)*weight)/weight.sum()),
                        nonfoot_vertical_mean_N=float(np.sum(ph[ix,47]*weight)/weight.sum())))
            else:
                states=z['plan_states'];pt=z['plan_times'];po=z['plan_origin']
                for horizon in [.02,.1,.2,.35]:
                    k=int(np.argmin(abs((pt[0]-pt[0,0])-horizon)));target=pt[:,k];valid=target<=t[-1]+1e-9
                    realq=np.stack([np.interp(target[valid],t,q[:,j]) for j in range(3)],axis=1)
                    realv=np.stack([np.interp(target[valid],t,v[:,j]) for j in range(6)],axis=1)
                    for origin,abstime,dq,dv,dw in zip(po[valid],target[valid],np.linalg.norm(states[valid,k,:3]-realq,axis=1),
                            np.linalg.norm(states[valid,k,19:22]-realv[:,:3],axis=1),np.linalg.norm(states[valid,k,22:25]-realv[:,3:6],axis=1)):
                        prediction_rows.append(dict(origin_s=float(origin),target_s=float(abstime),position_discrepancy_m=float(dq),velocity_discrepancy_m_s=float(dv),omega_discrepancy_rad_s=float(dw)))
    table(small/(name+'-support-10hz-bins.csv'),force_rows)
    table(small/(name+'-closed-loop-prediction.csv'),prediction_rows)
    sat=[dict(joint=joint,project_limit_Nm=float(limits[j]),actual_force_limited=bool(resolved['actuator_forcelimited'][j]),
        first_request_at_limit_s=sat_first[j],request_at_limit_s=float(saturation_seconds[j]),
        peak_request_Nm=float(max_request[j]),peak_actual_torque_Nm=float(max_actual[j])) for j,joint in enumerate(resolved['joint_names'])]
    table(small/(name+'-joint-saturation.csv'),sat)
    evidence=dict(attempt=name,control_samples=len(t),physical_duration_s=total_time,
        mean_foot_vertical_force_N=(foot_impulse/total_time).tolist(),mean_nonfoot_force_N=(nonfoot_impulse/total_time).tolist(),
        physical_peak_torque_Nm=max_actual.tolist(),max_applied_minus_nominal_target_rad=float(abs(a['q_target']-a['nominal_q_target']).max()),
        action_difference_meaning='Includes feedback and actuator-target Clamp; not unmodified feedback K dx.',
        prediction_comparison='Same absolute times against actual future with replanning and feedback; not a same-input model error.',
        source_commit=resolved['source_commit'],binary_sha256=resolved['agent_server']['sha256'])
    dump(small/(name+'-evidence.json'),evidence)
    ix=np.unique(np.r_[np.arange(0,len(t),50),len(t)-1])
    table(small/(name+'-continuous-10hz.csv'),[dict(time_s=float(t[j]),x_m=float(q[j,0]),y_m=float(q[j,1]),base_z_m=float(q[j,2]),
        speed_m_s=float(speed[j]),omega_rad_s=float(omega[j]),phase=int(a['phase'][j]),
        trunk_com_z_m=float(a['trunk_com_absolute_height'][j]),relative_height_residual_m=float(a['relative_height_residual'][j])) for j in ix])
    fig,axes=plt.subplots(5,1,figsize=(9,9),sharex=True,layout='constrained')
    axes[0].plot(t,q[:,0],color='#0072B2',lw=.8,label='Base x');axes[0].plot(t,a['body_target'][:,0],color='#D55E00',lw=.8,ls='--',label='Target x')
    axes[0].legend(ncol=2,loc='upper right');axes[0].set_ylabel('Route x (m)')
    axes[1].plot(t,q[:,2],color='#0072B2',lw=.7,label='Base');axes[1].plot(t,a['trunk_com_absolute_height'],color='#D55E00',lw=.7,ls='--',label='Trunk COM')
    axes[1].set_ylabel('Absolute height (m)');axes[1].legend(ncol=2,loc='upper right')
    axes[2].plot(t,speed,color='#0072B2',lw=.6);axes[2].axhline(.05,color='#D55E00',lw=.8,ls='--');axes[2].set_ylabel('Speed (m/s)')
    axes[3].plot(t,omega,color='#009E73',lw=.6);axes[3].axhline(.1,color='#D55E00',lw=.8,ls='--');axes[3].set_ylabel('Angular speed (rad/s)')
    ft=np.array([x['start_s'] for x in force_rows]);fm=np.array([x['foot_vertical_mean_N'] for x in force_rows]);fmin=np.array([x['foot_vertical_min_N'] for x in force_rows]);fmax=np.array([x['foot_vertical_max_N'] for x in force_rows])
    axes[4].fill_between(ft,fmin,fmax,color='#CC79A7',alpha=.25,label='Physical min/max in 0.1s bins')
    axes[4].plot(ft,fm,color='#882255',lw=.8,label='Time-weighted mean');axes[4].axhline(resolved['mass_kg']*9.81,color='0.3',lw=.8,ls='--',label='Weight')
    axes[4].set_ylabel('Foot vertical force (N)');axes[4].set_xlabel('Continuous physical time (s)');axes[4].legend(fontsize=7,ncol=3,loc='upper right')
    for ax in axes:
        ax.set_xlim(0,t[-1]);ax.grid(alpha=.15)
        if r['first_completion_s'] is not None:ax.axvline(r['first_completion_s'],color='0.3',ls=':',lw=.8)
    fig.suptitle(name+' | '+r['reason'].split(':')[0]+' | one continuous episode')
    folder=small/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    print(json.dumps(evidence),flush=True)

def summary(c):
    small=ROOT/c['outputs']['root'];idx=json.loads((small/'index.json').read_text());results=[];assets=[]
    for item in idx['attempts']:
        name=item['attempt'];path=small/(name+'.json')
        if not path.exists():continue
        r=json.loads(path.read_text());results.append(r)
        out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
        for file in sorted(out.glob('*.npz'))+list(out.glob('*.mp4'))+[out/'model.mjb',out/'agent-server.log',out/'model.xml' if r['stage']=='G0' else out/'model/G1-project-go2.xml']:
            assets.append(dict(name=name+'-'+file.name,path=str(file),size=file.stat().st_size,sha256=sha(file)))
    for file in sorted((ROOT/'work').glob('r8-build*.log')):
        assets.append(dict(name=file.name,path=str(file),size=file.stat().st_size,sha256=sha(file)))
    cols=['attempt','stage','purpose','arm','condition','mission','cohort','success','sustained_execution_success','reason','simulation_s','first_completion_s','completed_round_trips','wall_s']
    table(small/'episodes.csv',[{k:r.get(k) for k in cols} for r in results])
    holdcols=['start_s','observed_through_s','full_observation','all_criteria_fraction','violation_samples','interruptions','longest_stable_s','final_stable_s','max_speed_m_s','max_omega_rad_s','max_origin_distance_m']
    table(small/'holding.csv',[dict(attempt=r['attempt'],**{k:r['hold'].get(k) for k in holdcols}) for r in results])
    table(small/'arrivals.csv',[dict(attempt=r['attempt'],**{k:a[k] for k in ['lap','endpoint','arrival_s','stable_completion_s','stopping_s']}) for r in results for a in r['arrivals']])
    table(small/'timing.csv',[dict(attempt=r['attempt'],component=k,**v) for r in results for k,v in r['timing'].items() if v])
    with index_file(small,c) as idx:
        idx.update(assets=assets,actual_physical_starts=sum(x.get('physical_started',False) for x in idx['attempts']),
            initialization_failures_count=sum(x['status']=='initialization_failed' for x in idx['attempts']),
            missing_physical_records=[x['attempt'] for x in idx['attempts'] if x.get('physical_started') and 'result' not in x],
            completed_physical_records=len(results),configuration_sha256=sha(ROOT/CONFIG))
    print(json.dumps(dict(completed=len(results),assets=len(assets))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt');p.add_argument('--summary',action='store_true');args=p.parse_args()
    c=json.loads((ROOT/CONFIG).read_text())
    if args.attempt:analyze(args.attempt,c)
    if args.summary:summary(c)
