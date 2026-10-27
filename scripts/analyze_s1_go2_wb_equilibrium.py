#!/usr/bin/env python3
"""Compact R7 evidence from completed physical records; no physics integration."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_s1_go2_wb_baseline import ROOT,sha,index_file,dump
CONFIG='experiments/terrain_memory/s1-go2-wb-equilibrium.json'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def csvwrite(path,columns,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)


def analyze(name,c):
    small=ROOT/c['outputs']['root'];out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
    result=json.loads((small/(name+'.json')).read_text());resolved=json.loads((small/(name+'-resolved.json')).read_text())
    with np.load(out/'trajectory.npz') as a:
        t=a['time'];q=a['qpos'];v=a['qvel'];rpy=a['rpy'];grf=a['actual_grf'];target=a['q_target'];torque=a['torque']
        plan_states=a['plan_states'];pt=a['plan_times'];po=a['plan_origin']
        speed=np.linalg.norm(v[:,:3],axis=1);omega=np.linalg.norm(v[:,3:6],axis=1)
        ix=np.unique(np.r_[np.arange(0,len(t),50),len(t)-1])
        csvwrite(small/(name+'-continuous-10hz.csv'),
            ['time_s','phase','x_m','y_m','z_m','speed_m_s','omega_rad_s','roll_rad','pitch_rad','yaw_rad',
             'all_feet_contact','total_vertical_grf_N','max_abs_torque_Nm'],
            np.c_[t[ix],a['phase'][ix],q[ix,:3],speed[ix],omega[ix],rpy[ix],a['contact'][ix].all(axis=1),grf[ix,:,2].sum(axis=1),np.abs(torque[ix]).max(axis=1)])
        csvwrite(small/(name+'-prediction-tail.csv'),
            ['origin_s','terminal_s','span_s','terminal_speed_m_s','terminal_omega_rad_s','first_action_max_rad','last_action_max_rad'],
            np.c_[po,pt[:,-1],pt[:,-1]-pt[:,0],np.linalg.norm(plan_states[:,-1,19:22],axis=1),
                np.linalg.norm(plan_states[:,-1,22:25],axis=1),np.abs(a['plan_actions'][:,0]).max(axis=1),np.abs(a['plan_actions'][:,-1]).max(axis=1)])
        # Match actual absolute times only for the distinct closed-loop discrepancy diagnostic.
        rows=[]
        for horizon in [.02,.1,.2,.35]:
            k=int(np.argmin(abs((pt[0]-pt[0,0])-horizon)))
            absolute=pt[:,k];valid=absolute<=t[-1]+1e-9
            predicted=plan_states[valid,k];absolute=absolute[valid]
            realq=np.stack([np.interp(absolute,t,q[:,j]) for j in range(3)],axis=1)
            realv=np.stack([np.interp(absolute,t,v[:,j]) for j in range(6)],axis=1)
            rows.extend(np.c_[po[valid],absolute,np.linalg.norm(predicted[:,:3]-realq,axis=1),
                np.linalg.norm(predicted[:,19:22]-realv[:,:3],axis=1),np.linalg.norm(predicted[:,22:25]-realv[:,3:6],axis=1)].tolist())
        csvwrite(small/(name+'-closed-loop-prediction.csv'),
            ['origin_s','absolute_target_s','position_discrepancy_m','speed_discrepancy_m_s','omega_discrepancy_rad_s'],rows)
        evidence=dict(attempt=name,full_control_samples=len(t),diagnostic_stride=50,
            max_abs_torque_Nm=float(np.abs(torque).max()),max_speed_m_s=float(speed.max()),max_omega_rad_s=float(omega.max()),
            initial_two_seconds=dict(max_speed_m_s=float(speed[t<=2].max()),max_omega_rad_s=float(omega[t<=2].max())),
            prediction_nodes=int(pt.shape[1]),prediction_span_s=float(np.median(pt[:,-1]-pt[:,0])),
            source_commit=resolved['source_commit'],configuration_sha256=resolved['source_sha256'][CONFIG],
            binary_sha256=resolved['agent_server']['sha256'],closed_loop_note='Future actual commands include feedback and replanning; this is not the same-input dynamics comparison.')
        if result['stage']=='G1':
            limits=np.asarray(resolved['actuator_forcerange']);physical=a['physical']
            excess=np.maximum(np.maximum(physical[:,14:26]-limits[:,1],limits[:,0]-physical[:,14:26]),0.)
            evidence['original_force_limit_excess_Nm']=float(excess.max())
        figures=small/'figures';figures.mkdir(exist_ok=True)
        fig,axes=plt.subplots(4,1,figsize=(9,8),sharex=True,layout='constrained')
        axes[0].plot(t,q[:,0],label='Body x',color='#0072B2',lw=.8)
        axes[0].plot(t,a['body_target'][:,0],label='Target x',color='#D55E00',ls='--',lw=.8)
        axes[0].set_ylabel('Position (m)');axes[0].legend(loc='upper right',ncol=2)
        axes[1].plot(t,speed,color='#0072B2',lw=.7);axes[1].axhline(.05,color='#D55E00',ls='--',lw=.8);axes[1].set_ylabel('Base speed (m/s)')
        axes[2].plot(t,omega,color='#009E73',lw=.7);axes[2].axhline(.1,color='#D55E00',ls='--',lw=.8);axes[2].set_ylabel('Angular speed (rad/s)')
        axes[3].plot(t,grf[:,:,2].sum(axis=1),color='#CC79A7',lw=.6,label='Feet')
        if 'total_ground_force' in a.files:
            axes[3].plot(t,a['total_ground_force'][:,2],color='0.25',lw=.6,label='All ground contacts')
            evidence['nonfoot_ground_contact_samples_500Hz']=int((a['nonfoot_ground_contacts']>0).sum())
            evidence['max_nonfoot_ground_force_N']=float(np.linalg.norm(a['nonfoot_ground_force'],axis=1).max())
        else:
            evidence['nonfoot_ground_contact_samples_500Hz']=None
            evidence['max_nonfoot_ground_force_N']=None
        axes[3].legend(loc='upper right')
        axes[3].axhline(resolved['mass_kg']*9.81,color='0.3',ls='--',lw=.8)
        axes[3].set_ylabel('Recorded vertical force (N)');axes[3].set_xlabel('Physical time (s)')
        for ax in axes:
            ax.set_xlim(0,max(t[-1],.01));ax.grid(alpha=.15)
            if result['first_completion_s'] is not None:ax.axvline(result['first_completion_s'],color='0.2',ls=':',lw=.8)
        fig.suptitle(name+' | '+result['reason']+' | original completion: '+str(result['first_completion_s']))
        fig.savefig(figures/(name+'-mission.png'),dpi=160);fig.savefig(figures/(name+'-mission.pdf'));plt.close(fig)
    dump(small/(name+'-evidence.json'),evidence)
    print(json.dumps(evidence),flush=True)


def summary(c):
    small=ROOT/c['outputs']['root'];index=json.loads((small/'index.json').read_text());results=[];assets=[]
    for item in index['attempts']:
        name=item['attempt'];p=small/(name+'.json')
        if not p.exists():continue
        result=json.loads(p.read_text());results.append(result)
        out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
        for filename in ['trajectory.npz','startup.npz','complete-mission.mp4','model.xml','agent-server.log']:
            file=out/filename
            if not file.exists() and filename=='model.xml':file=out/'model/G1-project-go2.xml'
            if file.exists():assets.append(dict(name=name+'-'+filename,path=str(file),size=file.stat().st_size,sha256=sha(file)))
    for file in (ROOT/'outputs/terrain_memory/simulation'/c['run_id']).glob('*matched-input.npz'):
        assets.append(dict(name=file.name,path=str(file),size=file.stat().st_size,sha256=sha(file)))
    for filename in ['r7-build.log']:
        file=ROOT/'work'/filename;assets.append(dict(name=filename,path=str(file),size=file.stat().st_size,sha256=sha(file)))
    columns=['attempt','stage','purpose','arm','condition','mission','cohort','success','reason','simulation_s','first_completion_s','completed_round_trips','wall_s']
    csvwrite(small/'episodes.csv',columns,[[r.get(k) for k in columns] for r in results])
    csvwrite(small/'holding.csv',['attempt','full_observation','start_s','observed_through_s','all_criteria_fraction','violation_samples','interruptions','longest_stable_s','final_stable_s','max_speed_m_s','max_omega_rad_s','max_origin_distance_m'],
        [[r['attempt']]+[r['hold'][k] for k in ['full_observation','start_s','observed_through_s','all_criteria_fraction','violation_samples','interruptions','longest_stable_s','final_stable_s','max_speed_m_s','max_omega_rad_s','max_origin_distance_m']] for r in results])
    csvwrite(small/'arrivals.csv',['attempt','lap','endpoint','arrival_s','stable_completion_s','stopping_s'],
        [[r['attempt']]+[a[k] for k in ['lap','endpoint','arrival_s','stable_completion_s','stopping_s']] for r in results for a in r['arrivals']])
    csvwrite(small/'timing.csv',['attempt','component','count','total_s','p50_ms','p95_ms','max_ms','logical_deadline_s','deadline_misses'],
        [[r['attempt'],name]+[v[k] for k in ['count','total_s','p50_ms','p95_ms','max_ms','logical_deadline_s','deadline_misses']] for r in results for name,v in r['timing'].items() if v])
    with index_file(small,c) as idx:
        idx.update(assets=assets,completed_records=len(results),launch_budget_used=len(idx['attempts']),
            actual_physical_starts=sum(x.get('physical_started',False) for x in idx['attempts']),
            missing_records=[x['attempt'] for x in idx['attempts'] if not (small/(x['attempt']+'.json')).exists()],
            configuration_sha256=sha(ROOT/CONFIG),
            same_input_comparisons=[str(p.relative_to(small)) for p in small.glob('*matched-input.json')])
    print(json.dumps(dict(records=len(results),assets=len(assets),successes=sum(r['success'] for r in results))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt');p.add_argument('--summary',action='store_true');args=p.parse_args()
    c=json.loads((ROOT/CONFIG).read_text())
    if args.attempt:analyze(args.attempt,c)
    if args.summary:summary(c)
if __name__=='__main__':main()
