#!/usr/bin/env python3
"""Summarize the actual R9 material prerequisite attempts; no new simulation."""
import csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_s1_go2_wb_baseline import ROOT,dump,sha,git

RUN='tm-s1-go2-joint-r9'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN
OUT=ROOT/'outputs/terrain_memory/simulation'/RUN

def main():
    idx=json.loads((SMALL/'index.json').read_text())
    rows=[];series=[];assets=[]
    names=['dev02-cap018','dev03-cap0215','dev01-cap025-run02']
    for name in names:
        r=json.loads((SMALL/(name+'.json')).read_text())
        resolved=json.loads((SMALL/(name+'-resolved.json')).read_text())
        p=OUT/name
        a=np.load(p/'trajectory.npz')
        f=np.concatenate([np.load(p/n)['foot'] for n in r['material']['chunks']])
        ph=np.concatenate([np.load(p/x['path'].split('/')[-1])['physical']
            for x in r['chunks'] if x['stream']=='physical'])
        soft=f[:,2:6].astype(bool);bot=f[:,6:18].reshape(-1,4,3)
        vel=f[:,18:30].reshape(-1,4,3);force=f[:,30:42].reshape(-1,4,3)
        failed_leg=int(np.argmin(bot[-1,:,2]))
        mask=soft[:len(ph)].repeat(3,axis=1)
        agreement=float(np.max(abs(ph[:,30:42][mask]-f[:len(ph),30:42][mask])))
        events=[json.loads(x) for x in (p/'material-events.jsonl').read_text().splitlines()]
        row=dict(attempt=name,cap_m_s=r['requested_cap_m_s'],success=r['success'],
            reason=r['reason'],physics_end_s=r['simulation_s'],max_body_x_m=r['max_x_m'],
            failed_foot=['FL','FR','RL','RR'][failed_leg],
            failed_boundary_bottom_mm=float(bot[-1,failed_leg,2]*1000),
            max_abs_principal_strain=r['material']['max_abs_principal_strain'],
            first_soft_foot_s=float(f[np.flatnonzero(soft.any(axis=1))[0],0]),
            first_material_force_s=float(f[np.flatnonzero((force[:,:,2]>1).any(axis=1))[0],0]),
            peak_applied_soft_Fz_N=float(force[:len(ph),:,2].max()),
            rejected_step_soft_Fz_N=float(force[-1,failed_leg,2]),
            peak_loaded_downward_speed_m_s=float(np.max(np.where(force[:len(ph),:,2]>1,-vel[:len(ph),:,2],0))),
            max_constitutive_residual=r['material']['max_residual'],
            actual_soft_grf_discrepancy_N=agreement,
            physical_steps=len(ph),material_steps=len(f),states_500Hz=len(a['time']),
            world_cells=r['material']['world_cells'],material_events=len(events),
            cross_leg_revisits=sum(x.get('cross_leg',False) for x in events),
            wall_s=r['wall_s'],material_wall_s=r['material']['wall_s'],
            source_commit=r['source_commit'],model_sha256=resolved['model_mjb_sha256'],
            binary_sha256=resolved['agent_server']['sha256'])
        rows.append(row);series.append((r,a,f,bot,vel,force,failed_leg))
        for file in sorted(p.iterdir()):
            if file.is_file() and (file.suffix in ('.npz','.mp4','.jsonl','.mjb','.log')):
                assets.append(dict(name=name+'-'+file.name,path=str(file.relative_to(ROOT)),
                    size=file.stat().st_size,sha256=sha(file)))
        file=p/'model/G1-project-go2.xml'
        assets.append(dict(name=name+'-model.xml',path=str(file.relative_to(ROOT)),
            size=file.stat().st_size,sha256=sha(file)))
    with (SMALL/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    colors=['#0072B2','#D55E00','#009E73'];styles=['-','--','-.']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(9,6.4),layout='constrained')
    for (r,a,f,bot,vel,force,leg),color,style in zip(series,colors,styles):
        label=f"cap {r['requested_cap_m_s']:.3f} m/s"
        t=f[:,0];lag=(t-t[-1])*1000;last=lag>=-8
        axs[0,0].plot(a['time'],a['qpos'][:,0],color=color,ls=style,label=label)
        # Max within each 2 ms plot bin preserves the original 5 kHz peaks.
        n=(len(f)//10)*10
        indentation=np.max(np.where(f[:,2:6].astype(bool),np.maximum(-bot[:,:,2]*1000,0),0),axis=1)
        axs[0,1].plot(t[:n].reshape(-1,10)[:,-1],indentation[:n].reshape(-1,10).max(axis=1),
            color=color,ls=style)
        axs[0,1].plot(t[-1],indentation[-1],marker='x',color=color)
        axs[1,0].plot(lag[last],-bot[last,leg,2]*1000,color=color,ls=style)
        axs[1,0].plot(0,-bot[-1,leg,2]*1000,marker='x',color=color)
        applied=last.copy();applied[-1]=False
        axs[1,1].plot(lag[applied],force[applied,leg,2],color=color,ls=style)
        axs[1,1].plot(0,force[-1,leg,2],marker='x',color=color)
    axs[0,0].set(xlabel='Physical time (s)',ylabel='Body x (m)',xlim=(0,9),ylim=(0,3.15),
        title='A  All three stop during outbound travel')
    axs[0,0].axhline(3,color='.4',lw=.8,ls=':',label='Original far goal')
    axs[0,0].legend(frameon=False,loc='upper left')
    axs[0,1].set(xlabel='Boundary sample time (s)',ylabel='Maximum soft-foot indentation (mm)',
        xlim=(0,9),ylim=(0,1.23),title='B  Full recorded material loading')
    axs[1,0].set(xlabel='Time relative to rejected boundary (ms)',ylabel='Failed-foot indentation (mm)',
        xlim=(-8.2,.4),ylim=(-.25,1.23),title='C  Current total displacement reaches the limit')
    axs[1,1].set(xlabel='Time relative to rejected boundary (ms)',ylabel='Failed-foot material force Fz (N)',
        xlim=(-8.2,.4),ylim=(0,1450),title='D  Loaded response before domain termination')
    for ax in [axs[0,1],axs[1,0]]:
        ax.axhline(1,color='.3',lw=.8,ls=':')
    for ax in axs.flat:ax.grid(alpha=.15)
    fig.get_layout_engine().set(rect=(0,.055,1,.94))
    axs[0,0].set_yticks(np.arange(0,3.01,.5))
    axs[0,1].set_yticks(np.arange(0,1.21,.2))
    axs[1,0].set_yticks(np.arange(0,1.21,.2))
    axs[1,1].set_yticks(np.arange(0,1401,200))
    fig.text(.5,.012,'x marks the rejected constitutive boundary: retained in data, not integrated into the robot. One episode per cap.',
        ha='center',fontsize=8)
    fig.savefig(SMALL/'material-boundary.png',dpi=190)
    fig.savefig(SMALL/'material-boundary.svg')
    fig.savefig(SMALL/'material-boundary.pdf')
    fig.canvas.draw()
    bbox=fig.bbox
    outside=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(fig.canvas.get_renderer())
            if b.x0<bbox.x0 or b.y0<bbox.y0 or b.x1>bbox.x1 or b.y1>bbox.y1:outside.append(text.get_text())
    print('Figure text outside canvas:',outside)
    plt.close(fig)
    init=idx['attempts'][0]
    file=ROOT/init['log']
    assets.append(dict(name='initialization-dev01-cap025.log',path=str(file.relative_to(ROOT)),size=file.stat().st_size,sha256=sha(file)))
    now=idx.get('campaign_stopped_unix_s',time.time())
    idx.update(status='blocked_material_domain_publication_pending',
        actual_nontraining_physical_starts=3,nontraining_start_cap=32,
        development_physical_starts=3,development_start_cap=8,
        initialization_failures=1,training_starts=0,training_runs=0,training_transitions=0,
        formal_short_starts=0,formal_long_starts=0,active_attempts=0,
        material_route_completions=0,material_route_attempts=3,
        campaign_stopped_unix_s=now,campaign_wall_s=now-idx['campaign_started_unix_s'],
        campaign_stop_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),
        physical_seconds=sum(r['physics_end_s'] for r in rows),
        physics_steps=sum(r['physical_steps'] for r in rows),
        episode_wall_sum_s=sum(r['wall_s'] for r in rows),
        resource_execution=dict(max_simultaneously_integrating_environments=2,
            solver_workers_per_environment=2,available_cpu_threads=8,
            gpu='RTX 5080; no learned workload launched before the material prerequisite passed',
            gpu_training_runs=0,process_nice=0),
        source_commit=git(ROOT),assets=assets,summary=rows,
        prerequisite=dict(passed=False,
            reason='All three allowed constant caps (.18/.215/.25 m/s) reach the unchanged |principal strain|>.02 limit during actual soft-foot loading, before completing the outbound leg.',
            implementation_evidence='The native pre-attachment model and MPC binary SHA256 exactly match R8 S3; actual soft GRF and applied material-force records agree at every integrated step; parameters and .2 ms truth integration are unchanged; constitutive residuals remain below 1e-8.',
            domain_failure_semantics='The last attempted material boundary is saved with current robot time t and material time t+.0002. Its out-of-domain force is not applied or integrated into the robot. Each run therefore has one more material-boundary sample than physical integration steps.',
            permitted_scope='No evidenced adapter defect remains in these records. Changing the contact law, material domain, S3 gait/controller or physics step is outside the frozen R9 contract. No blind additional starts were used.',
            inference_limit='These three fixed intents do not prove impossibility for every time-varying allowed speed sequence; they did not establish the required feasible domain for PPO.'),
        unexecuted=dict(pann_observer='Not implemented/executed after the actual material prerequisite failed.',
            ppo='No new policies, optimizer updates, training curves or weights; all six streams unstarted.',
            formal='21 short and 3 gated long evaluations unstarted; no trained matched checkpoint exists.',
            reasons='Physical material prerequisite failed; 48 h and 8-start caps are not exhausted.'),
        conclusions=dict(joint_closed_loop='not_established',pann_changes_actions='not_evaluated',
            improves_full_mission='not_evaluated',material_history='Persistent world grid and cross-foot material events recorded until failure; no return traverse reached.'),
        recording=dict(actual_grf_Hz=5000,material_boundary_Hz=5000,
            material_cell_state_Hz=500,robot_state_control_Hz=500,video_count=3,
            videos='Complete recorded trajectories through real failure, not 60 s successes.',
            startup_planning='20 initial planning calls per physical episode, no plant integration during them.',
            units='Joint mechanical work, when calculated, is not battery energy.'),
        figure=dict(path='material-boundary.png',data='Original high-rate arrays, 2ms peak envelope only for overview panel B; no independent sample inference or confidence intervals.'))
    dump(SMALL/'index.json',idx)
    print(json.dumps(dict(attempts=rows,total_assets=len(assets),bytes=sum(x['size'] for x in assets),
        campaign_wall_s=idx['campaign_wall_s'],physical_seconds=idx['physical_seconds']),indent=2))
if __name__=='__main__':main()

