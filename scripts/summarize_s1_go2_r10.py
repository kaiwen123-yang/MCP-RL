#!/usr/bin/env python3
"""Actual R10 task tables, trajectories and publication inventory; no physics."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_s1_go2_wb_baseline import dump,sha,index_file
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-soft-landing-r10'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN
OUT=ROOT/'outputs/terrain_memory/simulation'/RUN


def table(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    idx=json.loads((SMALL/'index.json').read_text())
    rows=[];curves=[];assets=[]
    for attempt in sorted(idx['attempts'],key=lambda a:a['attempt']):
        name=attempt['attempt'];path=SMALL/(name+'.json');out=OUT/name
        if attempt['status']=='initialization_failed':
            rows.append(dict(attempt=name,status='initialization_failed',physical_started=False,
                success=False,reason=attempt['reason'],simulation_s=0.,physical_steps=0,
                counts_toward_development_allowance=True))
        elif path.exists():
            r=json.loads(path.read_text());z=json.loads((SMALL/(name+'-resolved.json')).read_text())
            with np.load(out/'trajectory.npz') as data:
                a={k:data[k] for k in ['time','qpos','qvel','rpy','torque','contact']}
            f=np.concatenate([np.load(out/n)['foot'] for n in r['material']['chunks']])
            count=r['physical_samples'];applied=f[:count];loaded=applied[:,32:42:3]>1
            build=Path(z['agent_server']['path']).parent.name
            row=dict(attempt=name,formulation=r['controller_formulation'],build=build,
                cap_m_s=r['requested_cap_m_s'],workers=z['workers'],status=r['status'],
                physical_started=True,success=r['success'],reason=r['reason'],
                simulation_s=r['simulation_s'],wall_s=r['wall_s'],physical_steps=count,
                counts_toward_development_allowance=True,max_body_x_m=r['max_x_m'],
                max_abs_body_y_m=float(abs(a['qpos'][:,1]).max()),min_body_z_m=r['min_body_height_m'],
                max_abs_roll_pitch_rad=float(abs(a['rpy'][:,:2]).max()),
                max_abs_principal_strain=r['material']['max_abs_principal_strain'],
                peak_applied_material_Fz_N=float(applied[:,32:42:3].max()),
                peak_loaded_downward_m_s=float(np.max(np.where(loaded,-applied[:,20:30:3],0))),
                last_attempted_bottom_min_mm=float(1000*f[-1,8:18:3].min()),
                material_newton_residual=r['material']['max_residual'],world_cells=r['material']['world_cells'],
                max_joint_limit_excess_rad=r['max_joint_limit_excess_rad'],
                planning_wall_s=r['timing']['planning']['total_s'],
                planning_p50_ms=r['timing']['planning']['p50_ms'],
                material_wall_s=r['material']['wall_s'],
                source_commit=r['source_commit'],binary_sha256=z['agent_server']['sha256'],
                model_sha256=z['model_mjb_sha256'])
            rows.append(row);curves.append((row,a,f))
        for file in sorted(out.rglob('*')):
            if file.is_file() and file.suffix in ('.npz','.mp4','.jsonl','.mjb','.log','.xml'):
                asset_name=name+'-'+str(file.relative_to(out)).replace('/','-')
                assets.append(dict(name=asset_name,path=str(file.relative_to(ROOT)),
                    size=file.stat().st_size,sha256=sha(file)))
    table(SMALL/'episodes.csv',rows)
    physical=[r for r in rows if r['physical_started']]
    complete=all(a['status'] in ('completed','initialization_failed') for a in idx['attempts'])
    finish_unix=max((SMALL/(r['attempt']+'.json')).stat().st_mtime for r in physical)
    summary=dict(development_attempts=len(idx['attempts']),initialization_failures=len(rows)-len(physical),
        physical_missions=len(physical),completed_original_missions=sum(r['success'] for r in physical),
        integrated_physical_steps=sum(r['physical_steps'] for r in physical),
        integrated_simulation_s=sum(r['simulation_s'] for r in physical),
        summed_task_wall_s=sum(r['wall_s'] for r in physical),
        finish_reason_counts={reason:sum(r['reason']==reason for r in physical) for reason in sorted({r['reason'] for r in physical})},
        elapsed_campaign_h=(finish_unix-idx['campaign_started_unix_s'])/3600,
        campaign_end_basis='Latest original physical-task result JSON mtime, including task final serialization; excludes subsequent analysis and publication.',
        development_allowance_used=len(idx['attempts']),development_allowance_limit=12,
        figures_are_individual_development_records_not_seed_statistics=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.labelsize':9,'legend.fontsize':8})
    fig,axes=plt.subplots(3,3,figsize=(10.8,8.2),sharex='col',sharey='row')
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.83,wspace=.16,hspace=.20)
    colors={.25:'#0072B2',.215:'#D55E00',.18:'#009E73'}
    styles={.25:'-',.215:'--',.18:':'}
    for row,a,f in curves:
        build=row['build'];cap=row['cap_m_s']
        col=0 if build in ('G','GN') else 1 if build in ('GNC','GNC-r2') else 2
        color=colors[cap];style=styles[cap]
        label=f"{build} / {cap:.3f}"
        if build=='G':style=':'
        if build=='GNC':color='#555555';style='-.'
        ai=np.unique(np.r_[np.arange(0,len(a['time']),5),len(a['time'])-1])
        fi=np.unique(np.r_[np.arange(0,len(f),10),len(f)-1])
        for rr,t,y in [(0,a['time'][ai],a['qpos'][ai,0]),
                       (1,a['time'][ai],a['qpos'][ai,2]),(2,f[fi,1],f[fi,-2])]:
            axes[rr,col].plot(t,y,color=color,ls=style,lw=1.25,label=label)
            axes[rr,col].plot(t[-1],y[-1],marker='x',color=color,ms=5,mew=1)
    for col,title in enumerate(['G and G + N','G + N + C: contact models','G + N + C: body support']):
        axes[0,col].set_title(title,pad=62,weight='bold')
        axes[0,col].legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,borderaxespad=0)
        axes[0,col].axhline(3,color='#777777',ls='--',lw=.7)
        axes[1,col].axhline(.12,color='#777777',ls='--',lw=.7)
        axes[2,col].axhline(.02,color='#777777',ls='--',lw=.7)
        axes[2,col].set_xlabel('Simulation time (s)')
        for rr in range(3):
            axes[rr,col].grid(alpha=.12)
            axes[rr,col].text(.025,.95,chr(97+rr*3+col),transform=axes[rr,col].transAxes,
                va='top',weight='bold')
    axes[0,0].set_ylabel('Base x (m)');axes[0,0].set_ylim(-.08,3.15)
    axes[1,0].set_ylabel('Base height (m)');axes[1,0].set_ylim(0,.36)
    axes[2,0].set_ylabel('Maximum |principal strain|\nthrough time t')
    axes[2,0].set_ylim(0,max(.022,max(r['max_abs_principal_strain'] for r in physical)*1.08))
    fig.text(.09,.025,'Each curve ends at the recorded stop. Material traces include the final rejected constitutive step.\nDashed guides: 3 m turnaround, 0.12 m height bound, 0.02 material domain. Other failure rules also apply.',fontsize=8)
    fig.savefig(SMALL/'development-trajectories.png',dpi=180)
    # Layout evaluation is optional tooling selected by the plotting workflow,
    # not a gate or test of the physical controller.
    if str(ROOT/'work/figure_tools') not in sys.path:sys.path.insert(0,str(ROOT/'work/figure_tools'))
    from visual_qa import audit_layout
    print(json.dumps(dict(figure_layout_issues=audit_layout(fig))))
    fig.savefig(SMALL/'development-trajectories.svg')
    for ax in axes.flat:
        for line in ax.lines:
            rgb=matplotlib.colors.to_rgb(line.get_color());gray=sum(v*w for v,w in zip(rgb,[.2126,.7152,.0722]))
            line.set_color((gray,gray,gray))
    fig.savefig(OUT/'development-trajectories-grayscale.png',dpi=150);plt.close(fig)
    for build in sorted({r['build'] for r in physical}):
        folder=ROOT/'work/r10-frozen'/build
        for file in [folder/'agent_server',folder/'identity.json']:
            assets.append(dict(name=build+'-'+file.name,path=str(file.relative_to(ROOT)),size=file.stat().st_size,sha256=sha(file)))
    library=ROOT/'work/r10-frozen/G/lib/libmujoco.so.3.2.3'
    assets.append(dict(name='runtime-libmujoco.so.3.2.3',path=str(library.relative_to(ROOT)),size=library.stat().st_size,sha256=sha(library)))
    for file in sorted((OUT/'observer-r9-replay').iterdir()):
        assets.append(dict(name='observer-r9-replay-'+file.name,path=str(file.relative_to(ROOT)),size=file.stat().st_size,sha256=sha(file)))
    contract=json.loads((ROOT/'experiments/terrain_memory/s1-go2-soft-landing-r10.json').read_text())
    with index_file(SMALL,dict(run_id=RUN)) as latest:
        latest['actual_summary']=summary;latest['episodes_table']='episodes.csv';latest['assets']=assets
        latest['prior_weights']=dict(reused_sha256=contract['joint_learning']['pann_sha256'],
            source_release='https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm02-memory-study-evidence-20261023',
            archive_member='selected/B_random1000--ideal-potentials.npz',new_PANN_training=0)
        latest['figures']=['development-trajectories.png','development-trajectories.svg']
        if complete and len(latest['attempts'])>=12 and not summary['completed_original_missions']:
            latest['status']='blocked_material_execution_development_allowance_exhausted'
            latest['campaign_execution_finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(finish_unix))
            latest['campaign_execution_wall_s']=finish_unix-latest['campaign_started_unix_s']
            latest['unexecuted_reason']='No same-version nominal material mission completed within the twelve development starts, including initialization failures.'
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
