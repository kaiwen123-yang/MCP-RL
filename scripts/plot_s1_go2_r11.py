#!/usr/bin/env python3
"""Source-backed R11 trajectory figures: each line is one actual completed task."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s1_go2_r11_protocol import ROOT,CONTRACT
from s1_go2_r11_runtime import index_file
SMALL=ROOT/CONTRACT['outputs']['compact_root'];LARGE=ROOT/CONTRACT['outputs']['large_root']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
STYLES=['-','--','-.',':',(0,(5,1)),(0,(3,1,1,1))]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
 'lines.linewidth':1.1,'savefig.dpi':300})
def save(fig,name):
    fig.canvas.draw()
    fig.savefig(SMALL/(name+'.png'),dpi=300)
    fig.savefig(SMALL/(name+'.svg'))
    plt.close(fig)
    return dict(png=name+'.png',svg=name+'.svg')
def main():
    from campaign_s1_go2_r11 import current
    index=current()
    records=[json.loads((SMALL/a['result']).read_text()) for a in index['attempts'] if a.get('result') and a['purpose']=='development']
    flags={a['attempt']:a for a in index['attempts']}
    for record in records:
        record['qualification_usable']=flags[record['attempt']].get('qualification_usable',True)
    figures={}
    for offset in range(0,len(records),6):
        selected=records[offset:offset+6]
        fig,axes=plt.subplots(4,1,figsize=(7.2,7.8),sharex=True,layout='constrained')
        for j,res in enumerate(selected):
            name=res['attempt'];a=np.load(LARGE/name/'trajectory.npz')
            foot=np.concatenate([np.load(LARGE/name/f)['foot'] for f in res['material']['chunks']])
            label=name.replace('-E30','').replace('-E60','')
            if not res['qualification_usable']:label+=' [parameter mismatch]'
            marker='s' if res['stage_status'] in ('technical_interruption','budget_truncated') else 'x' if not res['success'] else 'o' if res['qualification_usable'] else 'D'
            style=dict(color=COLORS[j],linestyle=STYLES[j])
            t=a['time']
            values=[a['qpos'][:,0],a['qpos'][:,2],np.max(abs(a['rpy'][:,:2]),axis=1)]
            for ax,y in zip(axes[:3],values):
                ax.plot(t,y,label=label,**style)
                ax.plot(t[-1],y[-1],marker=marker,color=COLORS[j],ms=5)
            axes[3].plot(foot[:,0],foot[:,46],**style)
            axes[3].plot(foot[-1,0],foot[-1,46],marker=marker,color=COLORS[j],ms=5)
        axes[0].axhline(3,color='#555555',ls=':',lw=.8);axes[0].set(ylabel='Base x (m)',ylim=(-.15,3.3),title='Individual development tasks; lines end at actual termination')
        axes[1].axhline(.12,color='#444444',ls=':',lw=.8);axes[1].set(ylabel='Base height (m)',ylim=(0,.42))
        axes[2].axhline(1.2,color='#444444',ls=':',lw=.8);axes[2].set(ylabel='Max |roll, pitch| (rad)',ylim=(0,1.3))
        axes[3].axhline(.02,color='#444444',ls=':',lw=.8);axes[3].set(ylabel='Evaluated peak |strain|',ylim=(0,max(.022,max(r['material']['max_abs_principal_strain'] for r in selected)*1.05)),xlabel='Physical time (s)')
        axes[3].set_xlim(0,max(30,max(r['evaluation_stage']['duration_s'] for r in selected)))
        for letter,ax in zip('abcd',axes):
            ax.set_title(letter,loc='left',fontweight='bold');ax.grid(alpha=.18,lw=.5)
        axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        name=f'development-trajectories-{offset//6+1:02d}'
        figures[name]=save(fig,name)
        figures[name]['caption']='Each trajectory is one declared development task, not a repeated-sample success estimate. Crosses mark failed task criteria or physical failure; squares mark technical/budget partial execution. Diamonds mark observed criteria met with a parameter mismatch and no valid qualification. Dotted lines show the3m endpoint and original physical/material limits; a contact failure can precede the height/tilt limits. Strain includes the evaluated, unapplied final boundary when a material-domain exit occurs.'
    if records:
        fig,axes=plt.subplots(2,1,figsize=(7.2,4.6),sharex=True,layout='constrained')
        for j,res in enumerate(records[:6]):
            a=np.load(LARGE/res['attempt']/'trajectory.npz');t=a['time']
            label=res['attempt'].replace('-E30','').replace('-E60','')
            axes[0].plot(t,a['body_target'][:,0]-a['qpos'][:,0],color=COLORS[j],ls=STYLES[j],label=label)
            axes[1].plot(t,a['qvel'][:,0],color=COLORS[j],ls=STYLES[j])
        axes[0].set(ylabel='Reference x - base x (m)',title='Moving-reference accumulation and actual longitudinal velocity')
        axes[1].set(ylabel='Actual base vx (m/s)',xlabel='Physical time (s)')
        axes[1].axhline(.215,color='#444444',ls=':',lw=.8,label='Fixed reference cap .215m/s')
        axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        axes[1].set_xlim(0,30)
        for ax in axes:ax.grid(alpha=.18,lw=.5);ax.axhline(0,color='#888888',lw=.5)
        figures['reference-accumulation']=save(fig,'reference-accumulation')
        figures['reference-accumulation']['caption']='Actual state and the goal sent to MPC, on the same uninterrupted physical clock. The.215m/s line is the fixed reference cap, not a hard physical velocity limit. Versions are separate development interventions.'
    # The R10 causal window is retained-source analysis, not an R11 formal prefix.
    table=LARGE/'r10-source-analysis/dev11-GNC-r3-cap0215-C-plan-contact.csv'
    old=Path('/root/autodl-tmp/MCP-RL-r10-gnc-r3/outputs/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/dev11-GNC-r3-cap0215')
    if table.exists():
        with table.open() as f:
            rows=[r for r in csv.DictReader(f) if int(float(r['leg']))==2 and float(r['origin_s'])>=32.519]
        origins=sorted({float(r['origin_s']) for r in rows})[-3:]
        foot=np.load(old/'material-0032.npz')['foot']
        select=(foot[:,0]>=32.51)&(foot[:,0]<=32.575)
        foot=foot[select];end=32.56719999995078
        fig,axes=plt.subplots(3,1,figsize=(7.2,5.6),sharex=True,layout='constrained')
        axes[0].plot(foot[:,0],1000*foot[:,14],color='black',label='Actual RL geometry')
        axes[1].plot(foot[:,0],foot[:,26],color='black')
        # The last boundary failed the domain test and its force was not applied.
        axes[2].plot(foot[:-1,0],foot[:-1,38],color='black')
        axes[2].plot(foot[-1,0],foot[-1,38],marker='x',color='black',ms=5,label='Rejected boundary force')
        for j,origin in enumerate(origins):
            chosen=[r for r in rows if float(r['origin_s'])==origin and 32.51<=float(r['time_s'])<=32.58]
            t=np.array([float(r['time_s']) for r in chosen])
            for ax,column,scale in zip(axes,['predicted_bottom_m','predicted_vz_m_s','predicted_C_Fz_N'],[1000,1,1]):
                ax.plot(t,[float(r[column])*scale for r in chosen],color=COLORS[j],ls=STYLES[j+1],label=f'C plan at {origin:.3f}s')
        axes[0].axhline(-1,color='#777777',ls=':',lw=.8)
        axes[0].set(ylabel='RL bottom (mm)',title='R10 dev11: fast landing is present in the optimized C plans')
        axes[1].set(ylabel='RL vz (m/s)')
        axes[2].set(ylabel='RL normal force (N)',xlabel='Absolute physical time (s)',xlim=(32.51,32.58))
        for ax in axes:ax.axvline(end,color='#777777',ls=':',lw=.8);ax.grid(alpha=.18,lw=.5);ax.ticklabel_format(useOffset=False,axis='x')
        axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False);axes[2].legend(handles=[axes[2].lines[1]],labels=['Rejected boundary force'],loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        figures['r10-dev11-last-RL-contact']=save(fig,'r10-dev11-last-RL-contact')
        figures['r10-dev11-last-RL-contact']['caption']='Exact saved C candidate FK and frozen causal force anchors, with2ms plan nodes and.2ms true material force boundaries. Predictions beyond the vertical termination marker are unobserved. The final boundary force is marked separately because the material-domain exit prevented its application. This is historical failure analysis, not R11 E30 qualification.'
    # Completed dev14: optimized support-foot suspension versus the real trajectory.
    path=LARGE/'dev14-S5-r12-E30'
    if (path/'trajectory.npz').exists():
        import mujoco
        tr=np.load(path/'trajectory.npz');pr=np.load(path/'prediction-000.npz')
        model=mujoco.MjModel.from_binary_path(str(path/'model.mjb'));data=mujoco.MjData(model)
        geom=model.geom('FR').id
        physical=np.load(path/'physical-000.npz')['physical']
        foot=np.concatenate([np.load(f)['foot'] for f in sorted(path.glob('material-[0-9]*.npz'))])
        fig,axes=plt.subplots(3,1,figsize=(7.6,6),sharex=True,layout='constrained')
        actual=(foot[:,0]>=9.75)&(foot[:,0]<=10.2)
        axes[0].plot(foot[actual,0],foot[actual,11]*1000,color='black',label='Actual FR bottom')
        actual_t=(tr['time']>=9.75)&(tr['time']<=10.2)
        axes[1].plot(tr['time'][actual_t],tr['qpos'][actual_t,2],color='black')
        actual_p=(physical[:,0]>=9.75)&(physical[:,0]<=10.2)
        axes[2].plot(physical[actual_p,0],physical[actual_p,44],color='black')
        rows=[]
        for n,origin in enumerate([9.8,9.9,9.98]):
            i=int(np.argmin(abs(pr['plan_origin']-origin)));height=[]
            for t,state in zip(pr['plan_times'][i],pr['plan_states'][i]):
                data.qpos[:]=state[:model.nq];mujoco.mj_kinematics(model,data)
                bottom=float(data.geom_xpos[geom,2]-model.geom_size[geom,0])
                height.append(bottom)
                rows.append(dict(origin_s=float(pr['plan_origin'][i]),time_s=float(t),
                    FR_bottom_m=bottom,base_height_m=float(state[2])))
            t=pr['plan_times'][i]
            axes[0].plot(t,np.array(height)*1000,color=COLORS[n],ls=STYLES[n+1],label=f'C plan {origin:.2f}s')
            axes[1].plot(t,pr['plan_states'][i,:,2],color=COLORS[n],ls=STYLES[n+1])
        with (SMALL/'dev14-predicted-stance-suspension.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        axes[0].set(ylabel='FR bottom above plane (mm)',title='dev14: nominal support loss is already predicted')
        axes[1].set(ylabel='Base height (m)')
        axes[2].set(ylabel='Actual total normal force (N)',xlabel='Absolute physical time (s)',xlim=(9.75,10.2))
        axes[0].axhline(0,color='#888888',lw=.6)
        axes[1].axhline(.12,color='#888888',ls=':',lw=.8)
        axes[2].axhline(15.206408*9.81,color='#888888',ls=':',lw=.8)
        for ax in axes:
            ax.grid(alpha=.18,lw=.5);ax.axvline(10.192,color='#777777',ls=':',lw=.8)
            ax.ticklabel_format(useOffset=False,axis='x')
        axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        figures['dev14-predicted-stance-suspension']=save(fig,'dev14-predicted-stance-suspension')
        figures['dev14-predicted-stance-suspension']['caption']='Actual1Hz/.65 gait confirmed by readback. FR is nominally in stance throughout this window. Saved2ms C plans predict support-foot suspension and body descent before actual body-fall termination10.192s. Candidate poses are read with forward kinematics only; no new robot integration. The real force trace uses.2ms integration-evaluation timestamps. Predicted states after actual termination are unobserved.'
    if (LARGE/'dev18-S5-r16-E30/trajectory.npz').exists():
        from matplotlib.patches import Rectangle
        fig,ax=plt.subplots(figsize=(7.4,4.5),layout='constrained')
        ax.add_patch(Rectangle((.8,-.6),1.4,1.2,facecolor='#56B4E9',alpha=.18,edgecolor='#0072B2',label='Original TM02 strip'))
        for n,name in enumerate(['dev17-S5-r15-E30','dev18-S5-r16-E30']):
            tr=np.load(LARGE/name/'trajectory.npz')
            label='dev17: effective band0 (type error)' if n==0 else 'dev18:75.25mm band; lateral bypass'
            ax.plot(tr['qpos'][:,0],tr['qpos'][:,1],color=COLORS[n],ls=STYLES[n],label=label)
            ax.plot(tr['qpos'][-1,0],tr['qpos'][-1,1],marker='x',color=COLORS[n])
        ax.plot([0,3],[0,0],color='#777777',ls=':',lw=.8,label='Original route reference')
        ax.scatter([0,3],[0,0],s=20,color='#444444')
        ax.set(xlabel='World base x (m)',ylabel='World base y (m)',
            title='Material passage is distinct from survival on a lateral rigid bypass')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18,lw=.5)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,ncol=1)
        figures['material-passage-versus-bypass']=save(fig,'material-passage-versus-bypass')
        figures['material-passage-versus-bypass']['caption']='Both are actual independent20-worker E30 development trajectories from the same initial state. dev17 accidentally used zero approach width; dev18 uses the declared75.25mm width. dev18 runs30s and reaches the far endpoint but exits the original strip side at9.300s, reaching|y|1.144624m. It is not a material-passage qualification. No new physical failure threshold or early task truncation was introduced.'
    with index_file(SMALL,CONTRACT) as idx:idx['figures']=figures
    print(json.dumps({'figures':list(figures),'completed_development_records':len(records)}),flush=True)
if __name__=='__main__':main()
