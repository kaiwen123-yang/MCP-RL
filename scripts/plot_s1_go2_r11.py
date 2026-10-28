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
    figures={}
    for offset in range(0,len(records),6):
        selected=records[offset:offset+6]
        fig,axes=plt.subplots(4,1,figsize=(7.2,7.8),sharex=True,layout='constrained')
        for j,res in enumerate(selected):
            name=res['attempt'];a=np.load(LARGE/name/'trajectory.npz')
            foot=np.concatenate([np.load(LARGE/name/f)['foot'] for f in res['material']['chunks']])
            label=name.replace('-E30','').replace('-E60','')
            style=dict(color=COLORS[j],linestyle=STYLES[j])
            t=a['time']
            values=[a['qpos'][:,0],a['qpos'][:,2],np.max(abs(a['rpy'][:,:2]),axis=1)]
            for ax,y in zip(axes[:3],values):
                ax.plot(t,y,label=label,**style)
                ax.plot(t[-1],y[-1],marker='x' if not res['success'] else 'o',color=COLORS[j],ms=5)
            axes[3].plot(foot[:,0],foot[:,46],**style)
            axes[3].plot(foot[-1,0],foot[-1,46],marker='x' if not res['success'] else 'o',color=COLORS[j],ms=5)
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
        figures[name]['caption']='Each trajectory is one declared development task, not a repeated-sample success estimate. Crosses mark true failures. Dotted lines show the3m endpoint and original physical/material limits; a contact failure can precede the height/tilt limits. Strain includes the evaluated, unapplied final boundary when a material-domain exit occurs.'
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
    with index_file(SMALL,CONTRACT) as idx:idx['figures']=figures
    print(json.dumps({'figures':list(figures),'completed_development_records':len(records)}),flush=True)
if __name__=='__main__':main()
