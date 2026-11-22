"""Representative archived FR early contact, its WBC command and a four-E600 event census."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from analyze_s1_w57_gait import ROOT,R12,REL,LEGS

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.65,
    'xtick.major.size':3,'ytick.major.size':3,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def main(qa=None):
    p=R12/'w57/p';figures=R12/'w57/figures';figures.mkdir(exist_ok=True)
    report=json.loads((p/'full-force-off-summary.json').read_text())
    # Same published B/FR episode diagnosed in W56, chosen before the four-case census.
    info=json.loads((p/'full-force-off-f-b-r60-e600.json').read_text())
    raw=ROOT.parent/'MCP-RL-w56-r60'/'outputs'/REL/info['source_attempt']
    allrows=[json.loads(l) for l in gzip.open(p/'impact-sequence-f-b-r60-e600.jsonl.gz','rt')]
    ev=next(x for x in allrows if x['leg']=='FR' and abs(x['loss_s']-7.8112)<.0001)
    td=ev['first_impact_after_apex_s']-ev['impact_minus_planned_touchdown_s']
    cols=json.loads((ROOT/info['original_result']).read_text())['physical_columns']
    with np.load(raw/'physical-000.npz') as z:
        data=z['physical'];selected=(data[:,0]>=td-.026)&(data[:,0]<=td+.125)
        ts=data[selected,0]
        height=(data[selected,cols.index('foot_position_FR_z_m')]-.022)*1000
        fz=data[selected,cols.index('grf_FR_z')]
    arr={}
    for j in sorted({int(round(float(t)/.002))//500 for t in (ts[0],ts[-1])}):
        with np.load(raw/'w55-f'/f'control-{j+1:05d}.npz') as z:
            for key in ('time','force','feet_position_reference','event_mode'):
                arr.setdefault(key,[]).append(z[key])
    arr={k:np.concatenate(v) for k,v in arr.items()};visible=(arr['time']>=ts[0])&(arr['time']<=ts[-1])
    current=visible & (arr['event_mode'][:,1]=='swing') & (arr['time']<=td+1e-9)
    x=(ts-td)*1000;ink='#225b76';control='#a64b2b';gray='#65727b'
    fig=plt.figure(figsize=(7.2,4.45))
    gs=fig.add_gridspec(2,2,width_ratios=(1.65,1),left=.115,right=.97,top=.72,bottom=.18,
                        hspace=.34,wspace=.38)
    top=fig.add_subplot(gs[0,0]);bottom=fig.add_subplot(gs[1,0],sharex=top);counts=fig.add_subplot(gs[:,1])
    fig.text(.045,.955,'Early contact at the end of a planned swing',fontsize=13,weight='bold',va='top')
    fig.text(.045,.888,'Front-right foot in baseline r60 B and the complete 5 kHz census across four original E600 runs.',
             fontsize=8,color='#39474f',va='top')
    for ax in (top,bottom):
        ax.axvline(0,ls=':',lw=.9,color=gray,zorder=1)
        ax.grid(axis='y',lw=.45,color='#d5dcdf');ax.set_axisbelow(True)
        ax.set_xlim(-26,125);ax.set_xticks([-20,0,25,50,75,100,125])
    measured,=top.plot(x,height,color=ink,lw=1.5,label='Measured at 5 kHz')
    controller,=top.plot((arr['time'][current]-td)*1000,(arr['feet_position_reference'][current,1,2]-.022)*1000,
             color=control,ls='--',lw=1.35,label='Controller at 500 Hz')
    top.tick_params(labelbottom=False);top.set_ylim(-.38,max(4.4,float(height.max())+.4));top.set_yticks([0,2,4])
    top.set_ylabel('Sole height (mm)')
    top.set_title('a   Foot motion',loc='left',fontsize=8.7,weight='bold',pad=5)
    fig.legend(handles=(measured,controller),loc='center left',bbox_to_anchor=(.108,.804),ncol=2,
               fontsize=7.3,handlelength=1.9,columnspacing=2.2,handletextpad=.6)
    bottom.plot(x,fz,color=ink,lw=1.25,label='Measured force')
    bottom.step((arr['time'][visible]-td)*1000,arr['force'][visible,1,2],where='post',ls='--',lw=1.2,
                color=control,label='WBC force')
    bottom.set_ylim(-9,192);bottom.set_yticks([0,80,160]);bottom.set_ylabel('Normal force (N)')
    bottom.set_xlabel('Time from committed touchdown (ms)')
    bottom.set_title('b   Measured load and command',loc='left',fontsize=8.7,weight='bold',pad=5)

    numbers=[];labels=['r55 A','r55 B','r60 A','r60 B']
    for label in labels:
        rev,case=label.lower().split();name=f'f-{case}-{rev}-e600'
        entry=json.loads((p/f'full-force-off-{name}.json').read_text())
        numbers.append([sum(a.get('small_force_off_intervals',0) for a in entry['phase_surface_leg_cells']
                            if a['leg']==leg) for leg in LEGS])
    numbers=np.array(numbers);palette=LinearSegmentedColormap.from_list('frequency',['#f3f6f7','#c8dce1','#398099','#12425b'])
    counts.imshow(numbers,cmap=palette,vmin=0,vmax=numbers.max(),aspect='auto',interpolation='nearest')
    counts.set_xticks(range(4),LEGS);counts.set_yticks(range(4),labels);counts.tick_params(length=0)
    counts.set_xticks(np.arange(-.5,4,1),minor=True);counts.set_yticks(np.arange(-.5,4,1),minor=True)
    counts.grid(which='minor',color='white',lw=1.2);counts.tick_params(which='minor',bottom=False,left=False)
    for s in counts.spines.values():s.set_visible(False)
    for (i,j),n in np.ndenumerate(numbers):
        counts.text(j,i,str(n),ha='center',va='center',fontsize=8.8,
                    color='white' if n>=numbers.max()*.70 else '#183642')
    counts.set_title('c   Events per 600 s run',loc='left',fontsize=8.7,weight='bold',pad=5)
    counts.set_xlabel('Foot',labelpad=6)
    fig.text(.045,.045,'1,047 events in 1,198.31 s of straight travel. Of these, 1,020 began on material. Time zero is the planned touchdown.',
             fontsize=7.1,color='#39474f')
    if qa:
        sys.path.insert(0,str(qa))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=qa/'w57-impact.alignment.json',
            overlay_svg=qa/'w57-impact.alignment.svg',strict=True)
    fig.savefig(figures/'gait-early-contact.pdf',facecolor='white')
    fig.savefig(figures/'gait-early-contact.svg',facecolor='white')
    fig.savefig(figures/'gait-early-contact.png',dpi=600,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qa',type=Path);main(ap.parse_args().qa)
