"""Compare short-window whole-foot errors with longer-horizon cell errors on the held A episode."""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.right':False,'axes.spines.top':False,'axes.spines.left':False,
                     'axes.linewidth':.65,'legend.frameon':False})


def read(name):
    return json.loads((OUT/'m'/name).read_text())


def main():
    summary=read('selection.json')['primary_and_secondary']['a']
    events=read('error-by-event.json')
    rows=[]
    for head in ('M0','M1'):
        rows.append([summary[head]['nonrelease_foot_force'],events['rows']['a'][head]['cross_foot_revisit']['future_force_N'],
                     summary[head]['release_to_reload_cell_force'],events['release_to_reload']['a'][head]['cross_foot_revisit']['next_cell_force_N']])
    fig,ax=plt.subplots(figsize=(7.08,3.55))
    fig.subplots_adjust(left=.37,right=.96,top=.70,bottom=.20)
    fig.text(.035,.93,'Force prediction on the held-out A episode',fontsize=13,weight='bold',va='top')
    fig.text(.035,.84,'Both observers use the same recorded boundary conditions. Future force is withheld.',fontsize=8.6,va='top')
    m0,m1='#4a5565','#006e70'
    legend=[Line2D([],[],ls='none',marker='o',markerfacecolor='white',markeredgecolor=m0,ms=5.5,label='M0  Forward observer'),
            Line2D([],[],ls='none',marker='D',markerfacecolor=m1,markeredgecolor=m1,ms=4.9,label='M1  Force-corrected observer')]
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.97,.785),ncol=2,handletextpad=.2,
               columnspacing=1.2,fontsize=7.8)
    labels=['20.2 ms  ·  Whole foot\nAll sampled loads','20.2 ms  ·  Whole foot\nCross-foot revisits',
            'Release to reload  ·  One cell\nAll matched reloads','Release to reload  ·  One cell\nCross-foot revisits']
    ys=[3.15,2.25,1.10,.20]
    for i,y in enumerate(ys):
        a,b=rows[0][i]['mae'],rows[1][i]['mae']
        ax.plot([a,b],[y,y],color='#a7aeb2',linewidth=1.4,zorder=2)
        ax.plot(a,y,ls='none',marker='o',mec=m0,mfc='white',mew=1.25,ms=6,zorder=3)
        ax.plot(b,y,ls='none',marker='D',mec=m1,mfc=m1,ms=5,zorder=4)
        ax.text(-.010,y,labels[i]+f"  (n={rows[0][i]['n']})",ha='right',va='center',fontsize=8)
    ax.axhline(1.67,color='#c9cecf',linewidth=.6)
    ax.set_xlim(0,.28);ax.set_ylim(-.3,3.55);ax.set_yticks([])
    ax.set_xticks([0,.05,.10,.15,.20,.25])
    ax.grid(axis='x',color='#e2e5e6',linewidth=.55,which='major')
    ax.set_axisbelow(True);ax.set_xlabel('Mean absolute normal-force error (N)',labelpad=5)
    fig.text(.035,.040,'n counts events in one episode. Cross-foot events form a subset of each corresponding group.',
             fontsize=7.3,color='#48535a')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'material-held-A.png',dpi=320,facecolor='white')
    fig.savefig(folder/'material-held-A.svg',facecolor='white')
    fig.savefig(folder/'material-held-A.pdf',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
