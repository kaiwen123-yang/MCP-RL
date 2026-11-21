"""Publication-friendly summary of independent saved-recording audit, no physics."""
from pathlib import Path
import argparse,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--audit',type=Path,required=True);P.add_argument('--originals',type=Path,required=True);a=P.parse_args()
j=json.loads((a.audit/'evidence-arrays.json').read_text());s=json.loads((a.audit/'evidence-summary.json').read_text());colors={'f-a-r55-e600':'#1f77b4','f-b-r55-e600':'#2ca02c','f-a-r60-e600':'#cf4d36','f-b-r60-e600':'#8c62aa'};names={k:k.replace('f-','').replace('-e600','').upper()for k in colors};plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18,'figure.dpi':150})
fig,ax=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
for k,c in colors.items():
 r=j['cases'][k];lap=r['roundtrip_completion_times_s'];ax[0,0].plot(lap,np.arange(1,11),'o-',ms=3,lw=1.4,color=c,label=names[k])
ax[0,0].axvline(540,color='#555',ls='--',lw=1,label='Original route deadline');ax[0,0].set(xlim=(0,600),ylim=(0,10.5),xlabel='Simulation time (s)',ylabel='Completed legal round trips',title='A  All four E600 records complete ten round trips');ax[0,0].legend(fontsize=8,loc='upper left');ax[0,0].yaxis.set_major_locator(MaxNLocator(integer=True))
for k,c in colors.items():
 f=next((a.originals/k).glob('job-*/trajectory.npz'))
 with np.load(f)as z:t=z['time'];tilt=np.max(abs(z['rpy'][:,:2]),axis=1)*180/np.pi
 # Max in each 1s bin retains every peak while making the visual readable.
 n=(len(t)-1)//500;peak=tilt[:n*500].reshape(n,500).max(1)
 ax[0,1].plot(np.arange(n),peak,lw=.9,alpha=.8,color=c,label=names[k])
ax[0,1].set(xlim=(0,600),ylim=(0,1.85),xlabel='Simulation time (s)',ylabel='Max |roll/pitch| in each 1 s bin (deg)',title='B  Posture remains bounded throughout the runs')
xx=np.arange(4);bottom=np.zeros(4)
for key,label,c in [('2','Two feet (diagonal)','#427ba1'),('3','Three feet','#8ab6c6'),('4','Four feet','#d3e0e4')]:
 vals=np.array([j['cases'][k]['travel']['support_count_fraction'][key]*100 for k in colors]);ax[1,0].bar(xx,vals,bottom=bottom,label=label,color=c,width=.65)
 for x,y,b in zip(xx,vals,bottom):ax[1,0].text(x,b+y/2,f'{y:.1f}%',ha='center',va='center',fontsize=9,color='white'if key=='2'else'#1a2630')
 bottom+=vals
ax[1,0].set(xticks=xx,xticklabels=[names[k]for k in colors],ylim=(0,115),ylabel='Fraction of outbound / return samples (%)',title='C  Actual support: predominantly diagonal trot');ax[1,0].legend(loc='upper center',ncol=3,fontsize=8);ax[1,0].text(.02,.02,'Fz > 1 N at 500 Hz; zero 0/1-support samples in these phases.',transform=ax[1,0].transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
batch='f-a-r60-e600';raw=next((a.originals/batch).glob('job-*/trajectory.npz')).parent;source=next((a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f/r60').glob(raw.name+'.json'));r=json.loads(source.read_text());cols={x:i for i,x in enumerate(r['physical_columns'])}
with np.load(raw/'physical-000.npz')as z:x=z['physical']
t=x[:,0];m=(t>=8.05)&(t<=9.25);legcols=['#4377b8','#cf4d36','#39a283','#9a6ab7']
for leg,col in zip(['FL','FR','RL','RR'],legcols):
 ax[1,1].plot(t[m],1000*(x[m,cols[f'foot_position_{leg}_z_m']]-.022),color=col,lw=1.3,label=leg)
 ax[2,0].plot(t[m],x[m,cols[f'grf_{leg}_z']],color=col,lw=1.1,label=leg)
for axis in [ax[1,1],ax[2,0]]:axis.axvspan(8.3794,8.4894,color='#cf4d36',alpha=.12);axis.set_xlim(8.05,9.25);axis.set_xlabel('Simulation time (s)');axis.legend(ncol=4,loc='upper right',fontsize=8)
ax[1,1].set(ylabel='Foot bottom above original plane (mm)',title='D  Real swings and small post-touchdown rebounds');ax[1,1].annotate('FR loses contact for 110 ms\nwhile bottom rises only ~2 mm',xy=(8.435,1.95),xytext=(8.1,18),fontsize=9,arrowprops={'arrowstyle':'->','color':'#333'})
ax[2,0].set(ylabel='Actual vertical ground reaction (N)',title='E  The rebound is a real unloading event (5 kHz)');ax[2,0].text(.02,.81,'At least two other feet keep supporting the body.',transform=ax[2,0].transAxes,fontsize=8,va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
records={(r['label'],r['level']):r for r in s['attempts']if r.get('source')};bw=.3;ix=np.array([0,1])
for rev,off,c in [('r55',-bw/2,'#467b9e'),('r60',bw/2,'#cf7a58')]:
 vals=[records[(f'F-{case}-{rev}','E600')]['source_material']['integral_slip_sq_s']for case in ['A','B']];ax[2,1].bar(ix+off,vals,width=bw,color=c,label=rev)
 for i,v in zip(ix+off,vals):ax[2,1].text(i,v+.002,f'{v:.4f}',ha='center',fontsize=9)
ax[2,1].set(xticks=ix,xticklabels=['Frozen actor A','Frozen actor B'],ylim=(0,.18),ylabel='Normalized integrated slip cost (s)',title='F  Material interface is stable; slip is not improved');ax[2,1].legend(loc='upper center',ncol=2,fontsize=8);ax[2,1].text(.02,.02,'Integral of loaded-foot mean (v_xy / 0.25 m/s)^2; not distance.',transform=ax[2,1].transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
fig.suptitle('S1 W56: stable task completion and actual gait quality',fontsize=16);fig.savefig(a.audit/'w56-gait-evidence.png',dpi=160);fig.savefig(a.audit/'w56-gait-evidence.svg');plt.close(fig)
