"""Plot already-recorded W53 actual trajectory/contact/material streams."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser();P.add_argument('--originals',type=Path,required=True);P.add_argument('--output',type=Path,default=Path(__file__).parent);a=P.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.titleweight':'bold'})
colors={'N':'#007b87','F':'#b45b18'}
D={}
for arm in ['N','F']:
 p=next(a.originals.glob('job-*w53-'+arm.lower()+'-a*'));tr=dict(np.load(p/'trajectory.npz'));ph=np.concatenate([np.load(f)['physical'] for f in sorted(p.glob('physical-*.npz'))]);mat=np.concatenate([np.load(f)['foot'] for f in sorted(p.glob('material-*.npz')) if 'final' not in f.name]);D[arm]=(p,tr,ph,mat)
fig,ax=plt.subplots(3,2,figsize=(13.5,9),sharex='col',gridspec_kw={'height_ratios':[1,1,.8]});fig.subplots_adjust(top=.88,hspace=.16,wspace=.23,bottom=.10)
fig.suptitle('W53: N recovered forward motion; F lost support before its solver stopped',x=.07,ha='left',y=.98,fontsize=18,fontweight='bold')
fig.text(.07,.935,'All actual recorded time | 500 Hz pose + 5,000 Hz contact stream | No new simulation',fontsize=11,color='#465264')
for j,arm in enumerate(['N','F']):
 p,tr,ph,mat=D[arm];col=colors[arm];t=tr['time'];counts=(ph[:,[32,35,38,41]]>1).sum(axis=1)
 ax[0,j].set_title('N / restored R8 motion kernel' if arm=='N' else 'F / contact-force MPC + WBC',loc='left',fontsize=13,pad=12)
 ax[0,j].plot(t,tr['qpos'][:,0],color=col,lw=2);ax[0,j].set_ylabel('Base x (m)')
 ax[1,j].plot(t,tr['qpos'][:,2],color=col,lw=2);ax[1,j].set_ylabel('Base height (m)');ax[1,j].set_ylim(.18,.32)
 ax[2,j].step(ph[:,0],counts,where='post',color=col,lw=.8);ax[2,j].set_ylabel('Feet with Fz > 1 N');ax[2,j].set_yticks([0,1,2,3,4]);ax[2,j].set_ylim(-.2,4.2);ax[2,j].set_xlabel('Physical time (s)')
 for i in range(3):ax[i,j].axvspan(0,2,color='#e3e7ec',alpha=.65,zorder=0);ax[i,j].set_xlim(0,float(t[-1])+.04)
 ax[0,j].text(.1,.90,'Initial Stand',transform=ax[0,j].transAxes,color='#5f6876',fontsize=10)
 if arm=='N':
  tm=6.027400000003871
  for i in range(3):ax[i,j].axvline(tm,color='#946b26',ls='--',lw=1)
  ax[0,j].annotate('First material force\nt = 6.0274 s',xy=(tm,np.interp(tm,t,tr['qpos'][:,0])),xytext=(3.8,.69),arrowprops={'arrowstyle':'-','color':'#946b26'},fontsize=10)
  ax[1,j].text(.46,.11,'Ends at z = 0.2904 m\nMaterial limit; no non-foot load',transform=ax[1,j].transAxes,fontsize=10,color=col)
 else:
  edges=np.diff(np.r_[False,counts==0,False].astype(int));st=np.flatnonzero(edges==1);en=np.flatnonzero(edges==-1)
  for stt,enn in zip(st,en):
   if ph[stt,0]<2:continue
   for i in [1,2]:ax[i,j].axvspan(ph[stt,0],ph[enn-1,0]+.0002,color='#cf3644',alpha=.12,zorder=0)
  ax[1,j].text(.08,.10,'Red bands: no foot above 1 N\nEnds at z = 0.2002 m',transform=ax[1,j].transAxes,fontsize=10,color='#963543')
fig.text(.07,.025,'The 1 N contact threshold is diagnostic only. Original termination rules and recorded physics are unchanged.',fontsize=10,color='#465264')
fig.savefig(a.output/'w53-motion-and-support.png',dpi=170);plt.close(fig)

p,tr,ph,mat=D['N'];m=(mat[:,0]>=6.462-1e-9);v=mat[m];x=(v[:,0]-6.464)*1000;applied=v[:,0]<tr['time'][-1]-1e-9
fig,axes=plt.subplots(2,2,figsize=(12,7.4),sharex=True);fig.subplots_adjust(top=.82,bottom=.11,hspace=.30,wspace=.27)
fig.suptitle('N: the final FL touchdown reaches the original material limit',x=.08,ha='left',y=.98,fontsize=18,fontweight='bold')
fig.text(.08,.925,'Saved 0.2 ms material evaluations | Hollow red point: rejected next material state; force never applied',fontsize=10.5,color='#465264')
fields=[(8,1000,'Foot-bottom height (mm)'),(20,1,'Vertical foot velocity (m/s)'),(32,1,'Material normal force (N)'),(46,1,'Maximum principal strain magnitude')]
for ax,(k,scale,label) in zip(axes.flat,fields):
 y=v[:,k]*scale;ax.plot(x[applied],y[applied],'-o',color=colors['N'],lw=1.8,ms=4);ax.plot(x[~applied],y[~applied],'o',mfc='white',mec='#c52e44',mew=2,ms=8);ax.set_ylabel(label);ax.axvline(0,color='#8791a1',ls=':',lw=1);ax.set_xlim(-2.1,1.8)
axes[0,0].axhline(-1,color='#c52e44',ls='--',lw=1);axes[0,0].text(-1.7,-.89,'-1 mm',color='#9d2738',fontsize=10)
axes[0,1].annotate('-0.956 m/s at touchdown',xy=(0,-.9558214886),xytext=(-1.7,-.55),arrowprops={'arrowstyle':'-','color':'#465264'},fontsize=10)
axes[1,0].annotate('Last applied: 1,194.56 N',xy=(1.4,1194.5643),xytext=(-1.75,850),arrowprops={'arrowstyle':'-','color':'#465264'},fontsize=10)
axes[1,1].axhline(.02,color='#c52e44',ls='--',lw=1);axes[1,1].annotate('Rejected: 0.0201782',xy=(1.6,.0201782),xytext=(-1.75,.0200),arrowprops={'arrowstyle':'-','color':'#c52e44'},fontsize=10,color='#9d2738');axes[1,1].set_ylim(0,.023)
for ax in axes[1]:ax.set_xlabel('Time from FL touchdown at t = 6.464 s (ms)')
fig.text(.08,.035,'Last integrated plant time: 6.4656 s. Rejected material state time: 6.4658 s. These clocks are kept separate.',fontsize=10,color='#465264')
fig.savefig(a.output/'n-final-impact.png',dpi=170);plt.close(fig)
print('Figures written.')
