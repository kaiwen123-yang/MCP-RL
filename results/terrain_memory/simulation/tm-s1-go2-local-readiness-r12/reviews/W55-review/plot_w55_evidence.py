"""Compact raw-recording comparison. Saved data only; no simulation."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--audit', type=Path, required=True)
_parser.add_argument('--originals', type=Path, required=True)
_args=_parser.parse_args()
ROOT=_args.audit
ORIGINALS=_args.originals
summary=json.loads((ROOT/'arrays-recomputed.json').read_text())
def read(tag):
 p=next((ORIGINALS/tag).glob('job-*/trajectory.npz'))
 with np.load(p) as z:return z['time'],z['qpos'],z['rpy']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(2,3,figsize=(15,8.1),layout='constrained')
old,new='#bc4749','#087e8b'
for col,seed in enumerate(['a','b']):
 for rev,color,ls in [('r53',old,'--'),('r55',new,'-')]:
  t,q,r=read(f'f-{seed}-{rev}-e60');sel=t>=30;skip=np.flatnonzero(sel)[::5]
  ax[0,col].plot(t[skip],q[skip,0],label=rev,color=color,ls=ls,lw=1.6)
  ax[1,col].plot(t[skip],q[skip,2],label=rev,color=color,ls=ls,lw=1.6)
  if rev=='r53':
   ax[0,col].scatter(t[-1],q[-1,0],c=color,s=30,zorder=4)
   ax[1,col].scatter(t[-1],q[-1,2],c=color,s=30,zorder=4)
   ax[1,col].annotate(f'WBC stop {t[-1]:.3f} s\nz = {q[-1,2]:.3f} m',xy=(t[-1],q[-1,2]),xytext=(t[-1]+2,.228),fontsize=9,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.7))
 ax[0,col].axhspan(-.15,.15,color='#a8c686',alpha=.22,label='terminal radius, x projection')
 diff=summary['r53_r55_comparisons'][seed]['first_qpos_difference_over_1e12_s']
 for row in [0,1]:
  ax[row,col].axvline(diff,color='#777',lw=.8,ls=':');ax[row,col].set_xlim(30,60);ax[row,col].set_xlabel('Simulation time (s)')
 ax[0,col].set_title(f'F-{seed.upper()} / E60: terminal return')
 ax[0,col].set_ylabel('Base x (m)');ax[0,col].legend(loc='upper right',fontsize=8)
 ax[1,col].set_ylabel('Base height z (m)');ax[1,col].set_ylim(.195,.303)
 ax[1,col].text(.03,.95,f'Identical qpos until {diff-.002:.3f} s',transform=ax[1,col].transAxes,va='top',fontsize=9)
t,q,r=read('f-a-r55-e300');sl=np.arange(0,len(t),20)
ax[0,2].plot(t[sl],q[sl,0],color=new,lw=1.2)
ax[0,2].axhspan(.8,2.2,color='#bfa48a',alpha=.2,label='material strip x = 0.8–2.2 m')
case=summary['cases']['f-a-r55-e300'];times=case['roundtrip_completion_times_s'];finish=case['first_completion_s']
for i,tm in enumerate(times,1):
 idx=np.argmin(abs(t-tm));ax[0,2].scatter(tm,q[idx,0],s=24,color='#32612d',zorder=4);ax[0,2].annotate(str(i),(tm,q[idx,0]),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=9)
ax[0,2].set_ylim(-.35,3.18);ax[0,2].set_ylabel('Base x (m)');ax[0,2].set_title('F-A r55 / E300: five legal round trips')
ax[0,2].legend(loc='upper right',fontsize=8)
ax[1,2].plot(t[sl],q[sl,2],color=new,lw=.8)
ax[1,2].set_ylabel('Base height z (m)');ax[1,2].set_ylim(.28,.303)
for row in [0,1]:
 ax[row,2].axvspan(finish,300,color='#a8c686',alpha=.24);ax[row,2].set_xlim(0,300);ax[row,2].set_xlabel('Simulation time (s)')
hold=case['hold_recomputed']
ax[1,2].text(.03,.96,f'Finish: {finish:.3f} s  |  stable hold: {hold["final_stable_s"]:.3f} s\nAll original hold checks: 100%\nHold max distance: {hold["max_origin_distance_m"]*100:.2f} cm\nHold max speed: {hold["max_speed_m_s"]*1000:.2f} mm/s\nWhole run max |roll/pitch|: {np.degrees(case["max_abs_roll_pitch_rad"]):.2f}°',transform=ax[1,2].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.88))
fig.suptitle('W55 saved 500 Hz recordings: r55 resolves the observed r53 terminal instability',fontsize=16,fontweight='bold')
fig.supxlabel('CRC + published SHA256 and size verified per selected source member. Lines visually downsampled; statistics use all samples.\nE300 physical 5 kHz stream was not downloaded; 500 Hz contact checks do not prove absence of sub-tick events.',fontsize=9,color='#444')
fig.savefig(ROOT/'w55-stability-evidence.png',dpi=180)
fig.savefig(ROOT/'w55-stability-evidence.svg')
print(str(ROOT/'w55-stability-evidence.png'))
