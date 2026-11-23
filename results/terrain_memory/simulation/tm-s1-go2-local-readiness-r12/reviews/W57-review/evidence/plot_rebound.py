from pathlib import Path
import argparse,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--media-dir',type=Path,required=True)
root=parser.parse_args().media_dir.resolve()
raw=root/'raw-p-a-e30/episode'
p=np.load(raw/'physical-000.npz')['physical'];t=p[:,0]
d=json.loads((root/'p-a-e30-independent-metrics.json').read_text())
e=next(v for v in d['selected_force_off_records'] if v['leg']=='FR' and 9.6<v['force_on_s']<9.7)
force=p[:,35];z=p[:,66]-.022;vz=p[:,78]
with np.load(raw/'w55-f/control-00010.npz') as c:
    ct=c['time'];ref=c['feet_position_reference'][:,1,2]-.022
mask=(t>=9.60)&(t<=9.82)
fig,axs=plt.subplots(3,1,figsize=(10,7.4),sharex=True,layout='constrained')
fig.suptitle('W57 P / A / E30: FR contacts early and rebounds\nOriginal 5 kHz physical rows; controller S5-w57-p-56b53ddc5b73',fontsize=13)
axs[0].plot(t[mask],force[mask],color='#0d6eac',lw=1.8,label='Measured normal force')
axs[0].axhline(1,color='#687889',ls=':',lw=1,label='1 N threshold')
axs[0].set_ylabel('Normal force (N)');axs[0].set_ylim(-6,185)
axs[0].text(9.67,155,'Impact: 167.49 N\n12.2 ms before committed touchdown',fontsize=10)
axs[0].legend(loc='upper right',fontsize=9)
axs[1].plot(t[mask],z[mask]*1000,color='#0d6eac',lw=1.8,label='Actual sphere bottom')
cm=(ct>=9.60)&(ct<=9.82)
axs[1].plot(ct[cm],ref[cm]*1000,color='#7a5195',lw=1.2,ls='--',label='WBC reference bottom')
axs[1].axhline(0,color='#687889',lw=.8);axs[1].set_ylabel('Bottom height (mm)')
axs[1].legend(loc='upper right',fontsize=9)
axs[2].plot(t[mask],vz[mask],color='#0d6eac',lw=1.8)
axs[2].axhline(0,color='#687889',lw=.8);axs[2].set_ylabel('Vertical velocity (m/s)')
axs[2].set_xlabel('Physical force-evaluation time (s)')
for ax in axs:
    ax.axvspan(e['loss_s'],e['reload_s'],color='#f5b971',alpha=.24,label='Fz <= 1 N gap')
    ax.axvline(e['force_on_s'],color='#c44e52',lw=1,ls=':')
    ax.axvline(e['force_on_s']-e['impact_minus_planned_td_ms']/1000,color='#555',lw=1,ls='--')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18);ax.set_xlim(9.60,9.82)
axs[1].text(9.688,7.,'102.4 ms below 1 N\n1.794 mm rebound apex',fontsize=10)
fig.savefig(root/'P-A-E30-FR-rebound-9.65s.png',dpi=170)
fig.savefig(root/'P-A-E30-FR-rebound-9.65s.pdf')
