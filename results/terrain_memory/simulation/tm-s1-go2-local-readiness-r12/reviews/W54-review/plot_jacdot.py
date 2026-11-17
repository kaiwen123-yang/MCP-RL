"""Plot independently recomputed saved-array kinematic identity residuals."""
from pathlib import Path
import argparse,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser(description=__doc__);P.add_argument('--audit-root',type=Path,default=Path(__file__).resolve().parent);args=P.parse_args();A=args.audit_root;d=json.loads((A/'jacdot-summary.json').read_text());rows=np.loadtxt(A/'jacdot-r52-central.csv',delimiter=',',skiprows=1);t=rows[:,0];before=rows[:,2:5];missing=rows[:,5:8];after=rows[:,8:11]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.65,1]});fig.subplots_adjust(top=.77,bottom=.22,wspace=.23)
fig.suptitle('A missing kinematic term matches the saved acceleration error',x=.07,y=.965,ha='left',fontsize=17,fontweight='bold');fig.text(.07,.875,'Independent saved-array check: no new simulator steps, QP solves, or corrected controller trial.',fontsize=10,color='#475569')
axes[0].scatter(t,before[:,2],s=23,facecolors='none',edgecolors='#3a5f9e',linewidths=1.1,label='Old vertical identity error');axes[0].scatter(t,missing[:,2],s=22,marker='x',color='#d68a20',linewidths=1.1,label='Missing (omega x v) vertical term');axes[0].scatter(t,after[:,2],s=7,color='#008a7a',label='Residual after adding term');axes[0].axhline(0,color='#7b8795',lw=.7);axes[0].set_xlabel('F r52 physical time (s)');axes[0].set_ylabel('Vertical acceleration (m/s²)');axes[0].legend(loc='upper left',fontsize=8.6);axes[0].set_xlim(2,4.9)
x=np.arange(2);b=[d[k]['central']['before_RMS_all_xyz'] for k in ['51','52']];c=[d[k]['central']['after_RMS_all_xyz'] for k in ['51','52']];bars=axes[1].bar(x-.18,b,.32,color='#3a5f9e',label='Before');bars2=axes[1].bar(x+.18,c,.32,color='#008a7a',label='After adding missing term');axes[1].set_xticks(x,['F r51 (160 samples)','F r52 (162 samples)']);axes[1].set_ylabel('XYZ component RMS (m/s²)');axes[1].set_ylim(0,3.6);axes[1].legend(fontsize=8.5,loc='upper left')
for xx,v in zip(x,b):axes[1].text(xx-.18,v+.055,f'{v:.3f}',ha='center',fontsize=10)
for xx,v in zip(x,c):axes[1].text(xx+.18,.07,f'{v:.5f}',ha='center',fontsize=10,color='#006e61')
fig.text(.07,.10,'Central difference uses saved velocities at t ± 2 ms; the saved J removes actual-versus-WBC generalized-acceleration mismatch.',fontsize=9,color='#475569');fig.text(.07,.055,'Scope: swing-foot records at MPC times, 2.002–4.880 s. Forward-difference checks also agree; numerical residuals differ by method.',fontsize=9,color='#475569');fig.savefig(A/'w54-jacdot-crosscheck.png',dpi=180);plt.close(fig)
