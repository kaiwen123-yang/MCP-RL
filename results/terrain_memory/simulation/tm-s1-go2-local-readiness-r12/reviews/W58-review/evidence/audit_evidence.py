"""Read-only W57/W58 evidence audit. No simulator or training is executed.
Required: pinned MCP-RL checkout, G1 original archive extracted as described in manifest.
"""
import argparse,pathlib,json,gzip,xml.etree.ElementTree as ET
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=pathlib.Path,required=True);parser.add_argument('--raw-g1',type=pathlib.Path);parser.add_argument('--out',type=pathlib.Path,required=True);args=parser.parse_args()
r=args.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
def read(p):return json.loads((r/p).read_text())
def pct(a,b):return 100*(a/b-1)
o={'source_commit':'ef500f31c7a96d94f83c9afb5c7a880864b447dc','w57_heads':{},'w57_PR_E600':{},'w57_mechanism_contrasts':[],'w58_gait':{},'w58_mechanism':{}}
train=read('w57/r/pilot02/training-actuals.json')
for name,v in train['training_heads'].items():
 o['w57_heads'][name]={k:v[k] for k in ['real_distinct','repeated_identifiers','case_counts','transitions','final_checkpoint']}
for case in 'ab':
 v=read(f'w57/mechanics/w57-m1-pr-{case}-original-e600-a01.json')
 o['w57_PR_E600'][case]={k:v[k] for k in ['phase_status','observed_physical_s','first_home_stable_s','peak_nonfoot_force_N','original_slip_integral_s','max_principal_strain','timing']}
 o['w57_PR_E600'][case]['all']=v['groups']['all'];o['w57_PR_E600'][case]['motion']=v['groups']['motion'];o['w57_PR_E600'][case]['post_home_hold_s']=v['observed_physical_s']-v['first_home_stable_s']
o['w57_original_E60']={}
for case in 'ab':
 records={}
 for arm in ['b','p','r','pr','blind']:
  v=read(f'w57/mechanics/w57-m1-{arm}-{case}-original-e60-a01.json');records[arm]={'home_s':v['first_home_stable_s'],'slip_integral_s':v['original_slip_integral_s'],'motion_CoT':v['groups']['motion']['positive_mechanical_CoT'],'motion_path_m':v['groups']['motion']['base_xy_path_m'],'max_abs_roll_pitch_deg':v['groups']['all']['max_abs_roll_pitch_deg'],'max_principal_strain':v['max_principal_strain']}
 for arm in ['p','r','pr','blind']:records[arm]['vs_B_pct']={k:pct(records[arm][k],records['b'][k])for k in ['slip_integral_s','motion_CoT','motion_path_m']}
 o['w57_original_E60'][case]=records
for v in read('w57/paired-mechanisms.json')['cases']:
 row={'case':v['case'],'history':v['condition'],'verified_cycles':{k:z['verified_cycles'] for k,z in v['observed'].items()}}
 obs=v['observed'];row['vs_B_pct']={arm:{m:pct(obs[arm][m],obs['B'][m]) for m in ['slip_integral_s','motion_positive_CoT','max_principal_strain']} for arm in ['P','R','PR']};row['R_vs_blind_pct']={m:pct(obs['R'][m],obs['blind'][m]) for m in ['slip_integral_s','motion_positive_CoT']};o['w57_mechanism_contrasts'].append(row)
for name,v in read('w58/g/selection.json')['case_levels'].items():
 o['w58_gait'][name]={'g0':v['g0'],'g1':v['g1'],'change_pct':{k:pct(v['g1'][k],v['g0'][k]) for k in ['off_s','positive_CoT','slip_s_per_m']}}
for mode in ['development','validation']:
 v=read(f'w58/{mode}-matrix.json');rs=v['episodes'];rows=[];eq=[]
 for case in ['A','B']:
  for h in ['short','recovered']:
   x={z['arm']:z for z in rs if z['case']==case and z['history']==h}
   if len(x)!=4:raise ValueError(x.keys())
   full,private=x['P2-full'],x['P2-private'];fields=['home_s','motion','material','original_slip_integral_s','requested_wait_s','history_action_evidence'];diff={k:[full[k],private[k]] for k in fields if full[k]!=private[k]};eq.append({'case':case,'history':h,'equal_fields':fields,'differing_fields':diff});rows.extend([{'case':case,'history':h,'arm':a,'realized':len(z['targets_realized']),'declared':z['targets_declared'],'status':z['status'],'home_s':z['home_s'],'history_action_evidence':z['history_action_evidence']} for a,z in x.items()])
 o['w58_mechanism'][mode]={'planned':v['planned'],'completed':v['completed_observation_windows'],'passed':v['passed'],'target_counts':rows,'full_private_pairs':eq}
if args.raw_g1:
 raw=args.raw_g1;meta=json.loads((raw/'receipt/w58-g1-m1-g1-a-original-e30-a01.json').read_text());cols=meta['physical_columns'];ix={x:i for i,x in enumerate(cols)};ph=np.load(raw/'episode/physical-000.npz')['physical'];traj=np.load(raw/'episode/trajectory.npz');t=ph[:,0];assert ph.shape==(150000,85);assert np.all(np.diff(t)>0)
 limits=[max(abs(float(x))for x in e.attrib['forcerange'].split()) for e in ET.parse(raw/'episode/model/G1-project-go2.xml').getroot().find('actuator')];af=ph[:,[ix[f'actuator_force_{i}'] for i in range(12)]]
 metrics={'physics_shape':list(ph.shape),'control_rows':len(traj['time']),'physics_dt_median_s':float(np.median(np.diff(t))),'min_base_height_m':float(traj['base_absolute_height'].min()),'max_abs_roll_pitch_deg':float(np.rad2deg(np.abs(traj['rpy'][:,:2])).max()),'nonfoot_samples':int(np.count_nonzero(ph[:,ix['nonfoot_ground_contacts']])),'max_actual_actuator_force_fraction_original_limit':float(np.max(np.abs(af)/np.array(limits))),'min_physics_time_s':float(t[0]),'max_physics_time_s':float(t[-1])}
 rows=[json.loads(x) for x in gzip.open(r/'w58/g/g1/w58-g1-m1-g1-a-original-e30-a01-landings.jsonl.gz','rt')];landing=max((x for x in rows if x['status']=='bearing_established'),key=lambda x:x['observed_off_before_bearing_s']);assert landing['leg']=='FR';li=ix['grf_FR_z'];m=(t>=landing['first_force_s'])&(t<landing['first_bearing_s']);f=ph[m];off=f[:,li]<=1;idx=np.where(np.r_[True,off[1:]!=off[:-1]])[0];last=np.r_[idx[1:],len(f)];intervals=[{'start_s':float(f[i,0]),'end_s':float(f[j-1,1]),'duration_s':float(j-i)*.0002} for i,j in zip(idx,last) if off[i]]
 metrics['largest_observed_off_landing']=landing;metrics['raw_recomputed_off_intervals']=intervals;metrics['raw_recomputed_total_off_s']=sum(x['duration_s'] for x in intervals);metrics['raw_recomputed_max_off_s']=max(x['duration_s'] for x in intervals);metrics['raw_recomputed_peak20ms_N']=float(ph[(t>=landing['first_force_s'])&(t<landing['first_force_s']+.02),li].max());metrics['raw_recomputed_max_rise_m']=float(ph[m,ix['foot_position_FR_z_m']].max()-ph[np.searchsorted(t,landing['first_force_s']),ix['foot_position_FR_z_m']]);o['g1_raw_audit']=metrics
 cparts=[np.load(raw/f'episode/w55-f/control-{i:05d}.npz')for i in [11,12]];ct=np.concatenate([c['time'] for c in cparts]);mode=np.concatenate([c['event_mode'][:,1] for c in cparts]);bearing=np.concatenate([c['load_bearing_contact'][:,1] for c in cparts]);mask=(ct>=10.9)&(ct<=11.03);ids=np.where(mask)[0];metrics['wbc_changes']=[{'time_s':float(ct[i]),'mode':str(mode[i]),'bearing':bool(bearing[i])}for i in ids if i==ids[0]or mode[i]!=mode[i-1]or bearing[i]!=bearing[i-1]]
 args.out.parent.mkdir(parents=True,exist_ok=True);m=(t>=10.9)&(t<=11.03);selected=['force_evaluation_time_s','post_integration_time_s','grf_FR_z','foot_position_FR_z_m','foot_velocity_FR_z_m_s'];np.savetxt(args.out.parent/'g1-FR-contact-window.csv',ph[m][:,[ix[c]for c in selected]],delimiter=',',header=','.join(selected),comments='')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 a=ph[m];tt=(a[:,0]-landing['first_force_s'])*1000;fig,ax=plt.subplots(3,1,figsize=(9,6.7),sharex=True,constrained_layout=True);ax[0].plot(tt,a[:,li],color='#146c94');ax[0].axhline(1,color='#999999',ls=':',lw=1);ax[0].set_ylabel('FR force (N)');ax[1].plot(tt,(a[:,ix['foot_position_FR_z_m']]-ph[np.searchsorted(t,landing['first_force_s']),ix['foot_position_FR_z_m']])*1000,color='#d27a22');ax[1].set_ylabel('Rise from contact (mm)');ax[2].plot(tt,a[:,ix['foot_velocity_FR_z_m_s']],color='#5b4794');ax[2].set_ylabel('FR vertical speed (m/s)');ax[2].set_xlabel('Time after first contact at 10.9192 s (ms)')
 for axis in ax:
  for inter in intervals:axis.axvspan((inter['start_s']-landing['first_force_s'])*1000,(inter['end_s']-landing['first_force_s'])*1000,alpha=.13,color='#bf2424')
  axis.axvline((landing['first_bearing_s']-landing['first_force_s'])*1000,color='#248232',ls='--',lw=1);axis.grid(alpha=.2)
 ax[0].set_title('W58 G1 / A / E30: early impact still rebounds before bearing\n5 kHz recorded physics; shaded: force <= 1 N; green: bearing gate')
 fig.savefig(args.out.parent/'g1-FR-contact-trace.png',dpi=160);plt.close(fig)
args.out.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');print(args.out)
