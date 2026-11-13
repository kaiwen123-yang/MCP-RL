#!/usr/bin/env python3
"""Recompute W46 trajectory statistics from an existing server/local original.
No download or raw copy is implemented; only the derived JSON is written.
The published audit summaries were originally computed from SHA/CRC-verified RAM bytes.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def first(t,mask):
    idx=np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['w46-a','w46-b'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True,help='Existing local trajectory.npz; no network fallback')
    args=ap.parse_args();case=args.case
    BASE=args.repo.resolve()/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    rows=json.loads((BASE/'w46/delivery'/case/'sources.json').read_text())
    x=next(x for x in rows if Path(x['source_path']).name=='trajectory.npz')
    raw=args.source.read_bytes()
    assert len(raw)==x['size'] and hashlib.sha256(raw).hexdigest()==x['sha256']
    url=f'https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w46-{case}-part01/'+x['archive']
    with np.load(args.source) as z:
        t=z['time'];q=z['qpos'];v=z['qvel'];rpy=z['rpy'];u=z['q_target'];tau=z['actuator_force'];f=z['feet'];F=z['actual_grf'];contact=z['contact'];n=len(t)
        moving=t>=2.002-1e-7
        du=np.max(np.abs(np.diff(u,axis=0)),axis=1);dtau=np.max(np.abs(np.diff(tau,axis=0)),axis=1)
        margin=.015811388300841896
        inside=(f[:,:,0]>=.8+margin)&(f[:,:,0]<=2.2-margin)&(np.abs(f[:,:,1])<=.6-margin)
        loaded=inside&(F[:,:,2]>0)
        gi=np.unravel_index(np.argmax(f[:,:,0]),f[:,:,0].shape)
        result={'case':case,'snapshot':'15ee620b8a909246a12baf999c136c2be299655b',
          'method':'Existing local trajectory read and size/SHA checked; no network, simulator, training or raw-file copy.',
          'source':{**x,'url':url,'raw_file_saved':False},
          'nodes':n,'all_numeric_finite':all(np.isfinite(z[k]).all() for k in z.files if np.issubdtype(z[k].dtype,np.number)),
          'simulation_s':float(t[-1]),'final_xyz':q[-1,:3].tolist(),'final_rpy':rpy[-1].tolist(),
          'body_x_max_m':float(q[:,0].max()),'body_y_absolute_max_m':float(np.abs(q[:,1]).max()),
          'min_z':{'value_m':float(q[:,2].min()),'time_s':float(t[np.argmin(q[:,2])])},
          'first_body_strip_entry_s':first(t,q[:,0]>=.8),'first_abs_roll_above_0_5_s':first(t,np.abs(rpy[:,0])>.5),
          'first_abs_pitch_above_0_5_s':first(t,np.abs(rpy[:,1])>.5),
          'first_z_below_0_2_s':first(t,q[:,2]<.2),
          'first_no_contact_after_stand_s':first(t,moving&(contact.sum(1)==0)),
          'first_full_foot_loaded_500Hz_s':first(t,loaded.any(1)),
          'loaded_nodes_by_leg':loaded.sum(0).tolist(),
          'max_foot_x':{'value_m':float(f[gi[0],gi[1],0]),'time_s':float(t[gi[0]]),'leg_index':int(gi[1]),'body_xyz':q[gi[0],:3].tolist(),'foot_xyz':f[gi[0],gi[1]].tolist()},
          'target_jump':{'max_rad':float(du.max()),'time_s':float(t[1:][du.argmax()]),'p50_p90_p99_rad':np.quantile(du,[.5,.9,.99]).tolist(),'count_gt_0_5':int((du>.5).sum()),'count_gt_1':int((du>1).sum())},
          'torque_jump_500Hz':{'max_Nm':float(dtau.max()),'time_s':float(t[1:][dtau.argmax()]),'count_ge_90_85':int((dtau>=90.85).sum())},
          'samples_at_requested_times':[],
          'limits':['0.5rad tilt and0.2m height are descriptive, not extra physical failure thresholds','500Hz force timestamp is preceding held integration; material5000Hz is required for first substep','Short flight is legal','No claim that one early jump is the unique cause of final failure']}
        for tt in [2,2.5,3,4,5,6,7,8,8.5,9,9.5,10,10.5,11,11.5,12,15,20,25,26.4]:
            if tt>t[-1]+.001:continue
            k=int(np.argmin(abs(t-tt)));result['samples_at_requested_times'].append({'time_s':float(t[k]),'xyz':q[k,:3].tolist(),'rpy':rpy[k].tolist(),'v_xyz':v[k,:3].tolist(),'contacts':contact[k].tolist(),'material_foot_inside_loaded':loaded[k].tolist()})
        if loaded.any():
            k,j=np.argwhere(loaded)[0]
            result['first_full_foot_loaded_state']={'time_s':float(t[k]),'leg':int(j),'body_xyz':q[k,:3].tolist(),'rpy':rpy[k].tolist(),'foot_xyz':f[k,j].tolist(),'Fz':float(F[k,j,2])}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
