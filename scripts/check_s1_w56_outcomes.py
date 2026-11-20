"""Independently recompute legal passages and terminal holds on the D57 r55 originals."""

import numpy as np
from record_s1_w56 import BASE,LARGE,WINDOW,read,save,stage_info


def summaries(raw,result):
    with np.load(raw/'trajectory.npz') as z:
        t,phase,xy,feet,force=(z[k] for k in ('time','phase','qpos','feet','actual_grf'))
        vel,rpy,contact=(z[k] for k in ('qvel','rpy','contact'))
        foot_vel=z['feet_velocity']
    cuts=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1,len(t)]
    passages=[]; lap_times=[]; by_phase=[]
    for start,end in zip(cuts,cuts[1:]):
        code=int(phase[start])
        if start and phase[start-1] in (4,7) and code in (5,8):
            lap_times.append(float(t[start]))
        s=slice(start,end)
        bearing=force[s,:,2]>0
        slip=np.linalg.norm(foot_vel[s,:,:2],axis=2)
        by_phase.append(dict(phase_code=code,start_s=float(t[start]),end_s=float(t[end-1]),
            min_z_m=float(xy[s,2].min()),max_abs_tilt_rad=float(abs(rpy[s,:2]).max()),
            max_speed_m_s=float(np.linalg.norm(vel[s,:3],axis=1).max()),
            load_weighted_xy_slip_speed_m_s=float((slip*np.where(bearing,force[s,:,2],0.)).sum()/force[s,:,2].clip(0).sum())
                if np.any(bearing) else None))
        if code not in (1,3):
            continue
        seg=xy[start:end]
        enter=np.flatnonzero(seg[:,0]>=.8 if code==1 else seg[:,0]<=2.2)
        en=start+int(enter[0]) if len(enter) else end
        ex=np.flatnonzero(xy[en:end,0]>=2.2 if code==1 else xy[en:end,0]<=.8)
        last=en+int(ex[0]) if len(ex) else end-1
        subset=slice(en,last+1)
        f=feet[subset];h=np.sqrt(.001)/2
        inside=(f[:,:,0]>=.8+h)&(f[:,:,0]<=2.2-h)&(abs(f[:,:,1])<=.6-h)
        loaded=bool(np.any(inside&(force[subset,:,2]>0)))
        side=np.flatnonzero(abs(xy[subset,1])>.6)
        passages.append(dict(direction='outbound' if code==1 else 'return',entry_s=float(t[en]) if len(enter) else None,
            exit_s=float(t[last]) if len(ex) else None,material_loaded=loaded,lateral_exit_s=float(t[en+side[0]]) if len(side) else None,
            completed=bool(len(ex) and not len(side) and loaded)))
    first=np.flatnonzero(phase==5)
    hold=None
    if len(first):
        tail=slice(first[0],len(t))
        good=(contact[tail].all(axis=1)&(np.linalg.norm(xy[tail,:2],axis=1)<.15)
             &(np.linalg.norm(vel[tail,:3],axis=1)<.05)&(np.linalg.norm(vel[tail,3:6],axis=1)<.1)
             &(abs(rpy[tail,:2]).max(axis=1)<.15))
        hold=dict(start_s=float(t[first[0]]),duration_s=float(t[-1]-t[first[0]]),
                  violations=int(np.count_nonzero(~good)),max_origin_distance_m=float(np.linalg.norm(xy[tail,:2],axis=1).max()),
                  max_linear_speed_m_s=float(np.linalg.norm(vel[tail,:3],axis=1).max()))
    official=result['legal_route']['passages']
    agreement=len(official)==len(passages) and all(all(
        (abs(p[k]-r[k])<1e-7 if isinstance(p[k],float) and isinstance(r[k],float) else p[k]==r[k])
        for k in p) for p,r in zip(passages,official))
    valid_roundtrips=sum(i<len(lap_times) and a['completed'] and b['completed'] and a['direction']=='outbound'
        and b['direction']=='return' for i,(a,b) in enumerate(zip(passages[::2],passages[1::2])))
    return dict(samples=len(t),physical_end_s=float(t[-1]),passages=passages,
        passages_agree_with_evaluator=agreement,complete_legal_roundtrips=int(valid_roundtrips),
        lap_completion_times_s=lap_times,hold_after_completion=hold,phase_metrics=by_phase,
        min_height_m=float(xy[:,2].min()),max_abs_roll_pitch_rad=float(abs(rpy[:,:2]).max()),
        evaluator_stage=result['stage_status'],evaluator_reason=result['reason'],peak_strain=result['material']['max_abs_principal_strain'])


def check():
    for case in ('A','B'):
        for level in ('E300','E600'):
            target=WINDOW/'baseline'/f'{case}-{level}.json'
            if target.exists(): continue
            row=stage_info(f'F-{case}-r55',BASE,level)
            if not row.get('source'): continue
            raw=BASE/LARGE/row['attempt']
            if not (raw/'trajectory.npz').is_file():continue
            result=read(BASE/row['source'])
            summary=summaries(raw,result)
            summary.update(attempt=row['attempt'],source_original=str(raw/'trajectory.npz'),
               target_laps=result['evaluation_stage']['target_circuits'],route_deadline_s=result['evaluation_stage']['route_deadline_s'],
               observation_duration_s=result['evaluation_stage']['duration_s'],official_success=result['success'])
            save(target,summary)
            print(row['attempt'],summary['complete_legal_roundtrips'],summary['passages_agree_with_evaluator'],flush=True)


if __name__=='__main__':
    check()
