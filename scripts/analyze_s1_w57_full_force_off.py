"""All 5kHz E600 force-off intervals, joined to the original 500Hz gait and WBC."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import resource
import time

import numpy as np

from analyze_s1_w57_gait import ROOT, R12, REL, REV, LEGS, quant

PHASE=('initial_stand','straight','turn','straight','endpoint','endpoint','endpoint','endpoint','turn')
STRATA=('rigid','material')


def native(obj):
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,np.ndarray):return obj.tolist()
    raise TypeError(type(obj).__name__)


def run(side,revision):
    clock=time.perf_counter();name=f'f-{side}-{revision}-e600'
    raw=next((ROOT.parent/REV[revision]/'outputs'/REL).glob(f'*-{side}-{revision}-*-E600-attempt01'))
    result=next(e['source'] for e in json.loads((R12/'w56/phase.json').read_text())['cases'] if e.get('attempt')==raw.name)
    meta=json.loads((ROOT/result).read_text());cols=meta['physical_columns'];total=meta['physical_samples']
    cdir=raw/'w55-f';cfg=json.loads((cdir/'manifest.json').read_text())['config']
    with np.load(raw/'trajectory.npz') as z:phase=z['phase']
    ticks=np.empty(total);fz=np.empty((total,4));pos=np.empty((total,4,3));vz=np.empty((total,4))
    touch=np.empty((total,4),bool)
    zs=[cols.index(f'grf_{l}_z') for l in LEGS]
    ps=[cols.index(f'foot_position_{l}_{a}_m') for l in LEGS for a in 'xyz']
    vs=[cols.index(f'foot_velocity_{l}_z_m_s') for l in LEGS]
    cs=[cols.index('contact_'+l) for l in LEGS]
    n=0;physical_files=[]
    for file in sorted(raw.glob('physical-*.npz')):
        with np.load(file) as z:
            a=z['physical'];slot=slice(n,n+len(a))
            ticks[slot]=a[:,0];fz[slot]=a[:,zs];pos[slot]=a[:,ps].reshape(-1,4,3)
            vz[slot]=a[:,vs];touch[slot]=a[:,cs]>0
            physical_files.append(dict(file=file.name,n=len(a),start_s=float(a[0,0]),end_s=float(a[-1,1])))
            n+=len(a)
    if n!=total or np.max(np.abs(np.diff(ticks)-.0002))>1e-7:raise ValueError('Incomplete physical clock')
    nctrl=len(phase);keys=('event_mode','confirmed_contact','load_bearing_contact','planned_contact','mode',
          'measured_grf','mpc_force','redistributed_force_target','force','transfer_fraction',
          'feet_acceleration_reference','foot_center_acceleration_achieved','tau_total','requested_pd',
          'feet_position_reference','contact_confirmation_count')
    control={};nrow=0
    for file in sorted(cdir.glob('control-*.npz')):
        with np.load(file) as z:
            n=len(z['time']);slot=slice(nrow,nrow+n)
            for k in keys:
                x=z[k]
                if k in ('measured_grf','mpc_force','redistributed_force_target','force','feet_acceleration_reference',
                         'foot_center_acceleration_achieved','feet_position_reference'):x=x[:,:,2]
                if k not in control:control[k]=np.empty((nctrl,*x.shape[1:]),x.dtype)
                control[k][slot]=x
            nrow+=n
    def frame(k,leg):
        if k<0 or k>=nrow:return None
        zt=(control['tau_total'][k,3*leg:3*leg+3]).tolist()
        return dict(t_s=float(k*.002),event_mode=str(control['event_mode'][k,leg]),
            wbc_mode=int(control['mode'][k,leg]),confirmed=bool(control['confirmed_contact'][k,leg]),
            bearing=bool(control['load_bearing_contact'][k,leg]),
            planned_stance=bool(control['planned_contact'][k,leg]),
            observed_Fz_N=float(control['measured_grf'][k,leg]),mpc_Fz_N=float(control['mpc_force'][k,leg]),
            redistributed_Fz_N=float(control['redistributed_force_target'][k,leg]),WBC_Fz_N=float(control['force'][k,leg]),
            transfer=float(control['transfer_fraction'][k]),confirmation_ticks=int(control['contact_confirmation_count'][k,leg]),
            requested_acceleration_z_m_s2=float(control['feet_acceleration_reference'][k,leg]),
            modeled_achieved_center_az_m_s2=float(control['foot_center_acceleration_achieved'][k,leg]),
            reference_center_z_m=float(control['feet_position_reference'][k,leg]),applied_hold_torque_Nm=zt)
    ev=[json.loads(l) for l in (cdir/'contact-events.jsonl').open()]
    masks=(fz>1.)
    half=np.sqrt(.001)/2
    code=np.array([{'initial_stand':0,'straight':1,'turn':2,'endpoint':3}[PHASE[i]] for i in range(len(PHASE))])
    group=code[phase[np.arange(total)//10]]
    rigid_or_soft=((pos[:,:,0]>=.8+half)&(pos[:,:,0]<=2.2-half)&(np.abs(pos[:,:,1])<=.6-half)).astype(int)
    phases=('initial_stand','straight','turn','endpoint')
    cases={};allrows=[];groups=defaultdict(lambda:Counter());tags=Counter();pulses=Counter()
    for leg,legname in enumerate(LEGS):
        region=rigid_or_soft[:,leg]
        loaded=masks[:,leg]
        lifted=[e for e in ev if e['kind']=='liftoff_committed' and legname in e['legs']]
        lift_times=np.array([e['time_s'] for e in lifted])
        confirmed=[e for e in ev if e['kind']=='actual_touchdown' and legname in e['legs']]
        confirm_times=np.array([e['time_s'] for e in confirmed])
        for ph,phase_name in enumerate(phases):
            for terrain,terrain_name in enumerate(STRATA):
                m=(group==ph)&(region==terrain);g=groups[legname,terrain_name,phase_name]
                g['physical_force_evaluations']=int(m.sum());g['bearing_physical_samples']=int(np.count_nonzero(m & loaded))
                g['observation_s']=g['physical_force_evaluations']*.0002
        cut=np.r_[0,np.flatnonzero(loaded[1:]!=loaded[:-1])+1,total]
        for j,(a,b) in enumerate(zip(cut[:-1],cut[1:])):
            if loaded[a]:continue
            ph=phases[group[a]];surface=STRATA[region[a]];g=groups[legname,surface,ph]
            if a==0 or b==total:
                g['left_or_right_censored_off_intervals']+=1;continue
            duration=(b-a)*.0002
            g['all_closed_force_off_intervals']+=1
            if duration<.04-1e-10:continue
            bottom=pos[a:b,leg,2]-.022;apex=float(bottom.max())
            if duration>.05+1e-10 and apex>.005:
                g['qualified_high_swings']+=1;continue
            # W56's duration/height discriminator is retained in each phase and substrate.
            # Cross-phase starts are not mixed into the within-phase counts.
            crossed=bool(np.any(group[a:b]!=group[a]));terrain_cross=bool(np.any(region[a:b]!=region[a]))
            cross_original=bool(np.any(phase[np.arange(a,b)//10] != phase[a//10]))
            g['small_force_off_intervals']+=1
            g['same_original_phase_small']+=int(not cross_original);g['same_substrate_small']+=int(not terrain_cross)
            p=cut[j-1]  # beginning of the positive force pulse preceding this loss
            loidx=int(np.searchsorted(lift_times,ticks[p]+1.e-8)-1)
            event=lifted[loidx] if loidx>=0 else None
            c0=max(0,p//10);c1=min(nrow,(a//10)+1);last=max(0,(a-1)//10)
            ids=np.arange(c0,c1);mode=control['event_mode'][ids,leg]
            capable=control['confirmed_contact'][ids,leg] & control['load_bearing_contact'][ids,leg] & control['planned_contact'][ids,leg]
            prior_bear=ids[capable]
            last_bearing=int(prior_bear[-1]) if len(prior_bear) else None
            was_swing=bool(np.any(mode=='swing'))
            early=bool(event is not None and ticks[p] < event['fixed_touchdown_s']-1.e-9 and was_swing)
            first_confirm_id=int(np.searchsorted(confirm_times,ticks[p]-.0002))
            did_confirm=first_confirm_id<len(confirmed) and confirm_times[first_confirm_id]<ticks[a]+1e-9
            actual_reversal=bool(vz[p,leg]<0. and vz[a,leg]>0.)
            # A planned liftoff following the last confirmed bearing is an intentional release.
            intervening_lift=False if last_bearing is None else bool(
                np.searchsorted(lift_times,ticks[a]+1e-8) > np.searchsorted(lift_times,.002*last_bearing+1e-8))
            if early and not len(prior_bear) and actual_reversal:
                kind='early_swing_contact_rebound'
            elif len(prior_bear) and not intervening_lift and control['planned_contact'][last,leg]:
                kind='loss_after_confirmed_bearing'
            else:
                kind='unresolved_or_planned_release'
            if kind=='unresolved_or_planned_release':
                detail=('committed_release_after_bearing' if intervening_lift else
                        'no_upward_reversal_at_force_off' if early and not actual_reversal else
                        'unconfirmed_stance_or_unidentified_previous_swing')
            else:detail=None
            g[kind]+=1;tags[kind]+=1
            cfirst=min(p//10,nrow-1);clast=min(a//10,nrow-1)
            record=dict(leg=legname,surface_at_loss=surface,phase_group_at_loss=ph,original_phase=int(phase[a//10]),
                crossed_original_phase=cross_original,crossed_phase_group=crossed,crossed_surface=terrain_cross,
                classification=kind,unresolved_subtype=detail,force_on_s=float(ticks[p]),loss_s=float(ticks[a]),
                reload_s=float(ticks[b]),off_s=duration,preceding_load_s=(a-p)*.0002,apex_bottom_m=apex,
                force_on_minus_planned_s=float(ticks[p]-event['fixed_touchdown_s']) if event else None,
                preceding_liftoff_id=event['event'] if event else None,
                preceding_liftoff_s=event['time_s'] if event else None,
                planned_touchdown_s=event['fixed_touchdown_s'] if event else None,
                physical_vz_at_on_and_off_m_s=[float(vz[p,leg]),float(vz[a,leg])],
                prior_pulse_peak_N=float(fz[p:a,leg].max()),prior_pulse_impulse_Ns=float(fz[p:a,leg].sum()*.0002),
                logged_contact_fraction_while_off=float(touch[a:b,leg].mean()),
                max_force_while_off_N=float(fz[a:b,leg].max()),min_supported_feet=int(masks[a:b].sum(1).min()),
                center_xy_displacement_while_off_m=float(np.linalg.norm(pos[b,leg,:2]-pos[a,leg,:2])),
                preceding_WBC_bearing_ticks=int(len(prior_bear)),prior_pulse_confirmed_event=did_confirm,
                last_bearing_control=frame(last_bearing,leg) if last_bearing is not None else None,
                control_at_force_on=frame(cfirst,leg),control_at_loss=frame(clast,leg))
            allrows.append(record)
    byclass={}
    for tag in tags:
        subset=[x for x in allrows if x['classification']==tag]
        byclass[tag]=dict(count=len(subset),duration_s=quant([v['off_s'] for v in subset]),
            bottom_apex_m=quant([v['apex_bottom_m'] for v in subset]),min_support_feet=min(v['min_supported_feet'] for v in subset),
            longest=max(subset,key=lambda x:x['off_s']),highest=max(subset,key=lambda x:x['apex_bottom_m']),
            highest_preceding_Fz=max(subset,key=lambda x:x['prior_pulse_peak_N']))
    review=json.loads((R12/'reviews/W56-review/evidence-force-dropouts.json').read_text())[name]
    old={(v['leg'],round(v['start_s']/.0002)) for v in review}
    rediscovered=[v for v in allrows if (v['leg'],round(v['loss_s']/.0002)) in old]
    dest=R12/'w57/p'
    with gzip.open(dest/f'full-force-off-{name}.jsonl.gz','wt') as stream:
        for x in allrows:stream.write(json.dumps(x,default=native)+'\n')
    res=dict(source_attempt=raw.name,original_result=result,physical_files=physical_files,
             physical_rows=total,physical_seconds=total*.0002,force_threshold_N=1.,radius_m=.022,
             source_legs=LEGS,old_selected_count=len(old),old_selected_crossmatched=len(rediscovered),
             old_subset_new_classification=dict(Counter(x['classification'] for x in rediscovered)),
             definition='Closed Fz<=1N gap of at least 40ms, excluding high swings when duration exceeds 50ms and sole apex exceeds 5mm. Continuous through all file boundaries; single disconnected initial/final intervals are censored.',
             classification='Confirmed-bearing loss requires a prior positive-force pulse with confirmed load-bearing planned stance and no later committed liftoff before loss. Early rebound requires preceding planned swing, measured impact before committed touchdown, unconfirmed bearing, and downward-to-upward measured vertical velocity. Remaining cases retain their measured mode.',
             denominator='Per-leg 5kHz physical-time exposure and actual load-bearing ticks, by actual sole interior and phase. Counts assigned by loss onset; crossing flags retained. Different original phases included for new stratified census.',
             classification_counts=dict(tags),by_class=byclass,phase_surface_leg_cells=[
                 dict(leg=leg,actual_surface=terrain,phase_group=phase_name,**counts) for (leg,terrain,phase_name),counts in sorted(groups.items())],
             peak_resident_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             cpu_s=time.process_time(),wall_s=time.perf_counter()-clock)
    (dest/f'full-force-off-{name}.json').write_text(json.dumps(res,indent=2,allow_nan=False,default=native)+'\n')
    print(name,total,len(allrows),dict(tags),'previous',len(old),len(rediscovered),res['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('side',choices=['a','b']);p.add_argument('revision',choices=REV)
    opt=p.parse_args();run(opt.side,opt.revision)
