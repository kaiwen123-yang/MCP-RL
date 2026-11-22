"""Resolve each archived candidate at the natural committed-apex to touchdown time scale."""
from collections import Counter,defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path
import resource
import time
import numpy as np

from analyze_s1_w57_gait import ROOT,R12,REL,REV,LEGS,quant


def main():
    wall=time.perf_counter();out=R12/'w57/p'
    combined=[];cells=defaultdict(Counter);percase={}
    for rev,origin in REV.items():
        for side in ('a','b'):
            label=f'f-{side}-{rev}-e600';base='full-force-off-'+label
            group=json.loads((out/(base+'.json')).read_text())
            for r in group['phase_surface_leg_cells']:
                key=(r['leg'],r['actual_surface'],r['phase_group'])
                cells[key].update({k:v for k,v in r.items() if k not in ('leg','actual_surface','phase_group')})
            root=ROOT.parent/origin/'outputs'/REL/group['source_attempt']
            with np.load(root/'trajectory.npz') as z:
                trfoot=z['feet']
            all_events=[json.loads(l) for l in (root/'w55-f/contact-events.jsonl').open()]
            ev={e['event']:e for e in all_events}
            search={(e['legs'][0],round(e['planned_end_s']/.002)):e for e in all_events if e['kind']=='touchdown_search'}
            cols=json.loads((ROOT/group['original_result']).read_text())['physical_columns']
            @lru_cache(maxsize=2)
            def physical(chunk):
                with np.load(root/f'physical-{chunk:03d}.npz') as z:
                    a=z['physical'];return a[:,[0]+[cols.index(f'grf_{l}_z') for l in LEGS]+
                                             [cols.index(f'foot_velocity_{l}_z_m_s') for l in LEGS]+
                                             [cols.index(f'foot_position_{l}_z_m') for l in LEGS]].copy()
            @lru_cache(maxsize=2)
            def wbc(i):
                with np.load(root/'w55-f'/f'control-{i+1:05d}.npz') as z:
                    return {k:z[k] for k in ('event_mode','confirmed_contact','load_bearing_contact','planned_contact',
                                               'feet_position_reference','mpc_force','force','measured_grf')}
            def measured(s,e):
                parts=[]
                for i in range(s//150000,e//150000+1):
                    block=physical(i);parts.append(block[max(0,s-i*150000):e-i*150000+1])
                return np.concatenate(parts)
            source=[json.loads(l) for l in gzip.open(out/(base+'.jsonl.gz'),'rt')]
            enriched=[]
            for original in sorted(source,key=lambda r:r['loss_s']):
                r={k:original[k] for k in ('leg','phase_group_at_loss','surface_at_loss','loss_s','reload_s',
                    'off_s','apex_bottom_m','classification','unresolved_subtype','preceding_liftoff_id','force_on_s',
                    'prior_pulse_peak_N','preceding_load_s','min_supported_feet')}
                r['case']=label;leg=LEGS.index(r['leg'])
                lift=ev.get(r['preceding_liftoff_id'])
                r['episode_class']=r['classification'];r['pattern']='single_preceding_force_pulse'
                if lift:
                    a=int(np.ceil(lift['fixed_apex_s']/.0002-1e-5)); b=round(r['loss_s']/.0002)
                    raw=measured(a,b);on=raw[:,1+leg]>1.
                    if on.any():
                        j=int(np.flatnonzero(on)[0]);seq=raw[j:];on=on[j:]
                        td=float(seq[0,0]);control_ids=range(round(td/.0002)//10,b//10+1)
                        got_bearing=False;was_swing=False
                        for i in control_ids:
                            z=wbc(i//500);kk=i%500
                            got_bearing|=bool(z['confirmed_contact'][kk,leg] and z['load_bearing_contact'][kk,leg])
                            was_swing|=z['event_mode'][kk,leg]=='swing'
                        changes=np.r_[0,np.flatnonzero(on[1:]!=on[:-1])+1,len(on)]
                        gaps=[(stop-start)*.0002 for start,stop in zip(changes[:-1],changes[1:]) if not on[start] and stop<len(on)]
                        r.update(first_impact_after_apex_s=td,actual_vz_impact_m_s=float(seq[0,5+leg]),
                            vz_at_loss_m_s=float(seq[-1,5+leg]),physical_sole_at_first_impact_m=float(seq[0,9+leg]-.022),
                            impact_to_last_force_off_s=float(seq[-1,0]-td),peak_impact_N=float(seq[:,1+leg].max()),
                            impact_force_impulse_Ns=float(seq[:-1,1+leg].sum()*.0002),
                            max_internal_under_1N_gap_s=max(gaps,default=0.),internal_under_1N_gaps=len(gaps),
                            any_confirmed_bearing_in_impact=got_bearing)
                        if (r['episode_class']=='unresolved_or_planned_release' and td<lift['fixed_touchdown_s'] and
                            was_swing and not got_bearing and seq[0,5+leg]<0. and seq[-1,5+leg]>0. and
                            gaps and max(gaps)<=.0002+1e-9):
                            r['episode_class']='early_swing_contact_rebound'
                            r['pattern']='brief_1N_threshold_recrossings_in_same_impact'
                        target=search.get((r['leg'],round(lift['fixed_touchdown_s']/.002)))
                        if target:
                            landing=float(target['landing'][2]);startz=float(trfoot[round(lift['time_s']/.002),leg,2])
                            apex=max(startz,landing)+.03;tau=(td-lift['fixed_apex_s'])/.10
                            if 0<=tau<=1.:
                                def pos(at):
                                    v=(at-lift['fixed_apex_s'])/.10
                                    return apex+(landing-apex)*v**3*(10.+v*(-15.+6.*v))
                                ref=pos(td)
                                vv=(landing-apex)/.10*30.*tau**2*(1.-tau)**2
                                i=int(np.floor((td+1.e-9)/.002));z=wbc(i//500);kk=i%500
                                r.update(committed_landing_sole_m=landing-.022,desired_sole_at_first_impact_m=ref-.022,
                                    actual_minus_desired_z_at_first_impact_m=float(seq[0,9+leg]-ref),
                                    desired_vz_at_first_impact_m_s=vv,
                                    impact_minus_planned_touchdown_s=td-lift['fixed_touchdown_s'],
                                    archived_swing_reference_reconstruction_error_m=(float(pos(i*.002)-z['feet_position_reference'][kk,leg,2])
                                     if z['event_mode'][kk,leg]=='swing' else None),
                                    impact_preceding_control_s=i*.002,impact_preceding_Fplan_N=float(z['mpc_force'][kk,leg,2]),
                                    impact_preceding_FWBC_N=float(z['force'][kk,leg,2]),
                                    impact_preceding_WBC_planned_stance=bool(z['planned_contact'][kk,leg]))
                enriched.append(r)
            outpath=out/f'impact-sequence-{label}.jsonl.gz'
            with gzip.open(outpath,'wt') as stream:
                for r in enriched:stream.write(json.dumps(r,allow_nan=False)+'\n')
            percase[label]=dict(physical_rows=group['physical_rows'],duration_s=group['physical_seconds'],
                                small_force_off=len(enriched),confirmed_bearing_loss=sum(r['episode_class']=='loss_after_confirmed_bearing' for r in enriched),
                                impact_sequence_class=dict(Counter(r['episode_class'] for r in enriched)),
                                first_pulse_class=group['classification_counts'])
            combined.extend(enriched)
    e=[r for r in combined if r['episode_class']=='early_swing_contact_rebound']
    extra=[r for r in e if r['pattern']=='brief_1N_threshold_recrossings_in_same_impact']
    model=[r for r in e if 'actual_minus_desired_z_at_first_impact_m' in r]
    for r in combined:
        cells[r['leg'],r['surface_at_loss'],r['phase_group_at_loss']]['complete_'+r['episode_class']]+=1
    def representative(rows,field):
        return max(rows,key=lambda x:x[field]) if rows else None
    outval=dict(schema='W57_original_E600_full_force_off_census_v1',scope='Four complete original E600 nominal episodes; Fz<=1N interval incidence, not independent replication.',
        physical_rows=sum(x['physical_rows'] for x in percase.values()),per_case=percase,
        total_small_force_off=len(combined),raw_last_pulse_classes=dict(Counter(x['classification'] for x in combined)),
        committed_swing_impact_classes=dict(Counter(x['episode_class'] for x in combined)),
        before_loss_patterns=dict(Counter(x['pattern'] for x in combined)),
        interpretation='Force-off intervals and original criterion unchanged. The 1N threshold can split a single hard-ground impact into fragments. From the committed apex, successive contact pulses separated only by a single force evaluation are kept visible; descending first impact, upward release and unchanged swing WBC determine the whole-interval classification.',
        terminal_and_turn_exposure=[dict(leg=l,surface=s,phase=ph,**v) for (l,s,ph),v in sorted(cells.items())],
        future_plan=dict(available=len(model),actual_minus_desired_at_impact_m=quant([r['actual_minus_desired_z_at_first_impact_m'] for r in model]),
                         desired_bottom_at_impact_m=quant([r['desired_sole_at_first_impact_m'] for r in model]),
                         desired_vz_at_impact_m_s=quant([r['desired_vz_at_first_impact_m_s'] for r in model]),
                         actual_vz_at_impact_m_s=quant([r['actual_vz_impact_m_s'] for r in model]),
                         desired_landing_bottom_m=quant([r['committed_landing_sole_m'] for r in model]),
                         force_on_minus_planned_s=quant([r['impact_minus_planned_touchdown_s'] for r in model]),
                         completed_WBC_targeted_stance=sum(r['impact_preceding_WBC_planned_stance'] for r in model),
                         maximum_reconstruction_vs_archived_WBC_ref_m=max((abs(r['archived_swing_reference_reconstruction_error_m']) for r in model if r['archived_swing_reference_reconstruction_error_m'] is not None),default=None)),
        ambiguous_last_pulse_context=dict(count=len(extra),elapsed_entire_impact_s=quant([r['impact_to_last_force_off_s'] for r in extra]),
                  peak_whole_impact_N=quant([r['peak_impact_N'] for r in extra]),max_internal_gap_s=quant([r['max_internal_under_1N_gap_s'] for r in extra])),
        extrema=dict(longest=representative(e,'off_s'),highest=representative(e,'apex_bottom_m'),
                     highest_force=representative(e,'peak_impact_N'),
                     threshold_recrossing=representative(extra,'peak_impact_N')),
        core_s=time.process_time(),wall_s=time.perf_counter()-wall)
    (out/'full-force-off-summary.json').write_text(json.dumps(outval,indent=2,allow_nan=False)+'\n')
    print(outval['committed_swing_impact_classes'],outval['ambiguous_last_pulse_context'],outval['future_plan'],flush=True)


if __name__=='__main__':main()
