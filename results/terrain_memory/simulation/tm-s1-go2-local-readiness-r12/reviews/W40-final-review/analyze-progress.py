#!/usr/bin/env python3
"""Recompute the final W40 compact evidence without running physics or a network.

Run from any directory. Writes progress-audit.json beside this script. Uses the
published W39 audit helpers for exactly the same lifetime/reward/GAE definitions.
Every raw file actually read is recorded separately by the optional raw audit.
"""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import math
import statistics as st
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
BASE = REPO / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
BASELINE = 'c438f5463659ab7c8d3a7f5eced251639e14b2ef'
spec = importlib.util.spec_from_file_location('w39_audit', HERE.parent / 'W39-review/analyze_results.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, summary, aggregate, numeric = old.read, old.summary, old.aggregate, old.numeric


def episode(result, seed, caps, events, transitions, values, resolved, small=None):
    row = old.episode(result, seed, caps, events)
    row.update(episode=transitions[0]['episode'] if transitions else None,
               transitions=len(transitions), reward_sum=sum(x['reward'] for x in transitions),
               current_pann_nonzero_rows=sum(any(x != 0 for x in v['pann_recovery_m']) for v in values),
               controller_artifact=resolved['r12_configuration'].get('controller_artifact',
                    resolved['r12_configuration'].get('controller_build')),
               nominal_workers=resolved['workers'], execution_cpus=resolved.get('execution_cpus'),
               controller_identity=resolved['r12_configuration'].get('controller_build'))
    row['first_nonzero_current_pann_action_start_s'] = next((t['time_s']-t['dt_s'] for t,v in zip(transitions, values)
             if any(x != 0 for x in v['pann_recovery_m'])), None)
    if transitions:
        row['terminal_transition'] = transitions[-1]
        row['terminal_value'] = values[-1]
    if small:
        row['small_episode'] = {k:v for k,v in small.items() if not isinstance(v,(dict,list))}
    if result.get('r29_failure'):
        failure=result['r29_failure']; q=failure['query']
        row['technical_failure']={k:q.get(k) for k in [
            'plan_id','query_id','projection_status','qp_solver_revision','qp_native_converged',
            'qp_native_kkt_scaled_residual','qp_refinement_attempted','qp_refinement_iterations',
            'qp_refinement_status','primal_residual_rad','kkt_scaled_residual']}
        row['technical_failure'].update(time_s=failure['time_s'],physics_advanced=failure['physics_advanced'])
    return row


def exposure_counts(transitions, values, exposure):
    """Match current actor observations; omit terminal successor-only queries."""
    queries=exposure.get('queries',[])
    counts=Counter(); mismatches=[]; rows=[]
    for t,v in zip(transitions,values):
        start=t['time_s']-t['dt_s']
        matches=[q for q in queries if abs(q['time_s']-start)<1e-7]
        if len(matches)!=1:
            mismatches.append({'serial':t['transition_serial'],'start_s':start,'query_matches':len(matches)})
            continue
        q=matches[0]; nonzero=any(x != 0 for x in v['pann_recovery_m'])
        if nonzero:
            category='nonzero_pann'
        elif q['all_outside']:
            category='all_queries_outside_strip'
        elif not any(inside and hist for inside,hist in zip(q['inside'],q['previous_true_load'])):
            category='inside_without_prior_true_load'
        elif any(inside and pulse for inside,pulse in zip(q['inside'],q['overlaps_prior_unobserved_pulse'])):
            category='zero_with_prior_unobserved_pulse'
        else:
            category='zero_despite_prior_true_load_not_explained_by_recorded_pulse'
        counts[category]+=1
        if nonzero!=any(q['feature_nonzero']):
            mismatches.append({'serial':t['transition_serial'],'feature_mismatch':True})
        if category.startswith('zero_'):
            rows.append({'serial':t['transition_serial'],'episode':t['episode'],'start_s':start,'query':q})
    return {'classification':dict(counts),'matched_actor_current_queries':sum(counts.values()),
            'all_recorded_queries':len(queries),'extra_successor_queries':len(queries)-sum(counts.values()),
            'mismatches':mismatches,'zero_history_cases':rows,
            'recorded_unobserved_pulses':len(exposure.get('missed_endpoint_pulses',[]))}


def build_stream(suffix,seed):
    jobid=f'job-20261106T103028Z-w40-{suffix}'
    root=BASE/'jobs'/jobid
    job=read(root/'job.json'); collection=read(root/'training/collection.json')
    rootcollection=read(root/'collection.json'); progress=read(root/'training/progress.json')
    stopped=read(root/'training/stopped.json'); learning=read(root/'training/learning.json')
    evaluation=read(root/'evaluation.json'); ledger=read(root/'attempt-ledger.json')
    index=read(root/'numeric-evidence/index.json'); contract=read(root/'execution-contract.json')
    read(root/'controller-identity.json'); read(root/'training/stream-identity.json')
    rows=[]; alltrans=[]; allvals=[]; checks=[]; exposures={}; ppoconfigs=[]
    start=285 if suffix=='a' else 281
    previous=8593
    for ep in range(start,collection['episode']+1):
        attempt=f'train-{jobid}-PANN_PPO_S5-s{seed}-ep{ep:05d}'
        result=read(BASE/(attempt+'.json')); resolved=read(BASE/(attempt+'-resolved.json'))
        ppoconfigs.append(resolved['r12_configuration']['r12_contract']['joint_learning']['ppo'])
        small=read(root/'training'/f'episode-{ep:05d}.json')
        caps=read(root/'episodes'/attempt/'cap-decisions.json')
        events=read(root/'numeric-evidence/episodes'/(attempt+'-events.json'))
        transitions=read(root/'numeric-evidence/training'/f'episode-{ep:05d}-transitions.jsonl')
        values=read(root/'numeric-evidence/training'/f'episode-{ep:05d}-pann-values.jsonl')
        exposure=read(root/'numeric-evidence/exposure'/attempt/'query-exposure-offline.json')
        row=episode(result,seed,caps,events,transitions,values,resolved,small)
        row['episode']=ep
        exposures[attempt]=exposure_counts(transitions,values,exposure)
        row['exposure']=exposures[attempt]
        delta=result['policy']['total_transitions']-previous
        checks.append({'episode':ep,'counter_delta':delta,'episode_transitions':small['transitions'],
              'caps':len(caps),'rows':len(transitions),'values':len(values),
              'counts_match':delta==small['transitions']==len(caps)==len(transitions)==len(values)})
        previous=result['policy']['total_transitions']
        rows.append(row); alltrans.extend(transitions); allvals.extend(values)
    batches=[]
    for update in learning['updates']:
        bt=[t for t in alltrans if t['behavior_updates']==update['update_count']-1]
        bv=[v for v in allvals if v['behavior_updates']==update['update_count']-1]
        batches.append({'update':update,'numeric':numeric(bt,bv),
          'gae_arithmetic':old.gae_arithmetic(bt,bv,update,ppoconfigs[0]['gamma'],ppoconfigs[0]['gae_lambda'])})
    evaluations=[]
    for attempt in evaluation['attempts']:
        result=read(BASE/(attempt+'.json')); resolved=read(BASE/(attempt+'-resolved.json'))
        caps=read(root/'episodes'/attempt/'cap-decisions.json')
        events=read(root/'numeric-evidence/episodes'/(attempt+'-events.json'))
        evdir=root/'numeric-evidence/evaluation'/attempt
        trans=[r for p in sorted(evdir.glob('*-transitions.jsonl')) for r in read(p)]
        vals=[r for p in sorted(evdir.glob('*-pann-values.jsonl')) for r in read(p)]
        row=episode(result,seed,caps,events,trans,vals,resolved)
        row['numeric']=numeric(trans,vals) if trans and vals else None
        exp=root/'numeric-evidence/exposure'/attempt/'query-exposure-offline.json'
        if exp.exists():row['exposure']=exposure_counts(trans,vals,read(exp))
        dep=root/'episodes'/attempt/'pann-action-dependence.json'
        if dep.exists():
            ds=read(dep);dif=[abs(x['with_pann_cap']-x['zero_feature_cap']) for x in ds]
            row['pann_action_dependence']={'rows':len(ds),'nonzero_differences':sum(x!=0 for x in dif),
                'max_abs_cap_difference_m_s':max(dif) if dif else None}
        evaluations.append(row)
    natural=[x for x in rows if x['natural_physical_terminal']]
    half=len(natural)//2
    expcount=Counter()
    for e in exposures.values():expcount.update(e['classification'])
    return {'job':jobid,'job_metadata':job,'collection':collection,'progress_checkpoint':progress['checkpoint'],
      'training_end':{k:v for k,v in stopped.items() if k in ['reason','last_stage_status','last_episode_reason',
          'last_error','phase_stop','stopped_unix_s','deadline_unix_s','initial_transitions','new_transitions',
          'new_updates','phase_new_transitions','phase_new_updates','continuous_environment_snapshot']},
      'root_collection_matches_training':all(rootcollection.get(k)==collection.get(k) for k in [
          'collected_transitions','updates','unupdated_fragment','phase_new_transitions','phase_new_updates']),
      'accounting_identity_valid':collection['collected_transitions']==collection['updates']*256+
          collection['unupdated_fragment']+collection['excluded_old_version_transitions'],
      'phase_sample_serials_contiguous':sorted(t['transition_serial'] for t in alltrans)==list(range(8594,collection['collected_transitions']+1)),
      'summary':aggregate(rows),'numeric':numeric(alltrans,allvals),'episodes':rows,
      'exposure_classification':dict(expcount),'exposure_mismatches':[m for e in exposures.values() for m in e['mismatches']],
      'updates':batches,'coverage_checks':checks,'all_sample_counts_match':all(x['counts_match'] for x in checks),
      'temporal_comparison':{'basis':'First and last equal-size halves of natural physical/material terminal episodes; technical and budget interruptions excluded.',
          'first_half':aggregate(natural[:half]),'last_half':aggregate(natural[-half:]),
          'omitted_middle_attempt':natural[half]['attempt'] if len(natural)%2 else None},
      'resolved_ppo_config':ppoconfigs[0],'all_resolved_ppo_configs_identical':all(x==ppoconfigs[0] for x in ppoconfigs),
      'learning_malformed_lines':learning['malformed_lines'],'evaluation_metadata':evaluation,'evaluations':evaluations,
      'source_commit_counts':dict(Counter(x['source_commit'] for x in rows)),
      'execution_fingerprints':dict(Counter(x['execution_fingerprint'] for x in rows)),
      'all_new_update_sample_sizes_256':all(b['numeric']['rows']==256 for b in batches),
      'numeric_index_training':{k:v for k,v in index['training'].items() if k!='files'}}


def main():
    streams={suffix:build_stream(suffix,seed) for suffix,seed in [('a',1),('b',2)]}
    rows=[r for s in streams.values() for r in s['episodes']]
    new=sum(s['collection']['phase_new_transitions'] for s in streams.values())
    start=min(s['job_metadata']['started_unix_s'] for s in streams.values())
    end=max(s['collection']['updated_unix_s'] for s in streams.values())
    raw=read(BASE/'raw-bundles/w40-20261107/manifest.json')
    rawfiles=[f for a in raw['archives'] for f in a['files']]
    oldreport=read(HERE.parent/'W39-review/results-audit.json')
    expcount=Counter()
    for s in streams.values():expcount.update(s['exposure_classification'])
    report={'schema':'W40_final_progress_audit_v1','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
       'review_baseline':BASELINE,'method':'Recomputed all 64 W40 training episodes and B E30, compact transitions and values, reward/GAE arithmetic, actor-time query exposure and job/update accounting. No new physics or neural-network forward pass.',
       'streams':streams,'combined':aggregate(rows),
       'failure_families':{family:aggregate([x for x in rows if x['reason_family']==family])
           for family in sorted(set(x['reason_family'] for x in rows))},
       'numeric_combined':{'rows':new,'trained_this_phase':sum(len(s['updates'])*256 for s in streams.values()),
          'unupdated_tail':sum(s['collection']['unupdated_fragment'] for s in streams.values()),
          'pann_current_nonzero_rows':sum(s['numeric']['pann_current_nonzero_rows'] for s in streams.values()),
          'pann_current_nonzero_fraction':sum(s['numeric']['pann_current_nonzero_rows'] for s in streams.values())/new,
          'reward_sum':sum(s['numeric']['reward_sum'] for s in streams.values()),
          'reward_formula_max_abs_error':max(s['numeric']['reward_formula_check']['max_absolute_error'] for s in streams.values()),
          'exposure_classification':dict(expcount)},
       'loaded_but_all_current_actor_pann_zero_episodes':sum(x['material_loaded'] and x['current_pann_nonzero_rows']==0 for x in rows),
       'natural_loaded_terminated_within_half_second':sum(x['natural_physical_terminal'] and x['time_first_load_to_terminal_s'] is not None and x['time_first_load_to_terminal_s']<=.5 for x in rows),
       'unique_tree_lineage':{'common_ancestor_samples':6545,'common_ancestor_updates':25,
          'unique_samples_after_W39':10641,'unique_samples_after_W40':10641+new,
          'unique_trained_samples_after_W40':10496+sum(len(s['updates'])*256 for s in streams.values()),
          'unique_historical_excluded':145,'not_one_combined_policy':True},
       'throughput':{'basis':'All unique phase samples / from common job start to final B training collection, includes repair pause, excludes B E30 and publication.',
          'phase_samples':new,'shared_elapsed_s':end-start,'shared_elapsed_h':(end-start)/3600,
          'transitions_per_h':new*3600/(end-start),'confounders':'Controller revision, resumed learning, qp1, 12 workers versus W39 8, GPU contention, unequal termination; not controlled speedup.'},
       'W39_comparison':{'combined':oldreport['W39']['combined'],'numeric_combined':oldreport['W39']['numeric_combined'],
          'throughput':oldreport['W39']['throughput']},
       'raw_delivery':{'status':raw['status'],'archives':len(raw['archives']),'source_files':len(rawfiles),
          'unique_source_paths':len(set(f['source_path'] for f in rawfiles)),
          'source_bytes':sum(f['source_size'] for f in rawfiles),'archive_bytes':sum(a['size'] for a in raw['archives']),
          'attempts':len(set(f['attempt'] for f in rawfiles if f.get('attempt'))),
          'verification':raw['verification'],'verified_archives':raw['verified_archives'],
          'declared_counts_match':len(rawfiles)==raw['source_files'] and sum(f['source_size'] for f in rawfiles)==raw['source_bytes'],
          'note':'Manifest delivery is not a claim that this audit downloaded every archive. Actual selected raw reads are in raw-progress-audit.json.'},
       'limitations':['W39 and W40 differ in controller, continued policy and worker allocation; descriptive comparison is not isolated causal evidence.',
          'Three W40 execution_error records are technical censored episodes, not body_fall nor successful survival; A final trained update36 has no E30 result.',
          'State at a technical stop is not a physically observed future; do not declare A314 inevitably falls.',
          'PANN current actor queries, next-state queries and simulator material load events are different denominators.',
          'Current feature-clearing action dependence is not a retrained causal ablation.',
          'No fixed-wall deadline was reached in either final training stop; removing deadlines will not repair a QP solver defect.',
          'All sampled transitions retain independent stream identity; do not merge optimizer histories or sum updates as one policy.']}
    report['coverage']={'files':len(old.READ),'bytes':sum(x['bytes'] for x in old.READ.values()),'read_files':dict(sorted(old.READ.items()))}
    output=HERE/'progress-audit.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'combined':report['combined'],'numeric_combined':report['numeric_combined'],
       'throughput':report['throughput'],'coverage':{k:v for k,v in report['coverage'].items() if k!='read_files'}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
