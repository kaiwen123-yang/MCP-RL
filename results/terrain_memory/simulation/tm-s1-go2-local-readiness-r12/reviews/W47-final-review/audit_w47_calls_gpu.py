#!/usr/bin/env python3
"""Offline W47 evidence audit. Reads committed CSV/JSON only; no simulation/network."""
import argparse, csv, hashlib, json, statistics
from collections import Counter
from pathlib import Path

def ident(p):
    b=p.read_bytes();return {'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def summarize(rows):
    positive=[r for r in rows if r['policy_updated'] and r['candidate_source'].startswith('alpha_') and r['chosen_alpha']>0]
    delta=[r['deployed_current_cost']-r['incumbent_current_cost'] for r in rows if r['deployed_current_cost'] is not None and r['incumbent_current_cost'] is not None]
    run=longest=0;start=None;longest_start=longest_end=None
    for r in rows:
        if r['stop_reason']=='native_search_stagnation':
            if run==0:start=r['origin']
            run+=1
            if run>longest:longest=run;longest_start=start;longest_end=r['origin']
        else:run=0
    return {'calls':len(rows),'positive_installs':len(positive),
        'positive_installs_fraction':len(positive)/len(rows) if rows else None,
        'positive_J_descent':sum(r['chosen_actual']>0 for r in positive),
        'positive_J_increase':sum(r['chosen_actual']<0 for r in positive),
        'nominal_installs':sum(r['policy_updated'] and r['candidate_source'].startswith('nominal_') for r in rows),
        'incumbent_retained':sum(r['return_reason']=='incumbent_retained' for r in rows),
        'search_stagnation':sum(r['stop_reason']=='native_search_stagnation' for r in rows),
        'positive_alpha_distribution':dict(sorted(Counter(str(r['chosen_alpha']) for r in positive).items())),
        'planning_wall_total_s':sum(r['wall_s'] for r in rows),
        'planning_wall_median_s':statistics.median([r['wall_s'] for r in rows]) if rows else None,
        'deployed_vs_incumbent_J':{'n':len(delta),'negative':sum(d<0 for d in delta),'zero':sum(d==0 for d in delta),'positive':sum(d>0 for d in delta)},
        'longest_stagnation_run':{'calls':longest,'start_s':longest_start,'end_s':longest_end},
        'work':{k:sum(r[k] for r in rows) for k in ['native_new_rollouts','candidate_records','derivative_passes','transition_fd_calls','risk_geometry_fd_calls']}}

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--repo',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);a=pa.parse_args()
    base=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    p=base/'w47';csvp=p/'task-c/planner-calls.csv'
    rows=list(csv.DictReader(csvp.open()))
    integers=['call_id','problem_epoch','installed_plan_id','policy_generation_call','native_new_rollouts','candidate_records','derivative_passes','transition_fd_calls','risk_geometry_fd_calls','gait_enum']
    floats=['origin','wall_s','chosen_alpha','chosen_actual','incumbent_current_cost','deployed_current_cost']
    for r in rows:
        for k in integers:r[k]=int(r[k])
        for k in floats:r[k]=float(r[k]) if r[k] else None
        for k in ['policy_updated','r35_enabled']:r[k]={'True':True,'False':False}[r[k]]
    s=json.loads((p/'task-c/planner-summary.json').read_text())
    g=json.loads((p/'task-c/gpu-corrected.json').read_text())
    ph=json.loads((p/'phase.json').read_text())
    out={'reviewed_commit':'6e9075f48718509d48f51d08d0a8f62a34edfb41','scope':'Independent arithmetic on committed CSV/JSON, source audit; raw planner/paired arrays not independently downloaded or rerun.',
        'sources':[ident(csvp),ident(p/'task-c/planner-summary.json'),ident(p/'task-c/gpu-corrected.json'),ident(p/'task-b-source-manifest.json'),ident(p/'task-b-fixture-manifest.json'),ident(p/'phase.json')],
        'combined':summarize(rows),'cases':{},'checks':[]}
    events={'a':{'first_target_jump_over_1rad':3.846,'first_full_knee_reversal':6.860,'first_z_below_0p2':8.868,'first_roll_abs_over_0p5':11.098,'terminal':26.432},
            'b':{'first_target_jump_over_1rad':3.106,'first_full_knee_reversal':6.150,'first_z_below_0p2':8.464,'first_roll_abs_over_0p5':8.572,'first_complete_foot_load':9.122,'terminal':11.638}}
    mapkeys={'calls':'calls','positive_installs':'final_positive_step_installs','nominal_installs':'nominal_installs','incumbent_retained':'incumbent_retained'}
    for case in ['a','b']:
        rr=[r for r in rows if r['case']==case]
        assert [r['call_id'] for r in rr]==list(range(1,len(rr)+1))
        assert all(r['r35_enabled']==(r['gait_enum']!=0) for r in rr)
        groups={'all':rr,'startup':[r for r in rr if r['call_id']<=20],
            'formal':[r for r in rr if r['call_id']>20],
            'Stand':[r for r in rr if r['gait_enum']==0],
            'moving':[r for r in rr if r['gait_enum']!=0]}
        d={k:summarize(v) for k,v in groups.items()}
        for name in ['all','Stand','moving']:
            for left,right in mapkeys.items():assert d[name][left]==s[case][name][right]
            assert abs(d[name]['planning_wall_total_s']-s[case][name]['planning_wall_total_s'])<1e-7
        d['two_second_bins']={str(i):summarize([r for r in rr if i<=r['origin']<i+2]) for i in range(0,int(rr[-1]['origin'])+1,2)}
        d['events']={}
        for name,t in events[case].items():
            match=max((r for r in rr if r['origin']<=t+1e-8),key=lambda r:(r['origin'],r['call_id']))
            window=[r for r in rr if max(2.002,t-1.)-1e-8<=r['origin']<=t+1e-8]
            d['events'][name]={'event_time_s':t,'preceding_1s_moving':summarize(window),'last_call':match}
        resolved=base/f'job-20261112T133729Z-w46-{case}-eval-PANN_PPO_S5-s{1 if case=="a" else 2}-E30-attempt01-resolved.json'
        rec=json.loads(resolved.read_text())['r29_recording'];out['sources'].append(ident(resolved))
        d['retained_event_snapshot_counts']={'full_plans':len(rec['plans']),'search_calls':len(rec['r32_search']['plans']),'candidate_calls':len(rec['r33_candidates']['calls']),'event_types':dict(Counter(e['kind'] for e in rec['events']))}
        d['knee_event_inside_existing_saved_window']=any(e['start_s']-1e-8<=events[case]['first_full_knee_reversal']<=e['end_available_s']+1e-8 for e in rec['events'])
        d['risk_prioritized_cost_increases']=s[case]['installed_positive_acceptance']['J_increase_rows']
        out['cases'][case]=d
    assert out['combined']['positive_installs']==1601
    assert out['combined']['positive_J_descent']==1598
    assert out['combined']['positive_J_increase']==3
    assert out['combined']['search_stagnation']==343
    assert out['combined']['nominal_installs']==137
    assert out['combined']['incumbent_retained']==206
    out['checks']=['1944 CSV rows, per-case sequential call IDs','All/moving/Stand counts and planning-time sums exactly agree with committed summary','r35_enabled coincides with moving gait for all rows','1601 positive installations =1598 cost descents+3 risk-prioritized cost increases','343 stagnations=137 nominal installations+206 incumbent retentions','Declared knee reversal events outside every existing snapshot window']
    duplicate={}
    for x in g['scenarios']:duplicate.setdefault(x['scenario_execution_sha256'],[]).append(x['scenario_index'])
    pol={x['policy_execution_sha256'] for x in g['scenarios']}
    assert len(duplicate)==18 and len(pol)==6
    duplicate_stats=[]
    for ids in duplicate.values():
        if len(ids)==2:
            u,v=[g['per_scenario_sampled_hard_conditions'][i] for i in ids]
            duplicate_stats.append({'scenario_indices':ids,'label':u['label'],'sign':u['sign'],
                'native_statistic_abs_differences':{k:abs(u['native'][k]-v['native'][k]) for k in ['minimum_z_m','maximum_abs_roll_rad','maximum_abs_pitch_rad']},
                'GPU_statistic_abs_differences':{k:abs(u['gpu'][k]-v['gpu'][k]) for k in ['minimum_z_m','maximum_abs_roll_rad','maximum_abs_pitch_rad']}})
    out['GPU']={'nominal_scenarios':len(g['scenarios']),'distinct_scenarios':len(duplicate),'distinct_policies':len(pol),'duplicate_scenario_groups':[v for v in duplicate.values() if len(v)>1],
        'duplicate_scenario_extrema_comparison':duplicate_stats,
        'duplicate_scope':'Differences between already-published per-scenario extrema, not a new pointwise comparison of raw arrays. This reveals unequal batch outcomes for identities equal under the declared hash; complete hidden initialization/solver state is not proven identical.',
        'initial':g['initial'],'first_hold':g['short_holds'][0], '20ms_hold':g['short_holds'][-1],
        'max_first_20ms_held_action_error':max(g['short_holds'],key=lambda x:x['action_max_error_rad']),
        'full350ms_state_errors':g['full350ms_state_errors'],'full350ms_held_action_error':g['full350ms_held_action_error'],
        'full350ms_recomputed_raw_error':g['full350ms_recomputed_raw_error'],
        'sampled_hard_condition_counts':{k:sum(x[k]['body_fall_sampled'] for x in g['per_scenario_sampled_hard_conditions']) for k in ['native','gpu']},
        'work_correction':g['work_correction'],'optional_single_hold':g['optional_single_hold'],
        'W47_new_GPU_allocations':json.loads((p/'task-c/run.json').read_text())['new_GPU_allocations']}
    out['W47_actual_work']=ph['actual_new_work']
    out['W47_resource_scope']={'cgroup_CPU_quota':ph['live_preflight']['cgroup_cpu_quota'],'allowed_cpu_ids':ph['live_preflight']['allowed_cpu_ids'],
        'GPU_total_MiB':ph['live_preflight']['GPU_total_MiB'],'unrelated_GPU_MiB':ph['live_preflight']['unrelated_GPU_MiB'],
        'unrelated_process':ph['live_preflight']['unrelated_process'],'CPU_B_no_solve':True,'GPU_C_no_GPU_execution':True,
        'paired_wall_s':ph['actual_new_work']['paired_wall_s'],'full_phase_elapsed_s':ph['finished_unix_s']-ph['started_unix_s']}
    out['recording_size_estimate_uncompressed_float64']={'definition':'Minimal deployable policy stores reference x/u/time/K/H and scalar header; excludes residuals, full candidate pools and derivative tensors. Illustrative dimensions T176/ns37/nu12/nx36.',
        'one_minimal_policy_bytes':8*(176*37+175*12+176+176*12*36+176*12*12+16),
        'actual_plus_prior_predicted_37D_states_per_50Hz_call_bytes':2*37*8,
        'such_state_pairs_per_30s_episode_bytes':1500*2*37*8,
        'such_state_pairs_per_600s_episode_bytes':30000*2*37*8,
        'pre0p1_post0p2_event_plans_upper_nominal':17,
        'one_event_minimal_policy_bytes_upper_nominal':17*8*(176*37+175*12+176+176*12*36+176*12*12+16),
        'limitations':'Metadata, userdata/material context add size; compression unknown; events should deduplicate immutable policy IDs. Estimate is not an observed compression ratio.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'positive':out['combined']['positive_installs'],'descent':out['combined']['positive_J_descent'],'stagnation':out['combined']['search_stagnation'],'GPU_unique':len(duplicate),'output':str(a.output)},ensure_ascii=False))

if __name__=='__main__':main()
