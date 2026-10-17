"""Execute the authorized no-training functional round on the existing task."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from joint_update_core import (Feedback,Case,run_round,full_parameter_credit,
                               merge_identical_rows,compare_cases)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def review_falsifiers():
    """Reuse the review's semantic counterexamples, not new task/environment data."""
    review=json.loads((ROOT/'notes/physical-task-instance-2026-10-17/independent-review-checks.json').read_text())
    make=lambda j,direct=0:{'response_jacobian':[[j]],'task_covector':[1.],
                            'prefix_direct':[0.],'suffix_direct':[direct]}
    g,parts=full_parameter_credit([make(1),make(-1)],[.5,.5])
    immobile,_=full_parameter_credit([make(0)],[1.])
    direct_only,_=full_parameter_credit([make(0,2)],[1.])
    conflicted=merge_identical_rows([
        {'facet':'review_active_max','value':0.,'gradient':[1.],'origins':['first']},
        {'facet':'review_active_max','value':0.,'gradient':[-1.],'origins':['second']}])
    ppo=review['ppo_acceptance_counterexample']
    base={'case_id':'review_logical_scalar','weight':1.,'evidence_class':'DECLARED_REVIEW_MATH',
          'task_version':'review_1_minus_d','sign':None,'status':'COMPLETED','task_time':1.}
    comparisons=[]
    for row in ppo['backtracks']:
        candidate={**base,'task_time':row['deployment_objective_1_minus_mean']}
        decision=compare_cases([base],[candidate])
        comparisons.append({'hypothetical_mean':row['hypothetical_mean'],'surrogate':row['surrogate'],
                            'gaussian_kl':row['gaussian_kl'],'task_comparison':decision['reason'],
                            'task_comparison_passed':decision['passed']})
    return {'source':'Previously declared independent-review mathematical counterexamples, not a PPO batch.',
            'same_lambda_opposite_j':{'individual': [x.tolist() for x in parts],'aggregate':g.tolist()},
            'zero_j_and_zero_direct':immobile.tolist(),'zero_j_nonzero_suffix_direct':direct_only.tolist(),
            'different_active_gradients_retained':conflicted,
            'ppo_guard_replay':{'source_delta_D':ppo['delta_D'],'original_strict_guard_rejections':len(ppo['backtracks']),
                               'all_original_surrogates_decrease':all(r['surrogate']<0 for r in ppo['backtracks']),
                               'all_original_KL_below_limit':all(r['gaussian_kl']<=ppo['delta_D'] for r in ppo['backtracks']),
                               'new_task_priority_comparisons':comparisons,
                               'replacement_rule':'No S-loss tolerance was added. Deterministic task is the final quality objective; S is monitored, stochastic improvement not guaranteed.'}}


def main():
    started=time.perf_counter();out=ROOT/'notes/joint-update-core-2026-10-17'
    protocol_hash=sha(out/'protocol.md')
    cases=(Case('positive_impulse',1),Case('negative_impulse',-1))
    deployed=Feedback(0.);anchor=Feedback(0.)
    input_identity=(deployed.identity,anchor.identity)
    supported=run_round(deployed,anchor,cases)
    unsafe=run_round(deployed,anchor,cases,external_candidate=Feedback(8/9))
    old_analytic=json.loads((ROOT/'notes/physical-task-instance-2026-10-17/analytic-results.json').read_text())
    witness=old_analytic['cases']['geometry_derived_joint_feedback'][0]
    first=witness['segments'][0]
    optimistic_response=[first['duration'],first['p_end'][0],first['v_end'][0],abs(first['p_end'][1])]
    model_only=run_round(deployed,anchor,cases,external_candidate=deployed,model_override=optimistic_response)
    availability={}
    for status in ('MODEL_UNSUPPORTED','SEARCH_UNRESOLVED','EVALUATION_TRUNCATED'):
        masked=(cases[0],Case('negative_impulse',-1,evidence=status))
        availability[status]=run_round(deployed,anchor,masked)
    all_rounds={'supported':supported,'unsafe_external_candidate':unsafe,'model_only':model_only,**availability}
    total={key:sum(r['budget'][key] for r in all_rounds.values()) for key in supported['budget']}
    falsifiers=review_falsifiers()
    assert (deployed.identity,anchor.identity)==input_identity
    assert supported['status']=='ACCEPTED_ANALYTIC_CASESET'
    assert unsafe['attempts'][0]['against_deployed']['reason']=='REJECT_HARD_CONTRACT'
    assert model_only['attempts'][0]['pair_check']['status']=='REJECT_MODEL_POLICY_MISMATCH'
    assert all(r['status']=='NO_FULL_SCOPE_UPDATE' and r['active_feedback']['r']==0 for r in availability.values())
    assert falsifiers['same_lambda_opposite_j']['aggregate']==[0.]
    assert len(falsifiers['different_active_gradients_retained'])==2
    assert all(c['task_comparison_passed'] for c in falsifiers['ppo_guard_replay']['new_task_priority_comparisons'])
    assert sha(out/'protocol.md')==protocol_hash
    report={'date':'2026-10-17','status':'completed_functional_rounds',
            'protocol_sha256':protocol_hash,'script_sha256':sha(__file__),
            'core_sha256':sha(ROOT/'src/joint_update_core.py'),
            'source_review_sha256':sha(ROOT/'notes/physical-task-instance-2026-10-17/independent-review-checks.json'),
            'rounds':all_rounds,'review_semantic_falsifiers':falsifiers,
            'input_policy_values_unchanged':True,'budget':total,'wall_seconds':time.perf_counter()-started,
            'review_arithmetic_budget':{'parameter_contributions':4,'merge_input_rows':2,
                                        'task_comparator_calls_on_existing_logical_records':4},
            'zero_operations':{'RL_training':0,'new_RL_batch':0,'neural_actor_evaluations':0,
                               'neural_weight_updates':0,'checkpoint_reads':0,'checkpoint_writes':0,
                               'robot_environment_rollouts':0,'hardware_operations':0},
            'interpretation':'The core actually proposes/evaluates/accepts or defers immutable analytic feedback values and pairs their responses. It is not PPO training or a robot learning result.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'round-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=plain)+'\n',encoding='utf-8')
    selected=supported['accepted_feedback']
    print(json.dumps({'accepted_feedback':selected,'supported_budget':supported['budget'],
                      'old_time':supported['attempts'][0]['against_deployed']['reference']['full_mean_task_time'],
                      'new_time':supported['attempts'][0]['against_deployed']['candidate']['full_mean_task_time'],
                      'unsafe_reason':unsafe['attempts'][0]['against_deployed']['reason'],
                      'model_only_reason':model_only['attempts'][0]['pair_check']['status'],
                      'availability_statuses':{k:r['status'] for k,r in availability.items()},
                      'all_rounds_budget':total,'wall_seconds':report['wall_seconds']},indent=2,default=plain))


if __name__=='__main__':main()
