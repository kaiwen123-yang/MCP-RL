"""Check one local credit packet and falsify exact reuse over its finite change.

Reads the already checked analytic instance. No trajectory, policy evaluation,
optimizer, training batch or parameter update is generated.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from control_task_credit import ResponsePacket,pullback,finite_discrepancy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'notes/physical-task-instance-2026-10-17'
    source=out/'analytic-results.json';data=json.loads(source.read_text(encoding='utf-8'))
    credit=data['credit_packet'];cfg=data['config']
    packet=ResponsePacket(np.array(credit['response']),5.,np.array(credit['downstream_task_covector']),
                          np.array([-.5]),np.array([[0.,0.,0.,1.]]),
                          'analytic-normalized-task-v1/frozen-lateral-first/exact-model')
    gradient=pullback(packet,np.array(credit['response_derivative'])[:,None])
    branch=data['cases']['geometry_derived_joint_feedback'][0]
    phase=branch['segments'][0]
    response=np.array([phase['duration'],phase['p_end'][0],phase['v_end'][0],abs(phase['p_end'][1])])
    tau,x,v,abs_y=response;a,L=cfg['a'],cfg['L']
    # One exact suffix value evaluation, from the same task and physical model.
    q=(L-x)/a+v*v/(2*a*a)
    task=float(tau+2*np.sqrt(q)-v/a)
    constraints=np.array([abs_y-cfg['b']])
    mismatch=finite_discrepancy(packet,response,task,constraints)
    refreshed_covector=np.array([1.,-1/(a*np.sqrt(q)),v/(a*a*np.sqrt(q))-1/a,0.])
    assert abs(gradient['task_gradient'][0]-credit['complete_task_direct_derivative'])<1e-12
    assert abs(task-branch['duration'])<1e-12
    assert mismatch['task_optimism']>.7  # Algebraic expected inequality, not a release tolerance.
    assert np.max(np.abs(mismatch['constraint_errors']))<1e-12
    report={'date':'2026-10-17','status':'completed_arithmetic_verification',
            'source_sha256':sha(source),'script_sha256':sha(__file__),
            'module_sha256':sha(ROOT/'src/control_task_credit.py'),
            'local_gradient':gradient['task_gradient'].tolist(),
            'local_constraint_gradients':gradient['constraint_gradients'].tolist(),
            'new_response':response.tolist(),
            'cached_task_prediction':mismatch['cached_task_value'],
            'exact_evaluated_task':task,'finite_change_optimism':mismatch['task_optimism'],
            'linear_clearance_constraint_error':mismatch['constraint_errors'].tolist(),
            'refreshed_task_covector':refreshed_covector.tolist(),
            'falsified_claim':'An exact local task gradient makes the old packet an exact finite-change task predictor.',
            'not_falsified':'The finite feedback witness still improves the exact declared task; same-class standard TO remains equal.',
            'scope':'No robot or PPO gradient. One physical allocation coordinate and one exact planar task.',
            'budget':{'stored_branches_read':1,'packet_pullbacks':1,'finite_discrepancy_checks':1,
                      'closed_form_suffix_value_and_covector_evaluations':1,
                      'new_trajectory_segments':0,'actor_evaluations':0,'robot_environment_rollouts':0,
                      'RL_batches':0,'parameter_updates':0,'optimizer_calls':0}}
    (out/'credit-packet-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
