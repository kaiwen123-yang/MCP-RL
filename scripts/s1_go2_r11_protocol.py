"""R11 independent stage semantics and exact-case predecessor qualification."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-go2-material-continuation-r11.json'
CONTRACT=json.loads((ROOT/CONFIG).read_text())
RUN=CONTRACT['run_id']
EVALUATION_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEVELS=['E30','E60','E300','E600']
CONDITIONS=['nominal','yaw_plus5','yaw_minus5']
STAGES={
 'E30':dict(level='E30',task_kind='original_route_short_stage',duration_s=30,target_circuits=None,route_deadline_s=None,far_hold_s=0,home_hold_s=2,final_hold_s=None),
 'E60':dict(level='E60',task_kind='original_single_mission',duration_s=60,target_circuits=1,route_deadline_s=60,far_hold_s=0,home_hold_s=2,final_hold_s=None),
 'E300':dict(level='E300',task_kind='continuous_finite',duration_s=300,target_circuits=5,route_deadline_s=270,far_hold_s=2,home_hold_s=2,final_hold_s=2),
 'E600':dict(level='E600',task_kind='continuous_finite',duration_s=600,target_circuits=10,route_deadline_s=540,far_hold_s=2,home_hold_s=2,final_hold_s=2)}
def digest(p):
    return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def stable_digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def stage(level,training_duration=None):
    if training_duration is None:return copy.deepcopy(STAGES[level])
    return dict(level=f'train{training_duration}',task_kind='continuing_acquisition',duration_s=training_duration,
        target_circuits=None,route_deadline_s=None,far_hold_s=2,home_hold_s=2,final_hold_s=None)
def case_identity(c,group,seed,condition,checkpoint=None,normalization=None):
    common=copy.deepcopy(c)
    for key in ['r11_stage','r10_contract','r11_contract','outputs','long_mission_protocol','campaign_budget',
                'r11_policy','r11_training','case_identity','case_key']:
        common.pop(key,None)
    common['task'].pop('maximum_episode_s',None)
    sources=['s1_go2_r11_runtime.py','s1_go2_r11_model.py','s1_go2_joint_material.py',
             's1_go2_wb_equilibrium.py','s1_go2_support_reference.py']
    if checkpoint:sources+=['s1_go2_r11_policy.py','s1_go2_r10_observer.py']
    identity=dict(protocol_revision=CONTRACT['protocol_revision'],group=group,seed=seed,
        condition=condition,initial_state=next(x for x in c['single_mission_protocol']['formal_conditions'] if x['id']==condition),
        initial_material='virgin ReferenceColumns, independent session',
        common_configuration_sha256=stable_digest(common),
        execution_sources={n:digest(ROOT/'scripts'/n) for n in sources},
        material_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/tm02').glob('*.py'))},
        binary_sha256=digest(ROOT/'work/r11-frozen'/c['controller_build']/'agent_server'),
        fixed_cap_m_s=c['fixed_cap_m_s'] if checkpoint is None else None,
        checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        normalizer_sha256=stable_digest(normalization) if normalization is not None else None)
    return identity,stable_digest(identity)
def prerequisite(c,index,purpose,terrain):
    level=c['r11_stage']['level']; key=c['case_key']
    if terrain=='rigid':
        if not index.get('frozen_controller'):raise ValueError('Rigid regression requires frozen S5')
        if any((a.get('case_identity') or {}).get('terrain')=='rigid' for a in index.get('attempts',[])):
            raise ValueError('The one declared rigid regression already launched')
        return None
    j=LEVELS.index(level)
    if j==0:return None
    prior=LEVELS[j-1]
    passes=[a for a in index.get('attempts',[]) if a.get('case_key')==key and
            a.get('evaluation_stage',{}).get('level')==prior and a.get('stage_status')=='passed' and a.get('qualification_usable',True)]
    if not passes:raise ValueError(f'{level} requires this exact case to pass independent {prior}')
    previous=passes[-1]['attempt']
    actual=json.loads((ROOT/CONTRACT['outputs']['compact_root']/(previous+'-resolved.json')).read_text())
    if actual['workers']!=c['evaluation_workers']:
        raise ValueError('The declared predecessor used different solver workers; re-establish qualification for the changed execution setting')
    return previous
def material_traversal(result,arrays):
    # Operationalize the user's passage through the unchanged rectangular strip.
    # This only scores the completed original route; it never terminates/steers physics.
    outbound=next((r['time_s'] for r in result['transitions'] if r['phase']=='outbound'),None)
    far=next((r['time_s'] for r in result['transitions'] if r['phase']=='turn'),None)
    t=arrays['time'];xy=arrays['qpos'][:,:2]
    selected=np.flatnonzero((t>=outbound-1e-8)&(t<=(far if far is not None else t[-1])+1e-8)) if outbound is not None else np.array([],int)
    west=next((int(i) for i in selected if xy[i,0]>=.8),None)
    east=next((int(i) for i in selected if west is not None and i>=west and xy[i,0]>=2.2),None)
    segment=np.arange(west,east+1 if east is not None else selected[-1]+1) if west is not None else np.array([],int)
    lateral=np.flatnonzero(abs(xy[segment,1])>.6) if len(segment) else np.array([],int)
    half=.015811388300841896
    feet=arrays['feet'][segment,:,:2]
    interior=(feet[:,:,0]>=.8+half)&(feet[:,:,0]<=2.2-half)&(abs(feet[:,:,1])<=.6-half)
    loaded=interior&(arrays['actual_grf'][segment,:,2]>0)
    material_loaded=bool(np.any(loaded))
    complete=bool(west is not None and east is not None and not len(lateral) and material_loaded)
    return dict(completed=complete,west_entry_s=float(t[west]) if west is not None else None,
        east_exit_s=float(t[east]) if east is not None else None,
        lateral_exit_s=float(t[segment[lateral[0]]]) if len(lateral) else None,
        max_abs_base_y_during_crossing_m=float(np.max(abs(xy[segment,1]))) if len(segment) else None,
        material_loaded_during_crossing=material_loaded,
        original_strip_x_m=[.8,2.2],original_strip_y_m=[-.6,.6],
        measurement='First outbound base-projection passage from the west to east strip boundary without a side exit, with actual material-loaded feet during that passage. Uses original strip geometry and500Hz recorded states; no extra physical termination threshold or route change.')

def evaluate_stage(result,c,joint,arrays):
    identity=c['r11_stage'];level=identity['level'];duration=identity['duration_s']
    result.update(evaluation_stage=identity,case_key=c.get('case_key'),case_identity=c.get('case_identity'),
        qualification=dict(predecessor_attempt=c.get('predecessor_attempt'),protocol_revision=CONTRACT['protocol_revision'],
            evaluator_sha256=EVALUATION_SOURCE_SHA256,
            solver_workers=c.get('evaluation_workers'),worker_setting_checked_against_predecessor=bool(c.get('predecessor_attempt'))))
    reason=result['reason'];full=result['simulation_s']>=duration-1e-7
    failure=reason.startswith(('material_domain_exit','body_fall','invalid_contact'))
    technical=reason=='execution_error'
    budget='budget_truncated' in reason or reason=='training_checkpoint_complete'
    far=[x for x in result['transitions'] if x['phase']=='turn']
    returning=[x for x in result['transitions'] if x['phase']=='return']
    result['stage_progress']=dict(far_arrival_s=far[0]['time_s'] if far else None,
        return_start_s=returning[0]['time_s'] if returning else None,
        actual_route_fraction=joint.route_progress,full_observation=full,
        actual_material_loaded=bool(np.any(joint.ground.ever_loaded)) if joint.ground.kind!='rigid' else False)
    result['stage_progress']['material_traversal']=material_traversal(result,arrays)
    if identity['task_kind']=='continuing_acquisition':
        result.update(success=None,stage_status='physical_material_failure' if failure else
            'technical_interruption' if technical else 'budget_truncated' if budget else
            'acquisition_boundary' if full else 'technical_interruption')
        return result
    if level=='E30':
        achieved=bool(full and far and result['stage_progress']['material_traversal']['completed'])
    elif level=='E60':
        achieved=bool(full and result['first_completion_s'] is not None)
    else:
        h=result['hold']
        achieved=bool(full and result['completed_round_trips']>=identity['target_circuits'] and
            result['first_completion_s'] is not None and result['first_completion_s']<=identity['route_deadline_s'] and
            h['endpoint_position_violation_samples']==0 and h['final_stable_s']>=2-1e-7)
    passed=bool(achieved and not failure and not technical and not budget)
    status='passed' if passed else 'physical_material_failure' if failure else 'technical_interruption' if technical else 'budget_truncated' if budget else 'task_criteria_unmet'
    result.update(success=passed,stage_status=status,sustained_execution_success=passed,
        original_mission_success=bool(result['first_completion_s'] is not None) if level=='E60' else None)
    if not failure and not technical and not budget:
        result['reason']='stage_passed' if passed else 'material_traversal_not_completed' if level=='E30' and not result['stage_progress']['material_traversal']['completed'] else 'stage_criteria_unmet'
    return result
