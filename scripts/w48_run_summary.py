"""Compact actual W48 attempt evidence, derived only from recorded same-attempt files."""
import json
from pathlib import Path
import numpy as np
from publish_s1_go2_w48 import ROOT,SMALL,LARGE,write
def summarize_attempt(attempt):
 out=LARGE/attempt;dest=SMALL/'w48'/'original-evidence'/(attempt+'.json')
 rp=SMALL/(attempt+'.json');result=json.loads(rp.read_text()) if rp.exists() else None
 calls=[json.loads(x) for x in (out/'planner-calls.jsonl').read_text().splitlines()] if (out/'planner-calls.jsonl').exists() else []
 boundaries=[json.loads(x) for x in (out/'w48-planner-boundaries.jsonl').read_text().splitlines()] if (out/'w48-planner-boundaries.jsonl').exists() else []
 events=json.loads((out/'w48-event-manifest.json').read_text()) if (out/'w48-event-manifest.json').exists() else {}
 positive=[x for x in calls if x['policy_updated'] and x['candidate_source'].startswith('alpha_') and x['chosen_alpha']>0]
 matched=[x for x in boundaries if x['previous_prediction']['status']=='available']
 first=next((x for x in matched if any(v!=0 for v in x['previous_prediction']['position_tangent']+x['previous_prediction']['velocity_difference'])),None)
 keys=['native_new_rollouts','candidate_records','derivative_passes','transition_fd_calls','risk_geometry_fd_calls','backward_attempts']
 summary=dict(attempt=attempt,status=result['stage_status'] if result else 'technical_interruption',
  original_result={k:result.get(k) for k in ['success','reason','error','simulation_s','wall_s','physical_samples','legal_route','stage_progress','qualification','max_x_m','min_body_height_m','completed_round_trips']} if result else None,
  planner_calls=len(calls),positive_installs=len(positive),positive_same_problem_cost_descent=sum(x['chosen_actual']>0 for x in positive),
  positive_risk_priority_cost_increase=sum(x['chosen_actual']<0 for x in positive),
  internal_stagnation=sum(x['stop_reason']=='native_search_stagnation' for x in calls),
  work={k:sum(x[k] for x in calls) for k in keys},new_PPO_samples=0,new_PPO_updates=0,
  plant_physics_substeps=result.get('physical_samples') if result else None,
  prediction_pairs=len(matched),prediction_unavailable=len(boundaries)-len(matched),
  first_nonzero_same_time_prediction_error=first,
  diagnostic_events=events.get('events',[]),object_bytes=events.get('actual_bytes'),
  same_attempt_record_readback=json.loads((out/'w48-first-record-readback.json').read_text()) if (out/'w48-first-record-readback.json').exists() else None,
  causal_conclusion='Original task status above; nonzero prediction error and diagnostic thresholds do not by themselves identify the cause of failure.')
 if matched:
  summary['prediction_error_extrema']={key:float(max(np.max(abs(np.asarray(x['previous_prediction'][key]))) for x in matched)) for key in
    ['base_position_m','base_orientation_rad','joint_position_rad','base_linear_velocity_m_s','base_angular_velocity_rad_s','joint_velocity_rad_s']}
 write(dest,summary);return dest
