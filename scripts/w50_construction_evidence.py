"""Read the executed finite objects and account for the W50 transfer algebra."""
import json
import numpy as np
from w50_common import LARGE,SMALL,LABELS,write,sha,parse_candidates
from w50_policy_io import read_policy,provenance_summary

def maxabs(x):return float(np.max(np.abs(x))) if np.size(x) else 0.
def analyze(label):
 d=LARGE/'task-b'/label
 f=json.loads((d/'first-homotopy.json').read_text());call=json.loads((d/'planner-calls.jsonl').read_text().splitlines()[-1])
 hb,b=read_policy(d/'first-baseline.bin');hn,n=read_policy(d/'first-backward.bin')
 cap=np.fromfile(d/'first-direction.bin',dtype='<f8');head=cap[:11];count,nu=map(int,head[3:5]);times=cap[11:11+count];k=cap[11+count:].reshape(count,nu)
 raw=np.fromfile(d/'first-centers.bin',dtype='<f8');ch=raw[:7];cnt,nua,nx=map(int,ch[1:4]);j=7
 center=raw[j:j+cnt*nua].reshape(cnt,nua);j+=cnt*nua
 L=raw[j:j+cnt*nua*nx].reshape(cnt,nua,nx);j+=cnt*nua*nx
 H=raw[j:].reshape(cnt,nua,nua)
 rows=parse_candidates(d/'snapshot.bin.r33-candidates');positives=[(idx,c) for idx,c in enumerate(rows) if c['metadata']['kind']==1]
 base=next(c for c in rows if c['metadata']['kind']==0 and c['metadata']['feedback_scale']==b['semantics'][5] and
  all(np.array_equal(c['arrays'][key],b[key]) for key in ['reference_states','reference_actions','reference_times','K','H']))
 checks=[]
 for idx,c in positives:
  h,p=read_policy(d/f'complete-candidate-{idx+1}.bin');a=c['metadata']['alpha']
  checks.append(dict(slot=c['metadata']['slot'],alpha=a,ff_max_abs_error=maxabs(p['reference_actions']-(center+a*k)),
   K_max_abs_error=maxabs(p['K'][:-1]-((1-a)*L+a*n['K'][:-1])),
   H_max_abs_error=maxabs(p['H'][:-1]-((1-a)*H+a*n['H'][:-1])),
   reference_states_equal_current_B_rollout=bool(np.array_equal(p['reference_states'],base['arrays']['states'])),
   reference_times_equal_current_B_rollout=bool(np.array_equal(p['reference_times'],base['arrays']['times'])),
   terminal_K_repeats_last_control=bool(np.array_equal(p['K'][-1],p['K'][-2])),
   terminal_H_repeats_last_control=bool(np.array_equal(p['H'][-1],p['H'][-2])),
   base_Qu_exact_same_backward=bool(np.array_equal(p['base_Qu'],n['base_Qu'])),
   base_Qxu_exact_same_backward=bool(np.array_equal(p['base_Qxu'],n['base_Qxu'])),
   active_exact_same_backward=bool(np.array_equal(p['base_active'],n['base_active'])),
   feedback_scale=float(p['semantics'][5]),delta_knots=len(p['delta_times']),metric_single_mu=provenance_summary(p)['metric_single_mu'],
   new_backward_mu=float(p['provenance'][24]),new_backward_attempt=int(p['provenance'][26])))
 hi,installed=read_policy(d/'installed.bin');source=call['candidate_source'];match=None
 for idx,c in enumerate(rows):
  h,p=read_policy(d/f'complete-candidate-{idx+1}.bin')
  if all(np.array_equal(p[key],installed[key]) for key in p):
   match=dict(record_index=idx,kind=c['metadata']['kind'],slot=c['metadata']['slot'],
    all_complete_arrays_exact=True,preinstall_plan_id=int(h[1]),installed_plan_id=int(hi[1]));break
 source_scope=dict(status='same backward capture includes full K/H/Qu/Qxu/active; k and times in direction',
  captured_mu=float(head[6]),captured_rate=float(head[7]),captured_attempt=int(head[5]),
  backward_object_mu=float(hn[8]),direction_time_matches_B_current_grid=bool(np.array_equal(times,base['arrays']['times'][:-1])))
 return dict(label=label,native_solves=0,new_physics=0,new_center_evaluations=0,
  source_capture=source_scope,raw_center_evaluations_from_executed_solve=int(ch[4]),raw_center_fixed_FD_scale=float(ch[5]),
  baseline=dict(J_from_cached_screen=base['metadata']['native_J'],J_from_homotopy=f['baseline_J'],
   cached_score_exact=bool(base['metadata']['native_J']==f['baseline_J']),
   alpha_zero_added_rollouts=0,positive_rollouts=call['native_new_rollouts']-10,
   preserved_reference_start=float(b['reference_times'][0]),preserved_reference_end=float(b['reference_times'][-1]),
   preserved_delta_knots=len(b['delta_times']),full_policy_sha256=sha(d/'first-baseline.bin')),
  candidates=checks,selected_source=source,installed_matches_scored_complete_object=match,
  installed_lifecycle=provenance_summary(installed,float(call['origin'])))

def main():
 events=[analyze(label) for label in LABELS]
 result=dict(schema='w50_executed_policy_construction_v1',status='completed',native_solves=0,new_physics=0,
  events=events,scope='Algebra and identity readback of already executed fixed-problem objects; no new solve, rollout, FD or deployment.',
  all_positive_formulas_exact=all(c['ff_max_abs_error']==c['K_max_abs_error']==c['H_max_abs_error']==0 for e in events for c in e['candidates']),
  all_winners_same_complete_objects=all(e['installed_matches_scored_complete_object'] is not None for e in events))
 write(SMALL/'policy-construction.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
if __name__=='__main__':main()
