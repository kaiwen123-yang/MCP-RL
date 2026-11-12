#!/usr/bin/env python3
"""Recompute W45 control audit using archived arrays only, never MuJoCo.

Requires Python 3 + NumPy. Source originals are identified by control-originals.json.
The two trajectory files are supplied separately as CASE-trajectory.npz.
"""
import argparse, collections, hashlib, json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from policy_arrays import raw_policy

FIXTURES=[('w45-a',148,136),('w45-a',153,140),('w45-b',155,141),('w45-b',158,144)]
IDENTITY=['accepted_states','accepted_actions','accepted_times','accepted_residual',
          'reference_states','reference_actions','reference_times','reference_residual',
          'K','H','base_Qu','base_Qxu','base_active']
LIMIT=np.tile([23.7,23.7,45.43],4)

def ma(x):return float(np.max(np.abs(x)))
def rows(p):return [json.loads(x)for x in p.read_text().splitlines()if x.strip()]
def intervals(mask,times,ends=None):
    start=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
    end=np.flatnonzero(mask&~np.r_[mask[1:],False])
    return [{'first_sample_s':float(times[a]),'last_sample_s':float(times[b]),'samples':int(b-a+1),
             **({'post_last_integration_s':float(ends[b]),'duration_s':float(ends[b]-times[a])}if ends is not None else{})}
            for a,b in zip(start,end)]

def prefix(raw,ns,nr):
    out=[];i=0
    while i<len(raw):
        h=raw[i:i+5];i+=5;nc=int(h[4]);d={'node':int(h[0]),'sub':int(h[1]),'before':float(h[2]),'after':float(h[3]),'ncontact':nc}
        for name,n in [('pre',ns),('post',ns),('raw_pd',12),('actuator',12),('passive',18),('residual',nr)]:
            d[name]=raw[i:i+n];i+=n
        d['contact']=raw[i:i+8*nc].reshape(nc,8);i+=8*nc;out.append(d)
    assert i==len(raw)
    return out

def cost_parts(residual,xml,weights):
    result={};offset=0
    terms=[e for e in ET.parse(xml).getroot().find('sensor')if e.tag=='user']
    for e,w in zip(terms,weights):
        dim=int(e.get('dim'));d=residual[:,offset:offset+dim];offset+=dim
        u=list(map(float,e.get('user').split()));norm=int(u[0]);p=u[4]if len(u)>4 else None
        if norm==0:v=.5*(d*d).sum(1)
        elif norm==2:v=np.sqrt((d*d).sum(1)+p*p)-p
        elif norm==6:v=(np.sqrt(d*d+p*p)-p).sum(1)
        else:raise ValueError(norm)
        result[e.get('name')]=float(w*np.mean(v))
    assert offset==residual.shape[1]
    return result

def whole_run(case,d,tr):
    calls=rows(d/'planner-calls.jsonl');feedback=rows(d/'feedback-actions.jsonl')
    execution={e['query_id']:e for e in rows(d/'action-execution.jsonl')if e['query_kind']=='applied_hold'}
    byplan={c['installed_plan_id']:c for c in calls if c['policy_updated']}
    applied=[]
    for f in feedback:
        if f['query_id']not in execution:continue
        e=execution[f['query_id']];c=byplan.get(f['plan_id'],{});ix=int(np.argmin(abs(tr['time']-e['time_s'])))
        applied.append({'time_s':e['time_s'],'query_id':f['query_id'],'plan_id':f['plan_id'],
            'call_id':c.get('call_id'),'source':c.get('candidate_source'),
            'effective_feedback_rad':abs(f['feedback_scaling'])*ma(f['raw_K_delta_x']),
            'projection_rad':ma(f['projection_delta']),'scale':f['feedback_scaling'],
            'alpha':f['action_step_alpha'],'target_match_error_rad':ma(np.array(f['u_held'])-tr['q_target'][ix]),
            'qp_converged':f['qp_converged'],'restoration':f['restoration'],
            'primal_residual_rad':f['primal_residual_rad'],'kkt_scaled':f['kkt_scaled_residual']})
    phy=np.load(d/'physical-000.npz')['physical'];excess=np.maximum(abs(phy[:,49:61])-LIMIT,0).max(1)
    nofeet=(phy[:,26:30].sum(1)==0)&(abs(phy[:,30:42]).max(1)<1e-10)
    flights=intervals(nofeet,phy[:,0],phy[:,1])
    result={'calls':len(calls),'all_r34_strict':all(c['r34_enabled']for c in calls),
        'source_counts':dict(collections.Counter(c['candidate_source'].split('_')[0]for c in calls)),
        'return_reason_counts':dict(collections.Counter(c['return_reason']for c in calls)),
        'stop_reason_counts':dict(collections.Counter(c['stop_reason']for c in calls)),
        'backward_attempt_counts':dict(collections.Counter(c['backward_attempts']for c in calls)),
        'native_scored_rollouts':sum(c['native_new_rollouts']for c in calls),
        'new_alpha0_installs':sum(c['policy_updated']and c['candidate_source'].startswith('alpha_')and c['chosen_alpha']==0 for c in calls),
        'alpha0_rejected_records':sum(x['kind']==1 and x['alpha']==0 and not x['eligible']and not x['replaced']for c in calls for x in c['r34_decisions']),
        'applied_holds':len(applied),'effective_feedback_gt_point5':sum(f['effective_feedback_rad']>.5 for f in applied),
        'effective_feedback_gt_point5_sources':dict(collections.Counter(f['source'].split('_')[0]for f in applied if f['effective_feedback_rad']>.5)),
        'projection_gt_point5':sum(f['projection_rad']>.5 for f in applied),
        'first_effective_feedback_gt_point1':next(f for f in applied if f['effective_feedback_rad']>.1),
        'first_effective_feedback_gt_point5':next(f for f in applied if f['effective_feedback_rad']>.5),
        'qp_failed':sum(not f['qp_converged']for f in applied),'restoration_count':sum(f['restoration']for f in applied),
        'max_primal':max(f['primal_residual_rad']for f in applied),'max_kkt_scaled':max(f['kkt_scaled']for f in applied),
        'max_query_target_match_error_rad':max(f['target_match_error_rad']for f in applied),
        'physical_samples':len(phy),'first_held_raw_PD_saturation_s':float(phy[np.flatnonzero(excess>1e-4)[0],0]),
        'held_raw_PD_saturation_fraction':float(np.mean(excess>1e-4)),
        'held_raw_PD_peak_excess_Nm':float(excess.max()),
        'physical_clamp_max_error_Nm':ma(phy[:,14:26]-np.clip(phy[:,49:61],-LIMIT,LIMIT)),
        'first_moving_nofoot_interval':next(f for f in flights if f['first_sample_s']>=2),
        'first_moving_nofoot_gt_20ms_interval':next(f for f in flights if f['first_sample_s']>=2 and f['duration_s']>=.02)}
    return result,calls,execution,feedback,phy

def fixture(case,callid,planid,d,tr,calls,execution,feedback,phy):
    call=next(c for c in calls if c['call_id']==callid)
    z=np.load(d/f'r33-candidates-call{callid:06d}.npz')
    inst=np.load(d/f'accepted-reference-gain-H-plan{planid:06d}.npz')
    headers={k.split('_')[0]:z[k]for k in z.files if k.endswith('_header')}
    nd=next(x for x in reversed(call['r34_decisions'])if x['kind']==0 and x['replaced'])
    wd=next(x for x in reversed(call['r34_decisions'])if x['replaced'])
    def key(dec):return next(k for k,h in headers.items()if int(h[2])==dec['kind']and int(h[3])==dec['slot'])
    nominal,selected=key(nd),key(wd)
    incumbent=key(next(x for x in call['r34_decisions']if x['source']=='incumbent'))
    identity={n:ma(z[selected+'_'+n]-inst[n])for n in IDENTITY}
    assert max(identity.values())==0
    def candidate(k):
        h=headers[k];s=z[k+'_accepted_states'];ts=z[k+'_accepted_times'];foot=z[k+'_foot_rows'].reshape(-1,29)
        feet=foot[:,1:].reshape(-1,4,7);nofeet=(feet[:,:,6].sum(1)==0)&(abs(feet[:,:,4:6]).max(axis=(1,2))<1e-10)
        parts=cost_parts(z[k+'_accepted_residual'],d/'G1-project-go2.xml',call['task_weights'])
        return {'candidate':k,'kind':int(h[2]),'slot':int(h[3]),'alpha':float(h[4]),'feedback_scale':float(h[10]),
            'native_cost':float(h[11]),'trunk_violation':float(h[12]),'cost_terms':parts,
            'cost_recompute_error':abs(sum(parts.values())-float(h[11])),
            'predicted_dx_m':float(s[-1,0]-s[0,0]),'predicted_dy_m':float(s[-1,1]-s[0,1]),
            'base_z_min_max_m':[float(s[:,2].min()),float(s[:,2].max())],
            'base_vz_min_max_m_s':[float(s[:,21].min()),float(s[:,21].max())],
            'terminal_z_vz':[float(s[-1,2]),float(s[-1,21])],
            'nofoot_zero_native_and_proxy_force_nodes':int(nofeet.sum()),
            'nofoot_intervals_at_2ms_nodes':intervals(nofeet,foot[:,0]),
            'native_K_max':ma(z[k+'_K']),'reference_origin_s':float(z[k+'_reference_times'][0])}
    fb={round(execution[f['query_id']]['time_s'],6):f for f in feedback if f['query_id']in execution}
    pre=prefix(inst['accepted_prefix_substeps'],int(inst['header'][4]),int(inst['header'][7]));firsthold=[]
    for q in pre[:10]:
        p=phy[np.argmin(abs(phy[:,0]-q['before']))]
        firsthold.append({'substep':q['sub'],'time_s':q['before'],'raw_PD_difference_Nm':ma(q['raw_pd']-p[49:61]),
            'native_ncontact':q['ncontact'],'physical_foot_contacts':p[26:30].astype(int).tolist()})
    node_rows=[]
    for j in [0,1,2,3,4,5,7,9,10,11,12,14]:
        t=float(inst['accepted_times'][j]);f=fb.get(round(t,6))
        if not f:continue
        ix=int(np.argmin(abs(tr['time']-t)));actual=np.r_[tr['qpos'][ix],tr['qvel'][ix]];pred=inst['accepted_states'][j]
        ff,native_feedback,_,_=raw_policy(t,pred,inst['reference_times'],inst['reference_states'],inst['reference_actions'],inst['K'],inst['header'][12])
        row={'node':j,'time_s':t,'same_installed_policy':f['plan_id']==planid,'actual_plan_id':f['plan_id'],
            'position_difference_max':ma(pred[:19]-actual[:19]),'velocity_difference_max':ma(pred[19:]-actual[19:]),
            'joint_position_difference_rad':ma(pred[7:19]-actual[7:19]),
            'joint_velocity_difference_rad_s':ma(pred[25:37]-actual[25:37]),
            'native_actual_target_difference_rad':ma(inst['accepted_actions'][j]-tr['q_target'][ix]),
            'native_effective_feedback_rad':ma(native_feedback),
            'actual_effective_feedback_rad':abs(f['feedback_scaling'])*ma(f['raw_K_delta_x']),
            'actual_projection_rad':ma(f['projection_delta']),
            'native_held_vs_feedforward_rad':ma(inst['accepted_actions'][j]-ff),
            'actual_held_vs_feedforward_rad':ma(np.array(f['u_held'])-ff),
            'actual_foot_contacts':tr['contact'][ix].astype(int).tolist(),
            'nominal_PD_same_actual_state_excess_Nm':ma(np.maximum(abs(60*(np.array(f['nominal_unclipped'])-actual[7:19])-5*actual[25:37])-LIMIT,0))}
        if j>0:
            prev=inst['accepted_states'][j-1]
            native_tau=np.clip(60*(inst['accepted_actions'][j]-pred[7:19])-5*pred[25:37],-LIMIT,LIMIT)
            previous_tau=np.clip(60*(inst['accepted_actions'][j-1]-prev[7:19])-5*prev[25:37],-LIMIT,LIMIT)
            row.update(native_adjacent_target_jump_rad=ma(inst['accepted_actions'][j]-inst['accepted_actions'][j-1]),
                native_adjacent_PD_torque_jump_Nm=ma(native_tau-previous_tau))
        if ix>0:
            row.update(actual_adjacent_target_jump_rad=ma(tr['q_target'][ix]-tr['q_target'][ix-1]),
                actual_adjacent_actuator_torque_jump_Nm=ma(tr['actuator_force'][ix]-tr['actuator_force'][ix-1]))
        if f['plan_id']==planid:
            _,af,dx,K=raw_policy(f['time'],actual,inst['reference_times'],inst['reference_states'],inst['reference_actions'],inst['K'],inst['header'][12])
            row.update(native_actual_feedback_difference_rad=ma(native_feedback-af),
                state_delta_reconstruction_error=ma(dx-np.array(f['state_delta'])),
                feedback_reconstruction_error=ma(af-np.array(f['raw_K_delta_x'])*f['feedback_scaling']),
                interpolated_K_max=ma(K))
        node_rows.append(row)
    held={}
    for q in pre:held.setdefault(q['node'],[]).append((q['raw_pd']+60*q['pre'][7:19]+5*q['pre'][25:37])/60)
    return {'case':case,'call_id':callid,'plan_id':planid,'origin_s':call['origin'],
        'source':call['candidate_source'],'comparison_tolerance':wd['comparison_tolerance'],
        'incumbent_candidate':incumbent,'incumbent_current_cost':call['incumbent_current_cost'],
        'incumbent_cost_terms':cost_parts(z[incumbent+'_accepted_residual'],d/'G1-project-go2.xml',call['task_weights']),
        'best_nominal':candidate(nominal),'selected':candidate(selected),
        'selected_installed_max_errors':identity,'first_2ms_held_prefix':firsthold,
        'native_hold_target_max_variation_rad':max(ma(np.asarray(a)-a[0])for a in held.values()),
        'native_prefix_clamp_max_error_Nm':max(ma(q['actuator']-np.clip(q['raw_pd'],-LIMIT,LIMIT))for q in pre),
        'nodes':node_rows,
        'interpretation_note':'Node10 at +20ms may use a newly replanned physical policy. Compare target/feedback causally only where same_installed_policy=true. Short flight is legal, not a task failure criterion.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--originals-dir',type=Path,required=True)
    ap.add_argument('--trajectories-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--provenance',type=Path,default=Path(__file__).with_name('control-originals.json'))
    args=ap.parse_args()
    provenance=json.loads(args.provenance.read_text());verified=0
    def check(path,size,sha):
        nonlocal verified
        assert path.stat().st_size==size,path
        digest=hashlib.sha256()
        with path.open('rb')as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
        assert digest.hexdigest()==sha,path
        verified+=1
    for item in provenance['members']:
        if item['required_for_recompute']:check(args.originals_dir/item['local_relative_path'],item['size'],item['sha256'])
    for item in provenance['trajectories']:check(args.trajectories_dir/item['local_name'],item['size'],item['sha256'])
    result={'source_commit':'3e3f69c440d6a33c14131563e5f485d88da3a803','verified_input_files':verified,
        'method':'Read-only JSON/NPZ array arithmetic; no simulator, solver, learned policy execution or new experiment.',
        'interpretation_limits':[
            'Large K-delta-x depends on the internal affine reference; it is not alone a task-failure or acceptance criterion. Inspect post-projection adjacent target/torque changes and actual-minus-accepted differences.',
            'Native accepted rollout is not policy reference. prediction-000.npz stores the feedback reference, not the accepted rollout; this audit uses complete accepted arrays.',
            'Short no-foot intervals are allowed by the original gait. They are reported as contact timing, never a new hard failure.',
            'A148 is a comparatively mild first-20ms contrast, not a demonstrated stable full-350ms nominal.',
            'Recorded qpos/qvel are insufficient for exact historical physics replay; warmstart/private/material state is not fully serialized.',
            'The early events do not establish that one event is a sufficient cause of the eventual body fall.'
        ],
        'whole_runs':{},'fixtures':[]}
    for case in ['w45-a','w45-b']:
        d=args.originals_dir/case;tr=np.load(args.trajectories_dir/f'{case}-trajectory.npz')
        whole,calls,execution,feedback,phy=whole_run(case,d,tr);result['whole_runs'][case]=whole
        for c,call,pid in FIXTURES:
            if c==case:result['fixtures'].append(fixture(c,call,pid,d,tr,calls,execution,feedback,phy))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'cases':len(result['fixtures']),
        'maximum_policy_identity_error':max(max(f['selected_installed_max_errors'].values())for f in result['fixtures']),
        'maximum_cost_recompute_error':max(f[w]['cost_recompute_error']for f in result['fixtures']for w in ['best_nominal','selected'])},indent=2))

if __name__=='__main__':main()
