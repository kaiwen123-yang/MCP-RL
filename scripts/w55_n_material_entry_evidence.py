"""Decompose the recorded stalled candidate family at the r54 material entry."""
import collections, json, xml.etree.ElementTree as ET
import numpy as np
import mujoco
from analyze_s1_w55_n import read_policy,records
from run_s1_w55_n import SMALL,LARGE,write

def main():
    case=json.loads((SMALL/'N-A-r54/E30.json').read_text());attempt=case['attempt']
    out=LARGE/attempt;native=out/'w54-n-native';calls=list(records(native/'calls.jsonl'))
    key=next(c for c in calls if c['full_candidates'] and c['time_s']>16.0)
    start=9.5;end=key['time_s'];win=[c for c in calls if start<=c['time_s']<=end+1e-8]
    candidates=[a for c in win for a in c['candidates']]
    forward=[a for a in candidates if a['dx_m']>0.]
    execute=[e for e in records(out/'w54-n-execution.jsonl') if start<=e['time_s']<=end+1e-8]
    d=np.array([e['diagnostic'] for e in execute]);active=(d[:,:,4]>0).any(axis=1)
    change=np.array([np.max(np.abs(np.array(e['raw_action'])-e['applied_action'])) for e in execute])
    support=active[:,None]&(d[:,:,10]>0)
    xml=out/'model/G1-project-go2.xml';m=mujoco.MjModel.from_xml_path(str(xml))
    sensors=[x for x in ET.parse(xml).getroot().find('sensor') if x.tag=='user']
    resolved=json.loads((SMALL/(attempt+'-resolved.json')).read_text());weights=resolved['applied_settings'][-1]['cost_weights']
    def term_values(saved):
        residual=saved['accepted_residual'];offset=0;terms={}
        for sensor in sensors:
            dim=int(sensor.get('dim'));params=list(map(float,sensor.get('user').split()));x=residual[:,offset:offset+dim];offset+=dim
            norm=int(params[0]);square=np.sum(x*x,axis=1)
            if norm==0:value=square*.5
            elif norm==2:value=np.sqrt(square+params[4]**2)-params[4]
            elif norm==6:value=(np.sqrt(x*x+params[4]**2)-params[4]).sum(axis=1)
            else:raise ValueError(norm)
            terms[sensor.get('name')]=float(np.mean(value)*weights[sensor.get('name')])
        return terms
    i=key['call_id'];accepted=read_policy(native/f'policy-{i:06d}.bin');ref=term_values(accepted)
    rows=[]
    errors=[]
    for a in key['candidates']:
        slot=a['slot'];path=native/f'call-{i:06d}-candidate-{slot:02d}.bin'
        saved=read_policy(path);terms=term_values(saved);cmd_excess=None
        if a['first_constraint_code']==4:
            node=np.searchsorted(saved['accepted_times'],a['first_constraint_time_s'])-1
            u=saved['applied_actions'][node]
            excess=np.maximum(m.actuator_ctrlrange[:,0]-u,u-m.actuator_ctrlrange[:,1])
            leg=int(excess.argmax())
            cmd_excess=dict(node=int(node),node_time=saved['accepted_times'][node],actuator=m.actuator(leg).name,
                applied_rad=float(u[leg]),original_interval_rad=m.actuator_ctrlrange[leg].tolist(),excess_rad=float(excess[leg]))
        errors.append(abs(sum(terms.values())-np.mean(saved['accepted_costs'])))
        rows.append(dict(slot=slot,generator=a['generator'],J=a['J'],valid=not a['failure'],dx=a['dx_m'],final_z=a['final_z_m'],
            first_constraint=a['first_constraint_code'],first_constraint_time=a['first_constraint_time_s'],
            weighted_terms=terms,weighted_delta_vs_installed={k:terms[k]-ref[k] for k in terms},command_violation=cmd_excess))
    contexts=list(records(out/'w54-n-context.jsonl'))
    observed=[x for x in contexts if start<=x['time_s']<=end+1e-8]
    output=dict(source_attempt=attempt,source_wall='natural running r54 held actions and native files; full ring was retained at the observed persistent stall without changing simulation inputs',
        source_key_call=i,source_key_time=end,interval=[start,end],original_goal='outbound',real_cases=len(win),
        installed_sources=dict(collections.Counter(c['selection_source'] for c in win)),
        actual_holds=len(d),E_active_holds=int(active.sum()),E_changed_holds=int((change>1e-6).sum()),
        min_active_predicted_support_change_N=float(np.min((d[:,:,17]-d[:,:,16])[support])) if support.any() else None,
        measured_body_x_range_m=[min(x['qpos'][0] for x in observed),max(x['qpos'][0] for x in observed)],
        commanded_vx_range_m_s=[min(x['command_velocity'][0] for x in observed),max(x['command_velocity'][0] for x in observed)],
        all_recorded_candidates=len(candidates),forward_candidates=len(forward),
        forward_by_first_constraint=dict(collections.Counter(a['first_constraint_code'] for a in forward)),
        installed_policy=dict(source=key['selection_source'],slot=key['winner'],J=accepted['accepted_J'],
            dx=accepted['accepted_states'][-1,0]-accepted['accepted_states'][0,0],terms=ref),
        full_candidates=rows,maximum_cost_reconstruction_error=max(errors),
        revision='Keep the original comparator/plant and the r54 recovery. Supply one lower-support-effort, task-directed alternative and canonicalize mapped controls inside the original bound.')
    write(SMALL/'r54-material-entry-candidates.json',output)
if __name__=='__main__':main()
