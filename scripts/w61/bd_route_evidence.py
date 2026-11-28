"""Measure a complete observed paired route at its original force and controller clocks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from gait_reachable_evidence import score


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def sections(fz,positions):
    stride=[]
    for leg in range(4):
        bearing=fz[:,leg]>1.
        transitions=np.flatnonzero(~bearing[:-1]&bearing[1:])+1
        for i in transitions[transitions+100<=len(fz)]:
            stride.append((float(fz[i:i+100,leg].max()),float(positions[i,leg,0])))
    return {k:{'n':len(v),'p95_p99_N':np.percentile(v,[95,99]).tolist() if v else None}
        for k,v in [('all',[f for f,x in stride]),('material_strip',[f for f,x in stride if .8<=x<=2.2]),
                    ('rigid_along_x',[f for f,x in stride if not .8<=x<=2.2])]}


def collect(args):
    name=args.name;root=args.root.resolve();raw=root/'raw'/name
    src=root/'episodes'/(name+'.json');r=json.loads(src.read_text())
    trace=np.load(raw/'trajectory.npz')
    cols=r['physical_columns']
    physical=np.concatenate([np.load(f)['physical'] for f in sorted(raw.glob('physical-*.npz'))],axis=0)
    z=physical[:,[cols.index('grf_'+leg+'_z') for leg in ('FL','FR','RL','RR')]]
    xyz=physical[:,[cols.index('foot_position_'+leg+'_'+c+'_m') for leg in ('FL','FR','RL','RR') for c in 'xyz']].reshape(-1,4,3)
    t=trace['time'];dt=np.diff(t,append=t[-1])
    vel=trace['qvel'];power=trace['torque']*vel[:,6:]
    q=trace['qpos'];travel=np.hypot(*np.diff(q[:,:2],axis=0).T).sum()
    feedback=trace['feedback_s'];feedback=feedback[np.isfinite(feedback)]
    impact=sections(z,xyz)
    renderer=importlib.util.spec_from_file_location('stride_evidence',args.renderer)
    render_module=importlib.util.module_from_spec(renderer);renderer.loader.exec_module(render_module)
    video=render_module.render(raw,{'plant_xml':str(raw/'model/G1-project-go2.xml'),'physical_s':r['simulation_s']},trace)
    target=score(raw,src,args.protocol)
    out=args.out/name;out.mkdir(exist_ok=True,parents=True)
    samples=dict(force_samples=len(physical),control_samples=len(t))
    assignment=json.loads((root/'jobs'/name/'contract.json').read_text())
    reg=assignment['w61']
    legals=r['legal_route']
    row=dict(attempt=name,runtime=json.loads((root/'runtime/pair-runtime.json').read_text()),
        assignment=dict(protocol=assignment['protocol_revision'],case=reg['case'],gait=reg['gait_role'],
          wbc=reg['role'],decoder_seed=reg.get('orbit_decoder_seed'),initial_condition=reg.get('initial_condition','nominal'),
          physical_window_s=reg.get('observation_window_s') or 60.,
          qualification=r['qualification'].get('role'),observed_stage_progress=r.get('stage_progress')),
        command_contract_sha256=digest(root/'jobs'/name/'contract.json'),actor_scope='Original W44 cap actor for named W59 baseline case',
        reason=r['reason'],status=r['stage_status'],success=r['success'],terminal_error=(r.get('error') or '').splitlines()[-1:],
        physics_s=r['simulation_s'],x_max=r['max_x_m'],trunk_z_min=r['min_body_height_m'],wall_s=r['wall_s'],samples=samples,
        legal_roundtrips=legals['completed_roundtrips'],legal_outbound=legals['legal_outbound_traversals'],
        legal_return=legals['legal_return_traversals'],side_exits=legals['side_exits'],travelled_m=float(travel),
        absolute_joint_work_J=float(np.sum(np.abs(power)*dt[:,None])),
        positive_joint_work_J=float(np.sum(np.maximum(power,0.)*dt[:,None])),
        velocity_tracking_ISE=float(np.sum((trace['command_velocity'][:,:2]-vel[:,:2])**2*dt[:,None])),
        feedback_ms_p50_p95_p99=np.percentile(feedback*1.e3,[50,95,99]).tolist(),
        force_initial20ms_at_5000Hz=impact,full_video=video,reachable_target_counts={k:target[k] for k in
        ('total_targets','target_source','target_pair','target_match','source_events','paired_events','matched_events')},
        files={p.name:dict(size=p.stat().st_size,sha256=digest(p)) for p in
            (raw/'trajectory.npz',raw/'material-events.jsonl',raw/'w55-f/contact-events.jsonl',src)})
    optional=raw/'support-bridges.json'
    if optional.is_file():row['bridges']=json.loads(optional.read_text())
    (out/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    (out/'reachable.json').write_text(json.dumps(target,indent=2,allow_nan=False)+'\n')
    print(name,r['simulation_s'],r['stage_status'],samples,row['reachable_target_counts'],flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('root','out','renderer','protocol'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('name');collect(p.parse_args())
