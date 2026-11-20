"""Replay the first r55 A/E30 replanning tick with confirmed soft/all-stance WBC."""
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from s1_w54_f_r52_backend import assemble_centroidal
from s1_w55_f_r55_backend import FController as R55
from s1_w56_f_r58_backend import FController as R58
from s1_w56_window import WindowObserver
from tm02.s1_contact import Ground
from tm02.s1_material_prediction import normal_reference, rollout_candidate

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
CASE = 'job-20261120T035811Z-w55-f-a-r55-eval-PANN_PPO_S5-s1-E30-attempt01'
SOURCE = ROOT.parent/'MCP-RL-w55-f'/'outputs'/RUN/CASE
OUTPUT = ROOT/'results'/RUN/'w56/c'
CFG = json.loads((ROOT/'experiments/terrain_memory/s1-w56-f-r58.json').read_text())
PARAMETERS = json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']


def observation(t):
    obs = WindowObserver(PARAMETERS,ROOT/'work/assets/B_random1000--ideal-potentials.npz')
    first = json.loads((SOURCE/'material-events.jsonl').open().readline())['time_s']
    # All cells are virgin before the first actual compression. Include the
    # entire preceding 2ms packet, with measured foot geometry in every sample.
    start = np.floor((first-.0002)/.002)*.002
    obs.time = start
    filename = SOURCE/f'material-{int(start):04d}.npz'
    with np.load(filename) as f:
        foot = f['foot']
        rows = foot[(foot[:,0]>=start-1e-9)&(foot[:,1]<=t+1e-9)]
        all_cells = f['cells']
        truth = all_cells[np.isclose(all_cells[:,0],t,rtol=0,atol=1e-8)]
    for j,r in enumerate(rows):
        obs.completed(r[0],r[1],r[6:18].reshape(4,3),r[18:30].reshape(4,3))
        if (j+1)%10==0:
            obs.observe(r[1],None,None)
    if not np.isclose(obs.time,t,rtol=0,atol=1e-9):
        raise ValueError('Saved packet did not reach the WBC consumption time')
    oracle = Ground({'id':'tm02_anchor'},PARAMETERS)
    oracle.law = obs.ground.law
    for key,idx in obs.ground.ids.items():
        oracle.footprint(np.asarray(obs.ground.xy[idx]))
    for row in truth:
        key = tuple(row[1:3].astype(int))
        if key in oracle.ids:
            i=oracle.ids[key]; oracle.e[i],oracle.q[i],oracle.stress[i]=row[3:5],row[5:7],row[7:9]
    return obs,oracle,rows[-1,6:18].reshape(4,3)


def solve(ctrl, model, data, p, row, mesh, obs, bottom, t):
    feet=np.array([model.geom(x).id for x in ['FL','FR','RL','RR']])
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    mujoco.mj_forward(model,data)
    with tempfile.TemporaryDirectory(dir=ROOT/'work/w56-c') as out:
        controller=ctrl(model,data,feet,model.body('trunk').id,Path(out),config=CFG)
        data.qpos[:]=p['qpos'];data.qvel[:]=p['qvel'];data.time=t
        model.geom_contype[feet]=np.where(mesh,0,model.geom_contype[feet])
        model.geom_conaffinity[feet]=np.where(mesh,0,model.geom_conaffinity[feet])
        mujoco.mj_forward(model,data)
        info=dict(material_observer=obs,force_boundary_bottom=bottom,stiffness_N_m=CFG['material_stiffness_N_m'])
        controller.current_material_info=info
        controller.current_soft=mesh
        mq=controller.model_quantities(data)
        # Restore recorded force points exactly: native contact generation at a
        # saved boundary can otherwise depend on the preceding mj_step.
        mq['contact_force_point']=row['contact_force_point']
        mq['Jforce']=row['J_contact_force']
        controller.anchor=row['stance_anchor'];controller.confirmed=row['confirmed_contact']
        controller.load_bearing=row['load_bearing_contact'];controller.moving=True
        controller.next_pair=np.setdiff1d(np.arange(4),row['transfer_support_pair'])
        controller.transfer_fraction=row['transfer_fraction']
        controller.transfer_start=t if controller.transfer_fraction else None
        controller.event_mode=row['event_mode']
        blocks=assemble_centroidal(controller.mass,p['inertia'],p['com'],p['feet'][0],p['x0'][:3],p['x0'][6:9],
                                  friction=CFG['force_pyramid_slope'])
        controller.last_plan=dict(time=t,forces=p['forces'],reference=p['state_reference'],state=p['states'],blocks=blocks)
        action=controller.wbc(t,data,mq,SimpleNamespace(phase='outbound'),row['planned_contact'],row['observed_contact'],
                              row['measured_grf'],mesh,info)
        result=controller.rows[-1]
        return dict(action=action, reference=result['feet_position_reference'],force=result['force'],
                    primal=float(result['solver_primal_residual']), dual=float(result['solver_dual_residual']))


def main():
    start=time.time()
    tick=210
    with np.load(SOURCE/'w55-f/control-00006.npz') as z:
        row={k:z[k][tick] for k in z.files}
    t=float(row['time']);num=int(row['plan_call'])
    with np.load(SOURCE/'w55-f'/f'mpc-{num:06}.npz') as z:
        p=dict(z)
    obs,oracle,bottom=observation(t)
    soft=row['contact_point_source']==2
    model=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'))
    data=mujoco.MjData(model)
    answers=[solve(ctrl,model,data,p,row,soft,obs,bottom,t) for ctrl in (R55,R55,R58)]
    a,b,c=answers
    maxdiff=lambda x,y:float(np.max(np.abs(x-y),initial=0.))
    learned=obs.ground
    active=np.flatnonzero((row['mode']==3)&soft)
    leg=int(active[0])
    target=float(row['redistributed_force_target'][leg,2]);actual=float(row['measured_grf'][leg,2])
    zero_increment=float(normal_reference(learned,bottom,leg,actual,actual,.002)[0])
    empty=Ground({'id':'tm02_anchor'},PARAMETERS);empty.law=learned.law
    modes={'learned_history':learned,'zero_current_history':empty,'oracle_initial_state_diagnostic_only':oracle}
    traces={}; finals={}; requests=[]
    xy=p['foot_center_plan'][:-1,:,:2];forces=p['forces'][:,:,2].copy()
    if np.min(forces) < -CFG['qp']['eps_abs']:raise ValueError('Saved MPC negative force exceeds primal tolerance')
    # Negative roundoff on open contact rows appears in the solver output.
    # Use the same nonnegative WBC target convention as the actual controller.
    forces=np.maximum(forces,0.)
    unchanged={k:v.copy() for k,v in [('e',learned.e),('q',learned.q)]}
    for label,g in modes.items():
        branch,trace=rollout_candidate(g,t,p['times'],xy,p['contact_schedule'],forces)
        traces[label]=trace
        finals[label]=branch
    # Independent alternate load request on the same causal clock, then repeat original.
    alt=forces.copy();alt[0,leg]=actual
    for f in (alt,forces):
        branch,trace=rollout_candidate(learned,t,p['times'],xy,p['contact_schedule'],f)
        requests.append((branch,trace))
    learned_ref=finals['learned_history']; alternate,repeated=requests[0][0],requests[1][0]
    cells=np.array(sorted(learned_ref.prediction_active),int)
    history_effects={}
    for label,trace in traces.items():
        diff=[maxdiff(np.asarray(a['displacement_m']),np.asarray(b['displacement_m']))
              for a,b in zip(trace,traces['learned_history'],strict=True)]
        history_effects[label]=dict(max_displacement_difference_vs_learned_m=max(diff),
                                    maximum_force_residual_N=max((max(map(abs,n['force_residual_N']),default=0) for n in trace)))
    values=dict(schema='w56_r58_material_execution_qualification_v1',source_attempt=CASE,
        source_mpc=num,source_control_file='control-00006.npz',row=tick,consume_s=t,publish_s=obs.time,
        observer_first_boundary_s=obs.packet_start,observer_last_boundary_s=obs.packet_last,
        observer_windows=obs.windows,soft_leg=leg,measured_N=actual,target_N=target,
        r55_repeat_mapped_input_rad=maxdiff(a['action'],b['action']),
        archived_r55_vs_reconstructed_mapped_input_rad=maxdiff(a['action'],row['mapped_u']),
        archived_r55_vs_reconstructed_reference_m=maxdiff(a['reference'],row['feet_position_reference']),
        r58_vs_r55_mapped_input_rad=maxdiff(c['action'],a['action']),
        r58_vs_r55_soft_ref_m=maxdiff(c['reference'],a['reference']),
        r58_vs_r55_wbc_force_N=maxdiff(c['force'],a['force']),
        numerical_residuals={label:[x['primal'],x['dual']] for label,x in zip(['r55','r58'],[a,c])},
        zero_force_difference_increment_m=zero_increment,history_comparisons=history_effects,
        candidate_nodes=len(p['times'])-1,joined_world_cells=len(cells),per_cell_forecast_steps=np.unique(learned_ref.prediction_last[cells]).tolist(),
        candidate_after_alternate_repeat_q=maxdiff(repeated.q,learned_ref.q),
        candidate_changed_load_final_q=maxdiff(alternate.q,learned_ref.q),
        source_state_change=max(maxdiff(getattr(learned,k),v) for k,v in unchanged.items()),
        candidate_nodes_by_history=traces,wall_s=time.time()-start)
    values['ready_for_original_E30']=bool(values['r58_vs_r55_mapped_input_rad']>max(values['r55_repeat_mapped_input_rad'],
        values['archived_r55_vs_reconstructed_mapped_input_rad']) and values['r58_vs_r55_soft_ref_m']>0 and
        zero_increment==0. and values['candidate_after_alternate_repeat_q']==0. and
        values['source_state_change']==0. and values['candidate_changed_load_final_q']>0. and
        values['per_cell_forecast_steps']==[len(p['times'])-1])
    dest=OUTPUT/'interface-qualification.json';temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(values,indent=2,allow_nan=False)+'\n');temp.replace(dest)
    print({k:v for k,v in values.items() if k!='candidate_nodes_by_history'},flush=True)


if __name__=='__main__':main()
