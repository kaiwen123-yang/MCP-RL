"""Complete pre-control state for the tick which actually published a legal request."""
import gzip
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import mujoco
import numpy as np
import torch


def fields(owner, exclude):
    return {key:value for key,value in vars(owner).items() if key not in exclude}


def integration(model, data):
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    buff = np.empty(mujoco.mj_stateSize(model,flag))
    mujoco.mj_getState(model,data,buff,flag)
    return buff


GROUND_LOGS = {'law','rate_hist','events'}
JOINT_EXCLUDE = {'model','data','native_model','native_data','fk_data','events_file','failures_file',
    'root','out','stream','agent','observer','ground','mission','records','cells','chunks','cap_decisions',
    'action_dependence','w59_learner'}
CONTROLLER_EXCLUDE = {'model','out','record_dir','cfg','settings','manifest','rows','events','timing',
    'drift_rows','current_material_info','current_mq','current_mission','snapshot_buffer','branch_override',
    'execution_wm','w59_learner','w59_joint','w59_data'}


def pre_tick(agent,joint,mission,step,previous_phase,first_completion,evaluation_route):
    # Every later legal publication follows a completed transfer and due time.
    # Store the state before the whole 2 ms tick, preserving its normal actor/mission/order.
    t = float(joint.data.time)
    agent.snapshot_buffer = None
    if agent.next_liftoff is None or agent.transfer_start is None:
        return
    if t+1e-9 < max(agent.next_liftoff,agent.transfer_start+agent.macro_request[1]):
        return
    observer = joint.observer
    state = dict(schema='W59_pre_control_legal_request_v1', time_s=t,step=step,
        previous_phase=previous_phase,first_completion=first_completion,
        mission=vars(mission), evaluation_route=evaluation_route,
        model_state=integration(joint.model,joint.data),
        contype=joint.model.geom_contype.copy(),conaffinity=joint.model.geom_conaffinity.copy(),
        ground=fields(joint.ground,GROUND_LOGS),belief=fields(observer.ground,GROUND_LOGS),
        observer=fields(observer,{'ground','native','marker','m0','m1_ground'}),
        secondary_belief=(dict(observer=fields(observer.m0,{'ground','native','marker'}),
                               ground=fields(observer.m0.ground,GROUND_LOGS)) if hasattr(observer,'m0') else None),
        joint=fields(joint,JOINT_EXCLUDE),
        stream_serial=joint.stream.transition_serial,stream_last_obs=joint.stream.last_observation,
        controller=fields(agent,CONTROLLER_EXCLUDE), current_ramp=agent.cfg['contact_events']['load_transfer_ramp_s'],
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        source_attempt=agent.out.name,source_build=agent.cfg['identity'])
    agent.snapshot_buffer = pickle.dumps(state,protocol=5)


def save_request(agent,t):
    folder = agent.out/'w59-legal-requests'
    folder.mkdir(exist_ok=True)
    name = f'request-{agent.request_count:05d}.pkl.gz'
    path = folder/name
    blob = gzip.compress(agent.snapshot_buffer, compresslevel=1,mtime=0)
    path.write_bytes(blob)
    with (folder/'manifest.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(request_id=agent.request_count,request_s=float(t),file=name,
            size=len(blob),sha256=hashlib.sha256(blob).hexdigest(),incoming=agent.preference_pair.tolist(),
            source=agent.out.name,source_build=agent.cfg['identity'],source_arm=agent.settings['arm'],
            source_request=agent.macro_request.tolist(),source_material=agent.settings['material'],
            source_model_sha256=agent.settings['model_sha256']))+'\n')
    agent.snapshot_buffer = None


def restore(config,agent,joint,mission):
    # Only project-produced snapshot bytes registered in the signed/nominal branch manifest are accepted.
    path = Path(config['snapshot'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != config['sha256']:
        raise ValueError('Saved legal request differs from the frozen registry')
    state = pickle.loads(gzip.decompress(blob))
    if state['schema'] != 'W59_pre_control_legal_request_v1':
        raise ValueError('Invalid request snapshot')
    if state['source_build'] != config['source_build']:
        raise ValueError('Snapshot source controller does not match the registration')
    model=joint.model
    model.geom_contype[:] = state['contype']
    model.geom_conaffinity[:] = state['conaffinity']
    mujoco.mj_setState(model,joint.data,state['model_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    vars(mission).update(state['mission'])
    vars(joint.ground).update(state['ground'])
    vars(joint.observer.ground).update(state['belief'])
    vars(joint.observer).update(state['observer'])
    if state.get('secondary_belief') is not None:
        shadow=state['secondary_belief']
        vars(joint.observer.m0).update(shadow['observer'])
        vars(joint.observer.m0.ground).update(shadow['ground'])
        joint.observer.m1_ground=joint.observer.ground
    vars(joint).update(state['joint'])
    vars(agent).update(state['controller'])
    agent.cfg['contact_events']['load_transfer_ramp_s']=state['current_ramp']
    joint.mission=mission
    joint.stream.transition_serial=state['stream_serial']
    joint.stream.last_observation=state['stream_last_obs']
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    agent.branch_override=(config['request_id'],config['request_s'],config['replacement'])
    return state
