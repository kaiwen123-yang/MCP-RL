"""Verify the two executed observation modes using each actor's saved W44 observations."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from run_s1_w55_f import ROOT, LARGE, digest, write
from s1_go2_r11_policy import PolicyMaterial, PolicyStream
from s1_go2_r11_runtime import Mission


def verify(case):
    seed=1 if case=='A' else 2
    base=LARGE/'training'/('w44-'+case.lower())/f'PANN_PPO_S5-seed{seed}'
    checkpoint=base/'checkpoint-014737.pt'
    contract=json.loads((base/'training-contract.json').read_text())
    config=json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
    config['pann']['weights_path']=str(ROOT/'work/assets/B_random1000--ideal-potentials.npz')
    if 'joint_learning' not in contract:
        contract['joint_learning']=json.loads((ROOT/contract['reuse']['runtime_physics_ppo_and_task_settings']).read_text())['joint_learning']
    task=json.loads((ROOT/'experiments/terrain_memory/s1-go2-wb-support.json').read_text())['task']
    source=base/'rollout-014226-014481.npz'
    with np.load(source) as data:
        samples=data['obs'].reshape(-1,658)
    index=int(np.argmax(abs(samples[:,-12:]).max(1)))
    raw=samples[index]
    mission=Mission(task)
    rows={}
    with tempfile.TemporaryDirectory(dir=ROOT/'work/w56-f') as testdir:
        stream=PolicyStream(ROOT,'PANN_PPO_S5',seed,contract,Path(testdir),training=False,checkpoint=checkpoint)
        joint=PolicyMaterial(ROOT,config,stream,0,30,full_recording=False)
        for mode in ('learned_history','zero_recovery_features'):
            joint.material_feature_mode=mode
            selected=joint.actor_observation(raw)
            expected=raw.copy()
            if mode=='zero_recovery_features':
                expected[-12:]=0
            assert np.array_equal(expected,selected) and np.any(raw[-12:])
            reference=.215+.035*float(stream.backend.act(torch.as_tensor(expected,device=stream.backend.config.device).reshape(1,-1),deterministic=True).action.item())
            joint.install_cap(0.,mission,raw,selected)
            observed=mission.task['command_speed_m_s']
            assert observed==reference==joint.cap==joint.pending['cap']
            assert np.array_equal(joint.pending['obs'].cpu().numpy()[0],expected)
            rows[mode]=dict(executed_cap_m_s=observed,independent_original_actor_cap_m_s=reference,
                            changed_indices=np.flatnonzero(selected!=raw).tolist(),record=joint.action_dependence[-1])
        stream.log.close()
        stream.learning.close()
    row=dict(case=case,seed=seed,checkpoint_sha256=digest(checkpoint),source=str(source.relative_to(ROOT)),
             rollout_sample=index,raw_recovery_tail=raw[-12:].tolist(),observations=658,
             unchanged_body_history_dimensions=646,checks=rows,both_modes_verified=True,
             qualification='Saved-input execution of actor and mission cap assignment; no physics integrated.')
    write(ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f'/f'r57-{case.lower()}-actor-selection.json',row)
    print(case,{k:v['executed_cap_m_s'] for k,v in rows.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=('A','B'))
    verify(p.parse_args().case)
