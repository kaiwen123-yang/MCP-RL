"""Independent four-point native-action check, using already frozen directions."""
from pathlib import Path
import json
import numpy as np
import torch
from audit_existing_neural_feedback import (ROOT, OUT, WORK, SEEDS, ARMS, INDICES,
                                           KEYS, NativeGeometry, mean, sha, text_sha, plain)

H, EPS = 1e-4, 1e-5


def signature(geometry, x, z, command, design):
    limits, znext = geometry.interval(x, z, command, design)
    bvec, P = geometry.env.B, geometry.P
    offset = geometry.env.f0(x)-znext
    ep = torch.stack((offset-geometry.env.E*geometry.cfg.disturbance_bound,
                      offset+geometry.env.E*geometry.cfg.disturbance_bound), 1)
    a = bvec@P@bvec
    b = 2*torch.einsum('i,ij,bkj->bk', bvec, P, ep)
    c = torch.einsum('bki,ij,bkj->bk', ep, P, ep)
    c -= (geometry.cfg.rho*torch.einsum('bi,ij,bj->b', x-z, P, x-z)+geometry.cfg.delta)[:, None]
    disc = b.square()-4*a*c
    sq = torch.sqrt(disc.clamp_min(0))
    q = -.5*(b+torch.where(b>=0, sq, -sq))
    r1 = q/a
    r2 = torch.where(q != 0, c/torch.where(q != 0, q, 1.), -b/(2*a))
    lower = torch.cat((torch.minimum(r1,r2), torch.full((len(x),1), -geometry.cfg.input_bound)), -1)
    upper = torch.cat((torch.maximum(r1,r2), torch.full((len(x),1), geometry.cfg.input_bound)), -1)
    lo = torch.where(limits.feasible, lower.argmax(-1), -1)
    hi = torch.where(limits.feasible, upper.argmin(-1), -1)
    return torch.stack((limits.feasible.long(), lo, hi), -1)


def native_action(geometry, parameters, x, z, command, design):
    obs, limits, znext = geometry.observations(x,z,command,design)
    s = torch.tanh(mean(parameters, obs))
    if not geometry.support:
        return s
    mid = torch.where(limits.feasible,(limits.lo+limits.hi)/2,0.)
    half = torch.where(limits.feasible,(limits.hi-limits.lo)/2,0.)
    geometry.env.x, geometry.env.batch = x,len(x)
    return torch.where(limits.feasible,mid+half*s,geometry.env.fallback(znext))/geometry.cfg.input_bound


def main():
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            path=ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}/{arm}.pt'
            original_hash=sha(path)
            state=torch.load(path,map_location='cpu',weights_only=True)['actor']
            p=torch.cat([state[k].reshape(-1) for k in KEYS]).double()
            with np.load(WORK/f'direction-{seed}-{arm}.npz',allow_pickle=False) as d:
                v=torch.tensor(d['direction']); expected=d['anchor_feedback_jvp']
            with np.load(WORK/f'replays/{seed}-{arm}.npz',allow_pickle=False) as d:
                geometry=NativeGeometry(d,arm.endswith('support'))
                x,z,c,design=[torch.tensor(d[name].reshape(-1,size)[INDICES],dtype=torch.float64)
                              for name,size in [('x',3),('z',3),('command',2),('raw_design',4)]]
            baseline=signature(geometry,x,z,c,design)
            fd=np.zeros((len(x),3)); stable=np.zeros((len(x),3),dtype=bool)
            for axis in range(3):
                dx=torch.zeros_like(x); dx[:,axis]=EPS*geometry.scale[axis]
                stable[:,axis]=((signature(geometry,x+dx,z,c,design)==baseline).all(-1)
                                &(signature(geometry,x-dx,z,c,design)==baseline).all(-1)).numpy()
                plus=(native_action(geometry,p+H*v,x+dx,z,c,design)-native_action(geometry,p+H*v,x-dx,z,c,design))
                minus=(native_action(geometry,p-H*v,x+dx,z,c,design)-native_action(geometry,p-H*v,x-dx,z,c,design))
                fd[:,axis]=((plus-minus)/(4*H*EPS)).numpy()
            error=fd[stable]-expected[stable]
            relative=float(np.linalg.norm(error)/max(np.linalg.norm(expected[stable]),1e-300))
            assert sha(path)==original_hash
            row={'seed':seed,'arm':arm,'checkpoint_unchanged':True,
                 'stable_coordinate_count':int(stable.sum()),'cross_branch_coordinate_count':int((~stable).sum()),
                 'max_abs_error_stable':float(np.max(np.abs(error))) if len(error) else None,
                 'relative_L2_error_stable':relative,'passes_declared_check':bool(relative<=1e-3),
                 'parameter_step':H,'normalized_state_step':EPS,'batched_native_action_calls':12}
            results.append(row); print(json.dumps(row),flush=True)
    r={'protocol_content_sha256':text_sha(OUT/'physical-map-check-protocol.md'),
       'script_content_sha256':text_sha(Path(__file__)),'cases':results,
       'summary':{'pass_count':sum(c['passes_declared_check'] for c in results),
                  'total_cases':len(results),'batched_native_action_calls':144,
                  'scalar_native_action_evaluations':144*128,'new_environment_transitions':0,
                  'MPC_calls':0,'training_updates':0,'policy_updates':0,'checkpoint_writes':0}}
    (OUT/'physical-map-check.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
