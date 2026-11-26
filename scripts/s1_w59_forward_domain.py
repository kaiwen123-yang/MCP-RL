"""Forward scientific admission at each physical horizon, independent of reward ordering."""
import json
from pathlib import Path
import numpy as np
import torch
from s1_w59_model_fields import STATE,INTERVAL,MASS

# This is an empirical support scale for prediction error, not a new plant limiter.
FOOT_SUPPORT_N=MASS*9.81/4
FOOT_WIDTH_M=np.sqrt(.001)
TORQUE_UPPER=45.43
EPS=1.e-6


def historical(root,kind,nodes,normal_max):
    files=[root/f'{kind}-{seed}-open-loop.json' for seed in (5901,5902,5903)]
    result={}
    for n in nodes:
        evidence=[]
        for f in files:
            if not f.exists():evidence.append(dict(file=f.name,pass_gate=False,reason='unavailable'));continue
            report=json.loads(f.read_text());horizons=report['per_horizon']
            # Interpolation at 0.35 s is shared evidence for its two bracketing model ticks.
            h=.35 if n in (17,18) else .02*n
            row=next((v for v in horizons if abs(v['horizon_s']-h)<1e-7),None)
            if row is None:evidence.append(dict(file=f.name,pass_gate=False,reason='horizon_unmeasured'));continue
            m=row['metrics'];force=m['foot_fz_MAE_N']['mean'];foot=m['foot_xyz_MAE_m']['mean']
            passed=(row['finite']==row['origins'] and row['observed']>0 and
                    np.isfinite(force) and np.isfinite(foot) and force<=normal_max)
            evidence.append(dict(file=f.name,pass_gate=bool(passed),horizon_s=row['horizon_s'],finite=row['finite'],
                origins=row['origins'],observed=row['observed'],foot_fz_MAE_N=force,foot_MAE_m=foot,
                foot_error_in_footprint_widths=foot/FOOT_WIDTH_M,force_error_in_nominal_single_foot_loads=force/FOOT_SUPPORT_N))
        result[str(n)]=dict(pass_gate=all(e['pass_gate'] for e in evidence),members=evidence)
    return result


def candidate_mask(result,normal_max):
    x,y,active=result['trajectory'][...,1:,:],result['interval'],result['valid_steps'].bool()
    finite=torch.isfinite(x).all(-1)&torch.isfinite(y).all(-1)
    # The realised force envelope comes from the complete W59 physical development ledger.
    # The original WBC total-torque bound and the solved force cone apply independently.
    solved=x[...,STATE['force_solved']].unflatten(-1,(4,3));z=solved[...,2]
    tol=EPS*(1.+solved.abs().amax(dim=(-2,-1)))
    cone=(solved[...,:2].abs()<=z[...,None]*(.5/(2.**.5))+tol[...,None,None]).all(-1).all(-1)
    normal=(z>=-tol[...,None]).all(-1)&(z<=normal_max+tol[...,None]).all(-1)
    measured=x[...,STATE['measured_force']].unflatten(-1,(4,3))[...,2]
    # The unilateral constraint applies to solved contact forces. Measured residual
    # predictions are also checked against the support actually observed in development.
    force_support=(measured.abs()<=normal_max+EPS).all(-1)
    load=(y[...,INTERVAL['load_mean']].abs()<=normal_max+EPS).all(-1)
    torque=y[...,INTERVAL['torque_peak_Nm']].squeeze(-1)<=TORQUE_UPPER+EPS
    lower=x.new_tensor([-1.0472,-1.5708,-2.7227]*4)
    upper=x.new_tensor([1.0472,3.4907,-.83776]*4)
    command=x[...,STATE['held_command']]
    joint=x[...,STATE['qpos']][...,7:]
    joint_domain=((joint>=lower-EPS)&(joint<=upper+EPS)&(command>=lower-EPS)&(command<=upper+EPS)).all(-1)
    ok=(~active)|(finite&cone&normal&force_support&load&torque&joint_domain)
    return ok.all(dim=0).all(dim=-1)
