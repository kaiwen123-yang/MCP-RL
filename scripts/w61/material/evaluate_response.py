"""Evaluate held-command and force-residual rollouts on independent whole W61 solver trajectories."""
import hashlib
import json
import time
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from s1_w59_model_fields import NS,STATE,DT
from s1_w59_model_rigid import RigidLinearization
from response_model import Response
from train_response import OUT,RAW,SPEC,write

MPC=ROOT.parent/'MCP-RL-w61-mpc/work/w61/mpc/w61-c-transfer-wall-v1'
MODEL=Path('/root/autodl-tmp/MCP-RL-w57/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57-m1-b-a-original-e30-a01/model/G1-project-go2.xml')
ORIGINS=(.60,1.16,1.74,2.32,2.90)


def fk(engine,state):
    m,d=engine.model,engine.data
    d.qpos[:]=state[STATE['qpos']];d.qvel[:]=state[STATE['qvel']]
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    jac=np.empty((3,m.nv))
    vel=[]
    for g in engine.gids:
        mujoco.mj_jacGeom(m,d,jac,None,g);vel.append(jac@d.qvel)
    state[STATE['feet']]=d.geom_xpos[engine.gids].ravel()
    state[STATE['foot_velocity']]=np.asarray(vel).ravel()
    return state


def one(member,learned,engine,log,i,horizons):
    state=np.zeros(NS,dtype='f')
    for k,field in [('qpos','qpos'),('qvel','qvel'),('measured_force','force'),('command','requested_velocity')]:
        state[STATE[k]]=log[field][i].ravel()
    state[STATE['held_command']]=log['ctrl'][i-1]
    state[STATE['time']]=log['time'][i]
    state=fk(engine,state)
    origin_fk_Linf=float(np.max(abs(state[STATE['feet']]-log['foot_xyz'][i].ravel())))
    predicted=[state.copy()]
    cause=None
    for it in range(1,101):
        try:
            mat=engine(state).astype('f')
            if not np.isfinite(mat).all():raise FloatingPointError('nonfinite rigid operation')
            out,_,_=member(torch.from_numpy(state[None]),torch.tensor([[0.,.050,0.]]),
                           torch.from_numpy(mat[None]),learned=learned)
            state=out[0].numpy()
            if not np.isfinite(state).all():raise FloatingPointError('nonfinite transition')
            state=fk(engine,state)
            if not np.isfinite(state).all():raise FloatingPointError('nonfinite kinematics')
        except (np.linalg.LinAlgError,FloatingPointError) as err:
            cause=dict(tick=it,error=str(err))
            break
        predicted.append(state.copy())
    pred=np.asarray(predicted).astype(float);rows=[]
    for h in horizons:
        pos=h/DT;lo=int(np.floor(pos+1e-7));hi=int(np.ceil(pos-1e-7));alpha=pos-lo
        j=i+round(h/.002)
        enough=hi<len(pred)
        seen=j<len(log['time'])
        r=dict(horizon=h,observed=seen,finite=bool(enough))
        if enough:
            s=(1.-alpha)*pred[lo]+alpha*pred[hi]
            joints=s[STATE['qpos']][7:]
            r['hard_joint_domain']=bool(np.all(joints>=member.lower.numpy()-1e-6) and np.all(joints<=member.upper.numpy()+1e-6))
            if seen:
                feet=s[STATE['feet']].reshape(4,3)
                r.update(foot_xyz_MAE_m=float(np.mean(abs(feet-log['foot_xyz'][j]))),
                         normal_force_MAE_N=float(np.mean(abs(s[STATE['measured_force']].reshape(4,3)[:,2]-log['force'][j,:,2]))),
                         height_error_m=float(abs(s[STATE['qpos']][2]-log['qpos'][j,2])),
                         joint_velocity_MAE_rad_s=float(np.mean(abs(s[STATE['qvel']][6:]-log['qvel'][j,6:]))))
        rows.append(r)
    return dict(origin_s=float(log['time'][i]),origin_FK_max_error_m=origin_fk_Linf,termination=cause,scores=rows)


def main():
    cfg=json.loads(SPEC.read_text());torch.set_num_threads(1)
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    binary=RAW/'C-xml.mjb';mujoco.mj_saveModel(model,str(binary))
    engine=RigidLinearization(binary)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for method in ('hybrid','mppi','local','vr_mppi'):
        case=MPC/('query1-'+method)
        while not (case/'result.json').exists():time.sleep(15)
        traj=case/'trajectory.npz'
        with np.load(traj) as z:log={k:z[k] for k in ('time','qpos','qvel','force','foot_xyz','ctrl','requested_velocity')}
        for seed in cfg['seeds']:
            while not (OUT/f'fit-{seed}.json').exists():time.sleep(5)
            cp=RAW/f'physical-response-{seed}.pt'
            d=torch.load(cp,map_location='cpu',weights_only=True)
            member=Response(d['model']['mean'],d['model']['scale'])
            member.load_state_dict(d['model']);member.eval()
            for learned in ((False,True) if seed==cfg['seeds'][0] else (True,)):
                tag=f'{method}-'+(f'residual-{seed}' if learned else 'held')
                receipt=OUT/(tag+'.json')
                if receipt.exists():continue
                tm=time.perf_counter()
                with torch.inference_mode():
                    rows=[one(member,learned,engine,log,int(round(t/.002)),cfg['horizons_s']) for t in ORIGINS if t<=log['time'][-1]]
                write(receipt,dict(runtime=rev,kind='residual' if learned else 'held',seed=seed if learned else None,
                    original_trajectory=str(traj),trajectory_sha256=hashlib.sha256(traj.read_bytes()).hexdigest(),
                    MJB_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),XML_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest() if learned else None,
                    registered_origins_s=ORIGINS,actual_origins=len(rows),rows=rows,wall_s=time.perf_counter()-tm,
                    gradients='CPU MuJoCo M/J/h and FK detached; no planning direction or path-gradient qualification',
                    scope='known rigid execution on an independently generated W61 controller; absolute 500Hz physical endpoint, same held command and nominal future request assumption'))
                print(tag,len(rows),flush=True)


if __name__=='__main__': main()
