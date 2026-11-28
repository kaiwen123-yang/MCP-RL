"""Real Go2 controller-in-the-loop M1 backend comparison with synchronous measured-force correction."""
import argparse
from collections import defaultdict,deque
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
FROZEN=Path('/root/autodl-tmp/MCP-RL-w59')
sys.path.insert(0,str(FROZEN/'scripts'))
from s1_go2_joint_material import JointMaterial
from s1_go2_r11_runtime import contact_state
from s1_w56_f_r60_backend import FController
from s1_w54_f_r52_backend import ControllerNumericalFailure
from s1_w57_material import W57MaterialObserver
from s1_w59_m1_hybrid import _HybridPacket,BUILD
from tm02.reference import IntegrationFailure
from tm02.s1_contact import StanceReferenceInfeasible

G1=Path('/root/autodl-tmp/MCP-RL-w57/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57-m1-b-a-original-e30-a01/model/G1-project-go2.xml')
PARAMS=FROZEN/'experiments/tm02/r2-multihistory.json'
CFG=FROZEN/'experiments/terrain_memory/s1-w56-f-r60.json'
WEIGHTS=Path('/root/autodl-tmp/MCP-RL/work/assets/B_random1000--ideal-potentials.npz')
LIB=BUILD/'propagate-one-column.so'
LEGS=('FL','FR','RL','RR')
TICK=.0002
DT=.002


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,data):
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class Packets:
    def __init__(self,original,model,arm,table):
        self.cpu=original
        self.gpu=_HybridPacket(model,LIB) if arm!='cpu' else None
        self.arm=arm
        self.prior=table
        self.samples=defaultdict(lambda:deque(maxlen=12))
        self.count=defaultdict(int)
        self.events=[]

    def w57_packet(self,*args):
        columns=int(args[1]);level=int(np.floor(np.log2(max(columns,1))))
        self.count[level]+=1
        if self.arm=='adaptive':
            prior=self.prior.get(str(level),[1.,2.])
            chosen=int(np.argmin(prior))
            pools=[self.samples[(level,b)] for b in (0,1)]
            if self.count[level]==2:chosen=1-chosen
            elif min(map(len,pools))>0:
                # Native synchronous time includes competing-device queue delay.
                chosen=int(np.argmin([np.median(v) for v in pools]))
                if self.count[level]%32==0:chosen=1-chosen
        else:chosen=int(self.arm=='gpu')
        start=time.perf_counter()
        code=(self.gpu if chosen else self.cpu).w57_packet(*args)
        elapsed=time.perf_counter()-start
        self.samples[level,chosen].append(elapsed)
        self.events.append((columns,chosen,elapsed))
        return code


def run(out,arm,case,config,*,controller_factory=FController,observer_factory=W57MaterialObserver):
    out.mkdir(exist_ok=False,parents=True)
    spec=json.loads(config.read_text())
    p=spec['cases'][case]
    start=time.perf_counter();cpu=time.process_time()
    model=mujoco.MjModel.from_xml_path(str(G1));d=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,d,model.key('home').id)
    d.qpos[0]+=p['x_m'];mujoco.mj_forward(model,d)
    feet=np.array([model.geom(l).id for l in LEGS]);trunk=model.body('trunk').id
    allowed={0}|{i for i in range(1,model.nbody) if i!=trunk and model.body_mocapid[i]<0}
    mat=JointMaterial(FROZEN,{})
    mat.attach(model,d,feet,allowed,out,model)
    obs=observer_factory(json.loads(PARAMS.read_text())['reference_material']['parameters'],WEIGHTS,mode='M1')
    router=Packets(obs.native,obs.ground.law.model,arm,spec['historical_A_batch_bin_prior_ms'])
    obs.native=router
    ctrl=controller_factory(model,d,feet,trunk,out,json.loads(CFG.read_text()))
    mission=SimpleNamespace(task={'turnaround_xy_m':[p['x_m']+.4,0],'return_xy_m':[p['x_m'],0]},
            phase='initial_stand',v=np.zeros(3),w=np.zeros(3),heading=0.,body_target=np.array([p['x_m'],0,.27]))
    reason='horizon';trace=[];timing=[];force_rows=[];steps=0
    last_fz=np.zeros(4);energy=0.;slip=0.;tracking=0.
    for tick in range(round(p['duration_s']/DT)):
        wall=time.perf_counter();t=float(d.time)
        mujoco.mj_forward(model,d)
        on,grf,invalid=mat.contacts(contact_state(model,d,feet,allowed,True))
        if t>=.36:
            mission.phase='return' if p['turn_s'] is not None and t>=p['turn_s'] else 'outbound'
            desired=(-1 if mission.phase=='return' else 1)*p['speed_m_s']
            mission.v[0]+=np.clip(desired-mission.v[0],-.5*DT,.5*DT)
            mission.v[1]=np.clip(-.8*d.qpos[1],-.1,.1)
            mission.body_target[:2]+=mission.v[:2]*DT
        a=time.perf_counter()
        try:
            obs.observe(t,mat.force_boundary_bottom,mat.velocity)
            b=time.perf_counter()
            info=dict(material_observer=obs,force_boundary_bottom=obs.published_bottom if obs.published_bottom is not None else mat.force_boundary_bottom,
                      stiffness_N_m=1227272.7272727273,force=grf)
            cmd=ctrl.update(t,d,mission,on,grf,mat.soft,info,replan=tick%10==0)
        except (ControllerNumericalFailure,StanceReferenceInfeasible,IntegrationFailure) as e:
            reason=type(e).__name__+': '+str(e);break
        c=time.perf_counter()
        trace.append((t,d.qpos.copy(),d.qvel.copy(),np.array(cmd)))
        d.ctrl[:]=cmd
        for micro in range(10):
            if not mat.before_physics(TICK,mission.phase):
                reason=mat.reason;break
            start_t=float(d.time)
            mujoco.mj_step(model,d)
            actual,force,invalid=mat.contacts(contact_state(model,d,feet,allowed,True))
            mat.after_physics(TICK,force)
            obs.completed(start_t,float(d.time),mat.force_boundary_bottom,mat.velocity,force[:,2])
            energy+=float(np.abs(d.actuator_force*d.qvel[6:]).sum())*TICK
            slip+=float((np.linalg.norm(mat.velocity[:,:2],axis=1)*actual).sum())*TICK
            tracking+=float(np.sum((d.qvel[:2]-mission.v[:2])**2))*TICK
            steps+=1;last_fz=force[:,2]
            if invalid or d.qpos[2]<.12:reason='nonfoot_or_body_fall';break
        force_rows.append(last_fz.copy())
        timing.append((t,(b-a)*1000,(c-b)*1000,(time.perf_counter()-wall)*1000))
        if reason!='horizon':break
    if hasattr(ctrl,'at_rollout_end'):
        ctrl.at_rollout_end(t=float(d.time),data=d,mission=mission,reason=reason)
    ctrl.flush();mat.flush();mat.events_file.close();mat.failures_file.close()
    wall=time.perf_counter()-start
    logs=np.asarray(timing)
    routing=np.asarray(router.events).reshape(-1,3)
    q=[v[1] for v in trace];v=[r[2] for r in trace]
    np.savez_compressed(out/'trajectory.npz',time=[r[0] for r in trace],qpos=q,qvel=v,control=[r[3] for r in trace],
        force_10th_substep=force_rows,clock_origins_and_elapsed_ms=logs,n_backend_seconds=routing)
    if router.gpu:router.gpu.close()
    normals=out/'w55-f/normal-execution.jsonl'
    nref=sum(1 for _ in normals.open()) if normals.is_file() else 0
    write(out/'result.json',dict(arm=arm,case=case,reason=reason,physical_s=float(d.time),planned_s=p['duration_s'],
         full_horizon=(steps==round(p['duration_s']/TICK)),steps_5kHz=steps,packets=len(router.events),
         gpu_packets=int(routing[:,1].sum()),cpu_packets=int(len(routing)-routing[:,1].sum()),normal_queries=nref,
         sourced_controller_config_sha256=digest(CFG),cpu_implementation=digest(FROZEN/'work/w57-m/material.so'),
         GPU_propagator_sha256=digest(LIB),GPU_correction_sha256=digest(BUILD/'correction.so'),weights_sha256=digest(WEIGHTS),
         previous_forward_equivalence_sha256=digest(BUILD/'full-queries/overall.json'),config_sha256=digest(config),
         x_m=float(d.qpos[0]),energy_J=energy,slip_m=slip,tracking_ISE=tracking,wall_s=wall,cpu_s=time.process_time()-cpu,
         maximum_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,full_sim_steps_s=steps/wall,
         measured_percentiles_p50_p95_p99_ms=dict(observer=np.percentile(logs[:,1],[50,95,99]).tolist(),
             controller=np.percentile(logs[:,2],[50,95,99]).tolist(),full_loop=np.percentile(logs[:,3],[50,95,99]).tolist()) if len(logs) else {},
         full_loop_2ms_misses=int((logs[:,3]>2).sum()) if len(logs) else 0,shared_host=True,deadline=None,
         wall_deadline_enabled=False,kill_on_conversation_end=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path);p.add_argument('case');p.add_argument('arm',choices=['cpu','gpu','adaptive'])
    p.add_argument('out',type=Path)
    a=p.parse_args();run(a.out,a.arm,a.case,a.config)
