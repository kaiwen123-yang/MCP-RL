"""Standard PPO sampling inside the retained S5/Go2/ReferenceColumns loop."""
from collections import deque
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from response_ppo.ppo_backend import PPOBackend, PPOConfig, Rollout
from s1_go2_r11_runtime import PHASES, contact_state
from s1_go2_joint_material import JointMaterial
from s1_go2_r10_observer import MaterialObserver

OBSERVATION_DIM=658
NORMALIZATION=dict(body_xy_m=3.,body_height_m=.3,body_linear_velocity_m_s=.25,
    body_angular_velocity_rad_s=1.,joint_position='range midpoint and half-width',
    joint_velocity_rad_s=10.,relative_feet_m=.3,foot_velocity_m_s=1.,
    force_N=15.206408*9.81/4,history_bottom_height_m=.05,
    pann_recovery_m=.001,history_frames=20,history_dt_s=.05,
    finite_task_channels='Always zero for every R11 training and evaluation duration',
    schema=['relative_goal_xy','base_z','base_quaternion_wxyz','relative_heading_sin_cos','base_velocity6',
        'joint_position12','joint_velocity12','relative_world_feet12','foot_velocity12',
        'contact4','GRF12','applied_cap','mission_phase9','gait_sin_cos',
        'finite_task_flag_and_remaining_fraction','completed_plan_age',
        'query_xy8','query_soft_mask4','history20x27','recovery4x3'],
    observation_clipping=False)


class PolicyStream:
    def __init__(self,root,group,seed,contract,directory,training=True,checkpoint=None):
        self.root,self.group,self.seed,self.contract=root,group,seed,contract
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.training=training
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        np.random.seed(seed);torch.manual_seed(seed)
        p=contract['joint_learning']['ppo']
        self.backend=PPOBackend(PPOConfig(obs_dim=OBSERVATION_DIM,action_bound=1.,
            hidden=(64,64),learning_rate=p['learning_rate'],initial_std=p['initial_std'],
            gamma=p['gamma'],gae_lambda=p['gae_lambda'],clip_epsilon=p['clip'],
            value_coef=p['value_coef'],entropy_coef=p['entropy_coef'],
            max_grad_norm=p['max_grad_norm'],epochs=p['epochs_per_update'],minibatch_size=64,device='cpu'))
        self.normalization=json.loads(json.dumps(NORMALIZATION))
        self.samples=0;self.effective_samples=0;self.buffer=[];self.episode=0
        self.stage_samples={0:0,1:0,2:0};self.physics_s=0.;self.begin=time.time()
        self.curriculum_stage=0;self.promotions=[];self.session_records=[]
        self.restored_without_environment=False
        self.target=contract['joint_learning']['standard_target_transitions_per_stream']
        self.observer_law=None
        if checkpoint is not None:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            self.backend.actor.load_state_dict(saved['actor']);self.backend.critic.load_state_dict(saved['critic'])
            self.normalization=saved['normalization'];self.samples=saved['samples']
            self.effective_samples=saved['effective_samples'];self.physics_s=saved['physics_s']
            self.stage_samples=saved['stage_samples'];self.episode=saved['episode']
            self.curriculum_stage=saved['curriculum_stage'];self.promotions=saved['promotions']
            self.session_records=saved['session_records'];self.backend.update_count=saved['updates']
            if training:
                self.backend.optimizer.load_state_dict(saved['optimizer'])
                torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
                self.buffer=saved.get('buffer',[])
                self.restored_without_environment=True
                self.session_records.append(dict(episode=self.episode,status='technical_interruption_on_restore',
                    continuous_environment_restored=False,checkpoint=str(checkpoint)))
                existing=[int(x.stem.split('-')[-1]) for x in self.directory.glob('episode-*.json')]
                self.episode=max([self.episode]+existing)
        self.log=(self.directory/'transitions.jsonl').open('a')
        self.learning=(self.directory/'learning.jsonl').open('a')

    def checkpoint(self,resume_only=False):
        p=self.directory/(f'resume-state-ep{self.episode:05d}.pt' if resume_only else f'checkpoint-{self.samples:06d}.pt')
        torch.save(dict(actor=self.backend.actor.state_dict(),critic=self.backend.critic.state_dict(),
            optimizer=self.backend.optimizer.state_dict(),normalization=self.normalization,
            samples=self.samples,stage_samples=self.stage_samples,seed=self.seed,group=self.group,
            effective_samples=self.effective_samples,physics_s=self.physics_s,episode=self.episode,
            curriculum_stage=self.curriculum_stage,promotions=self.promotions,session_records=self.session_records,
            updates=self.backend.update_count,continuous_environment_snapshot=False,buffer=self.buffer,
            session_boundary=resume_only,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            ppo=self.backend.provenance()),p.with_suffix('.tmp'))
        p.with_suffix('.tmp').replace(p)
        summary=dict(samples=self.samples,effective_samples=self.effective_samples,
            physics_s=self.physics_s,stage_samples=self.stage_samples,episodes_started=self.episode,
            curriculum_stage=self.curriculum_stage,promotions=self.promotions,session_records=self.session_records,
            updates=self.backend.update_count,wall_s=time.time()-self.begin,checkpoint=str(p),
            group=self.group,seed=self.seed)
        if resume_only:return
        progress=self.directory/'progress.json'
        progress.with_suffix('.tmp').write_text(json.dumps(summary,indent=2)+'\n')
        progress.with_suffix('.tmp').replace(progress)

    def transition(self,pending,observation,reward,terminated,truncated,dt,effective,stage,info):
        sample=pending['sample'];next_obs=torch.as_tensor(observation,dtype=torch.float32).reshape(1,-1)
        next_value=self.backend.value(next_obs)
        record=dict(obs=pending['obs'],next_obs=next_obs,raw_action=sample.raw_action,reward=torch.tensor([reward]),
            terminated=torch.tensor([terminated]),truncated=torch.tensor([truncated]),
            log_prob=sample.log_prob,value=sample.value,next_value=next_value)
        self.log.write(json.dumps(dict(time_s=info['time_s'],episode=self.episode,
            transition=self.samples+1,reward=reward,terminated=terminated,truncated=truncated,
            dt_s=dt,effective=effective,stage=stage,cap_m_s=pending['cap'],
            raw_action=float(sample.raw_action.item()),log_prob=float(sample.log_prob.item()),**info['metrics']))+'\n')
        if not self.training:return
        self.buffer.append(record);self.samples+=1;self.effective_samples+=int(effective)
        self.stage_samples[stage]+=1;self.physics_s+=dt
        if len(self.buffer)==256:
            batch={key:torch.stack([row[key] for row in self.buffer]) for key in record}
            np.savez_compressed(self.directory/f'rollout-{self.samples-255:06d}-{self.samples:06d}.npz',
                **{key:value.detach().cpu().numpy() for key,value in batch.items()})
            batch.pop('next_obs')
            loss=self.backend.update(Rollout(**batch));self.buffer.clear()
            self.learning.write(json.dumps(dict(samples=self.samples,wall_s=time.time()-self.begin,**loss))+'\n')
            self.log.flush();self.learning.flush();self.checkpoint()
            # Keep at most one 256-transition update ahead, preserving this live session.
            while True:
                deadline=json.loads((self.root/self.contract['outputs']['compact_root']/'training-deadline.json').read_text())['deadline_unix_s']
                amounts=[]
                for group in ('PPO_RAW_S5','PANN_PPO_S5'):
                    for seed in range(3):
                        path=self.directory.parent/f'{group}-seed{seed}'/'progress.json'
                        amounts.append(json.loads(path.read_text())['samples'] if path.exists() else 0)
                if self.samples-min(amounts)<=256 or time.time()>=deadline:break
                time.sleep(5)
            print(json.dumps(dict(group=self.group,seed=self.seed,samples=self.samples,
                physics_s=self.physics_s,wall_s=time.time()-self.begin,loss=loss['loss'])),flush=True)


class PolicyMaterial(JointMaterial):
    def __init__(self,root,r9,stream,stage,session_s,agent=None,full_recording=False):
        super().__init__(root,r9,.25,full_recording=full_recording)
        self.stream,self.stage,self.session_s,self.agent=stream,stage,session_s,agent
        self.group=stream.group;self.latest_plan=None;self.stop_requested=''
        self.observer=None;self.history=deque(maxlen=20);self.next_history=0.
        self.gait_phase=0.;self.last_control_time=0.;self.cadence=1.
        self.pending=None;self.last_snapshot=None;self.effective_time=0.;self.action_change=0.
        self.action_dependence=[]
        if self.group=='PANN_PPO_S5':
            parameters=json.loads((root/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
            self.observer=MaterialObserver(parameters,root/r9['pann']['weights_path'],law=stream.observer_law)
            stream.observer_law=self.observer.ground.law

    def attach(self,*args):
        super().attach(*args)
        self.fk_data=mujoco.MjData(self.native_model)
        self.qa=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.va=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        bounds=self.model.jnt_range[self.model.actuator_trnid[:,0]]
        self.qmid=bounds.mean(axis=1);self.qhalf=(bounds[:,1]-bounds[:,0])/2

    def measurements(self):
        bottom,velocity=self.kinematics()
        native=contact_state(self.model,self.data,self.feet,self.allowed,full=True)
        contact,grf,*_=self.contacts(native,full=True)
        return bottom,velocity,contact,grf

    def observation(self,t,mission,measurement):
        bottom,velocity,contact,grf=measurement
        query=np.zeros((4,2));mask=np.zeros(4,bool);feature=np.zeros((4,3));age=0.
        if self.latest_plan is not None:
            plan=self.latest_plan;j=int(np.argmin(abs(plan['times']-(t+.2))))
            state=plan['states'][j];self.fk_data.qpos[:]=state[:self.native_model.nq]
            mujoco.mj_kinematics(self.native_model,self.fk_data)
            query=self.fk_data.geom_xpos[self.feet,:2].copy()
            mask=np.array([self.ground.interior(p) for p in query])
            age=t-float(plan['times'][0])
            if self.observer is not None:feature,_=self.observer.recovery(query)
        goal=np.array([3.,0.]) if mission.phase in ('outbound','far_hold','turn') else np.zeros(2)
        phase=np.zeros(len(PHASES));phase[PHASES.index(mission.phase)]=1
        history=np.zeros((20,27));history[-len(self.history):]=np.asarray(self.history)
        d=self.data;mg=15.206408*9.81/4
        w,x,y,z=d.qpos[3:7]
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        heading_error=mission.heading-yaw
        return np.concatenate(((goal-d.qpos[:2])/3,[d.qpos[2]/.3],d.qpos[3:7],
            [np.sin(heading_error),np.cos(heading_error)],
            d.qvel[:6]/np.array([.25,.25,.25,1,1,1]),(d.qpos[self.qa]-self.qmid)/self.qhalf,
            d.qvel[self.va]/10,((bottom-d.qpos[:3])/.3).ravel(),velocity.ravel(),contact,
            (grf/mg).ravel(),[(self.applied_cap-.215)/.035],phase,
            [np.sin(self.gait_phase),np.cos(self.gait_phase),0.,
             0.,age/.35],(query/3).ravel(),mask,
            history.ravel(),(feature/.001).ravel())).astype(np.float32)

    def snapshot(self,t,mission):
        laps=getattr(mission,'laps',int(mission.phase=='complete'))
        return np.array([t,self.route_progress,self.indent_integral,self.slip_integral,
            self.action_change,laps,self.effective_time])

    def collect_pending(self,t,mission,observation,terminated=False,truncated=False,failure=False,timeout=False):
        current=self.snapshot(t,mission)
        if self.pending is not None:
            change=current-self.last_snapshot;dt=change[0]
            reward=5*change[1]-dt/60-change[2]/60-.1*change[3]/60-.01*change[4]+10*change[5]-10*failure-5*timeout
            self.stream.transition(self.pending,observation,float(reward),terminated,truncated,
                float(dt),bool(change[6]>0),self.stage,dict(time_s=float(t),metrics=dict(
                    signed_route_fraction=float(change[1]),indentation_sq_s=float(change[2]),
                    loaded_slip_sq_s=float(change[3]),completed_circuits=int(change[5]),
                    failure=failure,intrinsic_timeout=timeout)))
        self.last_snapshot=current;self.pending=None

    def control(self,t,mission):
        self.mission=mission
        measurement=self.measurements()
        if self.observer is not None:self.observer.observe(t,measurement[0],measurement[1])
        self.gait_phase+=2*np.pi*self.cadence*(t-self.last_control_time)
        self.cadence=mission.task['stand_settings' if mission.phase in
            ('initial_stand','terminal_stand','complete','far_hold','home_hold') else 'move_and_turn_settings']['Cadence']
        self.last_control_time=t
        if t>=self.next_history-1e-8:
            bottom,velocity,contact,grf=measurement
            self.history.append(np.r_[bottom[:,2]/.05,grf[:,2]/(15.206408*9.81/4),
                velocity[:,:2].ravel(),contact,self.data.qvel[:6]/[.25,.25,.25,1,1,1],
                (self.applied_cap-.215)/.035])
            self.next_history+=.05
        # Keep the real final successor with the preceding action. In particular,
        # a 60/300/600 s boundary must not create a zero-duration extra sample.
        if t>=self.session_s-1e-8:
            return
        tick=int(np.floor((t+1e-8)/.5))
        if tick!=self.last_actor_tick:
            observation=self.observation(t,mission,measurement)
            self.collect_pending(t,mission,observation)
            if self.stream.training:
                deadline=json.loads(self.training_deadline_path.read_text())['deadline_unix_s']
                if time.time()>=deadline:
                    self.stop_requested='training_budget_truncated';return
            if self.stream.training and self.stream.samples>=self.stream.target:
                self.stop_requested='training_checkpoint_complete';return
            obs=torch.as_tensor(observation).reshape(1,-1)
            sample=self.stream.backend.act(obs,deterministic=not self.stream.training)
            self.cap=.215+.035*float(sample.action.item())
            self.pending=dict(obs=obs,sample=sample,cap=self.cap)
            if self.observer is not None and not self.stream.training:
                zero=obs.clone();zero[:,-12:]=0
                alternate=self.stream.backend.act(zero,deterministic=True)
                self.action_dependence.append(dict(time_s=t,phase=mission.phase,
                    with_pann_cap=self.cap,zero_feature_cap=.215+.035*float(alternate.action.item())))
            self.cap_decisions.append(dict(time_s=t,tick=tick,requested_cap_m_s=self.cap,phase_before_update=mission.phase))
            self.last_actor_tick=tick
        mission.task['command_speed_m_s']=self.cap

    def before_physics(self,dt,phase):
        ok=super().before_physics(dt,phase)
        if ok and phase in ('outbound','return'):
            self.action_change+=((self.cap-self.applied_cap)/.035)**2
            self.applied_cap=self.cap
        return ok

    def after_physics(self,dt,grf):
        super().after_physics(dt,grf)
        if self.phase in ('outbound','return'):self.effective_time+=dt

    def finish(self,result):
        t=float(self.data.time);measurement=self.measurements()
        if self.observer is not None and t>self.observer.time:self.observer.observe(t,measurement[0],measurement[1])
        observation=self.observation(t,self.mission,measurement)
        physical_failure=result['reason'].startswith(('material_domain_exit','body_fall','invalid_contact'))
        timeout=False
        success=False
        truncated=not (physical_failure or timeout or success)
        self.collect_pending(t,self.mission,observation,physical_failure or timeout or success,truncated,physical_failure,timeout)
        result=super().finish(result)
        result.pop('development_intent',None)
        result['execution_intent']='Standard PPO in the actual S5/Go2/material loop'
        result['material']['observer_enabled']=self.observer is not None
        result['policy']=dict(group=self.group,seed=self.stream.seed,training=self.stream.training,
            total_transitions=self.stream.samples,stage=self.stage,session_limit_s=self.session_s,
            cap_to_reference_increment_latency_s=.002,
            terminated=physical_failure or timeout or success,truncated=truncated,
            effective_translation_s=self.effective_time,normalization=self.stream.normalization,
            observer=self.observer.timing() if self.observer is not None else None)
        if self.stream.training and not physical_failure:
            result.update(success=None,reason='external_session_truncation' if t>=self.session_s-1e-8 else result['reason'])
        if not self.full_recording:
            result['material'].update(cell_rate_Hz=None,foot_force_rate_Hz=None,
                event_recording='Compact training: event/rate arrays not collected')
        if self.action_dependence:
            (self.out/'pann-action-dependence.json').write_text(json.dumps(self.action_dependence,indent=2)+'\n')
        return result
