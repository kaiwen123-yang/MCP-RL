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


class LegalRoute:
    """Reward account on the original 500Hz strip passage criterion.

    Approach/crossing credit is provisional. A side exit or an uncompleted
    passage retracts that lap's positive credit and disables further positive
    progress until the next natural lap. Physics and formal evaluation are unchanged.
    """
    def __init__(self):
        self.progress=0.;self.lap_credit=0.;self.completed_laps=0
        self.raw_laps=0;self.valid=True;self.previous_phase=None;self.previous_x=None
        self.passages=[];self.current=None;self.leg_pass={}
        self.side_exits=0

    def invalidate(self):
        if self.valid:
            retract=max(self.lap_credit,0.)
            self.progress-=retract;self.lap_credit-=retract
        self.valid=False

    def update(self,t,mission,xy,bottom,grf):
        phase=mission.phase
        raw_laps=getattr(mission,'laps',int(phase=='complete'))
        if self.previous_phase in ('outbound','return'):
            direction=1 if self.previous_phase=='outbound' else -1
            if self.valid:
                change=direction*(float(xy[0])-self.previous_x)/6
                self.progress+=change;self.lap_credit+=change
        # Close the actual leg when the mission changes to its endpoint/turn.
        if self.current is not None and phase!=self.current['direction']:
            self.leg_pass[self.current['direction']]=self.current['completed']
            if not self.current['completed']:self.invalidate()
            self.current=None
        if raw_laps>self.raw_laps:
            self.completed_laps+=int(self.valid and self.leg_pass.get('outbound',False)
                                     and self.leg_pass.get('return',False))
            self.raw_laps=raw_laps
            self.lap_credit=0.;self.valid=True;self.leg_pass={}
        if phase in ('outbound','return'):
            if self.current is None:
                self.current=dict(lap=raw_laps+1,direction=phase,entry_s=None,exit_s=None,
                    lateral_exit_s=None,material_loaded=False,completed=False)
                self.passages.append(self.current)
            row=self.current
            enter=xy[0]>=.8 if phase=='outbound' else xy[0]<=2.2
            leave=xy[0]>=2.2 if phase=='outbound' else xy[0]<=.8
            if row['entry_s'] is None and enter:row['entry_s']=float(t)
            if row['entry_s'] is not None and row['exit_s'] is None:
                if abs(xy[1])>.6 and row['lateral_exit_s'] is None:
                    row['lateral_exit_s']=float(t);self.side_exits+=1;self.invalidate()
                half=.015811388300841896
                inside=(bottom[:,0]>=.8+half)&(bottom[:,0]<=2.2-half)&(abs(bottom[:,1])<=.6-half)
                row['material_loaded'] |= bool(np.any(inside&(grf[:,2]>0)))
                if leave:
                    row['exit_s']=float(t)
                    row['completed']=row['lateral_exit_s'] is None and row['material_loaded']
                    if not row['completed']:self.invalidate()
        self.previous_x=float(xy[0]);self.previous_phase=phase

    def result(self):
        return dict(reward_version='W36_legal_route_progress_and_lap_alignment',
            progress=self.progress,completed_roundtrips=self.completed_laps,
            legal_outbound_traversals=sum(x['direction']=='outbound' and x['completed'] for x in self.passages),
            legal_return_traversals=sum(x['direction']=='return' and x['completed'] for x in self.passages),
            side_exits=self.side_exits,raw_roundtrips=self.raw_laps,passages=self.passages,
            criterion='Original base strip x[.8,2.2],abs(y)<=.6 and actual interior-foot Fz>0 at500Hz; return uses reversed boundaries. Signed credit is provisional and retracted on illegal/incomplete passage. Formal evaluator unchanged.')


class PolicyStream:
    def __init__(self,root,group,seed,contract,directory,training=True,checkpoint=None):
        self.root,self.group,self.seed,self.contract=root,group,seed,contract
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.training=training
        self.controller_build=contract.get('reuse',{}).get('active_controller_build')
        self.reward_version=contract.get('learning',{}).get('reward_version','legacy_signed_route_and_raw_laps')
        self.phase_initial_samples=contract.get('learning',{}).get('phase_initial_transitions',0)
        self.phase_initial_updates=contract.get('learning',{}).get('phase_initial_updates',0)
        self.phase_id=contract.get('learning',{}).get('phase_id',contract.get('current_task'))
        self.excluded_old_version_transitions=0;self.controller_migration=None
        self.last_completed_update=None
        resources=contract.get('resources',{})
        torch.set_num_threads(resources.get('torch_threads',1));torch.set_num_interop_threads(1)
        np.random.seed(seed);torch.manual_seed(seed)
        p=contract['joint_learning']['ppo']
        self.backend=PPOBackend(PPOConfig(obs_dim=OBSERVATION_DIM,action_bound=1.,
            hidden=(64,64),learning_rate=p['learning_rate'],initial_std=p['initial_std'],
            gamma=p['gamma'],gae_lambda=p['gae_lambda'],clip_epsilon=p['clip'],
            value_coef=p['value_coef'],entropy_coef=p['entropy_coef'],
            max_grad_norm=p['max_grad_norm'],epochs=p['epochs_per_update'],minibatch_size=64,device=resources.get('torch_device','cpu')))
        self.normalization=json.loads(json.dumps(NORMALIZATION))
        self.samples=0;self.effective_samples=0;self.buffer=[];self.episode=0
        self.transition_serial=0;self.discarded_after_checkpoint=0
        self.stage_samples={0:0,1:0,2:0};self.physics_s=0.;self.begin=time.time()
        self.curriculum_stage=0;self.promotions=[];self.session_records=[]
        self.restored_without_environment=False;self.last_observation=None
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
            self.begin=saved.get('collection_started_unix_s',self.begin)
            self.excluded_old_version_transitions=saved.get('excluded_old_version_transitions',0)
            self.controller_migration=saved.get('controller_migration')
            self.last_completed_update=saved.get('last_completed_update')
            if training:
                self.backend.optimizer.load_state_dict(saved['optimizer'])
                torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
                if saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])
                self.buffer=[{k:v.to(self.backend.config.device) for k,v in row.items()} for row in saved.get('buffer',[])]
                if self.buffer and (saved.get('controller_build')!=self.controller_build or
                                    saved.get('reward_version','legacy_signed_route_and_raw_laps')!=self.reward_version):
                    boundary=contract['learning'].get('controller_version_boundary',{})
                    declared=root/boundary.get('parent_checkpoint','')
                    source=Path(checkpoint)
                    if not source.is_absolute():source=root/source
                    explicit=(self.controller_build=='S5-r26' and saved.get('controller_build')=='S5-r25'
                        and source.resolve()==declared.resolve() and saved['episode']==144
                        and saved['samples']==4497 and saved['updates']==17 and len(self.buffer)==145
                        and saved.get('reward_version')==self.reward_version
                        and self.excluded_old_version_transitions==0)
                    if not explicit:
                        raise ValueError('Preserve the saved old-version tail; it cannot enter a new-version on-policy update.')
                    self.controller_migration=dict(parent_checkpoint=str(source.relative_to(root)),
                        parent_release_url=boundary['parent_checkpoint_release_url'],
                        parent_controller_build='S5-r25',controller_build='S5-r26',reward_version=self.reward_version,
                        first_excluded_transition=4353,last_excluded_transition=4497,
                        excluded_transitions=145,used_in_new_updates=False,
                        restored=['actor','critic','optimizer','RNG','normalization','historical_counts'],
                        continuous_environment_restored=False,new_buffer_size=0)
                    excluded=self.directory/'excluded-r25-004353-004497.pt'
                    torch.save(dict(migration=self.controller_migration,buffer=saved['buffer']),excluded)
                    self.controller_migration['isolated_tail_path']=str(excluded.relative_to(root))
                    self.excluded_old_version_transitions=145
                    self.buffer=[]
                # A new physical session never inherits a pending successor.
                # Same-version buffers retain their already recorded next_obs/value.
                self.last_observation=None
                self.transition_serial=saved.get('transition_serial',self.samples)
                log_path=self.directory/'transitions.jsonl'
                if log_path.exists():
                    recorded=[json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
                    latest=max([self.transition_serial]+[row.get('transition_serial',row['transition']) for row in recorded])
                    self.discarded_after_checkpoint=latest-self.transition_serial
                    self.transition_serial=latest
                self.restored_without_environment=True
                self.session_records.append(dict(episode=self.episode,status='new_physical_session_on_checkpoint_restore',
                    continuous_environment_restored=False,checkpoint=str(checkpoint)))
                existing=[int(x.stem.split('-')[-1]) for x in self.directory.glob('episode-*.json')]
                prefix=f'train-{group}-s{seed}-ep'
                existing += [int(x.name[len(prefix):]) for x in self.directory.parent.parent.glob(prefix+'*') if x.is_dir()]
                # An interrupted started session also owns its episode number.
                self.episode=max([self.episode]+existing)
        self.log=(self.directory/'transitions.jsonl').open('a')
        self.learning=(self.directory/'learning.jsonl').open('a')

    def checkpoint(self,resume_only=False):
        p=self.directory/(f'resume-state-ep{self.episode:05d}.pt' if resume_only else f'checkpoint-{self.samples:06d}.pt')
        if not resume_only:
            self.last_completed_update=dict(path=str(p.relative_to(self.root)),samples=self.samples,
                updates=self.backend.update_count,phase_id=self.phase_id,
                controller_build=self.controller_build,reward_version=self.reward_version)
        torch.save(dict(actor=self.backend.actor.state_dict(),critic=self.backend.critic.state_dict(),
            optimizer=self.backend.optimizer.state_dict(),normalization=self.normalization,
            samples=self.samples,stage_samples=self.stage_samples,seed=self.seed,group=self.group,
            transition_serial=self.transition_serial,discarded_after_checkpoint=self.discarded_after_checkpoint,
            collection_started_unix_s=self.begin,
            effective_samples=self.effective_samples,physics_s=self.physics_s,episode=self.episode,
            curriculum_stage=self.curriculum_stage,promotions=self.promotions,session_records=self.session_records,
            updates=self.backend.update_count,continuous_environment_snapshot=False,buffer=self.buffer,
            last_observation=self.last_observation,session_boundary=resume_only,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.backend.config.device.startswith('cuda') else None,
            controller_build=self.controller_build,reward_version=self.reward_version,
            phase_initial_transitions=self.phase_initial_samples,
            phase_initial_updates=self.phase_initial_updates,phase_id=self.phase_id,
            excluded_old_version_transitions=self.excluded_old_version_transitions,
            controller_migration=self.controller_migration,last_completed_update=self.last_completed_update,
            ppo=self.backend.provenance()),p.with_suffix('.tmp'))
        p.with_suffix('.tmp').replace(p)
        summary=dict(samples=self.samples,effective_samples=self.effective_samples,
            physics_s=self.physics_s,stage_samples=self.stage_samples,episodes_started=self.episode,
            curriculum_stage=self.curriculum_stage,promotions=self.promotions,session_records=self.session_records,
            updates=self.backend.update_count,wall_s=time.time()-self.begin,checkpoint=str(p),
            group=self.group,seed=self.seed,unupdated_fragment=len(self.buffer),
            continuous_environment_snapshot=False,
            excluded_old_version_transitions=self.excluded_old_version_transitions,
            controller_migration=self.controller_migration,phase_id=self.phase_id,
            last_completed_update=self.last_completed_update)
        self.write_collection()
        progress=self.directory/'progress.json'
        progress.with_suffix('.tmp').write_text(json.dumps(summary,indent=2)+'\n')
        progress.with_suffix('.tmp').replace(progress)

    def write_collection(self):
        collected=self.directory/'collection.json'
        collected.with_suffix('.tmp').write_text(json.dumps(dict(
            collected_transitions=self.samples,transition_serial=self.transition_serial,
            completed_update_transitions=self.backend.update_count*256,updates=self.backend.update_count,
            unupdated_fragment=len(self.buffer),stage_samples=self.stage_samples,
            wall_s=time.time()-self.begin,updated_unix_s=time.time(),collection_started_unix_s=self.begin,
            physics_s=self.physics_s,episode=self.episode,curriculum_stage=self.curriculum_stage,
            discarded_after_checkpoint=self.discarded_after_checkpoint,
            excluded_old_version_transitions=self.excluded_old_version_transitions,
            controller_migration=self.controller_migration,phase_id=self.phase_id,
            last_completed_update=self.last_completed_update,
            controller_build=self.controller_build,reward_version=self.reward_version,
            phase_initial_transitions=self.phase_initial_samples,
            phase_new_transitions=self.samples-self.phase_initial_samples,
            phase_initial_updates=self.phase_initial_updates,
            phase_new_updates=self.backend.update_count-self.phase_initial_updates))+'\n')
        collected.with_suffix('.tmp').replace(collected)

    def transition(self,pending,observation,reward,terminated,truncated,dt,effective,stage,info):
        self.transition_serial+=1
        device=self.backend.config.device
        sample=pending['sample'];next_obs=torch.as_tensor(observation,dtype=torch.float32,device=device).reshape(1,-1)
        self.last_observation=next_obs.detach().cpu()
        next_value=self.backend.value(next_obs)
        record=dict(obs=pending['obs'],next_obs=next_obs,raw_action=sample.raw_action,reward=torch.tensor([reward],device=device),
            terminated=torch.tensor([terminated],device=device),truncated=torch.tensor([truncated],device=device),
            log_prob=sample.log_prob,value=sample.value,next_value=next_value)
        self.log.write(json.dumps(dict(time_s=info['time_s'],episode=self.episode,
            transition=self.samples+1,transition_serial=self.transition_serial,reward=reward,terminated=terminated,truncated=truncated,
            dt_s=dt,effective=effective,stage=stage,cap_m_s=pending['cap'],
            raw_action=float(sample.raw_action.item()),log_prob=float(sample.log_prob.item()),**info['metrics']))+'\n')
        self.log.flush()
        if not self.training:return
        self.buffer.append(record);self.samples+=1;self.effective_samples+=int(effective)
        self.stage_samples[stage]+=1;self.physics_s+=dt
        if len(self.buffer)==256:
            batch={key:torch.stack([row[key] for row in self.buffer]) for key in record}
            np.savez_compressed(self.directory/f'rollout-{self.samples-255:06d}-{self.samples:06d}.npz',
                **{key:value.detach().cpu().numpy() for key,value in batch.items()})
            batch.pop('next_obs')
            loss=self.backend.update(Rollout(**batch));self.buffer.clear()
            self.learning.write(json.dumps(dict(loss,cumulative_samples=self.samples,wall_s=time.time()-self.begin,
                controller_build=self.controller_build,reward_version=self.reward_version,
                phase_new_transitions=self.samples-self.phase_initial_samples,
                phase_new_updates=self.backend.update_count-self.phase_initial_updates))+'\n')
            self.log.flush();self.learning.flush();self.checkpoint()
            # R12 independent streams never wait for unstarted groups or seeds.
            if self.contract['joint_learning'].get('cross_stream_wait',True):
                while True:
                    deadline=json.loads((self.root/self.contract['outputs']['compact_root']/'training-deadline.json').read_text())['deadline_unix_s']
                    paths=list(self.directory.parent.glob('*/progress.json'))
                    amounts=[json.loads(path.read_text())['samples'] for path in paths]
                    if self.samples-min(amounts)<=256 or time.time()>=deadline:break
                    time.sleep(5)
            print(json.dumps(dict(group=self.group,seed=self.seed,samples=self.samples,
                physics_s=self.physics_s,wall_s=time.time()-self.begin,loss=loss['loss'])),flush=True)
        self.write_collection()


class PolicyMaterial(JointMaterial):
    def __init__(self,root,r9,stream,stage,session_s,agent=None,full_recording=False):
        super().__init__(root,r9,.25,full_recording=full_recording)
        self.stream,self.stage,self.session_s,self.agent=stream,stage,session_s,agent
        self.group=stream.group;self.latest_plan=None;self.stop_requested=''
        self.observer=None;self.history=deque(maxlen=20);self.next_history=0.
        self.gait_phase=0.;self.last_control_time=0.;self.cadence=1.
        self.pending=None;self.last_snapshot=None;self.effective_time=0.;self.action_change=0.
        self.action_dependence=[]
        self.legal_route=LegalRoute() if stream.reward_version=='W36_legal_route_progress_and_lap_alignment' else None
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
        progress=self.legal_route.progress if self.legal_route is not None else self.route_progress
        if self.legal_route is not None:laps=self.legal_route.completed_laps
        return np.array([t,progress,self.indent_integral,self.slip_integral,
            self.action_change,laps,self.effective_time])

    def collect_pending(self,t,mission,observation,terminated=False,truncated=False,failure=False,timeout=False):
        current=self.snapshot(t,mission)
        self.stream.last_observation=torch.as_tensor(observation).reshape(1,-1).detach().cpu()
        pending=self.pending;previous=self.last_snapshot
        self.last_snapshot=current;self.pending=None
        if pending is not None and current[0]>previous[0]:
            change=current-previous;dt=change[0]
            reward=5*change[1]-dt/60-change[2]/60-.1*change[3]/60-.01*change[4]+10*change[5]-10*failure-5*timeout
            self.stream.transition(pending,observation,float(reward),terminated,truncated,
                float(dt),bool(change[6]>0),self.stage,dict(time_s=float(t),metrics=dict(
                    signed_route_fraction=float(change[1]),indentation_sq_s=float(change[2]),
                    loaded_slip_sq_s=float(change[3]),completed_circuits=int(change[5]),
                    failure=failure,intrinsic_timeout=timeout,
                    reward_version=self.stream.reward_version,
                    raw_signed_route_fraction=self.route_progress,
                    raw_completed_circuits=getattr(mission,'laps',int(mission.phase=='complete')),
                    side_exits=self.legal_route.side_exits if self.legal_route else None)))

    def control(self,t,mission):
        self.mission=mission
        measurement=self.measurements()
        if self.legal_route is not None:
            self.legal_route.update(t,mission,self.data.qpos[:2],measurement[0],measurement[3])
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
            external_end=''
            if self.stream.training:
                deadline=json.loads(self.training_deadline_path.read_text())['deadline_unix_s']
                has_transition=self.pending is not None and t>self.last_snapshot[0]
                if time.time()>=deadline:external_end='training_budget_truncated'
                elif self.stream.samples+int(has_transition)>=self.stream.target:
                    external_end='training_checkpoint_complete'
            # The runtime checks physical failure immediately after integration.
            # Its finish path wins over this external boundary at the next tick.
            self.collect_pending(t,mission,observation,truncated=bool(external_end))
            if external_end:
                self.stop_requested=external_end;return
            obs=torch.as_tensor(observation,device=self.stream.backend.config.device).reshape(1,-1)
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
        if self.legal_route is not None:
            self.legal_route.update(t,self.mission,self.data.qpos[:2],measurement[0],measurement[3])
            result['legal_route']=self.legal_route.result()
        if self.observer is not None and t>self.observer.time:self.observer.observe(t,measurement[0],measurement[1])
        observation=self.observation(t,self.mission,measurement)
        physical_failure=result['reason'].startswith(('material_domain_exit','body_fall','invalid_contact'))
        timeout=False
        success=False
        truncated=not (physical_failure or timeout or success)
        self.collect_pending(t,self.mission,observation,physical_failure or timeout or success,truncated,physical_failure,timeout)
        result=super().finish(result)
        # Variable policy actions are recorded in cap-decisions/transitions.
        result.update(cap_profile='ppo_policy',requested_cap_m_s=None)
        result.pop('development_intent',None)
        result['execution_intent']='Standard PPO in the actual S5/Go2/material loop'
        result['material']['observer_enabled']=self.observer is not None
        result['policy']=dict(group=self.group,seed=self.stream.seed,training=self.stream.training,
            total_transitions=self.stream.samples,stage=self.stage,session_limit_s=self.session_s,
            cap_to_reference_increment_latency_s=None if hasattr(self,'reference_timing') else .002,
            reference_timing=getattr(self,'reference_timing',None),
            action_record='cap-decisions.json and training transitions.jsonl',
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
