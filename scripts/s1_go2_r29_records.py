"""Lossless W40 event snapshots; observes the live task without rerunning physics."""
from collections import deque
import json
import numpy as np

class R29Records:
    def __init__(self,out,model):
        self.out,self.model=out,model
        self.ring={};self.saved={};self.search_ring={};self.search_saved={};self.events=[];self.until=-1.
        self.r33_ring={};self.r33_saved={}
        self.offset=0;self.plan_offset=0;self.latest=None;self.applied=0;self.seen=set()
        self.failure=None
        self.actions=(out/'action-execution.jsonl').open('w')
        self.nonprogress=deque()
    def r33_plan(self,t):
        if getattr(self,'controller',None) not in ('S5-r33','S5-r34','S5-r35','S5-r36','S5-r37'):return
        path=self.out/'r29-current-plan.bin.r33-candidates'
        if not path.exists():return
        with path.open('rb') as source:
            first=np.fromfile(source,dtype='<f8',count=19)
            if not len(first):return
            call=int(first[1])
            if call in self.r33_ring or call in self.r33_saved:return
            source.seek(0)
            raw=np.fromfile(source,dtype='<f8')
        cursor=0;candidate=0;data={}
        while cursor<len(raw):
            header=raw[cursor:cursor+19];cursor+=19
            T,ns,nu,nx,nr=map(int,header[5:10])
            prefix=f'candidate{candidate:03d}_'
            data[prefix+'header']=header.copy()
            def take(key,shape):
                nonlocal cursor
                n=int(np.prod(shape))
                data[prefix+key]=raw[cursor:cursor+n].reshape(shape).copy();cursor+=n
            for role in ('accepted','reference'):
                for field,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr))]:
                    take(role+'_'+field,shape)
            take('K',(T,nu,nx));take('H',(T,nu,nu))
            take('base_Qu',(T,nu));take('base_Qxu',(T,nx,nu));take('base_active',(T,nu))
            for key in ('prefix_substeps','foot_rows'):
                count=int(raw[cursor]);cursor+=1
                take(key,(count,))
            candidate+=1
        if getattr(self,'controller',None) in ('S5-r35','S5-r36','S5-r37'):
            with (self.out/'r35-cost-components.jsonl').open('a') as log:
                for c in range(candidate):
                    prefix=f'candidate{c:03d}_';h=data[prefix+'header']
                    residual=data[prefix+'accepted_residual']
                    added=float(.5*np.sum(residual[:,-12:]**2)/len(residual))
                    log.write(json.dumps(dict(call_id=call,candidate=c,kind=int(h[2]),slot=int(h[3]),
                        J=float(h[11]),J_ref=added,J0=float(h[11])-added,
                        reference_identity='planner-calls.jsonl:r35_reference_id',
                        terminal_new_cost=float(.5*np.sum(residual[-1,-12:]**2))))+'\n')
        scale_path=self.out/'r29-current-plan.bin.r33-scale'
        scale=None
        if scale_path.exists():
            values=np.fromfile(scale_path,dtype='<f8')
            if len(values) and int(values[1])==call:
                scale=dict(header=values[:6].copy(),D=values[6:].reshape(int(values[2]),int(values[3])).copy())
        self.r33_ring[call]=dict(origin=float(first[15]),candidates=candidate,data=data,scale=scale)
        if t<=self.until:self.save_r33(call)

    def save_r33(self,call):
        if call in self.r33_saved:return
        row=self.r33_ring[call]
        name=f'r33-candidates-call{call:06d}.npz'
        np.savez_compressed(self.out/name,**row['data'])
        record=dict(call_id=call,origin=row['origin'],candidates=row['candidates'],file=name,scale_file=None)
        if row['scale'] is not None:
            name=f'r33-scale-call{call:06d}.npz'
            np.savez_compressed(self.out/name,**row['scale'])
            record['scale_file']=name
        self.r33_saved[call]=record

    def search_plan(self,t):
        p=self.out/'r29-current-plan.bin.search'
        if not p.exists():return
        raw=np.fromfile(p,dtype='<f8');h=raw[:16];call=int(h[1])
        if call in self.search_ring or call in self.search_saved:return
        T,ns,nu,nx=map(int,h[4:8]);cursor=16;data={'search_header':h.copy()}
        for key,shape in [
            ('nominal_states',(T,ns)),('nominal_actions',(T-1,nu)),
            ('k_unscaled',(T,nu)),('K',(T,nu,nx)),('H',(T,nu,nu)),
            ('Qu_unregularized',(T,nu)),('Qxu_unregularized',(T,nx,nu)),
            ('Vx',(T,nx)),('Vxx',(T,nx,nx)),('projected_gradient',(T,nu))]:
            n=int(np.prod(shape));data[key]=raw[cursor:cursor+n].reshape(shape).copy();cursor+=n
        self.search_ring[call]={'origin':float(h[3]),'data':data}
        if t<=self.until:self.save_search(call)
    def save_search(self,call):
        if call in self.search_saved:return
        row=self.search_ring[call];name=f'r32-search-call{call:06d}.npz'
        np.savez_compressed(self.out/name,**row['data'])
        self.search_saved[call]=dict(call_id=call,origin=row['origin'],file=name)
    def plan(self,t):
        self.r33_plan(t)
        self.search_plan(t)
        p=self.out/'r29-current-plan.bin'
        if not p.exists():return
        raw=np.fromfile(p,dtype='<f8');h=raw[:16]
        pid=int(h[1])
        if pid in self.ring or pid in self.saved:return
        T,ns,nu,nx,nr=map(int,h[3:8]);cursor=16
        data={'header':h.copy()}
        def take(key,shape):
            nonlocal cursor
            n=int(np.prod(shape));data[key]=raw[cursor:cursor+n].reshape(shape).copy();cursor+=n
        for role in ('accepted','reference'):
            for field,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr))]:
                take(role+'_'+field,shape)
        take('K',(T,nu,nx));take('H',(T,nu,nu));take('base_Qu',(T,nu))
        take('base_Qxu',(T,nx,nu));take('base_active',(T,nu));take('userdata',(int(h[14]),))
        if cursor<len(raw):
            count=int(raw[cursor]);cursor+=1;take('accepted_prefix_substeps',(count,))
        if int(h[0]) in (31,32,33):
            for key in ('r31_native_soft_substeps','r31_native_risk_peaks','r31_first_soft_AB'):
                count=int(raw[cursor]);cursor+=1;take(key,(count,))
            take('r31_held_proxy_A',(T-1,nx,nx))
            take('r31_held_proxy_B',(T-1,nx,nu))
        self.ring[pid]={'origin':float(h[2]),'last_ref':float(t),'data':data}
        if t<=self.until:self.save(pid)
        planner=self.out/'planner-records.jsonl'
        if planner.exists():
            with planner.open() as f:
                f.seek(self.plan_offset)
                rows=[json.loads(line) for line in f]
                self.plan_offset=f.tell()
            for row in rows:
                if row.get('controller_artifact') in ('S5-r33','S5-r34','S5-r35','S5-r36','S5-r37') and 'acceptance_mode' not in row:
                    # A scored nominal/structured archive winner has no alpha
                    # model ratio. Its recorded native violation defines this event.
                    if row['accepted_trunk_violation'] > 0:
                        self.event(t,'trunk_restoration',mode='scored_archive_restoration',
                            candidate_source=row['candidate_source'],plan_id=row['plan_id'])
                elif row['acceptance_mode']!='feasible_to_feasible_cost':
                    self.event(t,'trunk_restoration',mode=row['acceptance_mode'],plan_id=row['plan_id'])
        self.prune(t)
    def save(self,pid):
        if pid in self.saved:return
        row=self.ring[pid]
        name=f'accepted-reference-gain-H-plan{pid:06d}.npz'
        np.savez_compressed(self.out/name,**row['data'])
        self.saved[pid]=dict(plan_id=pid,origin=row['origin'],file=name)
    def prune(self,t):
        for call,row in list(self.r33_ring.items()):
            if row['origin']<t-.12:del self.r33_ring[call]
        for call,row in list(self.search_ring.items()):
            if row['origin']<t-.12:del self.search_ring[call]
        for pid,row in list(self.ring.items()):
            if row['origin']<t-.12 and row['last_ref']<t-.12:
                del self.ring[pid]
    def event(self,t,kind,first=False,**extra):
        if first and kind in self.seen:return
        self.seen.add(kind)
        start=max(0.,t-.1);end=t+.2
        self.events.append(dict(time_s=t,kind=kind,start_s=start,end_requested_s=end,**extra))
        self.until=max(self.until,end)
        for pid,row in self.ring.items():
            if row['origin']>=start or row['last_ref']>=start:self.save(pid)
        for call,row in self.search_ring.items():
            if row['origin']>=start:self.save_search(call)
        for call,row in self.r33_ring.items():
            if row['origin']>=start:self.save_r33(call)
    def query(self,t,kind,action=None,qpos=None,qvel=None):
        p=self.out/'feedback-actions.jsonl'
        if not p.exists():return None
        with p.open() as f:
            f.seek(self.offset)
            rows=[json.loads(line) for line in f]
            self.offset=f.tell()
        for row in rows:
            self.latest=row
            pid=row['plan_id']
            if pid in self.ring:self.ring[pid]['last_ref']=t
            link=dict(query_id=row['query_id'],time_s=t,plan_id=pid,query_kind=kind,
                      applied_action_id=None)
            if kind=='applied_hold':
                # Explicit numerical/initialization failure, never r28 fallback.
                if not row['metric_valid'] or not row['qp_converged']:
                    link['query_kind']='rejected_before_physics'
                    self.actions.write(json.dumps(link)+'\n');self.actions.flush()
                    self.event(t,'invalid_metric_or_QP')
                    self.failure=dict(schema='r29_rejected_action_v1',time_s=float(t),
                        physics_advanced=False,query=row,
                        rpc_target=np.asarray(action).tolist(),
                        qpos=np.asarray(qpos).tolist(),qvel=np.asarray(qvel).tolist())
                    # One small failure packet goes in compact JSON immediately;
                    # diagnosing a stopped stream must not await the raw queue.
                    plan=self.ring.get(pid)
                    if plan is not None:
                        nodes=row['interpolation_nodes'];fraction=row['interpolation_fraction']
                        metric=plan['data']['H']
                        self.failure['metric_H']=((1-fraction)*metric[nodes[0]]+
                                                  fraction*metric[nodes[1]]).tolist()
                        self.failure['metric_plan_id']=pid
                    else:
                        self.failure['metric_unavailable']='Referenced plan is absent from the retained event ring.'
                    (self.out/'qp-failure.json').write_text(json.dumps(self.failure,indent=2)+'\n')
                    raise RuntimeError('r29 action unavailable before physics: '+row['projection_status'])
                self.applied+=1;link['applied_action_id']=self.applied
                link['held_target']=np.asarray(action).tolist()
                self.actions.write(json.dumps(link)+'\n')
                center=np.asarray(row['center_before_projection'])
                nominal=np.clip(row['nominal_unclipped'],self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
                joints=self.model.actuator_trnid[:,0]
                raw_nom=60*(nominal-qpos[self.model.jnt_qposadr[joints]])-5*qvel[self.model.jnt_dofadr[joints]]
                sat_nom=(raw_nom<self.model.actuator_forcerange[:,0])|(raw_nom>self.model.actuator_forcerange[:,1])
                raw_center=60*(center-qpos[self.model.jnt_qposadr[joints]])-5*qvel[self.model.jnt_dofadr[joints]]
                sat_center=(raw_center<self.model.actuator_forcerange[:,0])|(raw_center>self.model.actuator_forcerange[:,1])
                if sat_nom.any():self.event(t,'first_nominal_PD_saturation',first=True)
                if (sat_center&~sat_nom).any():self.event(t,'feedback_added_saturation',first=True)
                if max(abs(np.asarray(row['raw_K_delta_x'])))>.5:self.event(t,'large_raw_feedback',first=True)
                if np.any(np.asarray(row['action_before_clip'])!=center):self.event(t,'first_target_clip',first=True)
                if row['restoration']:self.event(t,'actuator_restoration')
                self.actions.flush()
            else:self.actions.write(json.dumps(link)+'\n');self.actions.flush()
        self.prune(t)
        return self.latest
    def progress(self,t,qpos,command,contact):
        if abs(command[0])<=.05:
            self.nonprogress.clear()
            return
        self.nonprogress.append((t,float(qpos[0])))
        while self.nonprogress and self.nonprogress[0][0]<t-2.:self.nonprogress.popleft()
        if self.nonprogress and t-self.nonprogress[0][0]>=1.99 and abs(command[0])>.05 and abs(qpos[0]-self.nonprogress[0][1])<.01:
            self.event(t,'command_nonprogress_2s',first=True,dx_m=float(qpos[0]-self.nonprogress[0][1]),
                       command=np.asarray(command).tolist(),contacts=np.asarray(contact).tolist())
    def finish(self,t,reason):
        self.event(t,'terminal',reason=reason)
        for event in self.events:event['end_available_s']=min(event['end_requested_s'],t)
        result=dict(schema='r29_event_ring_v1',controller=getattr(self,'controller','S5-r29'),plans=list(self.saved.values()),
                    events=self.events,pre_window_s=.1,post_window_s=.2,terminal_s=t,
                    header_fields=['schema','plan_id','origin','horizon','ns','nu','nx','nr',
                      'regularization_used','regularization_type','nominal_feedback_scale',
                      'action_step_alpha','installed_feedback_scaling','representation','nuserdata','accepted_cost'],
                    plan_coverage='Only listed plans have full accepted/reference/K/H; no old reconstruction.',
                    raw_binary='r29-current-plan.bin is a current transport buffer, not a history.',
                    substep_clock='Each encoded row: node,substep,before,after,ncontact; pre-state,post-state,rawPD,actuator_force,qfrc_passive,residual; then ncontact*(geom1,geom2,6 contact-frame forces).')
        if getattr(self,'controller',None) in ('S5-r31','S5-r32','S5-r33','S5-r34','S5-r35','S5-r36','S5-r37'):
            result['r31_schema']=dict(
                model_roles='native geom/pair forward; independent smooth derivatives at native substeps',
                native_soft_substeps='(T-1,10,1+3*ns+nu): time,native_pre,held_u,native_post,soft_post',
                native_risk_peaks='(T-1,nrisk,2): native maximum,earliest active substep',
                first_soft_AB='(10,nx*nx+nx*nu): row-major one-step A_s then B_s at native states',
                held_proxy_A='(T-1,nx,nx)',held_proxy_B='(T-1,nx,nu)',
                scope='Approximate local model, no plant Jacobian or stability claim')
        if getattr(self,'controller',None) in ('S5-r32','S5-r33','S5-r34','S5-r35','S5-r36','S5-r37'):
            result['r32_search']=dict(plans=list(self.search_saved.values()),
                header_fields=['schema','call_id','problem_epoch','origin','horizon','ns','nu','nx',
                  'lambda_used','rate_used','backward_complete','failed_node','incumbent_plan_id',
                  'attempt','dV0_unscaled','dV1_unscaled'],
                scope='Last actual numerical attempt for each selected call; partial failed-backward arrays are not valid full Riccati results. Stops with no new attempt retain the prior search identity.',
                gradient='u-clip(u-Qu, original instantaneous target/PD bounds); independent of H/lambda, surrogate only')
        if getattr(self,'controller',None)in ('S5-r33','S5-r34','S5-r35','S5-r36','S5-r37'):
            result['r33_candidates']=dict(calls=list(self.r33_saved.values()),
                header_fields=['schema','call_id','kind','slot','alpha','horizon','ns','nu','nx','nr',
                    'feedback_scale','native_J','trunk_violation','failure','metric_valid','origin','mu',
                    'representation','input_plan_id'],
                kind={'0':'nominal','1':'alpha','2':'structured'},
                full_policy='Each candidate preserves its native accepted rollout and exact reference/actions/times/K/H/feedback scale; JSON planner calls carry selection and high-precision expected/actual descent.',
                foot_rows='Each row: time, then FL/FR/RL/RR each geom_xyz,foot_bottom_z,native_world_Fz,proxy_material_Fz,native_contact_count; after mj_step fields refer to its pre-integration node.',
                scale_header=['schema','call_id','horizon','nu','mean','floor'],
                scale_units='D is diagonal Hessian units cost/rad^2; mu is dimensionless.',
                scope='Bounded native event windows including non-winners. Missing scale means no matching scale was computed by this call, not a zero matrix.')
        (self.out/'r29-event-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        self.actions.close()
        return result
