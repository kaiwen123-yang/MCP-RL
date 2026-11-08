#!/usr/bin/env python3
"""W41 resident MJX event expansion, full horizon batches and native iLQG bridge."""
import argparse,json,os,subprocess,time,traceback,resource
from pathlib import Path
import numpy as np
import mujoco,jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jp
from mujoco.mjx._src import math as jmath
from s1_go2_w40_gpu import Predictor,state,diff
from s1_go2_w41_dynamics import CPUHold,event_rows,contact_record
ROOT=Path(__file__).resolve().parents[1]
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT=LARGE/'gpu/w41-gpu'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sync(x):return jax.block_until_ready(x)
class Resident:
    def __init__(self,model):
        self.pred=Predictor(model);p=self.pred;self.m=p.m
        self.local=jax.jit(self.at_cpu_prefixes)
        self.hold=jax.jit(self.held_fd)
        self.forward=jax.jit(self.held_forward)
    def perturbations(self,d):
        p=self.pred;eps=1e-6
        b=jax.tree.map(lambda x:jp.broadcast_to(x,(49,)+x.shape),d)
        q=b.qpos.at[1:4,:3].add(eps*jp.eye(3))
        quat=jax.vmap(lambda v:jmath.quat_integrate(d.qpos[3:7],v,eps))(jp.eye(3))
        q=q.at[4:7,3:7].set(quat).at[7:19,7:19].add(eps*jp.eye(12))
        v=b.qvel.at[19:37].add(eps*jp.eye(18))
        lo=jp.array(self.m.actuator_ctrlrange[:,0]);hi=jp.array(self.m.actuator_ctrlrange[:,1])
        fwd=(d.ctrl>=lo)&(d.ctrl+eps<=hi);back=(d.ctrl-eps>=lo)&(d.ctrl<=hi)
        sign=jp.where(fwd,1.,jp.where(back,-1.,0.))
        u=b.ctrl.at[37:49].add(eps*jp.diag(sign))
        return b.replace(qpos=q,qvel=v,ctrl=u),jp.concatenate([jp.ones(36),sign])
    def difference(self,y,base):
        return jp.concatenate([y[:3]-base[:3],jmath.quat_sub(y[3:7],base[3:7]),y[7:19]-base[7:19],y[19:]-base[19:]])
    def payload(self,bd,sign):
        ys=jp.concatenate([bd.qpos,bd.qvel],axis=1)
        ab=jax.vmap(lambda y:self.difference(y,ys[0]))(ys[1:]).T/1e-6*sign
        d=jax.tree.map(lambda x:x[0],bd)
        return dict(states=ys,AB=ab,contact_geom=d.contact.geom,contact_dist=d.contact.dist,
            contact_pos=d.contact.pos,contact_frame=d.contact.frame,contact_efc=d.contact.efc_address,
            solver_niter=d.solver_niter,efc_force=d.efc_force,qfrc_passive=d.qfrc_passive,
            actuator_force=d.actuator_force,qfrc_constraint=d.qfrc_constraint)
    def at_cpu_prefixes(self,d,prefix):
        def step(carry,x):
            inp=d.replace(qpos=x[:19],qvel=x[19:37],time=x[37])
            batch,sign=self.perturbations(inp)
            out=jax.vmap(self.pred.step_gpu)(batch)
            return carry,self.payload(out,sign)
        return jax.lax.scan(step,jp.array(0),prefix)[1]
    def step_with_fixed_contact_indices(self,d):
        out=self.pred.step_gpu(d)
        # MJX3.2.3 collision_driver creates these arrays from NumPy int64
        # under x64, while make_data initializes int32. Preserve the incoming
        # index representation at the scan boundary; all float64 dynamics,
        # contact values, finite differences and solver settings are unchanged.
        contact=out.contact.replace(geom1=out.contact.geom1.astype(d.contact.geom1.dtype),
            geom2=out.contact.geom2.astype(d.contact.geom2.dtype),
            geom=out.contact.geom.astype(d.contact.geom.dtype))
        return out.replace(contact=contact)
    def held_fd(self,d):
        def step(carry,_):
            batch,sign=self.perturbations(carry)
            out=jax.vmap(self.step_with_fixed_contact_indices)(batch)
            nextd=jax.tree.map(lambda x:x[0],out)
            return nextd,self.payload(out,sign)
        return jax.lax.scan(step,d,None,length=10)[1]
    def held_forward(self,d):
        def step(carry,_):
            nxt=self.step_with_fixed_contact_indices(carry)
            return nxt,jp.concatenate([nxt.qpos,nxt.qvel])
        return jax.lax.scan(step,d,None,length=10)[1]
    def data(self,row):return self.pred.gpu_data(row)
    def request(self,row,fd):
        return sync((self.hold if fd else self.forward)(self.data(row)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from',choices=['G2'])
    parser.add_argument('--resume-record',type=Path)
    parser.add_argument('--invocation')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);SMALL.mkdir(parents=True,exist_ok=True)
    if args.resume_from:
        if args.resume_record is None or not args.invocation:
            parser.error('G2 continuation requires its preserved result record and invocation ID')
        report=json.loads(args.resume_record.read_text())
        if len(report['stages'].get('G1',[]))!=report['new_events']:
            raise ValueError('G2 continuation requires the completed G1 record; do not rerun G1')
        if report.get('first_technical_error'):
            report.setdefault('technical_history',[]).append(dict(
                status=report['status'],updated_unix_s=report['updated_unix_s'],
                error=report.pop('first_technical_error'),record=str(args.resume_record.relative_to(ROOT))))
        report['G1_reuse']=dict(record=str(args.resume_record.relative_to(ROOT)),
            completed_events=len(report['stages']['G1']),recomputed=False)
        report['continuation']=dict(invocation=args.invocation,resume_from='G2',
            started_unix_s=time.time(),deadline_unix_s=None,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            change='Preserve contact geom1/geom2/geom integer dtypes across held FD and forward scan')
        report['version']='w41-resident-mjx323-fp64-contact-index-fix1'
        report['status']='G2_resuming'
    else:
        report=dict(task='w41-gpu',version='w41-resident-mjx323-fp64-contact-index-fix1',deadline_unix_s=None,
            started_unix_s=time.time(),status='G1_preparing',stages={},original8='retained gpu/w40-gpu; no recomputation',
            physical_plant='original CPU Go2/TM02',equivalent_backend=False,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            memory_budget_GiB=16,host_threads=2,compile_jobs=2,ppo_branch=False)
    def publish(status):
        report['status']=status;report['updated_unix_s']=time.time()
        report['host_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['device_memory_stats']=jax.devices()[0].memory_stats()
        for p in [OUT/'result.json',SMALL/'result.json']:write(p,report)
        print(json.dumps(dict(status=status,events=len(report['stages'].get('G1',[])),time=time.time())),flush=True)
    try:
        if args.resume_from:
            metadata=json.loads((OUT/'fixtures.json').read_text());row0=dict(metadata[0])
            with np.load(OUT/(row0['name']+'-gpu.npz')) as saved:
                row0.update(qpos=saved['input_prefix'][0,:19],qvel=saved['input_prefix'][0,19:37],
                    ctrl=saved['ctrl'],userdata=saved['userdata'],mocap_pos=saved['mocap_pos'],mocap_quat=saved['mocap_quat'])
            resident=Resident(ROOT/row0['source']/'model.mjb');m=resident.m
            publish('G2_resuming_completed_G1_reused')
        else:
            rows=event_rows();write(OUT/'fixtures.json',[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in rows])
            resident=Resident(ROOT/rows[0]['source']/'model.mjb');m=resident.m
            report['versions']=dict(mujoco=mujoco.__version__,jax=jax.__version__,mjx='3.2.3')
            report['new_events']=len(rows);report['stages']['G1']=[]
            publish('G1_compiling_resident_prefix_scan_49_inputs')
            for i,row in enumerate(rows):
                cpu=CPUHold(ROOT/row['source']/'model.mjb')
                mujoco.set_mjcb_contactfilter(cpu.cpu_filter);mujoco.set_mjcb_passive(cpu.cpu_passive)
                d=cpu.cpu_data(row);prefix=[];contacts=[]
                for j in range(10):
                    prefix.append(np.r_[d.qpos,d.qvel,d.time]);mujoco.mj_step(cpu.m,d)
                    contacts.append(contact_record(cpu.m,d))
                mujoco.set_mjcb_contactfilter(None);mujoco.set_mjcb_passive(None)
                begin=time.perf_counter();gd=resident.data(row);pp=jp.asarray(np.array(prefix));sync((gd,pp));h2d=time.perf_counter()-begin
                begin=time.perf_counter();out=sync(resident.local(gd,pp));elapsed=time.perf_counter()-begin
                begin=time.perf_counter();a={k:np.asarray(v) for k,v in out.items()};d2h=time.perf_counter()-begin
                reference=np.load(LARGE/'w41-dynamics/native-v2'/(row['name']+'.npz'))
                cpu_y=reference['local_FD_endpoints'];er=np.array([[diff(m,cpu_y[j,k],a['states'][j,k]) for k in range(49)] for j in range(10)])
                ab_cpu=np.concatenate([reference['A'],reference['B']],axis=2)
                ab_error=abs(ab_cpu-a['AB']);P=np.eye(36);Q=np.zeros((36,12))
                for ab in a['AB']:P,Q=ab[:,:36]@P,ab[:,:36]@Q+ab[:,36:]
                first=np.argwhere((abs(er[:,:,:18]).max(axis=2)>5e-7)|(abs(er[:,:,18:]).max(axis=2)>1e-5))
                reason='no state threshold crossing' if not len(first) else 'native/MJX contact or constraint-solve branch; first differing input saved, compare geometry/force/solver below'
                row_result=dict(fixture=row['name'],index=i,source=row['source'],time_s=row['time'],
                    first_difference=first[0].tolist() if len(first) else None,reason=reason,
                    position_max=float(abs(er[:,:,:18]).max()),velocity_max=float(abs(er[:,:,18:]).max()),
                    AB_max=float(ab_error.max()),first_call_includes_compilation=i==0,wall_s=elapsed,H2D_s=h2d,D2H_s=d2h,
                    aligned=bool(abs(er[:,:,:18]).max()<=5e-7 and abs(er[:,:,18:]).max()<=1e-5 and ab_error.max()<=5e-3))
                np.savez_compressed(OUT/(row['name']+'-gpu.npz'),**a,state_error=er,AB_error=ab_error,P=P,Q=Q,
                    input_prefix=np.array(prefix),ctrl=row['ctrl'],userdata=row['userdata'],mocap_pos=row['mocap_pos'],mocap_quat=row['mocap_quat'])
                write(OUT/(row['name']+'-native-contacts.json'),contacts)
                report['stages']['G1'].append(row_result);publish('G1_event_completed')
            report['equivalent_backend']=all(x['aligned'] for x in report['stages']['G1'])
            row0=rows[0]
        # The existing negative result can continue only under an explicit NEW
        # predictor identity; numerical tolerances remain the original values.
        report['predictor_identity']=('equivalent_mjx323' if report['equivalent_backend'] else 'new_mjx323_contact_predictor')
        source=ROOT/row0['source']
        plan_path=(ROOT/report['stages']['G2']['source'] if args.resume_from else sorted(source.glob('accepted-reference-gain-H-plan*.npz'))[0])
        with np.load(plan_path) as z:plan={k:z[k] for k in z.files}
        if not args.resume_from:
            report['stages']['G2']=dict(source=str(plan_path.relative_to(ROOT)),nodes=176,inputs_per_node=49,logical_inputs=8624,node_blocks=[])
        completed={entry['node'] for entry in report['stages']['G2']['node_blocks']}
        publish('G2_resident_full_horizon')
        for i in range(176):
            if i in completed:continue
            node_path=OUT/f'G2-node{i:03d}.npz'
            if node_path.exists():raise FileExistsError('Unindexed G2 output retained for inspection: '+str(node_path))
            r=dict(row0,time=float(plan['accepted_times'][i]),qpos=plan['accepted_states'][i,:19],
                qvel=plan['accepted_states'][i,19:],ctrl=plan['accepted_actions'][min(i,174)],userdata=plan['userdata'])
            t=time.perf_counter();out=resident.request(r,True);wall=time.perf_counter()-t
            # Keep complete derivatives and inputs, not just a timing scalar.
            np.savez_compressed(node_path,**{k:np.asarray(v) for k,v in out.items()},
                qpos=r['qpos'],qvel=r['qvel'],ctrl=r['ctrl'],userdata=r['userdata'],
                input_time_s=r['time'],mocap_pos=r['mocap_pos'],mocap_quat=r['mocap_quat'],node_index=i,wall_s=wall)
            report['stages']['G2']['node_blocks'].append(dict(node=i,wall_s=wall,microbatch=49))
            publish('G2_node_block_completed')
        # G3/G4 execute through the actual native cost/private-phase/iLQG outer
        # loop; the isolated predictor is explicit and never installed in A/B.
        from s1_go2_w41_gpu_bridge import integrated_evaluation
        integrated_evaluation(resident,report,publish)
        publish('completed_declared_gpu_stages')
    except Exception:
        if args.invocation and report['status'].startswith('G2') and 'r' in locals():
            failure=OUT/'resume-history'/args.invocation/'first-G2-failure-input.npz'
            np.savez_compressed(failure,**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},time_s=r['time'],node=i)
            report['first_failure_input']=str(failure.relative_to(ROOT))
        report['first_technical_error']=traceback.format_exc();publish('technical_interruption');raise
if __name__=='__main__':main()
