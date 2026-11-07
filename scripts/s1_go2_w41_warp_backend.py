#!/usr/bin/env python3
"""Actual original-problem callback adapter and independent MJWarp evaluation."""
import copy,json,os,select,socket,struct,subprocess,time,traceback,shlex
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from mujoco_warp._src import support as sup
ROOT=Path(__file__).resolve().parents[1]
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT=LARGE/'gpu/w41-gpu';SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
@wp.func
def inside(p:wp.vec3):
    return p[0]>=0.8158113883008419 and p[0]<=2.1841886116991582 and wp.abs(p[1])<=0.5841886116991581
@wp.kernel
def filter_contact(geom:wp.array[wp.vec2i],world:wp.array[int],n:wp.array[int],x:wp.array2d[wp.vec3],feet:wp.array[int],dist:wp.array[float]):
    i=wp.tid()
    if i>=n[0]:return
    w=world[i]
    for k in range(4):
        g=feet[k]
        if (geom[i][0]==g or geom[i][1]==g) and inside(x[w,g]):dist[i]=1.0e10
@wp.kernel
def material_force(x:wp.array2d[wp.vec3],size:wp.array2d[wp.vec3],body:wp.array[int],feet:wp.array[int],
    userdata:wp.array2d[float],vel:wp.array2d[float],parent:wp.array[int],root:wp.array[int],dofbody:wp.array[int],
    ancestor:wp.array2d[int],com:wp.array2d[wp.vec3],cdof:wp.array2d[wp.spatial_vector],
    stiffness:float,floor:float,out:wp.array2d[float]):
    w,dof=wp.tid();force=float(0.0)
    for k in range(4):
        g=feet[k];pos=x[w,g]
        if inside(pos):
            point=wp.vec3(pos[0],pos[1],pos[2]-size[0,g][0]);rest=float(0.0)
            for p in range(4):
                off=62+4*p
                overlap=wp.max(1.0-wp.abs(point[0]-userdata[w,off+1])/0.03162277660168379,0.0)*wp.max(1.0-wp.abs(point[1]-userdata[w,off+2])/0.03162277660168379,0.0)
                rest+=userdata[w,off]*overlap*userdata[w,off+3]
            fz=stiffness*wp.max(floor+rest-point[2],0.0);velocity=wp.vec3(0.0)
            for j in range(18):
                jac,jacr=sup.jac_dof(parent,root,dofbody,ancestor,com,cdof,point,body[g],j,w)
                velocity+=jac*vel[w,j]
            slip=wp.sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+0.0001)
            f=wp.vec3(-0.5*fz*velocity[0]/slip,-0.5*fz*velocity[1]/slip,fz)
            jac,jacr=sup.jac_dof(parent,root,dofbody,ancestor,com,cdof,point,body[g],dof,w)
            force+=wp.dot(jac,f)
    out[w,dof]+=force

def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def difference(m,a,b):
    out=np.empty(18);mujoco.mj_differentiatePos(m,out,1.,a[:19],b[:19]);return np.r_[out,b[19:]-a[19:]]
class WarpPredictor:
    def __init__(self,xml):
        tree=ET.parse(xml);root=tree.getroot();root.remove(root.find('sensor'))
        root.find('option').set('timestep','.0002');root.find('option').set('integrator','implicitfast')
        path=OUT/'mjwarp-predictor.xml';tree.write(path,encoding='unicode')
        m=self.m=mujoco.MjModel.from_xml_path(str(path));m.geom_solimp[:,0]=0;m.jnt_solimp[:,0]=0;m.pair_solimp[:,0]=0
        m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.wm=mjw.put_model(m);self.wm.opt.tolerance=float(m.opt.tolerance)
        self.d=mjw.make_data(m,nworld=49,nconmax=128,njmax=768)
        self.feet=wp.array([m.geom(n).id for n in ['FL','RL','FR','RR']],dtype=int)
        self.stiff=float(m.numeric('r10_local_contact').data[0]);self.floor=float(m.numeric('s1_flat_ground_height').data[0])
        def cf(wm,d):wp.launch(filter_contact,dim=d.naconmax,inputs=[d.contact.geom,d.contact.worldid,d.nacon,d.geom_xpos,self.feet],outputs=[d.contact.dist])
        def pf(wm,d):wp.launch(material_force,dim=(49,18),inputs=[d.geom_xpos,wm.geom_size,wm.geom_bodyid,self.feet,d.userdata,d.qvel,wm.body_parentid,wm.body_rootid,wm.dof_bodyid,wm.body_isdofancestor,d.subtree_com,d.cdof,self.stiff,self.floor],outputs=[d.qfrc_passive])
        self.wm.callback.contactfilter=cf;self.wm.callback.passive=pf;self.graph=None;self.eps=1e-3
    def load(self,row,fd):
        q=np.tile(row['qpos'],(49,1));v=np.tile(row['qvel'],(49,1));u=np.tile(row['ctrl'],(49,1));sign=np.ones(48)
        if fd:
            for k in range(48):
                if k<18:
                    delta=np.zeros(18);delta[k]=self.eps;mujoco.mj_integratePos(self.m,q[k+1],delta,1.)
                elif k<36:v[k+1,k-18]+=self.eps
                else:
                    j=k-36;lo,hi=self.m.actuator_ctrlrange[j];val=u[k+1,j]
                    step=self.eps if lo<=val and val+self.eps<=hi else (-self.eps if lo<=val-self.eps and val<=hi else 0.)
                    u[k+1,j]+=step;sign[k]=step/self.eps
        for name,array in [('qpos',q),('qvel',v),('ctrl',u),('userdata',np.tile(row['userdata'],(49,1))),
            ('time',np.full(49,row['time'])),('mocap_pos',np.tile(row['mocap_pos'],(49,1,1))),('mocap_quat',np.tile(row['mocap_quat'],(49,1,1)))]:
            target=getattr(self.d,name);wp.copy(target,wp.array(array,dtype=target.dtype))
        self.d.qacc_warmstart.zero_()
        return sign
    def request(self,row,fd,eps=None):
        if eps is not None:self.eps=eps
        row={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in row.items()};self.load(row,fd)
        if self.graph is None:
            mjw.step(self.wm,self.d);wp.synchronize();self.load(row,fd)
            with wp.ScopedCapture() as cap:mjw.step(self.wm,self.d)
            self.graph=cap.graph
        states=[];AB=[]
        for j in range(10):
            signs=self.load(row,fd) if fd else np.ones(48)
            wp.capture_launch(self.graph);wp.synchronize()
            ys=np.c_[self.d.qpos.numpy(),self.d.qvel.numpy()].astype(float);states.append(ys[0])
            if fd:
                AB.append(np.stack([difference(self.m,ys[0],x)/self.eps for x in ys[1:]],axis=1)*signs)
                row.update(qpos=ys[0,:19],qvel=ys[0,19:],time=row['time']+.0002)
        return np.asarray(states),np.asarray(AB)

def recv(conn,n):
    b=b''
    while len(b)<n:
        part=conn.recv(n-len(b))
        if not part:raise RuntimeError('incomplete native request')
        b+=part
    return b

def main():
    report=dict(status='preparing_actual_MJWarp_adapter',dtype='float32',deadline_unix_s=None,
        versions=dict(mujoco=mujoco.__version__,warp=wp.__version__),predictor_only=True,original_cpu_plant_unchanged=True,
        first_error=None,stages=[],fd_steps=[3e-4,1e-3,3e-3],selected_fd_step=1e-3)
    try:
        fixture=json.loads((OUT/'fixtures.json').read_text());xml=ROOT/fixture[0]['source']/'model/G1-project-go2.xml'
        pred=WarpPredictor(xml)
        report['solver']=dict(iterations=int(pred.m.opt.iterations),tolerance=float(pred.m.opt.tolerance),native_tolerance_retained=True)
        for i in [0,4,11,20,23,27]:
            z=np.load(LARGE/'w41-dynamics/native-v2'/(fixture[i]['name']+'.npz'));x=z['starts'][0]
            row=dict(time=fixture[i]['time'],qpos=x[:19],qvel=x[19:37],ctrl=x[37:],userdata=z['userdata'],mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'])
            values=[];measure=[]
            for eps in report['fd_steps']:
                t=time.perf_counter();states,ab=pred.request(row,True,eps);wall=time.perf_counter()-t
                values.append(ab);measure.append(dict(eps=eps,wall_s=wall,AB_max=float(abs(ab).max())))
                np.savez_compressed(OUT/f'mjwarp-{i:02d}-eps{eps:g}.npz',states=states,AB=ab,input=x,userdata=row['userdata'])
            report['stages'].append(dict(fixture=fixture[i]['name'],step_sensitivity=measure,
                adjacent_relative_AB_changes=[float(np.linalg.norm(values[j]-values[j+1])/max(np.linalg.norm(values[j+1]),1e-12)) for j in [0,1]],
                interpretation='Exploratory FP32 finite-difference sensitivity, not equivalent-FP64 qualification or an admission gate'))
            write(OUT/'mjwarp-result.json',report)
        pred.eps=report['selected_fd_step']
        config=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
        config.update(current_task='W41_GPU_MJWARP_FIXED_POLICY',protocol_revision='W41_MJWARP4_FP32_v1')
        config['reuse'].update(active_controller_build='S5-r30',controller_artifact='S5-r30-gpu-warp')
        config['resources']['mpc_workers']=2;config['learning'].update(phase_id='w41-gpu-mjwarp',stream_id='w41-gpu-eval')
        contract=OUT/'mjwarp-evaluation-contract.json';write(contract,config)
        path=ROOT/'work/w41-mjwarp.sock'
        if path.exists():path.rename(path.with_name('w41-mjwarp-'+str(time.time_ns())+'.sock'))
        server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));server.listen(4)
        env=dict(os.environ,MJPC_W41_GPU_SOCKET=str(path));checkpoint=LARGE/'training/w40-b/PANN_PPO_S5-seed2/checkpoint-009617.pt'
        report['evaluations']=[]
        for level in ['E30','E60']:
            attempt='w41-gpu-mjwarp-PANN_PPO_S5-s2-update37-'+level
            args=[str(ROOT/'work/envs/d57/bin/python'),'-u','scripts/run_s1_go2_r12.py','--contract',str(contract),'--attempt',attempt,
                '--build','S5-r30','--group','PANN_PPO_S5','--seed','2','--checkpoint',str(checkpoint),'--level',level,'--workers','2']
            cmd=['bash','-c','source scripts/autodl_d57_env.sh; exec '+shlex.join(args)]
            count=0;report['status']='independent_'+level+'_running';write(OUT/'mjwarp-result.json',report)
            with (OUT/(attempt+'.log')).open('w') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if not select.select([server],[],[],1)[0]:continue
                    conn=server.accept()[0];header=struct.unpack('8Q',recv(conn,64));mode=bool(header[1])
                    raw=np.frombuffer(recv(conn,153*8),dtype=np.float64).copy()
                    row=dict(time=float(raw[0]),qpos=raw[1:20],qvel=raw[20:38],ctrl=raw[38:50],mocap_pos=raw[50:53].reshape(1,3),mocap_quat=raw[53:57].reshape(1,4),userdata=raw[57:])
                    states,ab=pred.request(row,mode);response=np.r_[states.ravel(),ab.ravel()] if mode else states.ravel()
                    count+=1
                    if count<=10:np.savez_compressed(OUT/f'mjwarp-{level}-request{count:06d}.npz',input=raw,states=states,AB=ab)
                    conn.sendall(struct.pack('Q',len(response))+response.astype(np.float64).tobytes());conn.close()
                result_path=ROOT/config['outputs']['compact_root']/(attempt+'.json')
                result=json.loads(result_path.read_text()) if result_path.exists() else {'success':False,'stage_status':'technical_interruption','reason':'See native evaluation log'}
                report['evaluations'].append(dict(level=level,attempt=attempt,requests=count,exit_code=process.returncode,
                    result={k:result.get(k) for k in ['success','reason','stage_status','simulation_s','wall_s','legal_route','qualification']}))
                write(OUT/'mjwarp-result.json',report)
                if not result.get('success'):break
        server.close();report['status']='completed_independent_MJWarp_comparison'
    except Exception:
        report.update(status='technical_interruption',first_error=traceback.format_exc())
        if 'raw' in locals():np.savez_compressed(OUT/'mjwarp-first-failure-input.npz',input=raw)
        if 'process' in locals() and process.poll() is None:process.terminate();process.wait()
        raise
    finally:
        write(OUT/'mjwarp-result.json',report);write(SMALL/'mjwarp-result.json',report)
if __name__=='__main__':main()
