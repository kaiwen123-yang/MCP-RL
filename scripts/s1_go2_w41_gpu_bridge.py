"""The independent GPU predictor serves the actual CPU iLQG outer loop.
The task residual, private gait phase, candidate feedback, QP, (v,J) ranking and
original TM02 plant remain in their owning native/runtime implementations.
"""
import copy,gzip,json,os,select,socket,struct,subprocess,time,traceback
from pathlib import Path
import numpy as np
import jax
from s1_go2_w41_gpu import ROOT,LARGE,OUT,SMALL,write,sync

def recv_exact(conn,n):
    chunks=[]
    while n:
        b=conn.recv(n)
        if not b:raise RuntimeError('native predictor request ended before complete input')
        chunks.append(b);n-=len(b)
    return b''.join(chunks)

def unpack(conn):
    header=struct.unpack('8Q',recv_exact(conn,64));magic,mode,nq,nv,na,nu,nd,nm=header
    if magic!=0x573431475055 or (nq,nv,na,nu,nd,nm)!=(19,18,0,12,96,1):
        raise RuntimeError('W41 predictor request model differs from declared original Go2')
    v=np.frombuffer(recv_exact(conn,(1+nq+nv+na+nu+7*nm+nd)*8),dtype=np.float64).copy()
    row=dict(time=float(v[0]),qpos=v[1:20],qvel=v[20:38],ctrl=v[38:50],
        mocap_pos=v[50:53].reshape(1,3),mocap_quat=v[53:57].reshape(1,4),userdata=v[57:])
    return int(mode),row,v

def serve_evaluation(resident,command,env,prefix,report,publish):
    socket_path=ROOT/'work/w41-gpu.sock'
    if socket_path.exists():
        # A previous service created this named socket. Preserve its directory
        # entry as evidence before binding the new service; no process killing.
        socket_path.rename(socket_path.with_name('w41-gpu-'+str(time.time_ns())+'.sock'))
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(socket_path));server.listen(4)
    env=dict(env,MJPC_W41_GPU_SOCKET=str(socket_path))
    many_fd=jax.jit(jax.vmap(resident.held_fd));many_forward=jax.jit(jax.vmap(resident.held_forward))
    logpath=OUT/(prefix+'.log');requests=0;begin=time.monotonic();wall=[]
    with logpath.open('w') as log,(OUT/(prefix+'-predictor.jsonl')).open('w') as trace, gzip.open(OUT/(prefix+'-complete-predictor-input-output.bin.gz'),'wb',compresslevel=1) as evidence:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if not select.select([server],[],[],1)[0]:continue
                clients=[server.accept()[0]]
                if select.select([server],[],[],.003)[0]:clients.append(server.accept()[0])
                pending=[(c,*unpack(c)) for c in clients]
                for mode in sorted(set(x[1] for x in pending)):
                    group=[x for x in pending if x[1]==mode];rows=[x[2] for x in group]
                    t=time.perf_counter()
                    if len(group)==1:
                        output=resident.request(rows[0],bool(mode))
                        results=[output]
                    else:
                        batch=jax.tree.map(lambda *xs:np.stack(xs),*[resident.data(x) for x in rows])
                        output=sync((many_fd if mode else many_forward)(batch))
                        results=[jax.tree.map(lambda a:a[i],output) for i in range(len(group))]
                    elapsed=time.perf_counter()-t;wall.append(elapsed)
                    for (conn,_,row,raw),out in zip(group,results):
                        requests+=1
                        if mode:
                            arrays={k:np.asarray(v) for k,v in out.items()}
                            response=np.r_[arrays['states'][:,0,:].ravel(),arrays['AB'].ravel()]
                        else:
                            arrays={'states':np.asarray(out)};response=arrays['states'].ravel()
                        if requests<=10:
                            np.savez_compressed(OUT/f'{prefix}-request{requests:06d}.npz',input=raw,mode=mode,**arrays)
                        evidence.write(struct.pack('3Q',requests,mode,len(response))+raw.astype(np.float64).tobytes()+response.astype(np.float64).tobytes())
                        conn.sendall(struct.pack('Q',len(response))+response.astype(np.float64).tobytes());conn.close()
                        trace.write(json.dumps(dict(request=requests,mode=mode,time_s=row['time'],batch_nodes=len(group),wall_s=elapsed))+'\n');trace.flush()
                    if requests%100==0:
                        report['live_bridge']=dict(requests=requests,wall_s=time.monotonic()-begin,last_batch_wall_s=elapsed)
                        publish('G3_G4_native_outer_gpu_predictor_running')
        except Exception:
            # Save the complete first request before ending this explicitly
            # failed evaluation. Do not substitute a CPU prediction or retry.
            for i,x in enumerate(locals().get('pending',[])):
                np.savez_compressed(OUT/f'{prefix}-first-failure-{i}.npz',input=x[3],mode=x[1])
            write(OUT/(prefix+'-first-failure.json'),dict(error=traceback.format_exc(),command=command))
            process.terminate();process.wait();raise
        finally:server.close()
    return dict(exit_code=process.returncode,requests=requests,wall_s=time.monotonic()-begin,
        batch_wall_p50_s=float(np.median(wall)) if wall else None,log=str(logpath.relative_to(ROOT)))

def integrated_evaluation(resident,report,publish):
    config=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
    # The same immutable own update37 is used for the new predictor and native
    # comparison. Neither result is a repeated W40 job or an extra PPO lineage.
    checkpoint=LARGE/'training/w40-b/PANN_PPO_S5-seed2/checkpoint-009617.pt'
    if not checkpoint.is_file():raise FileNotFoundError('Declared immutable W40 B update37 checkpoint missing')
    report['stages']['G3']=dict(status='native_full_stack_integration',
        checkpoint=str(checkpoint.relative_to(ROOT)),private_state='New named fresh mission; original native private task phase advances through original TransitionLocked',
        cost='Original native101 residuals, weights, risk, ten-substep path maxima',
        candidates='Original10 feedback/action candidates, GPU forward in up-to-two-candidate blocks, native r30 QP and (v,J) sorting',
        derivatives='GPU local A/B + native node-start sensor C/D and native risk geometry Jacobians at GPU prefix states',
        fixture_identity='w41-gpu-new-full-stack-seed2-update37; not a historical planner replay',
        complete_candidate_transport='gzip binary sequence: uint64 request,mode,response_count;153 float64 input;response_count float64 output. All forward/derivative requests retained; native planner records own costs/ranking.')
    report['stages']['G4']=[]
    env=dict(os.environ);env['PYTHONPATH']=':'.join(x for x in env.get('PYTHONPATH','').split(':') if 'gpu-w40-python' not in x)
    for backend,artifact in [('native','S5-r30'),('mjx','S5-r30-gpu-mjx')]:
        contract=copy.deepcopy(config)
        contract.update(current_task='W41_GPU_'+backend.upper()+'_FIXED_POLICY',protocol_revision='W41_GPU_'+backend.upper()+'_v1')
        contract['reuse'].update(active_controller_build='S5-r30',controller_artifact=artifact)
        contract['resources']['mpc_workers']=2
        contract['learning'].update(phase_id='w41-gpu-'+backend,stream_id='w41-gpu-eval')
        file=OUT/(backend+'-evaluation-contract.json');write(file,contract)
        for level in ['E30','E60']:
            attempt='w41-gpu-'+backend+'-PANN_PPO_S5-s2-update37-'+level
            command=[str(ROOT/'work/envs/d57/bin/python'),'-u','scripts/run_s1_go2_r12.py',
                '--contract',str(file),'--attempt',attempt,'--build','S5-r30','--group','PANN_PPO_S5',
                '--seed','2','--checkpoint',str(checkpoint),'--level',level,'--workers','2']
            publish('G4_'+backend+'_'+level+'_running')
            if backend=='mjx':execution=serve_evaluation(resident,command,env,attempt,report,publish)
            else:
                log=OUT/(attempt+'.log');t=time.monotonic()
                with log.open('w') as f:proc=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                execution=dict(exit_code=proc.returncode,wall_s=time.monotonic()-t,log=str(log.relative_to(ROOT)))
            resultpath=ROOT/contract['outputs']['compact_root']/(attempt+'.json')
            result=json.loads(resultpath.read_text()) if resultpath.exists() else dict(stage_status='technical_interruption',reason='No physical result; see retained execution log')
            row=dict(backend=backend,level=level,attempt=attempt,execution=execution,
                result={k:result.get(k) for k in ['stage_status','reason','success','simulation_s','wall_s','legal_route','qualification','stage_progress']})
            report['stages']['G4'].append(row);publish('G4_'+backend+'_'+level+'_completed')
            if not result.get('success'):break
    report['stages']['G3']['status']='completed_actual_outer_loop'
    # G5 is one selected forward-throughput/contact comparison, never a
    # collection of unrelated algorithms. The new backend has its own identity.
    from s1_go2_w41_mjwarp import run_selected_alternative
    run_selected_alternative(report,publish)
