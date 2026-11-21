"""Exercise NVIDIA in-process GPU checkpoint and restore on the real B actor and packet."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path('/root/autodl-tmp/MCP-RL-w55-integration/work/r58-e300-stall')
BASE=Path('/root/autodl-tmp/MCP-RL-w56-material')
def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def workload(work):
    import numpy as np
    import torch
    from s1_go2_r11_policy import PolicyStream
    from selected_cells import DATA,WEIGHTS,replay
    from tm02.s1_material import LearnedColumns
    base=DATA/'training/w44-b/PANN_PPO_S5-seed2'
    contract=json.loads((base/'training-contract.json').read_text())
    if 'joint_learning' not in contract:
        contract['joint_learning']=json.loads((BASE/contract['reuse']['runtime_physics_ppo_and_task_settings']).read_text())['joint_learning']
    with np.load(base/'rollout-014226-014481.npz') as z:
        choices=z['obs'].reshape(-1,658)
    observation=choices[np.argmax(abs(choices[:,-12:]).max(1))]
    stream=PolicyStream(BASE,'PANN_PPO_S5',2,contract,work,training=False,checkpoint=base/'checkpoint-014737.pt')
    obs=torch.as_tensor(observation,device=stream.backend.config.device).reshape(1,-1)
    class RealCall:
        def __init__(self):
            self.law=LearnedColumns(WEIGHTS)
        def step(self,*args):
            self.last=args
            return self.law.step(*args)
    capture=RealCall()
    replay('b',capture)
    def measure():
        r=capture.law.step(*capture.last)
        sample=stream.backend.act(obs,deterministic=True)
        return dict(action=float(sample.action.item()),material_fp64_sha256=hashlib.sha256(
                    b''.join(a.tobytes() for a in r)).hexdigest(),rows=len(r[0]))
    write(work/'before.json',measure())
    calls=0
    while not (work/'release').exists():
        result=measure();calls+=1
    write(work/'after.json',dict(result=measure(),calls=calls))

class LockArgs(C.Structure):
    _fields_=[('timeoutMs',C.c_uint),('reserved0',C.c_uint),('reserved1',C.c_uint64*7)]

def cycle(pid):
    drv=C.CDLL('libcuda.so.1')
    drv.cuCheckpointProcessGetState.argtypes=[C.c_int,C.POINTER(C.c_int)]
    for suffix in ('Lock','Unlock','Checkpoint','Restore'):
        getattr(drv,'cuCheckpointProcess'+suffix).argtypes=[C.c_int,C.c_void_p]
    def state():
        s=C.c_int(-1);rc=drv.cuCheckpointProcessGetState(pid,C.byref(s))
        return [rc,s.value]
    results=[]
    def invoke(suffix,arg=None):
        start=time.time(); rc=getattr(drv,'cuCheckpointProcess'+suffix)(pid,arg)
        name=C.c_char_p();drv.cuGetErrorName(rc,C.byref(name))
        results.append(dict(call=suffix,code=rc,name=name.value.decode(),elapsed_s=time.time()-start,state=state()))
        return rc
    if drv.cuInit(0) or state()!=[0,0]:raise RuntimeError('Driver or state not ready')
    if invoke('Lock',C.byref(LockArgs(timeoutMs=2000)))==0:
        try:
            if invoke('Checkpoint')==0:
                invoke('Restore')
        finally:
            if state()==[0,1]:invoke('Unlock')
    return results

def main():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        work=Path(temporary)
        child=subprocess.Popen([sys.executable,'-u',__file__,'worker',str(work)])
        while not (work/'before.json').exists():
            if child.poll() is not None:raise RuntimeError('Rehearsal workload exited during initialization')
            time.sleep(.2)
        before=json.loads((work/'before.json').read_text())
        try:
            print('Running checkpoint rehearsal on disposable B material/actor PID',child.pid,flush=True)
            rows=cycle(child.pid)
        finally:
            (work/'release').touch()
        rc=child.wait()
        after=json.loads((work/'after.json').read_text()) if (work/'after.json').is_file() else None
        report=dict(pid=child.pid,source='B original W44 actor and frozen normalization; last mixed material input in selected-cell replay at 78.000 s',
            driver_calls=rows,child_exit=rc,before=before,after=after,
            successful_cycle=bool(len(rows)==4 and all(r['code']==0 for r in rows) and after and before==after['result']))
        write(HERE/'rehearsal.json',report)
        print(report,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:workload(Path(sys.argv[2]))
    else:main()
