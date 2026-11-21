"""Separate ongoing device work from new CPU-side GPU launches for the two original processes."""
if __name__=='__main__':
    import ctypes as C
    import json
    from pathlib import Path
    import subprocess
    import time
    from contextlib import ExitStack

    class LockArgs(C.Structure):
        _fields_=[('timeoutMs',C.c_uint),('reserved',C.c_uint),('pad',C.c_uint64*7)]

    drv=C.CDLL('libcuda.so.1')
    drv.cuCheckpointProcessGetState.argtypes=[C.c_int,C.POINTER(C.c_int)]
    drv.cuCheckpointProcessLock.argtypes=[C.c_int,C.POINTER(LockArgs)]
    drv.cuCheckpointProcessUnlock.argtypes=[C.c_int,C.c_void_p]

    def state(p):
        s=C.c_int(-1)
        if drv.cuCheckpointProcessGetState(p,C.byref(s))!=0: raise RuntimeError('No driver state')
        return s.value

    def release(p):
        rc=drv.cuCheckpointProcessUnlock(p,None)
        results['unlock'][p]=dict(code=rc,state=state(p),unix_s=time.time())

    pids=[648411,652013]
    results=dict(start_unix_s=time.time(),locked={},unlock={})
    with ExitStack() as stack:
        for p,case in zip(pids,['a','b']):
            cmd=Path('/proc',str(p),'cmdline').read_bytes().split(b'\0')
            attempt=cmd[cmd.index(b'--attempt')+1].decode()
            if f'-w56-f-{case}-r58-eval-' not in attempt or not attempt.endswith('E300-attempt01') or state(p)!=0:
                raise RuntimeError('Original attempt or current state changed')
            rc=drv.cuCheckpointProcessLock(p,C.byref(LockArgs(timeoutMs=2000)))
            if rc: raise RuntimeError((p,rc))
            stack.callback(release,p)
            results['locked'][p]=dict(state=state(p),attempt=attempt,unix_s=time.time())
        results['during']=subprocess.check_output(['nvidia-smi','pmon','-s','u','-c','6','-d','1'],text=True).splitlines()
        results['end_of_sampling_unix_s']=time.time()
    results['end_unix_s']=time.time()
    print(json.dumps(results),flush=True)
    Path('submission-boundary.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['code']!=0 or r['state']!=0 for r in results['unlock'].values()):
        raise RuntimeError('Check original CUDA process unlock states')
