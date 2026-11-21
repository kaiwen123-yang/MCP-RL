"""Temporarily quiesce only the peer's CUDA submissions, then restore its original state."""
if __name__=='__main__':
    import ctypes as C
    import json
    from pathlib import Path
    import subprocess
    import time
    import numpy as np
    from capture import CASE, D

    class Args(C.Structure):
        _fields_=[('timeoutMs',C.c_uint),('reserved0',C.c_uint),('reserved1',C.c_uint64*7)]

    drv=C.CDLL('libcuda.so.1')
    drv.cuCheckpointProcessGetState.argtypes=[C.c_int,C.POINTER(C.c_int)]
    drv.cuCheckpointProcessLock.argtypes=[C.c_int,C.POINTER(Args)]
    drv.cuCheckpointProcessUnlock.argtypes=[C.c_int,C.c_void_p]

    def state(pid):
        val=C.c_int(-1)
        if drv.cuCheckpointProcessGetState(pid,C.byref(val)):raise RuntimeError('CUDA state unavailable')
        return int(val.value)

    def target(case):
        s=json.loads((CASE/f'F-{case}-r58/E300.json').read_text())
        pid=s['pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if cmd[cmd.index(b'--attempt')+1].decode()!=s['attempt']:raise ValueError('Original PID changed')
        return pid,D/s['attempt']

    def mark(c):
        pid,raw=c
        with (raw/'w55-f/mpc-solves.jsonl').open('rb') as h:
            h.seek(max(0,h.seek(0,2)-8192))
            last=json.loads(h.read().split(b'\n')[-2])
        heart=json.loads((raw/'live.json').read_text())
        return dict(check_unix_s=time.time(),pid=pid,call=last['call'],mpc_clock_s=last['time_s'],
           physical_heartbeat_s=heart['physical_time_s'],driver_state=state(pid),
           cpu_runtime_ns=int(Path(f'/proc/{pid}/schedstat').read_text().split()[0]))

    cases=[target('A'),target('B')]
    rows=[]
    for active,peer in (cases,list(reversed(cases))):
        p,raw=active; other,_=peer
        f=raw/'w55-f/mpc-solves.jsonl'
        past=[json.loads(s) for s in f.read_text().splitlines()[-51:]]
        span=np.diff([(f.parent/x['file']).stat().st_mtime for x in past])
        duration=4*float(np.percentile(span,95))+2
        if state(other)!=0 or state(p)!=0:raise ValueError('Diagnostic requires original CUDA RUNNING states')
        rec=dict(active_pid=p,locked_peer_pid=other,prior_50_plan_intervals_p95_wall_s=float(np.percentile(span,95)),
             observation_window_s=duration,before=[mark(active),mark(peer)],samples=[])
        code=drv.cuCheckpointProcessLock(other,C.byref(Args(timeoutMs=2000)))
        if code:raise RuntimeError(('Peer did not quiesce',code,state(other)))
        try:
            start=time.monotonic()
            while time.monotonic()-start < duration:
                perf=subprocess.check_output(['nvidia-smi','pmon','-s','u','-c','1'],text=True).splitlines()
                rec['samples'].append(dict(active=mark(active),peer=mark(peer),per_process_SM=perf[2:]))
                time.sleep(2)
        finally:
            rec['peer_unlock_code']=drv.cuCheckpointProcessUnlock(other,None)
            rec['after_unlock']=[mark(active),mark(peer)]
            rows.append(rec);Path('exclusive.json').write_text(json.dumps(rows,indent=2)+'\n')
        if rec['peer_unlock_code'] or state(other)!=0:raise RuntimeError('Peer must be restored before another observation')
        print(p,rec['before'][0]['mpc_clock_s'],rec['after_unlock'][0]['mpc_clock_s'],state(other),flush=True)
