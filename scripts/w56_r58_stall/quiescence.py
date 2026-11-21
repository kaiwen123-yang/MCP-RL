"""One finite driver lock-acquisition check per original PID; never checkpoint a busy GPU."""
if __name__=='__main__':
    import ctypes as C
    import json
    from pathlib import Path
    import time

    class Args(C.Structure):
        _fields_=[('timeoutMs',C.c_uint),('reserved0',C.c_uint),('reserved1',C.c_uint64*7)]

    def state(pid):
        value=C.c_int(-1)
        rc=driver.cuCheckpointProcessGetState(pid,C.byref(value))
        return {'code':int(rc),'state':value.value}

    def error_name(code):
        name=C.c_char_p()
        driver.cuGetErrorName(code,C.byref(name))
        return name.value.decode() if name.value else str(code)

    driver=C.CDLL('libcuda.so.1')
    driver.cuCheckpointProcessGetState.argtypes=[C.c_int,C.POINTER(C.c_int)]
    driver.cuCheckpointProcessLock.argtypes=[C.c_int,C.POINTER(Args)]
    driver.cuCheckpointProcessUnlock.argtypes=[C.c_int,C.c_void_p]
    rows=[]
    for pid,tag in [(648411,b'-w56-f-a-r58-eval-'),(652013,b'-w56-f-b-r58-eval-')]:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        attempt=cmd[cmd.index(b'--attempt')+1]
        if tag not in attempt or b'E300-attempt01' not in attempt:raise ValueError('Attempt identity changed')
        before=state(pid)
        if before!={'code':0,'state':0}: raise ValueError((pid,before))
        args=Args(timeoutMs=2000)
        start=time.time()
        rc=driver.cuCheckpointProcessLock(pid,C.byref(args))
        unlock=None
        try:
            row=dict(pid=pid,attempt=attempt.decode(),start_unix_s=start,driver_lock_timeout_ms=2000,
                     acquisition_returned_unix_s=time.time(),return_code=rc,return_name=error_name(rc))
        finally:
            if rc==0:
                unlock=driver.cuCheckpointProcessUnlock(pid,None)
        row.update(unlock_return_code=unlock,state_before=before,state_after=state(pid),end_unix_s=time.time())
        print(json.dumps(row),flush=True); rows.append(row)
        Path('quiescence.json').write_text(json.dumps(rows,indent=2)+'\n')
        if row['state_after']!=before or unlock not in (None,0):
            raise RuntimeError('Check driver state before proceeding to another PID')
