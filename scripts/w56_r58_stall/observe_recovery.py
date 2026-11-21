"""Record actual progress of both preserved original attempts if their clocks resume."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from capture import CASE, D, M, REL

REPORT=M/'results'/REL/'w56/c/r58-e300-clock-recovery.json'
DIAG=Path(__file__).resolve().parents[2]/'results'/REL/'w56/c/r58-e300-stall.json'


def sample(case):
    job=json.loads((CASE/('F-'+case['case']+'-r58')/'E300.json').read_text())
    pid=case['pid']
    if job.get('status')!='running' or job.get('pid')!=pid:
        return None
    try:
        cmd=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    if case['attempt'].encode() not in cmd: return None
    raw=D/case['attempt']
    heart=json.loads((raw/'live.json').read_text())
    if heart['pid']!=pid: return None
    with (raw/'w55-f/mpc-solves.jsonl').open('rb') as stream:
        stream.seek(max(0,stream.seek(0,2)-8192))
        rows=stream.read().split(b'\n')
    # A write in progress can leave the last line incomplete.
    solved=json.loads(rows[-2])
    return dict(checked_unix_s=time.time(),heartbeat_unix_s=heart['unix_s'],
                physical_s=heart['physical_time_s'],mpc_s=solved['time_s'],call=solved['call'])


def advanced(new,old):
    return all(new[k]>old[k] for k in ('physical_s','mpc_s','call'))


def main():
    cases=json.loads(DIAG.read_text())['cases']
    marks=[[dict(checked_unix_s=c['proc_live_measurement']['end_unix_s'],heartbeat_unix_s=c['heartbeat']['unix_s'],
                 physical_s=c['heartbeat']['physical_time_s'],mpc_s=c['last_mpc']['time_s'],call=c['last_mpc']['call'])] for c in cases]
    while len(marks[0])<3:
        new=[sample(c) for c in cases]
        if any(x is None for x in new):
            print('Original PID or run state changed before two joint progress intervals; consult the official stage records.',flush=True)
            return
        if all(advanced(a,history[-1]) for a,history in zip(new,marks,strict=True)):
            for history,value in zip(marks,new,strict=True): history.append(value)
        if len(marks[0])<3: time.sleep(30)
    proof=dict(status='both_original_clocks_advanced_twice_after_observed_stall',
        cases=[dict(attempt=c['attempt'],pid=c['pid'],samples=m) for c,m in zip(cases,marks,strict=True)],
        deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    with (M/'work/w56-f/git.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        REPORT.write_text(json.dumps(proof,indent=2)+'\n')
        path=str(REPORT.relative_to(M))
        subprocess.run(['git','add','--',path],cwd=M,check=True)
        subprocess.run(['git','commit','--only','-m','Verify renewed physics-clock and MPC progress of both original r58 E300 attempts','--',path],cwd=M,check=True)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
