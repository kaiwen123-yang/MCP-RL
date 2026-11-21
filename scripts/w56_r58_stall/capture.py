"""Capture r58 job-owned clocks and per-attempt CPU/I/O samples without attaching."""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

M=Path('/root/autodl-tmp/MCP-RL-w56-material')
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
D=M/'outputs'/REL
CASE=M/'results'/REL/'w56/f'
HERE=Path('/root/autodl-tmp/MCP-RL-w55-integration/work/r58-e300-stall')
now=lambda:datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def jsonl(p):
    with p.open() as h: return [json.loads(line) for line in h]

def proc(pid):
    path=Path('/proc')/str(pid)
    if not path.exists(): return None
    io={k:int(v) for k,v in (line.split(':') for line in (path/'io').read_text().splitlines())}
    return dict(unix_s=time.time(),io=io,sched_runtime_ns=int((path/'schedstat').read_text().split()[0]),
        tracer=int(next(l for l in (path/'status').read_text().splitlines() if l.startswith('TracerPid:')).split()[1]))

def observe(letter):
    state=json.loads((CASE/('F-'+letter+'-r58')/'E300.json').read_text())
    d=D/state['attempt'];wc=d/'w55-f'; pid=state['pid']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if state['attempt'].encode() not in cmd:raise ValueError('PID does not run recorded attempt')
    last=jsonl(wc/'mpc-solves.jsonl')[-1]
    returns=jsonl(wc/'normal-execution.jsonl')
    end=returns[-1]['consume_s']
    arr=wc/last['file']; heart=json.loads((d/'live.json').read_text())
    resources=[r for r in jsonl(D/'jobs'/state['job']/'resources.jsonl') if r['root_pid']==pid]
    after=next(r for r in resources if r['unix_s']>=(wc/'mpc-solves.jsonl').stat().st_mtime+120)
    recent=resources[-1]
    deltas={k:recent[k]-after[k] for k in ('unix_s','user_cpu_s','system_cpu_s','read_bytes','write_bytes')}
    return dict(attempt=state['attempt'],case=letter,pid=pid,root=str(d),original_commit=subprocess.check_output(['git','-C',str(M),'rev-parse','HEAD'],text=True).strip(),
        last_mpc=last,mpc_packet_sha256=hashlib.sha256(arr.read_bytes()).hexdigest(),
        last_mpc_file_write_unix_s=(wc/'mpc-solves.jsonl').stat().st_mtime,heartbeat=heart,
        last_soft_reference_s=end,legs_logged_at_last_soft_reference=[x['leg'] for x in returns if x['consume_s']==end],
        last_complete_material_block=sorted(d.glob('material-[0-9]*.npz'))[-1].name,
        resource_interval_begin_unix_s=after['unix_s'],resource_interval_end_unix_s=recent['unix_s'],resource_delta=deltas,
        published_wbc_failure=(wc/'solver-failure.json').exists(),material_failure_bytes=(d/'material-failures.jsonl').stat().st_size,
        proc=proc(pid))

if __name__=='__main__':
    record=dict(captured_beijing=now(),cases=[observe('A'),observe('B')])
    record['nvidia_process_sample']=subprocess.check_output(['nvidia-smi','pmon','-c','1'],text=True).splitlines()
    print(json.dumps(record,indent=2))
