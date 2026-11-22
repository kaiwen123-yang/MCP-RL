"""Record W57 process usage and actual advancing simulation clocks, on D57."""
import json
import os
from pathlib import Path
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN/'w57'
RAW=ROOT/'outputs/terrain_memory/simulation'/RUN
LAYOUT=struct.Struct('<Qdd40sQ')


def sample():
    physics={}
    for f in (SMALL/'attempts').glob('*.json'):
        row=json.loads(f.read_text())
        if row['status']!='running':continue
        out=RAW/f.stem
        marker=out/'r60-current-call.bin'
        if marker.exists():
            data=marker.read_bytes()
            if len(data)==LAYOUT.size:
                first,t,when,call,second=LAYOUT.unpack(data)
                if first==second:
                    physics[f.stem]=dict(pid=row['pid'],physical_s=t,call_unix_s=when,
                                          call=call.rstrip(bytes(1)).decode())
    procs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cwd=os.readlink(p/'cwd')
            if '/MCP-RL-w57' not in cwd:continue
            fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
            exe=(p/'comm').read_text().strip()
            if exe not in ('python','python3','ffmpeg','gh','main'):continue
            procs.append(dict(pid=int(p.name),ppid=int(fields[1]),command=exe,cwd=cwd,
                              cpu_s=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                              rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
        except (OSError,ValueError):
            continue
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],text=True,capture_output=True)
    dev=[int(v.strip()) for v in gpu.stdout.strip().split(',')] if not gpu.returncode else None
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
                         '--format=csv,noheader,nounits'],text=True,capture_output=True)
    return dict(unix_s=time.time(),physics=physics,processes=procs,gpu_percent_used_total_MiB=dev,
                gpu_compute_pid_used_MiB=apps.stdout.strip().splitlines(),
                cgroup_memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))


if __name__=='__main__':
    dest=RAW/'w57'/'resource-trace.jsonl'
    dest.parent.mkdir(exist_ok=True)
    with dest.open('a',buffering=1) as log:
        while True:
            log.write(json.dumps(sample(),separators=(',',':'))+'\n')
            if json.loads((SMALL/'index.json').read_text()).get('status')=='finished':break
            time.sleep(20)
