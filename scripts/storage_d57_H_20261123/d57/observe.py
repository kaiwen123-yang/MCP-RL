"""Measure real D57 payload writers, CPU and shared disk latency during this archive only."""
import json,os,subprocess,time
from pathlib import Path
from inventory import CTL,BG,now,save
from packet import recent_physics
def sample():
    procs={};hz=os.sysconf('SC_CLK_TCK')
    for p in Path('/proc').glob('[0-9]*'):
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if b'packet.py' not in a or b'pack' not in a or os.readlink(p/'cwd')!=str(CTL):continue
            io={k:int(v) for k,v in (s.split(':') for s in (p/'io').read_text().splitlines())}
            st=(p/'stat').read_text().rsplit(') ',1)[1].split()
            procs[int(p.name)]=(io['wchar'],io['rchar'],int(st[11])+int(st[12]))
        except (OSError,ValueError):continue
    md=next((list(map(int,l.split()[3:14])) for l in Path('/proc/diskstats').read_text().splitlines() if l.split()[2]=='md0'),None)
    z=Path('/sys/fs/cgroup')
    total=float(dict(l.split() for l in (z/'cpu.stat').read_text().splitlines())['usage_usec'])/1e6
    return time.monotonic(),procs,md,total
def main():
    previous=sample();peak=0;maxrate=0.;samples=0
    with (CTL/'real-throughput.jsonl').open('a',buffering=1) as out:
        while not (CTL/'receiver-closed.json').exists():
            time.sleep(8)
            t,p,md,cpu=sample();old,op,od,oc=previous;dt=t-old
            streams={pid:values[0]-op[pid][0] for pid,values in p.items() if pid in op and values[0]>op[pid][0]}
            rate=sum(streams.values())/dt/2**20
            v=dict(at=now(),payload_writer_pids=list(streams),writing_in_both_samples=len(streams),alive_pack_processes=len(p),
                   D57_pipe_MiB_s=round(rate,2),cpu_cores_used=round((cpu-oc)/dt,2),physics=recent_physics(),**BG)
            if md and od:
                delta=[y-x for x,y in zip(od,md)]
                v.update(md0_read_await_ms=round(delta[3]/max(1,delta[0]),2),md0_write_await_ms=round(delta[7]/max(1,delta[4]),2),
                    md0_device_busy_pct=round(100*delta[9]/(dt*1000),2),md0_io_in_flight=md[8])
            peak=max(peak,len(streams));maxrate=max(maxrate,rate);samples+=1
            out.write(json.dumps(v,separators=(',',':'))+'\n');save('real-throughput.json',
                dict(samples=samples,peak_concurrent_nonempty_payload_writers=peak,peak_sample_MiB_per_s=round(maxrate,2),last=v,**BG))
            previous=(t,p,md,cpu)
if __name__=='__main__':main()
