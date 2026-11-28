"""Sample external co-tenancy, the batch service and participating collectors without entering their loops."""
from pathlib import Path
import json,os,subprocess,time
from paired import RAW,DEST,SPEC
CG=Path('/sys/fs/cgroup')
def fields(n):return dict(x.split() for x in (CG/n).read_text().splitlines())
def main():
    cfg=json.loads(SPEC.read_text())
    dest=RAW/'resource-samples.jsonl'
    while not all((DEST/f'{w}-comparison.json').exists() for w in cfg['windows']):
        sample=dict(unix_s=time.time(),monotonic_s=time.monotonic(),cgroup_cpu_us=int(fields('cpu.stat')['usage_usec']),
                    cgroup_memory_current_bytes=int((CG/'memory.current').read_text()),inactive_file_bytes=int(fields('memory.stat')['inactive_file']),
                    memory_max_bytes=int((CG/'memory.max').read_text()),io_pressure=(CG/'io.pressure').read_text().splitlines()[0])
        smi=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if smi.returncode==0:sample['device_util_pct_used_and_free_mib']=[int(x.strip()) for x in smi.stdout.split(',')]
        active={}
        for p in Path('/proc').glob('[0-9]*'):
            try:
                cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if ('/collectors/collector.py ' not in cmd and '/collectors/broker.py ' not in cmd) or 'python' not in cmd.split(' ',1)[0]:continue
                rss=next(l for l in (p/'status').read_text().splitlines() if l.startswith('VmRSS:'))
                active[p.name]={'rss_kib':int(rss.split()[1]),'role':'broker' if 'broker.py' in cmd else 'collector','args':cmd.split(' ')[-6:-1]}
            except (FileNotFoundError,StopIteration,ProcessLookupError):continue
        sample['processes']=active
        with dest.open('a') as f:f.write(json.dumps(sample,separators=(',',':'))+'\n')
        time.sleep(10)
if __name__=='__main__':main()
