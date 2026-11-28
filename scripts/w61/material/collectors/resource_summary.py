"""Summarize actual shared-host telemetry on the physical wall interval of each arm."""
import json,subprocess,time
import numpy as np
from paired import RAW,DEST,SITE,SPEC,write


def main():
    file=RAW/'resource-samples.jsonl';target=DEST/'resources.json'
    cfg=json.loads(SPEC.read_text());done=json.loads(target.read_text()) if target.exists() else {}
    while len(done)<2*len(cfg['windows']):
        for w in cfg['windows']:
            for a in ('cpu','gpu_batch'):
                name=w+'-'+a
                if name in done:continue
                paths=[f for f in (DEST/(name+'-a02.json'),DEST/(name+'.json')) if f.is_file()]
                if not paths:continue
                item=json.loads(paths[0].read_text())
                if item['status']!='complete':continue
                from pathlib import Path
                d=Path(item['raw_directory'])
                clocks=[np.load(d/c/'benchmark-timing.npz')['frames'][:,1]*1e-9 for c in ('A','B')]
                start=min(t[0] for t in clocks);end=max(t[-1] for t in clocks)
                samples=[p for p in (json.loads(x) for x in file.read_text().splitlines()) if start<=p['monotonic_s']<=end]
                if len(samples)<2:continue
                dt=samples[-1]['monotonic_s']-samples[0]['monotonic_s']
                device=[p['device_util_pct_used_and_free_mib'] for p in samples if 'device_util_pct_used_and_free_mib' in p]
                done[name]=dict(samples=len(samples),active_wall_s=end-start,sampled_wall_span_s=dt,
                    shared_cgroup_mean_cores=(samples[-1]['cgroup_cpu_us']-samples[0]['cgroup_cpu_us'])/dt*1e-6,
                    shared_GPU_util_pct_min_median_max=np.percentile(np.array(device)[:,0],[0,50,100]).tolist() if device else None,
                    device_free_vram_min_mib=min(x[2] for x in device) if device else None,
                    available_cgroup_memory_min_gib=min((p['memory_max_bytes']-p['cgroup_memory_current_bytes']+p['inactive_file_bytes'])/2**30 for p in samples),
                    collector_RSS_max_mib=max([v['rss_kib']/1024 for p in samples for v in p['processes'].values() if v['role']=='collector'] or [0]),
                    broker_RSS_max_mib=max([v['rss_kib']/1024 for p in samples for v in p['processes'].values() if v['role']=='broker'] or [0]),
                    scope='shared cgroup/device utilization; process RSS sampled every 10s; initial CPU prefix monitoring started after boot')
                write(target,done)
                subprocess.run(['git','add','--sparse','--',str(target.relative_to(SITE))],cwd=SITE,check=True)
                if subprocess.run(['git','diff','--cached','--quiet'],cwd=SITE).returncode==1:
                    subprocess.run(['git','commit','-qm','Record contemporaneous D57 resource overlap for '+name],cwd=SITE,check=True)
        time.sleep(10)


if __name__=='__main__':main()
