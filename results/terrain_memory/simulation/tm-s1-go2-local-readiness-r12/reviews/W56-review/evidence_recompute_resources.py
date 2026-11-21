"""Recompute process-tree resource observations without running any experiment."""
from pathlib import Path
import argparse, json, numpy as np
p=argparse.ArgumentParser()
p.add_argument('--originals',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out={}
for f in sorted(a.originals.glob('*/job-snapshot/resources.jsonl')):
 rows=[json.loads(line)for line in f.read_text().splitlines()if line.strip()]
 cores=[]
 for before,after in zip(rows,rows[1:]):
  if set(before['pids'])!=set(after['pids']):continue
  dt=after['unix_s']-before['unix_s']
  dc=(after['user_cpu_s']+after['system_cpu_s'])-(before['user_cpu_s']+before['system_cpu_s'])
  if dt>0 and dc>=0:cores.append(dc/dt)
 out[f.parent.parent.name]=dict(samples=len(rows),coverage_start_unix_s=rows[0]['unix_s'],coverage_end_unix_s=rows[-1]['unix_s'],max_rss_GiB=max(r['rss_bytes']for r in rows)/2**30,cpu_cores_interval_quantiles=dict(zip(['p05','p50','p95','max'],np.percentile(cores,[5,50,95,100]).tolist())),no_hot_segment_isolation=True,source=str(f))
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
