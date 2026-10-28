#!/usr/bin/env python3
"""Render the full recorded R10 attempts in independent GPU contexts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-soft-landing-r10'


def main():
    small=ROOT/'results/terrain_memory/simulation'/RUN
    index=json.loads((small/'index.json').read_text());jobs=[]
    for row in index['attempts']:
        if row['status']!='completed' or not row['physical_started']:continue
        name=row['attempt'];out=ROOT/'outputs/terrain_memory/simulation'/RUN/name
        if (out/'complete-mission.mp4').exists():continue
        resolved=json.loads((small/(name+'-resolved.json')).read_text())
        build=Path(resolved['agent_server']['path']).parent.name
        form='GNC' if build.startswith('GNC') else build
        jobs.append((name,build,form))
    def render(job):
        name,build,form=job
        command=[sys.executable,str(ROOT/'scripts/run_s1_go2_r10.py'),
            '--attempt',name,'--form',form,'--build',build,'--render']
        with (ROOT/'work'/('r10-render-'+name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return dict(attempt=name,render_exit_code=result.returncode)
    with ThreadPoolExecutor(3) as pool:
        for result in pool.map(render,jobs):print(json.dumps(result),flush=True)


if __name__=='__main__':main()
