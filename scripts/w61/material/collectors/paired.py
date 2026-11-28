"""Run two different frozen E policy collectors by their real entry point on one atomic pair of slots."""
import argparse,hashlib,json,os,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
SPEC=ROOT/'experiments/terrain_memory/w61/material/e-packet-v1.json'
LEARN=Path('/root/autodl-tmp/MCP-RL-w61-learning')
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/material')
SITE=Path('/root/autodl-tmp/MCP-RL-w61-material')
STUDY='w61-e-packet-throughput-v1'
RAW=SITE/'work/w61/material'/STUDY
DEST=SITE/REL/STUDY


def write(p,data):
    t=p.with_suffix('.pending');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)


def run(window,arm):
    import numpy as np
    cfg=json.loads(SPEC.read_text());win=cfg['windows'][window]
    groups=json.loads(os.environ['W61_PAIR_CPUS'])
    if len(groups)!=2 or len(os.environ['W61_SLOTS'].split(','))!=2:raise RuntimeError('Need two atomically reserved physical slots')
    done=DEST/f'{window}-{arm}.json'
    if done.exists():return
    out=RAW/f'{window}-{arm}';out.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONPATH=f"{LEARN}/src:{LEARN}/scripts:{LEARN}/scripts/w61/learning")
    start=time.monotonic();service=None
    with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='w61-ep-') as tmp:
        ipc=Path(tmp)
        if arm!='cpu':
            service=subprocess.Popen([sys.executable,'-u',str(HERE/'broker.py'),str(ipc),str(SPEC),str(out)],env=env,stdout=(out/'broker.log').open('w'),stderr=subprocess.STDOUT)
            while not (ipc/'ready').exists():
                if service.poll() is not None:raise RuntimeError('Broker failed during initialization, see broker.log')
                time.sleep(.05)
        ready=time.monotonic()
        def one(case,group):
            cmd=['taskset','-c',','.join(map(str,group)),sys.executable,'-u',str(HERE/'collector.py'),
                                    str(SPEC),arm,case,window,str(out/case),str(ipc)]
            with (out/(case+'.log')).open('w') as f:proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            return proc.returncode
        with ThreadPoolExecutor(max_workers=2) as ex:
            jobs=[ex.submit(one,c,g) for c,g in zip(('A','B'),groups)];codes=[f.result() for f in jobs]
        collectors_end=time.monotonic()
        if service is not None:
            if any(codes) and service.poll() is None:service.terminate()
            service.wait()
        wall=time.monotonic()-start
    cases={}
    for case in ('A','B'):
        logfile=out/case/'result.json'
        if not logfile.exists():continue
        data=json.loads(logfile.read_text())
        info={k:data[k] for k in ('physical_s','physical_steps','requested','observed','reason','wall_s','error','progress','actor_sha','max_rss_gib','cpu_h','control_full_ms_p50_p95_p99','force_peak_percentiles','physical_failure')}
        events=[json.loads(s) for s in (out/case/'events.jsonl').read_text().splitlines()] if (out/case/'events.jsonl').exists() else []
        info['complete_events']=sum(e['end_kind'] not in ('truncated','external_batch_cut') for e in events)
        a=np.load(out/case/'benchmark-timing.npz');frames=a['frames'];packets=a['packets']
        times=frames[:,0];mono=frames[:,1]*1e-9
        info['nonempty_packets']=len(packets)
        info['fused_packets']=int(packets[:,3].sum()) if len(packets) else 0
        info['soft_windows']={}
        for left,right in cfg['soft_windows_s']:
            label=f'{left:g}-{right:g}'
            if times[-1]+1.e-7<right:continue
            i,k=np.searchsorted(times,[left-1e-7,right-1e-7])
            rows=packets[(packets[:,1]>left-1e-7)&(packets[:,1]<right+1e-7)] if len(packets) else packets
            info['soft_windows'][label]=dict(start_monotonic_s=float(mono[i]),end_monotonic_s=float(mono[k]),wall_s=float(mono[k]-mono[i]),
                events=sum(left<e['end_s']<=right+1e-7 and e['end_kind']!='truncated' for e in events),
                nonempty_packets=len(rows),fused_packets=int(rows[:,3].sum()) if len(rows) else 0,
                min_max_cells=[int(rows[:,2].min()),int(rows[:,2].max())] if len(rows) else None,
                feedback_chain_ms_p50_p95_p99=np.percentile(np.diff(mono[i:k+1])*1000,[50,95,99]).tolist())
        cases[case]=info
    elapsed=collectors_end-ready
    complete=sum(v['complete_events'] for v in cases.values())
    rec=dict(window=window,arm=arm,phase=win['phase'],status='complete' if not any(codes) and
             len(cases)==2 and all(r['reason']!='technical' for r in cases.values()) else 'technical',
             slots=os.environ['W61_SLOTS'],cpu_groups=groups,group_wall_s=wall,group_collector_wall_s=elapsed,
             broker_startup_s=ready-start,complete_events_s=complete/wall,
             exit_codes=codes,cases=cases,actual_source=cfg['original_collector_sha256'],actor_checkpoint_sha256=win['checkpoint_sha256'],
             raw_directory=str(out),new_formal_training_steps=0,deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    if (out/'broker.npz').exists():
        b=np.load(out/'broker.npz')['dispatch'];rec['batch_width_histogram']={str(i):int(((b[:,2]==1)&(b[:,3]==i)).sum()) for i in (1,2)}
        rec['mixed_worker_kernel_launches']=int(b[:,2].sum())
    write(done,rec)
    print(window,arm,rec['status'],wall,complete,flush=True)
    if any(codes):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('window');p.add_argument('arm');a=p.parse_args();run(a.window,a.arm)
