"""Read D57 W57 handoff, the real collectors, their binaries and all W57 panes without taking a lease."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
W57=ROOT.parent/'MCP-RL-w57'
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
R=W57/'results'/REL/'w57'
RAW=W57/'outputs'/REL
PILOT=W57/'work/w57-r/pilot02'
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
SH=ZoneInfo('Asia/Shanghai')


def read(path):
    return json.loads(Path(path).read_text())


def proc_info(pid):
    proc=Path('/proc')/str(pid)
    try:
        return dict(pid=pid,cwd=os.readlink(proc/'cwd'),
                    args=proc.joinpath('cmdline').read_bytes().decode().strip('\x00').split('\x00'),
                    start_ticks=int(proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]))
    except FileNotFoundError:
        return None


def w57_state(verify=False):
    fmt='#{session_name} #{pane_pid} #{pane_dead}'
    panes=subprocess.check_output(['tmux','list-panes','-a','-F',fmt],text=True)
    sessions={n:proc_info(int(p)) for n,p,dead in (line.split() for line in panes.splitlines())
              if dead=='0' and 'w57' in n.lower()}
    di={p.name:read(p) for p in sorted((R/'dispatch').glob('*.json'))}
    heads={}
    for head in ('full','blind'):
        path=PILOT/head
        u=read(path/'published.json')
        parts={c:sum(1 for _ in f.open()) if f.is_file() else 0
               for c in 'AB' for f in [path/f'batch-{u["version"]:03d}-{c}.jsonl']}
        ck=Path(u['checkpoint'])
        heads[head]=dict(**u,uncommitted_rows=parts,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                         collector_exits=[read(f) for f in sorted(path.glob('collector-exit-*.json'))])
    rows=[read(p) for p in (R/'attempts').glob('*.json')]
    active=[]
    for row in rows:
        if 'ended_unix_s' in row:continue
        item={k:row.get(k) for k in ('attempt','status','controller','case','source_commit','pid','runner_pid',
                                     'slot','tmux','checkpoint_sha256','started_unix_s','settings')}
        item['physics_process']=proc_info(row.get('pid'));item['launcher_process']=proc_info(row.get('runner_pid'))
        last=RAW/row['attempt']/'live.json'
        item['last_physics']=read(last) if last.is_file() else None
        if verify and item['physics_process']:
            src=Path(item['physics_process']['cwd'])
            hashes=row['active_sources']
            item.update(worktree_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),
                        sources_and_compiled_count=len(hashes),source_or_binary_mismatches=[
                k for k,v in hashes.items() if hashlib.sha256((src/k).read_bytes()).hexdigest()!=v],
                compiled_sha256={k:v for k,v in hashes.items() if k.endswith('.so')})
            argv=item['physics_process']['args']
            if '--checkpoint' in argv:
                actual=Path(argv[argv.index('--checkpoint')+1])
                item['installed_speed_checkpoint']=dict(path=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
        active.append(item)
    pair=read(R/'r/pilot02/two-slot-queue.json')
    idx=read(R/'index.json')
    return dict(observed_at=datetime.now(SH).isoformat(),workspace=str(W57),panes=sessions,heads=heads,
                active_physics=active,primary=di['frozen-evaluations.json']['stage'],
                private_short_A=di['private-short-A.json']['status'],dispatcher_files=di if verify else sorted(di),
                pair_dispatch=pair['status'],pair_owner=proc_info(pair['queue_pid']),final=(PILOT/'final.json').is_file(),
                distinct_attempts=len(rows),status=idx['status'],original_qualifications=idx['original_qualifications'],
                overlap_measurement=idx['measurement'],publication=dict(verified_assets=idx['verified_release_assets'],
                verified_bytes=idx['release_bytes']),**BG)


if __name__=='__main__':
    print(json.dumps(w57_state(verify=True),indent=2))

