"""Record two new physical-clock intervals per case at the relevant r60 milestones."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
REPORT=ROOT/'results'/REL/'w56/c/r60-physical-progress.json'
WORKTREE=ROOT.parent/'MCP-RL-w56-r60'
SMALL=WORKTREE/'results'/REL/'w56/f'
LARGE=WORKTREE/'outputs'/REL


def milestones():
    """Yield only stages that have advanced beyond a measured physical event."""
    recovery=ROOT/'results'/REL/'w56/c/r58-e300-recovery-outcome.json'
    if not recovery.exists():
        return
    former={c['case']:c['last_solved_mpc_s'] for c in json.loads(recovery.read_text())['cases']}
    for case,old_end in former.items():
        for level in ('E30','E300'):
            status=SMALL/f'F-{case}-r60'/(level+'.json')
            if not status.is_file():continue
            stage=json.loads(status.read_text())
            if not stage.get('attempt'):continue
            raw=LARGE/stage['attempt']
            if not (raw/'live.json').is_file():continue
            heart=json.loads((raw/'live.json').read_text())
            if heart['pid']!=stage.get('pid'):continue
            if level=='E30':
                path=raw/'material-events.jsonl'
                if not path.is_file():continue
                with path.open() as events:
                    first=next((e for line in events if line.endswith('\n') and
                                (e:=json.loads(line))['type']=='load'),None)
                if first is None:continue
                threshold=first['time_s']
            else:
                threshold=old_end
            if heart['physical_time_s']>threshold+.5:
                yield case,level,threshold,stage,raw,heart


def record():
    data=json.loads(REPORT.read_text()) if REPORT.exists() else {'checks':{}}
    changed=False
    for case,level,threshold,stage,raw,heart in milestones():
        name=case+'-'+level
        if name in data['checks']:continue
        after=[]
        with (raw/'w55-f/mpc-solves.jsonl').open() as handle:
            for line in handle:
                if not line.endswith('\n'):break
                r=json.loads(line)
                if r['time_s']<threshold or r['status_val'] not in (1,2):continue
                if after and r['time_s']-after[-1]['mpc_s']<.245:continue
                after.append(dict(mpc_s=r['time_s'],call=r['call'],
                     original_write_unix_s=(raw/'w55-f'/r['file']).stat().st_mtime))
                if len(after)==3:break
        if len(after)<3:continue
        if not all(y['original_write_unix_s']>x['original_write_unix_s'] for x,y in zip(after,after[1:])):
            raise ValueError('The stored MPC packets do not establish separate wall-time observations')
        data['checks'][name]=dict(attempt=stage['attempt'],pid=stage.get('pid'),physical_event_s=threshold,
          event='first original material load' if level=='E30' else 'last solved MPC time of the corresponding interrupted r58 E300',
          solved_packets=after,observed_heartbeat=heart)
        changed=True
    if changed:
        tmp=REPORT.with_suffix('.w60-tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n')
        tmp.replace(REPORT)
