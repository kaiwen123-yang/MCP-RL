"""Record two new physical-clock intervals per case at the relevant r60 milestones."""
import json
import os
import signal
import struct
import time
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
    changed=compare_recorded_arms(data) or changed
    if changed:
        tmp=REPORT.with_suffix('.w60-tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n')
        tmp.replace(REPORT)

def sample_active_calls(previous):
    """Capture the registered Python stack once if one exact r60 call stops returning while its thread consumes CPU."""
    active={}
    for case in ('A','B'):
        for level in ('E30','E60','E300','E600'):
            path=SMALL/f'F-{case}-r60'/(level+'.json')
            if not path.exists():continue
            stage=json.loads(path.read_text())
            if stage.get('status')!='running' or not stage.get('pid'):continue
            pid=stage['pid'];attempt=stage['attempt'];raw=LARGE/attempt
            spec=raw/'r60-call-format.json';proc=Path('/proc',str(pid))
            if not spec.is_file():continue
            cfg=json.loads(spec.read_text())
            if cfg['pid']!=pid or cfg['signal']!='SIGUSR1':continue
            try:
                cmd=(proc/'cmdline').read_bytes().split(b'\0')
                if b'--attempt' not in cmd or cmd[cmd.index(b'--attempt')+1].decode()!=attempt:continue
                cpu=int((proc/'schedstat').read_text().split()[0])
                sigcgt=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines()
                               if line.startswith('SigCgt:')),16)
            except FileNotFoundError:
                continue
            if not sigcgt & (1<<(signal.SIGUSR1-1)):continue
            seq,t,wall,op,seq2=struct.unpack(cfg['format'],(raw/'r60-current-call.bin').read_bytes())
            if seq!=seq2:continue
            now=dict(attempt=attempt,pid=pid,sequence=seq,physical_s=t,
                     operation=op.rstrip(b'\0').decode(),entry_unix_s=wall,observed_unix_s=time.time(),cpu_ns=cpu)
            prior=previous.get(attempt)
            active[attempt]=now
            if not prior or prior['sequence']!=seq or prior['pid']!=pid or cpu<=prior['cpu_ns']:continue
            file=raw/'r60-call-stack-evidence.json'
            if file.exists() and json.loads(file.read_text())['observation']['sequence']==seq:continue
            # Only a process that installed its own opt-in traceback handler is signalled.
            try:os.kill(pid,signal.SIGUSR1)
            except ProcessLookupError:continue
            evidence=dict(previous_sample=prior,observation=now,signal='SIGUSR1',stack_log=str(raw/cfg['stack']),
                scope='Same call marker at separate observations with additional main-thread CPU time. Collects diagnostics and leaves the original qualification untouched.')
            tmp=file.with_suffix('.partial')
            tmp.write_text(json.dumps(evidence,indent=2)+'\n')
            tmp.replace(file)
            print('Captured a repeated r60 call marker',attempt,op,t,flush=True)
    return active


def compare_recorded_arms(data):
    """Compare original r58 & r60 outputs on their common 500 Hz lattice."""
    import numpy as np
    comparisons=data.setdefault('r58_r60_matched_originals',{})
    changed=False
    baseline=ROOT.parent/'MCP-RL-w56-material'
    for case in ('A','B'):
        for level in ('E30','E60'):
            tag=case+'-'+level
            if tag in comparisons:continue
            pair=[]
            for tree,build in ((baseline,'r58'),(WORKTREE,'r60')):
                root=tree/'results'/REL/'w56/f'
                stage=root/f'F-{case}-{build}'/(level+'.json')
                if not stage.is_file():break
                attempt=json.loads(stage.read_text()).get('attempt')
                if not attempt:break
                result=root/build/(attempt+'.json')
                raw=tree/'outputs'/REL/attempt/'trajectory.npz'
                if not result.is_file() or not raw.is_file():break
                pair.append((json.loads(result.read_text()),raw))
            if len(pair)!=2:continue
            (old,ro),(new,rn)=pair
            with np.load(ro) as a,np.load(rn) as b:
                t0,t1=a['time'],b['time']
                common=min(len(t0),len(t1))
                different=np.flatnonzero(t0[:common]!=t1[:common])
                if len(different):common=int(different[0])
                row=dict(r58_attempt=old['attempt'],r60_attempt=new['attempt'],
                    r58_wall_s=old['wall_s'],r60_wall_s=new['wall_s'],r58_stage=old['stage_status'],r60_stage=new['stage_status'],
                    r58_samples=len(t0),r60_samples=len(t1),identical_prefix_clock_samples=common,
                    common_end_s=float(t0[common-1]) if common else None)
                if common:
                    qa=a['qpos'][:common];qb=b['qpos'][:common]
                    row.update(max_body_position_difference_m=float(np.linalg.norm(qb[:,:3]-qa[:,:3],axis=1).max()),
                        max_joint_angle_difference_rad=float(abs(qb[:,7:]-qa[:,7:]).max()),
                        max_torque_difference_Nm=float(abs(b['torque'][:common]-a['torque'][:common]).max()),
                        max_grf_component_difference_N=float(abs(b['actual_grf'][:common]-a['actual_grf'][:common]).max()),
                        max_requested_cap_difference_m_s=float(abs(b['requested_speed_cap'][:common]-a['requested_speed_cap'][:common]).max()),
                        samples_with_different_phase=int(np.count_nonzero(a['phase'][:common]!=b['phase'][:common])),
                        samples_with_different_contacts=int(np.count_nonzero((a['contact'][:common]!=b['contact'][:common]).any(axis=1))))
                comparisons[tag]=row;changed=True
    return changed
