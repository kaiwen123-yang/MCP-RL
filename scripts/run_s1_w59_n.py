"""Admit the qualified r59 build via the W56 shared slots, retain and publish E30."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import run_s1_w56_n as old
from analyze_s1_w55_n import measure,records
from publish_s1_go2_w54 import ROOT,LARGE,write

REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/n')
SMALL=ROOT/REL
RESULTS=SMALL/'r59'
CASE=SMALL/'N-A-r59'
WORK=ROOT/'work/w56-n-r59'
F=ROOT.parent/'MCP-RL-w56-f'
TMUX='s1-w56-n-a-r59-e30'
QUEUE=SMALL/'queue-r59.json'
old.SMALL,old.RESULTS,old.PLAN=SMALL,RESULTS,'6da12efc2cb85dcc99e53243dca489c9966e4969'

def read(path):return json.loads(path.read_text())

def commit(message,paths):
    names=[str(p.relative_to(ROOT)) for p in paths if p.is_file()]
    subprocess.run(['git','add','--',*names],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*names],cwd=ROOT).returncode:
        subprocess.run(['git','commit','--only','-m',message,'--',*names],cwd=ROOT,check=True)

def update_queue(status,**kwargs):
    state=read(QUEUE) if QUEUE.exists() else dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    state.update(status=status,tmux=TMUX,updated_unix_s=time.time(),**kwargs)
    write(QUEUE,state)

def execute():
    job=old.declaration('S5-r59','A','W56')
    write(CASE/'E30.json',dict(status='declared',job=job['job_id'],tmux=TMUX,level='E30',
                              deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
    update_queue('dispatched',job=job['job_id'],slot=int(os.environ['S1_W56_SLOT']),affinity=sorted(os.sched_getaffinity(0)))
    commit('Declare N/A r59 original E30 in its assigned physical slot', [CASE/n for n in ('job.json','execution-contract.json','E30.json')]+[QUEUE])
    return old.run(CASE,'E30',finish_stage=False)

def finish():
    state=read(CASE/'E30.json');attempt=state['attempt'];stage=RESULTS/(attempt+'.json')
    files=[CASE/'E30.json',RESULTS/'index.json',stage,RESULTS/(attempt+'-resolved.json')]
    update_queue('physics_ended',case_status=state['status'],attempt=attempt,ended_unix_s=state['ended_unix_s'])
    commit('Record original N/A r59 E30 physical outcome',files+[QUEUE])
    if not stage.exists():return
    from render_s1_go2_r12 import render
    import render_s1_go2_r12 as renderer
    renderer.SMALL=RESULTS;render(attempt)
    analysis=measure(CASE,'E30',RESULTS)
    native=LARGE/attempt/'w54-n-native'
    nominal=[x for x in records(native/'nominal-candidates.jsonl') if x['generator']=='trot_support_handoff']
    calls=list(records(native/'calls.jsonl'))
    analysis['handoff_candidate']=dict(proposed=len(nominal),feasible=sum(x['feasible'] for x in nominal),
        installs=sum(x['selection_source']=='trot_support_handoff' for x in calls))
    write(CASE/'E30-actual-analysis.json',analysis)
    state['video']=read(LARGE/attempt/'video-asset.json');write(CASE/'E30.json',state)
    write(LARGE/'jobs'/state['job']/'E30-runtime.json',state)
    commit('Record r59 support exchanges, first constraints and full video evidence', [CASE/'E30.json',CASE/'E30-actual-analysis.json'])
    subprocess.run([sys.executable,str(F/'scripts/deliver_s1_w56_stage.py'),'--workspace',str(ROOT),'--case','N-A-r59','--level','E30'],cwd=F,check=True)
    update_queue('published',delivery='w56/delivery/n-a-r59-e30/delivery.json')
    commit('Record verified N/A r59 E30 publication', [QUEUE])

def queue():
    report=read(SMALL/'same-input-r59.json')
    two=[x for x in report['comparison'] if x['mode']=='trot_support_handoff']
    if ([r['new_code'] for r in report['matrix_witness']]!=[0,0,0,0,7]
        or not all(r['first_failed_node'] is None and min(r['foot_transitions'])>=1 for r in two)
        or two[0]['dx_m']<=0 or report['mapping']['maximum_command_bound_excess_rad']>0
        or report['mapping']['native_wheel_delta_rad']>1e-8):
        raise RuntimeError('Saved support and mapping evidence incomplete')
    if CASE.exists() or QUEUE.exists():raise RuntimeError('r59 was already submitted; inspect its exact queue')
    update_queue('waiting_for_shared_physical_slot',priority='N',runner_pid=os.getpid(),workers=10,
                 stage='E30',case='A',build='S5-r59',same_input='w56/n/same-input-r59.json')
    commit('Queue qualified N/A r59 after F priority under the two-task limit',[QUEUE])
    result=subprocess.run([sys.executable,'-u',str(F/'scripts/s1_w56_slot.py'),'--priority','N','--label','N-A-r59-E30','--',
                           sys.executable,'-u',__file__,'--physical'],cwd=ROOT)
    if (CASE/'E30.json').is_file() and 'ended_unix_s' in read(CASE/'E30.json'):finish()
    elif result.returncode:raise RuntimeError(f'N dispatcher exited {result.returncode}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');p.add_argument('--finish',action='store_true')
    args=p.parse_args()
    if args.physical:sys.exit(execute())
    if args.finish:finish()
    else:queue()
