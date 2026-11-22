"""Run W57's two original E60 collectors and the shared timing learner on D57."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from s1_w57_learning import (ROOT, SPEC, SPEC_HASH, STEPS, atomic_json, batch_path, checkpoint_path,
                             common_initialization, close_interrupted_sequence, jsonline, read_rows, train_batch)
from s1_w57_training_evidence import node_evidence


REL = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
COMPACT = ROOT/REL/'w57'
DEFAULT = ROOT/'work/w57-r/learning'
BACKGROUND = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial(campaign):
    campaign.mkdir(parents=True,exist_ok=True)
    target = campaign/'common-initialization.pt'
    if not target.exists():
        common_initialization(target)
    saved = torch.load(target,map_location='cpu',weights_only=False)
    if saved['spec_sha256'] != SPEC_HASH or saved['head'] != 'common' or saved['samples'] != 0:
        raise ValueError('The common initialization does not belong to this timing spec.')
    print(json.dumps(dict(common_initialization=str(target),sha256=sha(target),samples=0)),flush=True)
    return target


def source_identity(observer):
    parent = ROOT/REL/'w56/f'
    actors = {c:read(parent/f'F-{c}-r60/job.json')['checkpoint_sha256'] for c in SPEC['cases']}
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                observer=observer,planner='P0',task='original_E60_virgin',speed_actors=actors,
                schema=SPEC['identity'],timing_spec_sha256=SPEC_HASH,
                reward='W38_legal_route_terminal_settlement_event_integrals',
                optimizer='Tianshou PPOPolicy.learn, event-duration GAE')


def open_run(campaign,head,observer,target):
    run = campaign/head
    run.mkdir(parents=True,exist_ok=True)
    identity = source_identity(observer)
    path = run/'run.json'
    if path.exists():
        existing = read(path)
        if existing['identity'] != identity or existing['declared_ceiling'] != target:
            raise ValueError('An existing learning run cannot change its frozen identity or matched budget.')
        return run
    common = campaign/'common-initialization.pt'
    payload = torch.load(common,map_location='cpu',weights_only=False)
    if payload['spec_sha256'] != SPEC_HASH:
        raise ValueError('Common initialization and runtime schema disagree.')
    payload.update(head=head,identity=identity)
    torch.save(payload,checkpoint_path(run,0))
    manifest = dict(head=head,identity=identity,common_initialization=str(common),
                    common_sha256=sha(common),declared_ceiling=target,case_allocation_per_update=STEPS,
                    cumulative_full_nodes=SPEC['nodes'],common_random_number_rule=SPEC['ppo']['exploration_rng'],
                    **BACKGROUND)
    for case in SPEC['cases']:
        atomic_json(run/f'collector-{case}.json',dict(mode='collect',case=case,head=head,run_dir=str(run)))
    atomic_json(path,manifest)
    atomic_json(run/'published.json',dict(version=0,finished=False))
    return run


def process_start(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):
        return None
    return None if tail[0]=='Z' else int(tail[19])


def owns_attempt(pid, attempt):
    if not pid:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split(chr(0))
    except (FileNotFoundError,ProcessLookupError):
        return False
    return ('--attempt' in args and args[args.index('--attempt')+1] == attempt)


def reconcile_episode(run,case,existing,receipt_dir):
    if not existing:
        return
    attempt = existing[-1]['attempt']
    ends_file = run/f'episode-ends-{case}.jsonl'
    matching = [item for item in read_rows(ends_file) if item['attempt']==attempt]
    if matching:
        last = matching[-1]
        if last['exit_code'] or last['status']=='technical_interruption':
            close_interrupted_sequence(run,case,attempt,last['reason'] or 'collector_technical_exit')
        return
    receipt_path = receipt_dir/(attempt+'.json')
    receipt = read(receipt_path) if receipt_path.exists() else {}
    # A collector supervisor can be lost while its exact original runner is still alive.
    # Allow that episode to finish or pass through the existing policy barrier.
    while (owns_attempt(receipt.get('runner_pid'),attempt) or owns_attempt(receipt.get('pid'),attempt)):
        time.sleep(1.)
        receipt = read(receipt_path)
    status = receipt.get('status')
    interrupted = status in (None,'declared','running','technical_interruption')
    reason = receipt.get('reason') or ('lost_exact_physics_process' if interrupted else str(status))
    jsonline(ends_file,dict(attempt=attempt,unix_s=time.time(),exit_code=int(interrupted),
             simulation_s=receipt.get('simulation_s'),status=status,reason=reason,adopted=True))
    if interrupted:
        close_interrupted_sequence(run,case,attempt,reason)


class ExistingCollector:
    """An exact same-case process that remains live after its learner supervisor exits."""
    def __init__(self,run,case,state):
        self.run,self.case,self.pid,self.start = run,case,state['pid'],state['proc_start']

    def poll(self):
        if process_start(self.pid)==self.start:
            return None
        path = self.run/f'collector-exit-{self.case}-{self.start}.json'
        return read(path)['exit_code'] if path.exists() else 1

    def wait(self):
        while self.poll() is None:
            time.sleep(1.)
        return self.poll()


def collector_owner(run,case):
    lock = (run/f'owner-{case}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid = os.getpid()
    start = process_start(pid)
    atomic_json(run/f'owner-{case}.json',dict(pid=pid,proc_start=start,case=case,
                head=run.name,campaign=str(run.parent),started_unix_s=time.time(),**BACKGROUND))
    code = 1
    try:
        collector(run,case)
        code = 0
    finally:
        atomic_json(run/f'collector-exit-{case}-{start}.json',dict(exit_code=code,
                    ended_unix_s=time.time()))
        lock.close()


def collector(run,case):
    manifest = read(run/'run.json')
    receipt_dir = COMPACT/'attempts'
    existing = read_rows(run/f'episodes-{case}.jsonl')
    reconcile_episode(run,case,existing,receipt_dir)
    episode = len(existing)
    while True:
        state = read(run/'published.json')
        if state['finished']:
            return
        version = state['version']
        if (run/f'ready-{version:03d}-{case}.json').exists():
            time.sleep(1.)
            continue
        count_at_start = version*STEPS + len(read_rows(batch_path(run,version,case)))
        episode += 1
        slug = run.parent.name.lower().replace(' ','-')
        attempt = f'w57-{slug}-{manifest["head"]}-{case.lower()}-e60-{episode:04d}'
        # Keep every attempted physical session, including a partial one, under its own identity.
        if (receipt_dir/(attempt+'.json')).exists():
            raise FileExistsError(f'The attempt {attempt} already exists; inspect its collector receipt.')
        cmd = [sys.executable,'-u','scripts/run_s1_w57.py','episode','--case',case,'--arm','R',
               '--task','training','--level','E60','--material',manifest['identity']['observer'],
               '--attempt',attempt,'--timing-config',str(run/f'collector-{case}.json')]
        jsonline(run/f'episodes-{case}.jsonl',dict(attempt=attempt,case=case,version_at_start=version,
                 unix_s=time.time(),command=cmd,**BACKGROUND))
        code = subprocess.call(cmd,cwd=ROOT)
        receipt = read(receipt_dir/(attempt+'.json')) if (receipt_dir/(attempt+'.json')).exists() else {}
        jsonline(run/f'episode-ends-{case}.jsonl',dict(attempt=attempt,unix_s=time.time(),exit_code=code,
                 simulation_s=receipt.get('simulation_s'),status=receipt.get('status'),reason=receipt.get('reason')))
        if code or receipt.get('status')=='technical_interruption':
            close_interrupted_sequence(run,case,attempt,receipt.get('reason') or 'collector_technical_exit')
            raise RuntimeError(f'Collector {case}, attempt {attempt}: {receipt.get("reason")}, exit {code}.')
        # A genuine zero-decision natural failure cannot be repaired by repeatedly launching virgin copies.
        ended = read(run/'published.json')
        end_version = ended['version']
        end_count = end_version*STEPS + len(read_rows(batch_path(run,end_version,case)))
        if not ended['finished'] and end_count == count_at_start:
            raise RuntimeError(f'No new distinct timing transition in natural attempt {attempt}.')
        jsonline(run/f'count-{case}.jsonl',dict(attempt=attempt,count=end_count))


def record_node(run,version,device,known_identity):
    evaluation,matches = node_evidence(run,version,known_identity)
    manifest = read(run/'run.json')
    if manifest['head']=='blind':
        proceed = evaluation['distinct'] < manifest['declared_ceiling']
    else:
        proceed = evaluation['distinct'] < manifest['declared_ceiling'] and evaluation['eligible']
    evaluation.update(version=version,continue_same_identity=proceed,
                      observed_unix_s=time.time(),source_checkpoint=str(checkpoint_path(run,version)),
                      checkpoint_sha256=sha(checkpoint_path(run,version)),device=device,
                      uses_mechanism_holdout=False)
    tag = f'node-{evaluation["distinct"]:05d}'
    atomic_json(run/(tag+'.json'),evaluation)
    with gzip.open(run/(tag+'-matched-loads.jsonl.gz'),'wt') as f:
        for m in matches:
            f.write(json.dumps(m,allow_nan=False)+'\n')
    evidence = COMPACT/'r'/run.parent.name/run.name
    evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/(tag+'.json'),evaluation)
    print(json.dumps(evaluation,ensure_ascii=False),flush=True)
    return proceed


def train(run,processes,device):
    manifest = read(run/'run.json')
    authority = manifest['identity']
    while not read(run/'published.json')['finished']:
        version = read(run/'published.json')['version']
        while not all((run/f'ready-{version:03d}-{c}.json').exists() for c in SPEC['cases']):
            for case,process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f'Collector {case} exited before the shared policy barrier (code {process.returncode}).')
            time.sleep(1.)
        update = run/f'update-{version+1:03d}.json'
        if update.exists():
            if not checkpoint_path(run,version+1).exists():
                raise ValueError('A saved update is missing its shared checkpoint.')
            metrics = read(update)
        else:
            metrics = train_batch(run,version,manifest['head'],authority,device=device)
        version += 1
        number = version*2*STEPS
        print(json.dumps(metrics),flush=True)
        compact = COMPACT/'r'/run.parent.name/run.name
        compact.mkdir(parents=True,exist_ok=True)
        atomic_json(compact/f'update-{version:03d}.json',metrics)
        proceed = True
        if number in SPEC['nodes'] or number == manifest['declared_ceiling']:
            proceed = record_node(run,version,device,authority==source_identity(authority['observer']))
        atomic_json(run/'published.json',dict(version=version,finished=not proceed,
                    cumulative_distinct=number,checkpoint=str(checkpoint_path(run,version)),
                    observed_unix_s=time.time()))
    state = read(run/'published.json')
    final_checkpoint = checkpoint_path(run,state['version'])
    atomic_json(run/'frozen-evaluation.json',dict(mode='evaluate',head=manifest['head'],
                                                  checkpoint=str(final_checkpoint)))
    return read(run/f'node-{state["cumulative_distinct"]:05d}.json')


def run_head(campaign,head,observer,target,affinity,device):
    run = open_run(campaign,head,observer,target)
    finished = read(run/'published.json')['finished']
    processes = {}
    for case,cores in zip(SPEC['cases'],affinity):
        owner = run/f'owner-{case}.json'
        state = read(owner) if owner.exists() else None
        if (state and state['case']==case and state['head']==head and state['campaign']==str(campaign)
                and process_start(state['pid'])==state['proc_start']):
            processes[case] = ExistingCollector(run,case,state)
            continue
        if finished:
            reconcile_episode(run,case,read_rows(run/f'episodes-{case}.jsonl'),COMPACT/'attempts')
            continue
        command = ['taskset','-c',cores,sys.executable,'-u',__file__,'collector',
                   '--campaign',str(campaign),'--head',head,'--case',case]
        with (run/f'collector-{case}.log').open('a') as log:
            processes[case] = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,S1_W56_SLOT=case))
    jsonline(run/'dispatch.jsonl',dict(supervisor_pid=os.getpid(),
        collectors={k:p.pid for k,p in processes.items()},affinity=affinity,
        started_unix_s=time.time(),**BACKGROUND))
    node = train(run,processes,device)
    # A final mid-episode cut was declared before acquisition. Remaining ticks run the last
    # deterministic policy in that same episode and own zero additional training transitions.
    for case,p in processes.items():
        if p.wait():
            raise RuntimeError(f'Collector {case} did not retain its natural final episode.')
    return node


def series(campaign,observer,affinity,device):
    initial(campaign)
    lock = (campaign/'learner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    full = run_head(campaign,'full',observer,SPEC['maximum_distinct_full'],affinity,device)
    blind = None
    if full['explicit_blind_informative']:
        blind = run_head(campaign,'blind',observer,full['distinct'],affinity,device)
    result = dict(full=full['distinct'],blind=blind['distinct'] if blind else 0,
                  full_last_checkpoint=full['source_checkpoint'],
                  blind_last_checkpoint=blind['source_checkpoint'] if blind else None,
                  old_actor_updates=0,selection='Final eligible training node, no mechanism holdout',
                  complete_unix_s=time.time(),**BACKGROUND)
    atomic_json(campaign/'final.json',result)
    atomic_json(COMPACT/'r'/campaign.name/'final.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('init','series','collector'))
    parser.add_argument('--campaign',type=Path,default=DEFAULT)
    parser.add_argument('--observer',choices=('M0','M1'))
    parser.add_argument('--affinity',nargs=2,metavar=('A_CORES','B_CORES'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--head',choices=('full','blind'))
    parser.add_argument('--case',choices=('A','B'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    campaign_path = args.campaign.resolve()
    if args.action=='init':
        initial(campaign_path)
    elif args.action=='collector':
        collector_owner(campaign_path/args.head,args.case)
    else:
        if args.observer is None or args.affinity is None:
            parser.error('series requires a frozen observer and two assigned core sets')
        series(campaign_path,args.observer,args.affinity,args.device)

