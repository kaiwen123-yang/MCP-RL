"""Run each W56 frozen F case on its own original qualification ladder on D57."""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import analyze_s1_w55_f as analysis
import render_s1_go2_r12 as renderer
from run_s1_w55_f import ROOT, RUN, LARGE, declaration, digest, run, write

REL = Path('results/terrain_memory/simulation') / RUN
CASES = ROOT / REL / 'w56/f'
PLAN = '6da12efc2cb85dcc99e53243dca489c9966e4969'
PYTHON = sys.executable
SHARED = ROOT.parent / 'MCP-RL-w56-f'
WORK = ROOT / 'work/w56-f'


def read(path):
    return json.loads(path.read_text())


def commit(paths, message):
    with (WORK / 'git.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = [str(p.relative_to(ROOT)) for p in paths if p.is_file()]
        subprocess.run(['git','add','--',*files], cwd=ROOT, check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*files],cwd=ROOT).returncode:
            subprocess.run(['git','commit','--only','-m',message,'--',*files],cwd=ROOT,check=True)


def new_session(name, command, log):
    script = ('source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh academic'
              ' && export NUMEXPR_NUM_THREADS=1 && exec ' + shlex.join(command)
              + ' > ' + shlex.quote(str(log)) + ' 2>&1')
    subprocess.run(['tmux','new-session','-d','-s',name,'-c',str(ROOT),script],check=True)


def readiness(build, case):
    if build == 'S5-r57':
        evidence = read(CASES / f'r57-{case.lower()}-actor-selection.json')
        if not evidence['both_modes_verified'] or evidence['case'] != case:
            raise ValueError('Recorded frozen actor did not verify the executed feature selector')
    else:
        proof = 'r60-interface-qualification.json' if build == 'S5-r60' else 'interface-qualification.json'
        evidence = read(ROOT / REL / 'w56/c' / proof)
        if build == 'S5-r60':
            config='experiments/terrain_memory/s1-w56-f-r60.json'
            hashes=evidence['source_sha256']
            for source in [config, *read(ROOT/config)['material_compiled_artifacts']]:
                if digest(ROOT/source) != hashes[source]:
                    raise ValueError(f'WBC qualification and original executable differ: {source}')
        if evidence['ready_for_original_E30'] is not True:
            raise ValueError('The window and WBC interface has not met the formal consumption gate')


def queue(build):
    revision=build.removeprefix('S5-')
    for case in ('A','B'):
        readiness(build,case)
        label=f'F-{case}-{revision}'
        case_dir=CASES/label
        declaration(build,case,window='W56',small=CASES,plan=PLAN,results=CASES/revision)
        row=read(case_dir/'job.json')
        name='s1-w56-'+label.lower()+'-ladder'
        queue_state=dict(job=row['job_id'],case=case,tmux=name,deadline=None,wall_deadline_enabled=False,
                         kill_on_conversation_end=False,levels=(['E30','E60'] if revision=='r57' else ['E30','E60','E300','E600']),
                         status='waiting_for_A_E30' if label=='F-B-r58' else 'waiting_for_original_slot',
                         registered_unix_s=time.time())
        write(case_dir/'queue.json',queue_state)
        commit([case_dir/'job.json',case_dir/'execution-contract.json',case_dir/'queue.json'],
               f'Declare W56 {label} frozen original ladder')
        new_session(name,[PYTHON,'-u',__file__,'ladder',label],WORK/(label+'-ladder.log'))


def ladder(label):
    case_dir=CASES/label
    queue_path=case_dir/'queue.json'
    ticket=read(queue_path)
    case=read(case_dir/'job.json')
    revision=case['controller_build'].removeprefix('S5-')
    if label=='F-B-r58':
        other=CASES/'F-A-r58/E30.json'
        while not other.exists() or read(other)['status'] in ('waiting_for_original_slot','running'):
            time.sleep(10)
        if read(other)['status']!='passed':
            ticket.update(status='not_run_A_E30_did_not_pass')
            write(queue_path,ticket)
            commit([queue_path],'Record W56 F-B-r58 not started after A E30')
            return
    for level in ticket['levels']:
        stage=case_dir/(level+'.json')
        if stage.exists():
            raise FileExistsError(f'Original stage already declared: {stage}')
        write(stage,dict(status='waiting_for_original_slot',job=case['job_id'],tmux=ticket['tmux'],level=level,
                         deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
        status=subprocess.run([PYTHON,'-u',str(SHARED/'scripts/s1_w56_slot.py'),'--priority','F',
            '--label',label+'-'+level,'--',PYTHON,'-u',__file__,'physical',label,level]).returncode
        state=read(stage)
        paths=[stage,CASES/revision/'index.json']
        if 'attempt' in state:
            paths.extend((CASES/revision).glob(state['attempt']+'*.json'))
        commit(paths,f'Record W56 {label} {level} original result')
        if 'attempt' in state and (CASES/revision/(state['attempt']+'.json')).is_file():
            session='s1-w56-evidence-'+label.lower()+'-'+level.lower()
            new_session(session,['taskset','-c','20-21',PYTHON,'-u',__file__,'evidence',label,level],WORK/(session+'.log'))
        if status or state['status']!='passed':
            break
    ticket.update(status='ladder_ended',last_stage=level,last_status=state['status'],ended_unix_s=time.time())
    write(queue_path,ticket)
    commit([queue_path],f'Record W56 {label} ladder terminal state')


def evidence(label,level):
    case_dir=CASES/label
    stage=case_dir/(level+'.json')
    state=read(stage)
    attempt=state['attempt']
    compact=CASES/label.rsplit('-',1)[1]
    if label=='F-A-r60' and level=='E30':
        artifacts=read(case_dir/'job.json')['controller_artifact_identity']['material_compiled_sha256']
        # Retain one exact executable with its first original, including in the D57 Release ZIP.
        for relative, expected in artifacts.items():
            source=ROOT/relative
            archived=LARGE/attempt/'material-executable'/source.name
            archived.parent.mkdir(exist_ok=True)
            if not archived.exists():
                os.link(source,archived)
            if digest(archived)!=expected:
                raise ValueError(f'Original material executable differs from archived file: {archived}')
    renderer.SMALL=compact
    renderer.render(attempt)
    state['video']=read(LARGE/attempt/'video-asset.json')
    write(stage,state)
    write(LARGE/'jobs'/state['job']/(level+'-runtime.json'),state)
    analysis.SMALL=compact
    result=analysis.analyze(attempt)
    from compare_s1_w56_cap import compare
    pair=compare(case_dir,level)
    commit([stage,compact/(attempt+'-analysis.json'),pair],f'Record W56 {label} {level} video and paired analysis')
    subprocess.run([PYTHON,'-u',str(SHARED/'scripts/deliver_s1_w56_stage.py'),'--workspace',str(ROOT),
                    '--case',label,'--level',level],check=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('queue').add_argument('build',choices=('S5-r57','S5-r58','S5-r60'))
    sub.add_parser('ladder').add_argument('label')
    for command in ('physical','evidence'):
        stage=sub.add_parser(command)
        stage.add_argument('label')
        stage.add_argument('level',choices=('E30','E60','E300','E600'))
    args=parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.action=='queue':
        queue(args.build)
    elif args.action=='ladder':
        ladder(args.label)
    elif args.action=='physical':
        folder=CASES/args.label
        sys.exit(run(folder,args.level,small=CASES/args.label.rsplit('-',1)[1],render=False))
    else:
        evidence(args.label,args.level)
