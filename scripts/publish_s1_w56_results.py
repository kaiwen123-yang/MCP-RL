"""Serial D57 publication of the already qualified W56 branches and their stage evidence."""
import fcntl
import json
import subprocess
import time

from check_s1_w56_outcomes import check
import s1_w60_clock_evidence as r60_progress
from record_s1_w56 import CASES,HOST,N,REL,ROOT,derive,read

BRANCHES=('codex/w56-f','codex/w56-native','codex/w56-material','codex/w56-r60','codex/w56-r58-prefix','codex/w60-prefix')
WRITER=HOST/'MCP-RL-w55-integration'
LOCK=WRITER/'work/w55-integration.lock'


def git(*args):
    return subprocess.check_output(['git',*map(str,args)],cwd=ROOT,text=True,
                                   stderr=subprocess.PIPE).strip()


def versions():
    # Only stage transitions, new receipts or a lane commit change the snapshot.
    # A heartbeat updates several times per minute and is sampled at these events.
    refs=[git('rev-parse',ref) for ref in BRANCHES]
    files={}
    for label,tree,levels in CASES:
        lane,_,build=label.split('-')
        window='w55' if build=='r55' else 'w56'
        folder=tree/REL/window/lane.lower()
        for case in folder.glob('**/'+label):
            for path in [case/'queue.json',*(case/(v+'.json') for v in levels)]:
                if path.exists():
                    files[str(path)]=path.stat().st_mtime_ns
    for path in (N/REL/'w56/n').glob('queue-*.json'):
        files[str(path)]=path.stat().st_mtime_ns
    for path in (ROOT/REL/'w55/delivery').glob('f-*-r55-e600/delivery.json'):
        files[str(path)]=path.stat().st_mtime_ns
    advancing=[case+'-'+level for case,level,*_ in r60_progress.milestones()]
    return [refs,files,advancing]


def merge_evidence():
    for ref in BRANCHES:
        base=git('merge-base','HEAD',ref)
        if base==git('rev-parse',ref):
            continue
        paths=git('diff','--name-only',base,ref).splitlines()
        if not all(p.startswith(str(REL/'w56')+'/') for p in paths):
            raise ValueError('Unintegrated source change on '+ref+' needs serial source integration')
        git('merge','--no-edit',ref)


def publish():
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        git('fetch','-q','origin','main')
        git('merge','--no-edit','origin/main')
        merge_evidence()
        check()
        r60_progress.record()
        index=derive()
        paths=[REL/'w56/index.json',REL/'w56/phase.json',REL/'w56/baseline',
               REL/'current.json','experiments/terrain_memory/s1-go2-local-readiness-r12.json']
        if r60_progress.REPORT.is_file():
            paths.append(r60_progress.REPORT.relative_to(ROOT))
        git('add','--',*paths)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode==1:
            git('commit','--only','-m','Derive W56 task qualifications, evidence and active queues from D57 originals','--',*paths)
        head=git('rev-parse','HEAD')
        git('push','origin','HEAD:main')
        git('fetch','-q','origin','main')
        git('merge-base','--is-ancestor',head,'origin/main')
        print(json.dumps(dict(verified_commit=head,snapshot=index['snapshot_unix_s'],
                              qualifications=index['original_qualification'],status=index['status'])),flush=True)
        return index


def watch():
    if ROOT != WRITER:
        raise ValueError('Run the stage publisher from the shared W55/W56 integration worktree '+str(WRITER))
    marker=ROOT/'work/w56-published-inputs.json'
    last=read(marker) if marker.exists() else None
    while True:
        current=versions()
        if current!=last:
            try:
                index=publish()
            except subprocess.CalledProcessError as error:
                detail=error.stderr or ''
                transient=any(text in detail.lower() for text in ('timed out','timeout','tls',
                    'could not resolve','connection reset','connection refused','connection was reset','http 502','http 503','http 504'))
                if error.cmd[1] not in ('fetch','push') or not transient:
                    raise
                print(f'GitHub {error.cmd[1]} transport: {detail.strip()}. The next poll fetches main before any push.',flush=True)
                time.sleep(30)
                continue
            last=current
            marker.write_text(json.dumps(current))
            if index['status']=='declared_ladders_concluded_and_delivered':
                return
        time.sleep(30)


if __name__=='__main__':
    watch()
