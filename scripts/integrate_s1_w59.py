"""Non-force integration of closed W59 results and the single Shanghai worklog."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from collect_s1_w59 import R, ROOT, collect
from publish_s1_go2_w54 import write

WORK=ROOT/'work/w59'
ZONE=ZoneInfo('Asia/Shanghai')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def integrate():
    with (WORK/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        collect()
        rel=str(R.relative_to(ROOT))
        git('add','--','docs/S1_DAILY_WORKLOG.md',rel,':(exclude)'+rel+'/episodes/**',
            ':(exclude)**/*.lock',':(exclude)**/*.tmp*',':(exclude)**/*.pending')
        if subprocess.call(['git','diff','--cached','--quiet'],cwd=ROOT)==1:
            git('commit','-m','Publish W59 measured results and Shanghai worklog')
    git('fetch','origin','main')
    git('merge','--no-edit','origin/main')
    git('push','origin','HEAD:refs/heads/main')
    head=git('rev-parse','HEAD')
    tip=git('ls-remote','origin','refs/heads/main').split()[0]
    if head!=tip:
        git('fetch','origin','main')
        subprocess.run(['git','merge-base','--is-ancestor',head,'origin/main'],cwd=ROOT,check=True)
    write(WORK/'last-push.json',dict(commit=head,remote_head=tip,checked_shanghai=datetime.now(ZONE).isoformat(timespec='seconds')))


def signature():
    # A node, an attempt ending or a verified asset is a publication event.
    # Live physical-clock ticks do not create a stream of Git commits.
    d={}
    for folder in ('attempts','delivery','model','action','learning','mechanism','dispatch'):
        for p in (R/folder).glob('**/*.json'):
            if p.name.endswith('-progress.json') or p.name.startswith('live'):
                continue
            s=p.stat()
            d[str(p.relative_to(R))]=(s.st_size,s.st_mtime_ns)
    return d


def main():
    with (WORK/'integration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            now=signature()
            if now!=previous:
                try:
                    integrate()
                except subprocess.CalledProcessError as e:
                    write(WORK/'integration-failure.json',dict(command=e.cmd,exit_code=e.returncode,
                         observed_shanghai=datetime.now(ZONE).isoformat(timespec='seconds')))
                    if (ROOT/'.git').is_file() and git('ls-files','-u'):
                        raise
                    print('Git integration pending',e,flush=True)
                else:
                    previous=now
            if (R/'outcome.json').exists() and (R/'delivery-ended.json').exists():
                return
            time.sleep(120)


if __name__=='__main__':
    main()
