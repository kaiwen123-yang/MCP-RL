"""Score and directly publish closed W59 research outputs while new physics continues."""
import fcntl
import json
import os
import time

from clip_s1_w59_episode import excerpts
from collect_s1_w59 import collect
from publish_s1_w59 import STATE, WORK, ensure_release, episode, publish
from publish_s1_w59_branches import publish_source
from score_s1_w59_episode import run as score


def verified(label):
    p=STATE/'delivery'/(label+'.json')
    return p.is_file() and json.loads(p.read_text()).get('status')=='verified'


def collect_locked():
    with (WORK/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        collect()


def tick():
    for p in sorted((STATE/'attempts').glob('*.json')):
        row=json.loads(p.read_text())
        if not row.get('ended_unix_s') or row.get('task_mode') in ('branch','training'):
            continue
        name=p.stem
        ordinary=row['status']!='technical_interruption' and row.get('physical_samples',0)>0
        if ordinary and not (STATE/'gait/phases'/(name+'.json')).is_file():
            score(name)
        if not verified(name):
            episode(name)
        if ordinary and row.get('case')=='A' and row.get('task_mode')=='original' and row.get('level') in ('E60','E300','E600'):
            label=name+'-gait-excerpts'
            if not verified(label):
                target=excerpts(name)
                publish(label,[(f.name,f) for f in sorted(target.iterdir()) if f.is_file()])
    registry=STATE/'action/registry-v2.json'
    if registry.is_file():
        reg=json.loads(registry.read_text())
        for source in reg['states']['development']:
            publish_source(source,'development',reg['signed'])
    collect_locked()


def main():
    with (WORK/'deliver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_release()
        while True:
            tick()
            if (STATE/'outcome.json').exists():
                return
            time.sleep(30)


if __name__=='__main__':
    main()
