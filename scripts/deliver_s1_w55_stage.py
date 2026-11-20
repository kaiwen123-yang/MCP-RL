"""Collect one concluded W55 stage, including its original video and per-PID resources."""
import argparse
import json
from pathlib import Path

from publish_s1_go2_w55 import RUN, STAGING, publish
from publish_s1_go2_w54 import write


def deliver_stage(worktree, label, level):
    lane = label[0].lower()
    compact = worktree / 'results/terrain_memory/simulation' / RUN / 'w55' / lane
    case = compact / label
    status = json.loads((case / (level + '.json')).read_text())
    if status['status'] in ('declared', 'running'):
        raise ValueError(f'{label}/{level} is still active')
    attempt = status['attempt']
    large = worktree / 'outputs/terrain_memory/simulation' / RUN
    raw = large / attempt
    zero_step_error = status.get('reason') == 'process_exited_without_completed_result'
    if not zero_step_error and not (raw / 'video-asset.json').exists():
        raise ValueError(f'{attempt} recorded video is not ready')
    job = large / 'jobs' / status['job']
    group = (label + '-' + level).lower()
    snapshot = STAGING / group / 'job-snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    resource_stream = job / 'resources.jsonl'
    if resource_stream.exists():
        with resource_stream.open() as source, (snapshot / 'resources.jsonl').open('w') as selected:
            for line in source:
                if json.loads(line)['root_pid'] == status['pid']:
                    selected.write(line)
    write(snapshot / 'stage.json', status)
    roots = [snapshot]
    if raw.exists():
        roots.append(raw)
    for filename in ('job.json', 'execution-contract.json', attempt+'.log', level+'-runner.log',
                     level+'-runtime.json'):
        path = job / filename
        if path.is_file():
            roots.append(path)
    policy = large / 'evaluation-policy' / attempt
    if policy.is_dir():
        roots.append(policy)
    publish(group, roots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--level', required=True)
    args = parser.parse_args()
    deliver_stage(args.workspace.resolve(), args.case, args.level)
