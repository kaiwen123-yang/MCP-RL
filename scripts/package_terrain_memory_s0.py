#!/usr/bin/env python3
"""Package new S0 arrays on the execution host; keep the video a separate asset.

Old weights stay at their original Release. Upload with gh on this host and
verify the returned asset size and SHA256; no download verification is needed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tm02.s0 import sha256
from tm02.evaluate import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/terrain_memory/s0-contact-oracle.json')
    parser.add_argument('--release-tag', default='tm-s0-contact-oracle-evidence-20261023')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    out, compact = (ROOT / config['outputs'][key] for key in ('large_root', 'compact_root'))
    index_path = compact / 'index.json'
    index = json.loads(index_path.read_text())
    index['run_metadata'] = json.loads((out / 'run.json').read_text())
    index['run_metadata'].pop('config')
    index['run_metadata']['config_file'] = args.config
    index.pop('policy_summary', None)
    index['policy_summary_csv'] = 'policy-summary.csv'
    index['full_run_provenance'] = config['run_id'] + '/run.json inside the trajectory archive'
    index['run_metadata']['hardware'] = dict(
        cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')),
        available_cpus=len(os.sched_getaffinity(0)), simulation_workers=config['execution']['workers'],
        accelerator='CPU only; no training')
    if index['counts']['complete_two_pad_predictions'] != config['budget']['total_complete_two_pad_predictions']:
        raise ValueError('The fixed S0 experiment is not complete.')
    archive = out / (config['run_id'] + '-trajectories.tar.gz')
    if not archive.exists():
        with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
            for name in ('trajectories', 'events.json', 'contexts-actions.json', 'run.json', 'requirements-lock.txt'):
                tar.add(out / name, arcname=config['run_id'] + '/' + name)
    video = out / 'complete-task-gamma0025.mp4'
    base_url = 'https://github.com/kaiwen123-yang/MCP-RL/releases'
    assets = [dict(filename=p.name, execution_host_path=str(p), bytes=p.stat().st_size,
                   sha256=sha256(p), url=base_url+'/download/'+args.release_tag+'/'+p.name)
              for p in (archive, video)]
    index['publication'] = dict(release_url=base_url+'/tag/'+args.release_tag, assets=assets,
        uploaded=False, weights_repackaged=False, user_computer_secondary_archive=False,
        contents='All 6048 complete two-pad time series and per-candidate continuation records; event catalog, run provenance and environment lock. Video is a separate asset. No existing R2 weights included.')
    write_json(index_path, index)
    print(json.dumps(index['publication'], indent=2), flush=True)


if __name__ == '__main__':
    main()
