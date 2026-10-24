#!/usr/bin/env python3
"""Run the frozen S0 72-context contact/oracle experiment; resume saved tasks."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/terrain_memory/s0-contact-oracle.json')
    parser.add_argument('--run', default='tm-s0-contact-oracle-20261023')
    parser.add_argument('--source-output-root', help='Existing R2 output; weights are read in place, not copied.')
    parser.add_argument('--phase', choices=('all', 'simulate', 'results', 'figures'), default='all')
    args = parser.parse_args()
    from tm02.s0 import catalog, commands, sha256, simulate
    from tm02.evaluate import write_json
    config = json.loads((ROOT / args.config).read_text())
    base = json.loads((ROOT / config['base']['material_config']).read_text())
    if args.run != config['run_id']:
        raise ValueError('This entry point executes the fixed configured S0 run ID.')
    source = Path(args.source_output_root) if args.source_output_root else ROOT / config['base']['source_output_root']
    out, compact = (ROOT / config['outputs'][k] for k in ('large_root', 'compact_root'))
    out.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'run.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config_sha256'] != sha256(ROOT / args.config):
            raise ValueError('The saved experiment configuration differs; do not mix runs.')
    else:
        weights = []
        for predictor in config['predictors']:
            if predictor['type'] == 'frozen_learned_material':
                path = source / predictor['weights_relative_to_source_run']
                digest = sha256(path)
                if digest != predictor['weights_sha256']:
                    raise ValueError(f'Frozen input digest differs: {path}')
                weights.append(dict(predictor=predictor['id'], path=str(path), sha256=digest,
                                    original_release=config['base']['release_url'], seed=0))
        manifest = dict(run=args.run, config=config, config_sha256=sha256(ROOT / args.config),
                        base_numerics=base['numerics'], weights=weights, host=platform.node(),
                        python=sys.version, cpu=platform.processor(), invocations=[], new_training_runs=0)
        contexts, actions = catalog(config)
        write_json(out / 'events.json', [commands(config, c, a)[1] for c in contexts for a in actions])
        write_json(out / 'contexts-actions.json', dict(contexts=contexts, actions=actions))
        (out / 'requirements-lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    invocation = dict(command=' '.join(sys.argv), phase=args.phase,
                      start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in
                                     [Path(__file__), *sorted((ROOT / 'src/tm02').glob('*.py'))]})
    manifest['invocations'].append(invocation)
    write_json(manifest_path, manifest)
    begin = time.monotonic()
    if args.phase in ('all', 'simulate'):
        simulate(config, base, source, out)
    if args.phase in ('all', 'results'):
        from tm02.s0_results import summarize
        summarize(config, out, compact)
    if args.phase in ('all', 'figures'):
        from tm02.s0_figures import figures
        figures(config, out, compact)
    invocation.update(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      elapsed_s=time.monotonic()-begin)
    write_json(manifest_path, manifest)
    print(json.dumps(dict(phase=args.phase, completed=True, elapsed_s=invocation['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
