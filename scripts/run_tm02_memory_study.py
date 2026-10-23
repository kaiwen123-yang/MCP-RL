#!/usr/bin/env python3
"""Execute the fixed third-round TM02 material-memory study, resuming real artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/tm02/r2-memory-study.json')
    parser.add_argument('--run', default='tm02-memory-study-20261023')
    parser.add_argument('--phase', choices=['all', 'prepare', 'train', 'diagnose', 'evaluate', 'figures'], default='all')
    args = parser.parse_args()
    if args.phase == 'all':
        for phase in ('prepare', 'train', 'diagnose', 'evaluate', 'figures'):
            environment = dict(os.environ, JAX_PLATFORMS='cuda' if phase == 'train' else 'cpu',
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            subprocess.run([sys.executable, '-u', str(Path(__file__)), '--config', args.config,
                            '--run', args.run, '--phase', phase], cwd=ROOT, env=environment, check=True)
        return

    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    config_path = ROOT / args.config
    study = json.loads(config_path.read_text(encoding='utf-8'))
    base = json.loads((ROOT / study['base']['config']).read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'experiments/tm02/r2-revisit.json').read_text(encoding='utf-8'))
    out = ROOT / base['outputs']['large_root'] / args.run
    old = ROOT / base['outputs']['large_root'] / study['base']['run']
    compact = ROOT / study['outputs']['compact_reproduction_root'] / args.run
    out.mkdir(parents=True, exist_ok=True)
    run_file = out / 'run.json'
    source_files = [Path(__file__), *sorted((ROOT / 'src/tm02').glob('*.py'))]
    invocation = dict(command=' '.join(sys.argv), phase=args.phase,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        execution=study['execution'],
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_files})
    snapshot = out / 'source_snapshots' / (args.phase + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    for source in [*source_files, config_path, ROOT / study['base']['config'],
                   ROOT / 'experiments/tm02/r2-revisit.json']:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    invocation['source_snapshot'] = snapshot.relative_to(out).as_posix()
    if run_file.exists():
        run = json.loads(run_file.read_text())
    else:
        run = dict(run=args.run, config=study,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            host=platform.node(), python=sys.version, jax=jax.__version__,
            hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                              '--format=csv,noheader'], text=True).strip(),
            cpu=platform.processor(), invocations=[], output=str(out))
        (out / 'requirements-lock.txt').write_text(
            subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    run['invocations'].append(invocation)
    run_file.write_text(json.dumps(run, indent=2, ensure_ascii=False) + '\n')
    begin = time.monotonic()
    if args.phase == 'prepare':
        from tm02.memory_data import prepare_data
        prepare_data(study, base, protocol, old, out)
    elif args.phase == 'train':
        from tm02.memory_training import train_study
        train_study(study, base, out, old)
    elif args.phase == 'diagnose':
        from tm02.memory_evaluate import diagnose_study
        diagnose_study(study, base, out, old, ROOT, workers=study['execution']['evaluation_workers'])
    elif args.phase == 'evaluate':
        from tm02.memory_evaluate import evaluate_study
        evaluate_study(study, base, out, old, ROOT, workers=study['execution']['evaluation_workers'])
    elif args.phase == 'figures':
        from tm02.memory_figures import study_figures
        study_figures(study, out, compact)
    invocation['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    invocation['elapsed_s'] = time.monotonic() - begin
    run_file.write_text(json.dumps(run, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(phase=args.phase, completed=True, elapsed_s=invocation['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
