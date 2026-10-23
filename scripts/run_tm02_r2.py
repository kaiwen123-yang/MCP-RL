#!/usr/bin/env python3
"""Run the fixed TM02 material budget on the project environment.

work/envs/tm02/bin/python scripts/run_tm02_r2.py --config experiments/tm02/r2-rnn.json --run tm02-rnn-20261023
The same command resumes the saved optimizer state; evaluation uses final models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import numpy as np
import optax
from tm02.material import initialize, normalization, loss, save_weights
from tm02.reference import Reference, random_path


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def prepare_data(config, out):
    if (out/'observations.npz').exists():
        return
    reference = Reference(config['reference_material']['parameters'])
    obs, truth = {}, {}
    for kind in ('train', 'evaluation'):
        t, e, kt, ke = random_path(config['data'], kind == 'evaluation')
        result = reference.path(t, e, config['numerics'])
        obs.update({f'{kind}_time': t, f'{kind}_strain': e,
                    f'{kind}_knot_time': kt, f'{kind}_knot_strain': ke})
        truth.update({f'{kind}_{key}': value for key, value in result.items()})
    noise = np.random.default_rng(config['data']['stress_noise']['seed']).normal(
        0, config['data']['stress_noise']['noisy_std_MPa'], (config['data']['train']['increments'], 6))
    obs['ideal_stress'] = truth['train_stress'][1:]
    obs['noisy_stress'] = obs['ideal_stress']+noise
    obs['training_noise'] = noise
    np.savez_compressed(out/'observations.npz', **obs)
    # True internal variables and evaluation stress are NEVER passed to training.
    np.savez_compressed(out/'reference.npz', **truth)
    print('Generated complete 200-increment training and independent 250-increment evaluation histories.', flush=True)


def train(config, out):
    obs = np.load(out/'observations.npz')
    tc = config['training']
    runs = [(condition, seed) for condition in tc['conditions']
            for seed in tc['initialization_seeds']]
    nrun, total = len(runs), tc['iterations_per_initialization']
    chunk = config['execution']['checkpoint_every_iterations']
    dt = np.diff(obs['train_time'])
    e = obs['train_strain']
    norms = [normalization(e, obs[f'{cond}_stress'], dt) for cond, seed in runs]
    def stack(trees):
        return jax.tree.map(lambda *x: jnp.asarray(np.stack(x), dtype=jnp.float32), *trees)
    norm = stack(norms)
    w = stack([initialize(seed) for cond, seed in runs])
    stresses = jnp.asarray(np.stack([obs[f'{cond}_stress'] for cond, seed in runs]), dtype=jnp.float32)
    eps, dtj = jnp.asarray(e[1:], dtype=jnp.float32), jnp.asarray(dt, dtype=jnp.float32)
    schedule = optax.exponential_decay(tc['initial_learning_rate'],
                                     tc['lr_decay_every_iterations'], tc['lr_decay_factor'], staircase=True)
    optimizer = optax.adam(schedule)
    opt = optimizer.init(w)
    curves = np.full((total, nrun, 2), np.nan, dtype=np.float64)
    checkpoint = out/'checkpoint.pkl'
    iteration, elapsed = 0, 0.
    if checkpoint.exists():
        with checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        w, opt = jax.tree.map(jnp.asarray, (saved['weights'], saved['optimizer']))
        iteration, elapsed = saved['iteration'], saved['elapsed_s']
        curves[:iteration] = saved['curves']
        print(f'Resuming all {nrun} independent runs at iteration {iteration}.', flush=True)
    per_model = jax.vmap(jax.value_and_grad(loss, has_aux=True), (0, 0, None, 0, None))
    def step(carry, _):
        weights, state = carry
        (values, terms), gradients = per_model(weights, norm, eps, stresses, dtj)
        updates, state = optimizer.update(gradients, state, weights)
        weights = optax.apply_updates(weights, updates)
        return (weights, state), terms
    advance = jax.jit(lambda weights, state: jax.lax.scan(step, (weights, state), None, length=chunk))
    while iteration < total:
        begin = time.monotonic()
        (w, opt), terms = advance(w, opt)
        terms = np.asarray(terms)
        elapsed += time.monotonic()-begin
        curves[iteration:iteration+chunk] = terms
        iteration += chunk
        saved = dict(weights=jax.device_get(w), optimizer=jax.device_get(opt),
                     iteration=iteration, elapsed_s=elapsed, curves=curves[:iteration], runs=runs)
        temporary = out/'checkpoint.tmp'
        with temporary.open('wb') as stream:
            pickle.dump(saved, stream, protocol=5)
        temporary.replace(checkpoint)
        sums = terms[-1].sum(axis=1)
        print(json.dumps(dict(iteration=iteration, of=total, independent_runs=nrun,
                              min_loss_MPa=float(np.min(sums)), median_loss_MPa=float(np.median(sums)),
                              max_loss_MPa=float(np.max(sums)), elapsed_s=round(elapsed, 2))), flush=True)
        if not np.all(np.isfinite(terms)):
            raise FloatingPointError('Nonfinite actual training loss; failed checkpoint retained.')
    final_loss = jax.jit(jax.vmap(loss, (0, 0, None, 0, None)))(w, norm, eps, stresses, dtj)
    totals, final_terms = map(np.asarray, final_loss)
    finaldir = out/'final'
    finaldir.mkdir(exist_ok=True)
    records = []
    for i, (condition, seed) in enumerate(runs):
        wi, ni = jax.tree.map(lambda x: np.asarray(x[i]), (w, norm))
        stem = f'{condition}-seed-{seed:02d}'
        save_weights(finaldir/f'{stem}.npz', wi, ni)
        with (finaldir/f'{stem}-optimizer.pkl').open('wb') as stream:
            pickle.dump(jax.tree.map(lambda x: np.asarray(x[i]) if np.ndim(x) else np.asarray(x), opt), stream, protocol=5)
        np.savez_compressed(finaldir/f'{stem}-loss.npz', iteration=np.arange(total),
                            stress_MAE=curves[:, i, 0], biot_MAE=curves[:, i, 1],
                            final_terms=final_terms[i])
        records.append(dict(condition=condition, seed=seed, iterations=iteration,
                            final_training_loss_MPa=float(totals[i]),
                            final_stress_MAE_MPa=float(final_terms[i, 0]),
                            final_biot_MAE_MPa=float(final_terms[i, 1]), weights=f'final/{stem}.npz'))
    selection = {cond: min((r for r in records if r['condition']==cond),
                          key=lambda r: r['final_training_loss_MPa']) for cond in tc['conditions']}
    selected_dir = out/'selected'
    selected_dir.mkdir(exist_ok=True)
    for condition, chosen in selection.items():
        i = runs.index((condition, chosen['seed']))
        wi, ni = jax.tree.map(lambda x: np.asarray(x[i]), (w, norm))
        potentials = {name: value for name,value in wi.items() if name != 'rnn'}
        chosen['potential_weights'] = f'selected/{condition}-potentials.npz'
        save_weights(out/chosen['potential_weights'], potentials, ni)
    write_json(out/'training.json', dict(runs=records, selection=selection, elapsed_s=elapsed,
                                       rule=tc['selection'], total_optimizer_updates=total*nrun))
    print('All final weights and optimizer states saved; selection uses only final training loss.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/tm02/r2-rnn.json')
    parser.add_argument('--run', default='tm02-rnn-20261023')
    parser.add_argument('--phase', choices=['all', 'train', 'evaluate', 'evaluate-frozen', 'revisit', 'diagnose'], default='all')
    parser.add_argument('--revisit-config', default='experiments/tm02/r2-revisit.json')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    multihistory = 'comparison' in config
    out = ROOT/config['outputs']['large_root']/args.run
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'run.json').exists():
        source_files = [Path(__file__), *sorted((ROOT/'src/tm02').glob('*.py'))]
        write_json(out/'run.json', dict(command=' '.join(sys.argv),
            config=config, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            host=platform.node(), python=sys.version, jax=jax.__version__, devices=[str(d) for d in jax.devices()],
            start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), output=str(out)))
        (out/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    if args.phase in ('all', 'train'):
        if multihistory:
            from tm02.multihistory import prepare_data as prepare_many, train as train_many
            prepare_many(config, out)
            train_many(config, out)
        else:
            prepare_data(config, out)
            train(config, out)
    if args.phase == 'evaluate-frozen':
        from tm02.multihistory_evaluate import evaluate_frozen
        evaluate_frozen(config, out, ROOT/config['outputs']['compact_root']/args.run, ROOT)
    if args.phase in ('all', 'evaluate'):
        if multihistory:
            from tm02.multihistory_evaluate import evaluate
            evaluate(config, out, ROOT/config['outputs']['compact_root']/args.run, ROOT)
        else:
            from tm02.evaluate import evaluate
            evaluate(config, out, ROOT/config['outputs']['compact_root']/args.run)
    if args.phase in ('all', 'revisit'):
        if multihistory:
            from tm02.multihistory_revisit import revisit
        else:
            from tm02.evaluate import revisit
        revisit_config = json.loads((ROOT/args.revisit_config).read_text(encoding='utf-8'))
        revisit(config, revisit_config, out, ROOT, args.run)
    if args.phase == 'diagnose':
        from tm02.evaluate import diagnose_selected_prediction
        diagnose_selected_prediction(config, out, ROOT/config['outputs']['compact_root']/args.run)


if __name__ == '__main__':
    main()
