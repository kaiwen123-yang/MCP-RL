"""Fixed Fig.14 histories and full-dataset Adam updates, reusing TM02 physics."""
import copy
import hashlib
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .material import initialize, normalization, loss, save_weights
from .reference import Reference, random_path


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare_data(config, out):
    if (out/'data.json').exists():
        return
    reference = Reference(config['reference_material']['parameters'])
    observations, truth = {}, {}
    for kind in ('train', 'evaluation'):
        histories = []
        responses = []
        for index, seed in enumerate(config['data'][kind]['strain_seeds']):
            data_config = copy.deepcopy(config['data'])
            data_config[kind]['strain_seed'] = seed
            t, e, kt, ke = random_path(data_config, kind == 'evaluation')
            histories.append((t, e))
            responses.append(reference.path(t, e, config['numerics']))
            observations[f'{kind}_knot_time_{index:03d}'] = kt
            observations[f'{kind}_knot_strain_{index:03d}'] = ke
        observations[f'{kind}_time'] = np.stack([h[0] for h in histories])
        observations[f'{kind}_strain'] = np.stack([h[1] for h in histories])
        observations[f'{kind}_seeds'] = np.asarray(config['data'][kind]['strain_seeds'])
        truth.update({f'{kind}_{key}': np.stack([r[key] for r in responses])
                      for key in responses[0]})
    noise_config = config['data']['stress_noise']
    noise = np.stack([np.random.default_rng(seed).normal(
        noise_config['mean_MPa'], noise_config['noisy_std_MPa'],
        (config['data']['train']['increments'], 6))
        for seed in noise_config['seeds_by_training_path']])
    observations['ideal_stress'] = truth['train_stress'][:, 1:]
    observations['noisy_stress'] = observations['ideal_stress'] + noise
    observations['training_noise'] = noise
    np.savez_compressed(out/'observations.npz', **observations)
    np.savez_compressed(out/'reference.npz', **truth)
    write_json(out/'data.json', dict(
        training_histories=len(config['data']['train']['strain_seeds']),
        heldout_histories=len(config['data']['evaluation']['strain_seeds']),
        frozen_before_training=True,
        arrays={name: dict(sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
                          bytes=(out/name).stat().st_size)
                for name in ('observations.npz', 'reference.npz')},
        axes='history,time,physical_component',
        actual_noise_mean_MPa=float(noise.mean()),
        actual_noise_std_MPa=float(noise.std())))
    print('Saved fixed 100 x 100 training and 20 x 250 held-out histories before training.', flush=True)


def train(config, out):
    obs = np.load(out/'observations.npz')
    tc = config['training']
    runs = [(condition, seed) for condition in tc['conditions']
            for seed in tc['initialization_seeds']]
    total = tc['iterations_per_initialization']
    chunk = config['execution']['checkpoint_every_iterations']
    group_size = config['execution']['initialization_batch_size']
    micro_size = config['execution']['path_microbatch_size']
    e = obs['train_strain']
    dt = np.diff(obs['train_time'], axis=1)
    paths, steps = dt.shape
    # These scales are computed once using the entire training set only.
    norms = {condition: normalization(e, obs[f'{condition}_stress'], dt)
             for condition in tc['conditions']}
    write_json(out/'normalization.json', {
        condition: {key: np.asarray(value).tolist() for key, value in norm.items()}
        for condition, norm in norms.items()})

    def stack(trees):
        return jax.tree.map(lambda *x: jnp.asarray(np.stack(x), dtype=jnp.float32), *trees)

    eps_batches = jnp.asarray(e[:, 1:].reshape(-1, micro_size, steps, 6), dtype=jnp.float32)
    dt_batches = jnp.asarray(dt.reshape(-1, micro_size, steps), dtype=jnp.float32)
    schedule = optax.exponential_decay(tc['initial_learning_rate'],
        tc['lr_decay_every_iterations'], tc['lr_decay_factor'], staircase=True)
    optimizer = optax.adam(schedule)

    def path_mean(weights, norm, eps, stresses, dts):
        _, terms = jax.vmap(loss, (None, None, 0, 0, 0))(
            weights, norm, eps, stresses, dts)
        terms = jnp.mean(terms, axis=0)
        return terms.sum(), terms

    per_model = jax.vmap(jax.value_and_grad(path_mean, has_aux=True),
                         (0, 0, None, 0, None))

    def dataset_gradient(weights, norm, stress_batches):
        zeros = jax.tree.map(jnp.zeros_like, weights)
        def accumulate(carry, inputs):
            gradient_sum, term_sum = carry
            eps, stress, dts = inputs
            (_, terms), gradient = per_model(weights, norm, eps, stress, dts)
            return (jax.tree.map(jnp.add, gradient_sum, gradient), term_sum+terms), None
        (gradient, terms), _ = jax.lax.scan(accumulate,
            (zeros, jnp.zeros((stress_batches.shape[1], 2), dtype=jnp.float32)),
            (eps_batches, stress_batches, dt_batches))
        # Each loss is already a path/time/component mean. No second batch division.
        count = eps_batches.shape[0]
        return jax.tree.map(lambda x: x/count, gradient), terms/count

    def advance(weights, state, norm, stresses, length):
        def step(carry, _):
            w, opt = carry
            gradient, terms = dataset_gradient(w, norm, stresses)
            updates, opt = optimizer.update(gradient, opt, w)
            return (optax.apply_updates(w, updates), opt), terms
        return jax.lax.scan(step, (weights, state), None, length=length)

    advance = jax.jit(advance, static_argnames=('length',))
    finaldir = out/'final'
    finaldir.mkdir(exist_ok=True)
    checkpointdir = out/'checkpoints'
    checkpointdir.mkdir(exist_ok=True)
    records, elapsed_total = [], 0.
    for start in range(0, len(runs), group_size):
        group = runs[start:start+group_size]
        nrun = len(group)
        norm = stack([norms[cond] for cond, seed in group])
        w = stack([initialize(seed) for cond, seed in group])
        stress = np.stack([obs[f'{cond}_stress'] for cond, seed in group])
        stresses = jnp.asarray(stress.reshape(nrun, -1, micro_size, steps, 6).swapaxes(0, 1),
                               dtype=jnp.float32)
        opt = optimizer.init(w)
        curves = np.full((total, nrun, 2), np.nan, dtype=np.float64)
        checkpoint = checkpointdir/f'group-{start:02d}.pkl'
        iteration, elapsed = 0, 0.
        if checkpoint.exists():
            with checkpoint.open('rb') as stream:
                saved = pickle.load(stream)
            w, opt = jax.tree.map(jnp.asarray, (saved['weights'], saved['optimizer']))
            iteration, elapsed = saved['iteration'], saved['elapsed_s']
            curves[:iteration] = saved['curves']
            print(f'Resuming group {start} at full-dataset update {iteration}.', flush=True)
        while iteration < total:
            begin = time.monotonic()
            length = min(chunk, total-iteration)
            (w, opt), terms = advance(w, opt, norm, stresses, length)
            terms = np.asarray(terms)
            elapsed += time.monotonic()-begin
            curves[iteration:iteration+length] = terms
            iteration += length
            saved = dict(weights=jax.device_get(w), optimizer=jax.device_get(opt),
                         iteration=iteration, elapsed_s=elapsed, curves=curves[:iteration],
                         runs=group, paths_per_update=paths,
                         path_microbatch_size=micro_size)
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(saved, stream, protocol=5)
            temporary.replace(checkpoint)
            sums = terms[-1].sum(axis=1)
            print(json.dumps(dict(group_start=start, independent_runs=nrun,
                iteration=iteration, of=total, paths_per_update=paths,
                min_loss_MPa=float(sums.min()), median_loss_MPa=float(np.median(sums)),
                max_loss_MPa=float(sums.max()), elapsed_s=round(elapsed, 2))), flush=True)
            if not np.all(np.isfinite(terms)):
                raise FloatingPointError('Nonfinite training loss; actual checkpoint retained.')
        # Selection is at the final weights after update 20,000, using training only.
        _, final_terms = jax.jit(dataset_gradient)(w, norm, stresses)
        final_terms = np.asarray(final_terms)
        totals = final_terms.sum(axis=1)
        for i, (condition, seed) in enumerate(group):
            wi, ni = jax.tree.map(lambda x: np.asarray(x[i]), (w, norm))
            stem = f'{condition}-seed-{seed:02d}'
            save_weights(finaldir/f'{stem}.npz', wi, ni)
            with (finaldir/f'{stem}-optimizer.pkl').open('wb') as stream:
                pickle.dump(jax.tree.map(lambda x: np.asarray(x[i]) if np.ndim(x) else np.asarray(x), opt),
                            stream, protocol=5)
            np.savez_compressed(finaldir/f'{stem}-loss.npz', iteration=np.arange(total),
                stress_MAE=curves[:, i, 0], biot_MAE=curves[:, i, 1], final_terms=final_terms[i])
            records.append(dict(condition=condition, seed=seed, iterations=iteration,
                final_training_loss_MPa=float(totals[i]),
                final_stress_MAE_MPa=float(final_terms[i, 0]),
                final_biot_MAE_MPa=float(final_terms[i, 1]), weights=f'final/{stem}.npz'))
        elapsed_total += elapsed
    selection = {cond: min((r for r in records if r['condition'] == cond),
                 key=lambda r: r['final_training_loss_MPa']) for cond in tc['conditions']}
    from .material import load_weights
    (out/'selected').mkdir(exist_ok=True)
    for condition, chosen in selection.items():
        weights, norm = load_weights(out/chosen['weights'])
        chosen['potential_weights'] = f'selected/{condition}-potentials.npz'
        save_weights(out/chosen['potential_weights'], weights, norm)
    write_json(out/'training.json', dict(runs=records, selection=selection,
        elapsed_s=elapsed_total, rule=tc['selection'], total_optimizer_updates=total*len(runs),
        paths_per_update=paths, full_sequence_increments=steps,
        initialization_batch_size=group_size, path_microbatch_size=micro_size))
    print('All 50 final models, optimizer states and complete loss curves saved.', flush=True)
