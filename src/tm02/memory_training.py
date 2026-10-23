"""Three fixed-data arms using the existing complete-history TM02 trainer.

Each arm's observations.npz is prepared before this module runs. It contains
train_time, train_strain and ideal/noise0/noise1/noise2_stress arrays. No reference
internal state or final-evaluation data is read by training or model selection.
"""
import hashlib
import json
import shutil

import numpy as np

from .material import load_weights, save_weights
from .multihistory import train, write_json


CELLS = ('ideal', 'noise0', 'noise1', 'noise2')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(arm, cell, source, prefix, reused):
    result = dict(source)
    result.update(arm=arm, cell=cell,
                  condition='ideal' if cell == 'ideal' else 'noisy',
                  noise_realization=None if cell == 'ideal' else int(cell[-1]),
                  reused=reused)
    result['weights'] = f"{prefix}/{source['weights']}"
    result.pop('potential_weights', None)
    stem = result['weights'][:-4]
    result['optimizer'] = f'{stem}-optimizer.pkl'
    result['loss_curve'] = f'{stem}-loss.npz'
    return result


def _reuse_a(config, out, old_out, arm_out):
    """Copy the ten exact archived models, retaining the old final losses."""
    old = json.loads((old_out/'training.json').read_text(encoding='utf-8'))
    records = []
    destination = arm_out/'final'
    destination.mkdir(parents=True, exist_ok=True)
    for cell, condition in (('ideal', 'ideal'), ('noise0', 'noisy')):
        for seed in config['training']['initialization_seeds']:
            source_record = next(r for r in old['runs']
                                 if r['condition'] == condition and r['seed'] == seed)
            old_stem = f'{condition}-seed-{seed:02d}'
            new_stem = f'{cell}-seed-{seed:02d}'
            source_paths = {}
            for suffix in ('.npz', '-optimizer.pkl', '-loss.npz'):
                source = old_out/'final'/f'{old_stem}{suffix}'
                target = destination/f'{new_stem}{suffix}'
                shutil.copyfile(source, target)
                source_paths[target.relative_to(out).as_posix()] = {
                    'source': str(source), 'sha256': _sha256(source)}
            renamed = dict(source_record, weights=f'final/{new_stem}.npz')
            record = _record('A_random100', cell, renamed,
                             arm_out.relative_to(out).as_posix(), True)
            record['source_run'] = config['base']['run']
            record['source_artifacts'] = source_paths
            records.append(record)
    return records


def train_study(config, base_config, out, old_out):
    """Train 50 new models and select 12 cells using final training loss only.

    ``out/training/<arm>/observations.npz`` contains each frozen arm. The shared
    multihistory trainer batches independent models and accumulates the mean
    gradient of all paths before each Adam update, including all 1000 B paths.
    ``old_out`` points to the extracted second-round material run directory.
    """
    tc = config['training']
    old_norms = {}
    norm_sources = {}
    for condition in ('ideal', 'noisy'):
        source = old_out/'final'/f'{condition}-seed-00.npz'
        with np.load(source) as weights:
            old_norms[condition] = {
                key.split('/', 1)[1]: weights[key].copy()
                for key in weights.files if key.startswith('norm/')}
        norm_sources[condition] = dict(source=str(source), sha256=_sha256(source))
    write_json(out/'normalization.json', {
        condition: {key: np.asarray(value).tolist() for key, value in norm.items()}
        for condition, norm in old_norms.items()})

    records, arm_compute = [], []
    for arm in config['arms']:
        arm_id = arm['id']
        arm_out = out/'training'/arm_id
        active_cells = ('noise1', 'noise2') if arm_id == 'A_random100' else CELLS
        arm_config = {'training': {
            'conditions': list(active_cells),
            'initialization_seeds': tc['initialization_seeds'],
            'iterations_per_initialization': tc['iterations_per_model'],
            'initial_learning_rate': tc['initial_learning_rate'],
            'lr_decay_every_iterations': tc['lr_decay_every_iterations'],
            'lr_decay_factor': tc['lr_decay_factor'],
            'selection': tc['selection']},
            'execution': {key: base_config['execution'][key] for key in (
                'checkpoint_every_iterations', 'initialization_batch_size',
                'path_microbatch_size')},
            'data': {'observations': 'observations.npz',
                     'histories_per_optimizer_update': arm['histories'],
                     'increments_per_history': tc['increments_per_history'],
                     'normalization': 'Frozen second-round values read directly from saved weights'}}
        # Execution sizes affect memory and speed only, not the experiment.
        arm_config['execution'].update(config.get('execution', {}))
        arm_config['execution']['path_microbatch_size'] = min(
            arm['histories'], arm_config['execution']['path_microbatch_size'])
        fixed_norms = {
            cell: old_norms['ideal' if cell == 'ideal' else 'noisy']
            for cell in active_cells}
        if not (arm_out/'training.json').exists():
            write_json(arm_out/'training-config.json', arm_config)
            train(arm_config, arm_out, fixed_norms=fixed_norms)
        trained = json.loads((arm_out/'training.json').read_text(encoding='utf-8'))
        prefix = arm_out.relative_to(out).as_posix()
        if arm_id == 'A_random100':
            records.extend(_reuse_a(config, out, old_out, arm_out))
        records.extend(_record(arm_id, r['condition'], r, prefix, False)
                       for r in trained['runs'])
        arm_compute.append(dict(
            arm=arm_id, new_models=len(trained['runs']),
            elapsed_s=trained['elapsed_s'],
            optimizer_updates=trained['total_optimizer_updates'],
            paths_per_update=trained['paths_per_update'],
            increments_per_path=trained['full_sequence_increments'],
            history_gradient_evaluations=(trained['total_optimizer_updates']
                                          * trained['paths_per_update']),
            additional_final_training_loss_history_evaluations=(
                len(trained['runs'])*trained['paths_per_update']),
            initialization_batch_size=trained['initialization_batch_size'],
            path_microbatch_size=trained['path_microbatch_size']))

    selected = out/'selected'
    selected.mkdir(exist_ok=True)
    selection = {}
    for arm in config['arms']:
        for cell in CELLS:
            chosen = dict(min((r for r in records
                               if r['arm'] == arm['id'] and r['cell'] == cell),
                              key=lambda r: r['final_training_loss_MPa']))
            model_id = f"{arm['id']}--{cell}"
            chosen['model_id'] = model_id
            chosen['potential_weights'] = f'selected/{model_id}-potentials.npz'
            weights, norm = load_weights(out/chosen['weights'])
            save_weights(out/chosen['potential_weights'], weights, norm)
            chosen['weights_sha256'] = _sha256(out/chosen['weights'])
            chosen['potential_weights_sha256'] = _sha256(out/chosen['potential_weights'])
            selection[model_id] = chosen

    result = dict(
        runs=records, selection=selection, selection_frozen=True,
        rule=tc['selection'], initialization_seeds=tc['initialization_seeds'],
        selected_models=len(selection), study_models=len(records),
        reused_models=sum(r['reused'] for r in records),
        new_models=sum(not r['reused'] for r in records),
        total_optimizer_updates=sum(a['optimizer_updates'] for a in arm_compute),
        elapsed_s=sum(a['elapsed_s'] for a in arm_compute),
        history_gradient_evaluations=sum(a['history_gradient_evaluations'] for a in arm_compute),
        arm_compute=arm_compute, normalization_sources=norm_sources,
        compute_scope='New training only; reused models are excluded. B is not equal compute.',
        selection_scope='Final training loss only, separately within each arm and noise cell; no final evaluation inputs are read.')
    write_json(out/'training.json', result)
    print('Completed 50 new models; 10 archived models reused; 12 training-only selections frozen.',
          flush=True)
    return result
