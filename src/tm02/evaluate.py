"""Full-history, RNN-free integration and the frozen 36-rollout transfer."""
import csv
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from .material import forces, load_weights, auxiliary, COMPONENTS
from .reference import newton, Reference, IntegrationFailure


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strain_residual(q, w, norm, e, qprev, dt):
    _, tau, tauhat = forces(w, norm, e, q, (q-qprev)/dt)
    return tauhat-tau


def stress_residual(x, w, norm, stress, qprev, dt):
    e, q = x[:6], x[6:]
    pred, tau, tauhat = forces(w, norm, e, q, (q-qprev)/dt)
    return jnp.concatenate((pred-stress, tauhat-tau))


STRAIN_FUN = jax.jit(strain_residual)
STRAIN_JAC = jax.jit(jax.jacfwd(strain_residual))
STRESS_FUN = jax.jit(stress_residual)
STRESS_JAC = jax.jit(jax.jacfwd(stress_residual))
FORCES = jax.jit(forces)


def integrate(w, norm, time, command, settings, stress_control=False):
    # Deployment signature has no evaluation stress labels or reference q.
    n = len(time)
    e, q, stress = (np.full((n, 6), np.nan) for _ in range(3))
    e[0] = q[0] = stress[0] = 0.
    residual, iterations = np.full(n, np.nan), np.full(n, -1, int)
    residual[0], iterations[0] = 0., 0
    failure = None
    for i in range(1, n):
        dt = time[i]-time[i-1]
        fun, jac = (STRESS_FUN, STRESS_JAC) if stress_control else (STRAIN_FUN, STRAIN_JAC)
        initial = np.r_[e[i-1], q[i-1]] if stress_control else q[i-1]
        args = (w, norm, command[i], q[i-1], dt)
        try:
            x, residual[i], iterations[i] = newton(
                lambda z: fun(z, *args), lambda z: jac(z, *args), initial, settings, i)
        except (IntegrationFailure, np.linalg.LinAlgError) as error:
            failure = dict(step=i, time_s=float(time[i]), error=str(error))
            break
        e[i], q[i] = (x[:6], x[6:]) if stress_control else (command[i], x)
        stress[i] = np.asarray(FORCES(w, norm, e[i], q[i], (q[i]-q[i-1])/dt)[0])
    return dict(time=time, strain=e, q=q, stress=stress,
                residual=residual, newton_iterations=iterations), failure


def evaluate(config, out, compact):
    from .figures import reproduction_figures
    begin = time.monotonic()
    training = json.loads((out/'training.json').read_text())
    observations = np.load(out/'observations.npz')
    # Integration consumes only this independently generated strain/time path.
    time_eval = observations['evaluation_time']
    strain_eval = observations['evaluation_strain']
    predictiondir = out/'predictions'
    predictiondir.mkdir(exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    records, predictions = [], {}
    for row in training['runs']:
        condition, seed = row['condition'], row['seed']
        stem = f'{condition}-seed-{seed:02d}'
        w, norm = load_weights(out/row['weights'])
        result, failure = integrate(w, norm, time_eval, strain_eval, config['numerics'])
        np.savez_compressed(predictiondir/f'{stem}.npz', **result)
        # Label access begins only AFTER the full free prediction has finished.
        with np.load(out/'reference.npz') as truth:
            target = truth['evaluation_stress']
        record = dict(row, integration_complete=failure is None,
                      failure=failure, stress_scale_MPa=float(norm['s_s']),
                      completed_increments=int(np.isfinite(result['stress'][1:, 0]).sum()),
                      MAE_MPa=None, RMSE_MPa=None, normalized_RMSE=None)
        if failure is None:
            error = result['stress'][1:]-target[1:]
            mae, rmse = np.mean(np.abs(error), axis=0), np.sqrt(np.mean(error**2, axis=0))
            record.update(MAE_MPa=float(np.mean(np.abs(error))),
                          RMSE_MPa=float(np.sqrt(np.mean(error**2))),
                          normalized_RMSE=float(np.sqrt(np.mean(error**2))/norm['s_s']),
                          per_component_MAE_MPa=mae.tolist(), per_component_RMSE_MPa=rmse.tolist(),
                          max_residual_MPa=float(np.max(result['residual'])),
                          max_Newton_iterations=int(np.max(result['newton_iterations'])))
        records.append(record)
        predictions[stem] = result
        write_json(predictiondir/f'{stem}.json', record)
        print(json.dumps({k: record[k] for k in ('condition','seed','integration_complete','MAE_MPa','RMSE_MPa','failure')}), flush=True)
    flatrows = []
    for row in records:
        flat = {k: row[k] for k in ('condition','seed','iterations','final_training_loss_MPa',
                                   'final_stress_MAE_MPa','final_biot_MAE_MPa','integration_complete',
                                   'completed_increments','MAE_MPa','RMSE_MPa','normalized_RMSE','stress_scale_MPa')}
        flat['failure'] = json.dumps(row['failure']) if row['failure'] else ''
        for i, component in enumerate(COMPONENTS):
            for metric in ('MAE', 'RMSE'):
                flat[f'{metric}_{component}_MPa'] = row[f'per_component_{metric}_MPa'][i] if row['integration_complete'] else None
        flatrows.append(flat)
    write_csv(compact/'all_initializations.csv', flatrows)
    selected = {}
    for condition, choice in training['selection'].items():
        selected[condition] = next(r for r in records if r['condition']==condition and r['seed']==choice['seed'])
        stem = f"{condition}-seed-{choice['seed']:02d}"
        pred = predictions[stem]
        curves = []
        with np.load(out/'reference.npz') as truth:
            target = truth['evaluation_stress']
        for i, t in enumerate(time_eval):
            row = dict(time_s=t)
            for k, c in enumerate(COMPONENTS):
                row[f'strain_{c}'] = strain_eval[i,k]
                row[f'reference_stress_{c}_MPa'] = target[i,k]
                row[f'predicted_stress_{c}_MPa'] = pred['stress'][i,k]
            curves.append(row)
        write_csv(compact/f'{condition}-selected-full-path.csv', curves)
    choice = training['selection']['ideal']
    w, norm = load_weights(out/choice['weights'], include_rnn=True)
    dt = np.diff(observations['train_time'])
    q, rate = jax.jit(auxiliary)(w['rnn'], norm, observations['train_strain'][1:],
                                observations['ideal_stress'], dt)
    # This diagnostic is AFTER selection, and the true q is never a training label.
    with np.load(out/'reference.npz') as truth:
        true_q = truth['train_q']
    qaux = np.vstack((np.zeros(6), np.asarray(q)))
    np.savez_compressed(out/'auxiliary-diagnostic.npz', time=observations['train_time'],
                        reference_q=true_q, auxiliary_q=qaux, strain=observations['train_strain'])
    write_csv(compact/'ideal-auxiliary-full-path.csv',
              [dict(time_s=float(t), **{f'{name}_q_{c}': float(a[i,k])
                    for name,a in [('reference',true_q),('auxiliary',qaux)]
                    for k,c in enumerate(COMPONENTS)}) for i,t in enumerate(observations['train_time'])])
    index = dict(kind='TM02 v1 selected RNN method reimplementation',
                 paper='https://arxiv.org/html/2401.14270v1',
                 run=out.name, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                 training_runs=len(records), iterations_each=config['training']['iterations_per_initialization'],
                 total_optimizer_updates=training['total_optimizer_updates'],
                 free_prediction_complete=sum(r['integration_complete'] for r in records),
                 selected=selected, selection_rule=training['rule'],
                 evaluation_increments=len(time_eval)-1, evaluation_dt_s=config['data']['evaluation']['dt_s'],
                 evaluation_reference='noise-free for both conditions',
                 deployment_inputs=['strain', 'time increments', 'own previous internal state'],
                 auxiliary_RNN_in_deployment=False, reference_internal_variables_in_deployment=False,
                 training_elapsed_s=training['elapsed_s'], evaluation_elapsed_s=time.monotonic()-begin,
                 raw_artifacts=str(out), numerics=config['numerics'],
                 implementation_choices=config['implementation_choices'],
                 reference_interpretation=config['reference_material']['constitutive_interpretation'],
                 scope='Figures 8-10 RNN portions only; fixed project paths and seeds, not recovered author arrays')
    write_json(compact/'index.json', index)
    reproduction_figures(out, compact, records, selected, predictions)
    print('Full original-method evaluation and figures saved.', flush=True)


def history(config, preload, candidate, delay):
    dt = config['dt_s']
    phases = [preload, candidate,
              dict(duration_s=delay, points_t_s_sigma11_MPa=[[0,0],[delay,0]]),
              config['common_reload_probe']]
    times, loads, labels = [0.], [0.], [0]
    for p, phase in enumerate(phases):
        local = np.arange(1, round(phase['duration_s']/dt)+1)*dt
        knots = np.asarray(phase['points_t_s_sigma11_MPa'])
        offset = times[-1]
        times.extend((offset+local).tolist())
        loads.extend(np.interp(local, knots[:,0], knots[:,1]).tolist())
        labels.extend([p]*len(local))
    command = np.zeros((len(times), 6))
    command[:,0] = loads
    return np.array(times), command, np.array(labels)


def revisit(config, protocol, source, root, run):
    from .figures import revisit_figures
    original = json.loads((root/config['outputs']['compact_root']/run/'index.json').read_text())
    training = json.loads((source/'training.json').read_text())
    out = root/protocol['outputs']['large_root']/run
    compact = root/protocol['outputs']['compact_root']/run
    out.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    write_json(out/'run.json', dict(config=protocol, source_run=str(source),
                                  selected_weights=training['selection'], numerics=config['numerics'],
                                  command='scripts/run_tm02_r2.py --phase revisit --run '+run,
                                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
    models = {condition: load_weights(source/row['weights']) for condition,row in training['selection'].items()}
    reference = Reference(config['reference_material']['parameters'])
    records, rollouts = [], {}
    for preload in protocol['preloading_histories']:
        for delay in protocol['recovery_delays_s']:
            for candidate in protocol['current_action_candidates']:
                history_id = f"{preload['id']}-delay-{delay:g}-{candidate['id']}"
                t, cmd, phase = history(protocol, preload, candidate, delay)
                truth = reference.path(t, cmd, config['numerics'], stress_control=True)
                for model in ('reference', 'ideal', 'noisy'):
                    if model == 'reference':
                        result, failure = truth, None
                    else:
                        w, norm = models[model]
                        result, failure = integrate(w, norm, t, cmd, config['numerics'], stress_control=True)
                    np.savez_compressed(out/f'{history_id}-{model}.npz', **result,
                                        prescribed_stress=cmd, phase=phase)
                    rollouts[(history_id,model)] = result
                    reload_start = preload['duration_s']+candidate['duration_s']+delay
                    probe = t >= reload_start-1e-10
                    row = dict(history=history_id, preload=preload['id'], delay_s=delay,
                               candidate=candidate['id'], model=model, integration_complete=failure is None,
                               failure=failure, peak_compression=None, integrated_compression_s=None,
                               reload_strain_MAE=None, peak_compression_error=None, integrated_compression_error_s=None)
                    if failure is None:
                        compression = np.maximum(0,-result['strain'][probe,0])
                        true_compression = np.maximum(0,-truth['strain'][probe,0])
                        integral = float(np.trapezoid(compression, t[probe]))
                        true_integral = float(np.trapezoid(true_compression, t[probe]))
                        row.update(peak_compression=float(np.max(compression)), integrated_compression_s=integral,
                                   reload_strain_MAE=float(np.mean(np.abs(result['strain'][probe]-truth['strain'][probe]))),
                                   reload_epsilon11_MAE=float(np.mean(np.abs(result['strain'][probe,0]-truth['strain'][probe,0]))),
                                   peak_compression_error=float(np.max(compression)-np.max(true_compression)),
                                   integrated_compression_error_s=integral-true_integral,
                                   max_residual_MPa=float(np.max(result['residual'])))
                    records.append(row)
                    print(json.dumps(row), flush=True)
    tol = protocol['metrics']['ranking_tie_tolerance_absolute_strain']
    pairs = []
    def rank(difference):
        return 'tie' if abs(difference) <= tol else ('early' if difference < 0 else 'late')
    for preload in protocol['preloading_histories']:
        for delay in protocol['recovery_delays_s']:
            relevant = {(r['model'],r['candidate']):r for r in records if r['preload']==preload['id'] and r['delay_s']==delay}
            refdiff = relevant['reference','early']['peak_compression']-relevant['reference','late']['peak_compression']
            pair = dict(preload=preload['id'], delay_s=delay, reference_difference=refdiff, reference_rank=rank(refdiff))
            for model in ('ideal','noisy'):
                early, late = relevant[model,'early'], relevant[model,'late']
                if early['integration_complete'] and late['integration_complete']:
                    diff = early['peak_compression']-late['peak_compression']
                    pair.update({f'{model}_difference':diff, f'{model}_rank':rank(diff),
                                 f'{model}_rank_matches':rank(diff)==rank(refdiff)})
                else:
                    pair.update({f'{model}_difference':None, f'{model}_rank':'integration_failure',
                                 f'{model}_rank_matches':False})
            pairs.append(pair)
    write_csv(compact/'candidate_ranking.csv', pairs)
    # Union columns includes failed rollouts without dropping them.
    fields = list(dict.fromkeys(k for row in records for k in row))
    write_csv(compact/'all_rollouts.csv', [{k:r.get(k) for k in fields} for r in records])
    summary = dict(kind='Project material revisit transfer; no robot-control benefit claim',
                   complete_histories=12, material_rollouts=len(records),
                   successful_rollouts=sum(r['integration_complete'] for r in records), additional_training_runs=0,
                   selected=training['selection'], pairs=pairs,
                   rank_matches={m:sum(p[f'{m}_rank_matches'] for p in pairs) for m in ('ideal','noisy')},
                   ranking_tie_tolerance=tol, failures=[r for r in records if not r['integration_complete']],
                   raw_artifacts=str(out), original_method_index=str(root/config['outputs']['compact_root']/run/'index.json'))
    write_json(compact/'index.json', summary)
    revisit_figures(protocol, compact, rollouts, pairs)
    print('All 36 prescribed full-history material rollouts and six candidate pairs saved.', flush=True)
