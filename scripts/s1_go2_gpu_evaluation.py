"""Shared launch arguments and evidence-based GPU stage completion.

This module has no simulator imports: command construction and the distinction
between a child process ending and physical evaluation can be checked directly.
"""

def evaluation_command(root, contract, attempt, checkpoint, level, started_unix_s):
    return [str(root / 'work/envs/d57/bin/python'), '-u', 'scripts/run_s1_go2_r12.py',
            '--contract', str(contract), '--attempt', attempt, '--build', 'S5-r30',
            '--group', 'PANN_PPO_S5', '--seed', '2', '--checkpoint', str(checkpoint),
            '--level', level, '--workers', '2',
            '--evaluation-start-unix', str(started_unix_s)]


def physical_started(row):
    return (row.get('result', {}).get('simulation_s') or 0) > 0


def technical_failure(row):
    execution = row.get('execution', row)
    return (execution.get('exit_code', 0) != 0
            or row.get('result', {}).get('stage_status') == 'technical_interruption')


def summarize_completion(report):
    """Do not turn missing physical results into successful integration."""
    stages = report['stages']
    g4 = stages.get('G4', [])
    mjx = [row for row in g4 if row.get('backend') == 'mjx']
    integration_observed = any(physical_started(row)
                               and row.get('execution', {}).get('requests', 0) > 0
                               for row in mjx)
    if 'G3' in stages:
        stages['G3']['status'] = ('physical_execution_observed' if integration_observed
                                  else 'physical_integration_not_observed')
        stages['G3']['mjx_predictor_requests'] = sum(
            row.get('execution', {}).get('requests', 0) for row in mjx)
    failures = [dict(stage='G4', backend=row.get('backend'), level=row.get('level'),
                     attempt=row.get('attempt'), reason=row.get('result', {}).get('reason'))
                for row in g4 if technical_failure(row)]
    g5 = stages.get('G5', {})
    g5_result = g5.get('result', {})
    if g5 and (g5.get('exit_code', 0) != 0
               or g5_result.get('status', '').startswith('technical_interruption')):
        failures.append(dict(stage='G5', reason=g5_result.get('first_error')
                             or g5_result.get('status')))
    g5_evaluations = g5_result.get('evaluations', [])
    failures.extend(dict(stage='G5', level=row.get('level'), attempt=row.get('attempt'),
                         reason=row.get('result', {}).get('reason'))
                    for row in g5_evaluations if technical_failure(row))
    active_stages = ['G5'] if report.get('continuation', {}).get('resume_from') == 'G5' else ['G4', 'G5']
    active_backends = stages.get('G3', {}).get('requested_backends', ['native', 'mjx'])
    active_failures = [row for row in failures if row['stage'] in active_stages
                       and (row['stage'] != 'G4' or row.get('backend') in active_backends)]
    completion = dict(workflow_finished=True, technical_failures=failures,
                      active_stages=active_stages, active_technical_failures=active_failures,
                      active_g4_backends=active_backends if 'G4' in active_stages else [],
                      active_invocation_status='technical_interruption' if active_failures else 'completed',
                      g4_physical_evaluations_started=sum(map(physical_started, g4)),
                      g5_physical_evaluations_started=sum(map(physical_started, g5_evaluations)),
                      mjx_full_stack_execution_observed=integration_observed,
                      equivalent_backend=bool(report.get('equivalent_backend', False)),
                      closed_loop_successful_attempts=[row['attempt'] for row in
                          g4 + g5_evaluations if row.get('result', {}).get('success')])
    report['completion'] = completion
    return 'completed_with_technical_failures' if failures else 'completed_declared_gpu_stages'
