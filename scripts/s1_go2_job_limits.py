"""Explicit optional wall limits; None means completion by experimental goals.

The 2026-11-07 user instruction removes background completion deadlines. It
does not remove physical episode durations, numerical convergence checks,
sample/update targets, or per-request network timeouts. Historical records keep
their original finite values unless an explicit continuation adopts this policy.
"""
import time

GOAL_POLICY = 'sample_update_goal_no_wall_deadline'
AUTHORIZATION = 'User instruction 2026-11-07: no fixed background completion time'


def goal_based(contract):
    return contract.get('execution', {}).get('background_job', {}).get('wall_deadline_enabled') is False


def expired(deadline, now=None):
    return deadline is not None and (time.time() if now is None else now) >= deadline


def request_timeout(deadline, maximum):
    """A bounded individual I/O request, without imposing a task deadline."""
    return maximum if deadline is None else max(.001, min(maximum, deadline-time.time()))


def adopt_goal_policy(contract):
    background = contract.setdefault('execution', {}).setdefault('background_job', {})
    background.update(wall_deadline_enabled=False, completion_policy=GOAL_POLICY,
                      default_maximum_wall_s=None, authorization=AUTHORIZATION)
    evaluation = contract['control_and_evaluation']['post_training_evaluation']
    evaluation.update(maximum_wall_s=None, wall_deadline_enabled=False)


def remove_wall_deadline(record, now=None):
    """Retain the former clock in the record before an authorized continuation."""
    prior = record.get('deadline_unix_s')
    if prior is not None:
        record.setdefault('wall_limit_history', []).append(dict(
            previous_deadline_unix_s=prior,
            changed_unix_s=time.time() if now is None else now,
            authorization=AUTHORIZATION))
    record.update(deadline_unix_s=None, completion_policy=GOAL_POLICY,
                  wall_deadline_enabled=False)
