"""Pure response-packet arithmetic; no policy, training, or parameter updates.

A task covector is a local derivative, not a finite-change performance
certificate. Callers must preserve its policy/model/task/context version and
re-evaluate the actual suffix objective and prefix constraints after a change.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResponsePacket:
    response: np.ndarray
    task_value: float
    task_covector: np.ndarray
    constraint_values: np.ndarray
    constraint_covectors: np.ndarray
    context_version: str


def pullback(packet: ResponsePacket, response_jacobian: np.ndarray) -> dict:
    """Pull task/facet credit into whatever variables the supplied J describes.

    J has shape (response_dimension, variable_dimension). These variables are
    not automatically actor weights; the analytic witness uses one physical
    allocation coordinate, explicitly identified by its caller.
    """
    jac = np.asarray(response_jacobian, dtype=float)
    return {'task_gradient': np.asarray(packet.task_covector) @ jac,
            'constraint_gradients': np.asarray(packet.constraint_covectors) @ jac}


def linear_prediction(packet: ResponsePacket, response: np.ndarray) -> dict:
    delta = np.asarray(response, dtype=float)-np.asarray(packet.response)
    return {'task_value':float(packet.task_value+np.asarray(packet.task_covector)@delta),
            'constraint_values':np.asarray(packet.constraint_values)+np.asarray(packet.constraint_covectors)@delta}


def finite_discrepancy(packet: ResponsePacket, response: np.ndarray,
                       evaluated_task: float, evaluated_constraints: np.ndarray) -> dict:
    """Measure cache error without inventing an acceptance tolerance or claim."""
    linear = linear_prediction(packet,response)
    return {'cached_task_value':linear['task_value'],
            'evaluated_task_value':float(evaluated_task),
            'task_optimism':float(evaluated_task-linear['task_value']),
            'constraint_errors':np.asarray(evaluated_constraints)-linear['constraint_values'],
            'context_version':packet.context_version}
