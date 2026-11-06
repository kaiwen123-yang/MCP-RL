"""Pure algebra counterexamples for the W40 plan; no simulator or training.

Run from any directory with Python's standard library. The examples are not
claims about an unrecorded W39 Hessian. They disprove general identities used
by the proposed r29 formula and illustrate the replacement's exact meaning.
"""

import json


def matvec(h, x):
    return [sum(a * b for a, b in zip(row, x)) for row in h]


def objective(h, center, u):
    d = [a - b for a, b in zip(u, center)]
    return 0.5 * sum(a * b for a, b in zip(d, matvec(h, d)))


h = [[2.0, 1.0], [1.0, 2.0]]
center = [1.0, 0.0]
projected = [0.2, 0.4]
axis_clip = [0.2, 0.0]

# Original one-dimensional backward QP:
# min .5*d^2 - 2*d, -1 <= d <= 1 => k=1, active K_old=0.
# The withdrawn centered QP replaces its gradient by delta_x; its nominal
# multiplier is zero, and it releases the old active axis for delta_x > 0.
counterexamples = {
    "scope": "pure algebra only; no new physics, no W39 H reconstruction",
    "old_box_active_gain": {
        "H": 1.0,
        "Q_u": -2.0,
        "Q_ux": 1.0,
        "nominal_u": 0.0,
        "bounds": [-1.0, 1.0],
        "backward_k": 1.0,
        "backward_K": 0.0,
        "delta_x": 0.1,
        "alpha_half_old_action": 0.5,
        "alpha_half_withdrawn_QP_action": 0.4,
        "alpha_one_old_action": 1.0,
        "alpha_one_withdrawn_QP_action": 0.9,
        "original_QP_at_delta_x_action": 1.0,
    },
    "interpolation": {
        "H_nodes": [1.0, 4.0],
        "C_nodes": [-1.0, -8.0],
        "K_nodes": [1.0, 2.0],
        "weight": 0.5,
        "interpolated_K": 1.5,
        "minus_inverse_interpolated_H_times_interpolated_C": 1.8,
    },
    "metric_projection_changes_unsaturated_axis": {
        "H": h,
        "old_target_clipped_center": center,
        "new_PD_upper_first_axis": 0.2,
        "projection": projected,
        "axis_clip": axis_clip,
        "projection_objective": objective(h, center, projected),
        "axis_clip_objective": objective(h, center, axis_clip),
        "gradient_at_projection": matvec(h, [a-b for a,b in zip(projected, center)]),
    },
    "fixed_axis_elimination": {
        "H": h,
        "center": [0.0, 0.0],
        "fixed_axis_0": 1.0,
        "correct_free_axis_1": -0.5,
        "wrong_free_axis_if_H_FA_u_A_is_dropped": 0.0,
    },
}

print(json.dumps(counterexamples, ensure_ascii=False, indent=2))
