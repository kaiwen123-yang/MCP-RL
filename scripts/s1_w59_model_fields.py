"""Common physical-unit W59 state and supervision layout at the 20 ms force clock."""
from collections import OrderedDict
import numpy as np

DT = .020
MASS = 15.206408
PHASES = ('initial_stand','outbound','turn','return','terminal_stand','complete','far_hold','home_hold','turn_home')
BG = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)


def layout(fields):
    out = OrderedDict()
    offset = 0
    for name, size in fields:
        out[name] = slice(offset, offset + size)
        offset += size
    return out, offset


STATE, NS = layout([
    ('qpos',19), ('qvel',18), ('feet',12), ('foot_velocity',12),
    ('measured_force',12), ('held_command',12), ('force_reference',12),
    ('force_solved',12), ('confirmed',4), ('bearing',4), ('transfer',1),
    ('incoming',4), ('request_age',1), ('nominal_due_in',1), ('active',1),
    ('target_xy',8), ('target_in',4), ('target_valid',4), ('phase',9),
    ('command',3), ('goal',3), ('hold',1), ('progress',1), ('laps',1),
    ('time',1), ('version',4), ('memory',8*4), ('memory_valid',8),
    ('request_valid',1), ('control_valid',1), ('route_valid',1)
])
INTERVAL, NI = layout([
    ('touch',4), ('rebound',4), ('bearing_on',4), ('load_mean',4),
    ('load_peak',4), ('sink_peak',4), ('slip_m',4), ('work_J',1),
    ('z_min',1), ('tilt_max_rad',1), ('torque_peak_Nm',1), ('request',1),
    ('handover',1), ('handover_at',1), ('hard_failure',1), ('truncated',1),
    ('complete',1), ('hold_broken',1), ('progress_delta',1),
    ('indentation_sq_s',1), ('old_slip_sq_s',1), ('cap_change',1), ('task_reward',1)
])
ACTION_NAMES = ('due_offset_s','transfer_ramp_s','rho')
NOMINAL = np.array([0,.05,0], dtype=np.float32)
# Each of 8 current-foot / committed-target queries stores loaded area fraction,
# unrelaxed released depth, released age and observed released area fraction.
MEMORY_FIELDS = ('active_fraction','released_depth_m','released_age_s','released_fraction')
CELL = .02
HALF = np.sqrt(.001)/2


def pack_state(**fields):
    out = np.zeros(NS, dtype=np.float32)
    for name, sl in STATE.items():
        if name not in fields:
            raise KeyError('missing W59 state field: ' + name)
        out[sl] = np.asarray(fields[name]).reshape(sl.stop-sl.start)
    return out
