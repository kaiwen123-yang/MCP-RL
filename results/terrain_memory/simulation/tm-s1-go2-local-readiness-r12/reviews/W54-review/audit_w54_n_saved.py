"""Pure saved-array audit of W54 N-r48 execution map; no simulator or solves.

Torque changes compare E to the same original torque/input-clipped raw command.
This separates E's correction from clipping already required by the actuator.
"""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np

P = argparse.ArgumentParser(description=__doc__)
P.add_argument('--originals', type=Path, required=True)
P.add_argument('--output', type=Path, required=True)
a = P.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
ORIGINAL = a.originals/'job-20261117T052531Z-w54-n-a-r48-eval-PANN_PPO_S5-s1-E30-attempt01'
OUT = a.output/'n-saved-execution-audit.json'


def rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if np.size(a) else None


def main():
    tr = np.load(ORIGINAL/'trajectory.npz', allow_pickle=False)
    rows = [json.loads(s) for s in (ORIGINAL/'w54-n-execution.jsonl').read_text().splitlines()]
    ts = np.array([x['time_s'] for x in rows])
    raw = np.array([x['raw_action'] for x in rows])
    applied = np.array([x['applied_action'] for x in rows])
    diag = np.array([x['diagnostic'] for x in rows])
    n = len(rows)
    assert np.allclose(ts, tr['time'][:n], atol=1e-10, rtol=0)
    q, v = tr['qpos'][:n, 7:], tr['qvel'][:n, 6:]
    actuators = list(ET.parse(ORIGINAL/'model/G1-project-go2.xml').getroot().find('actuator'))
    urange = np.array([[float(z) for z in a.get('ctrlrange').split()] for a in actuators])
    trange = np.array([[float(z) for z in a.get('forcerange').split()] for a in actuators])
    assert all(float(a.get('gear')) == 1. for a in actuators)
    lo = np.maximum(trange[:, 0], 60*(urange[:, 0]-q)-5*v)
    hi = np.minimum(trange[:, 1], 60*(urange[:, 1]-q)-5*v)
    tau0 = np.clip(60*(raw-q)-5*v, lo, hi)
    tauE = np.clip(60*(applied-q)-5*v, lo, hi)
    dtau = tauE-tau0
    gate_active = np.any(diag[:, :, 4] > 1e-10, axis=1)
    support = tr['actual_grf'][:n, :, 2] > 10
    support_joint = np.repeat(support, 3, axis=1)
    first_nonfoot = float(tr['time'][np.flatnonzero(tr['nonfoot_ground_contacts'] > 0)[0]])
    pre = (ts >= 5.864-1e-9) & (ts < first_nonfoot-1e-9) & gate_active
    support_changes = np.any((np.abs(dtau)>1e-8) & support_joint, axis=1)
    result = dict(
        source=str(ORIGINAL), scope='Saved arrays and XML only; no physics, Jacobian evaluation, transition FD or solver.',
        torque_comparison='PD torque from applied E command minus raw PD torque clipped to the same original torque/input intersection; unit gear verified.',
        first_nonfoot_contact_time_s=first_nonfoot,
        pre_nonfoot_active_ticks=int(pre.sum()),
        pre_nonfoot_active_ticks_modifying_a_current_support_leg=int((pre & support_changes).sum()),
        pre_nonfoot_torque_correction_all_joints_RMS_Nm=rms(dtau[pre]),
        pre_nonfoot_torque_correction_support_joints_RMS_Nm=rms(dtau[pre][support_joint[pre]]),
        pre_nonfoot_max_abs_support_joint_correction_Nm=float(np.max(np.abs(dtau[pre][support_joint[pre]]))),
        original_intersection_empty_count=int(np.count_nonzero(lo>hi)),
        examples=[],
    )
    # Model-vs-actual foot acceleration is diagnostic only: free articulated
    # acceleration deliberately omits contact. This does not isolate causality.
    a_actual = np.full((n, 4), np.nan)
    times = tr['time']
    vel = tr['feet_velocity'][:, :, 2]
    a_actual[1:n-1] = (vel[2:n]-vel[:n-2])/(times[2:n]-times[:n-2])[:, None]
    continuous_support = support.copy()
    continuous_support[1:] &= support[:-1]
    continuous_support[:-1] &= support[1:]
    quiet = continuous_support & (ts[:, None]>=5.864) & (ts[:, None]<first_nonfoot)
    quiet[0] = quiet[-1] = False
    result['continuous_support_pre_nonfoot'] = dict(
        foot_ticks=int(quiet.sum()),
        recorded_free_model_acceleration_RMS_m_s2=rms(diag[:, :, 6][quiet]),
        actual_central_velocity_difference_acceleration_RMS_m_s2=rms(a_actual[quiet]),
        model_minus_actual_RMS_m_s2=rms((diag[:, :, 6]-a_actual)[quiet]),
        limitation='Continuous >10 N support only checks neighboring 2 ms ticks; stored central differences are finite-time and contact dynamics are omitted by the recorded model by construction.',
    )
    for t in [5.896, 6.0, 6.38, 6.58, 6.9]:
        k=int(np.argmin(abs(ts-t)))
        result['examples'].append(dict(
            time_s=float(ts[k]), body_height_m=float(tr['qpos'][k,2]),
            actual_normal_forces_N=tr['actual_grf'][k,:,2].tolist(),
            support_gt10N=support[k].tolist(), gates=diag[k,:,4].tolist(),
            raw_clipped_torque_Nm=tau0[k].tolist(), E_command_torque_Nm=tauE[k].tolist(),
            correction_Nm=dtau[k].tolist(),
            raw_modeled_normal_acceleration_m_s2=diag[k,:,8].tolist(),
            E_modeled_normal_acceleration_m_s2=diag[k,:,6].tolist(),
            actual_central_normal_acceleration_m_s2=a_actual[k].tolist(),
            foot_gaps_m=diag[k,:,0].tolist(), foot_normal_velocity_m_s=diag[k,:,1].tolist(),
            raw_action=raw[k].tolist(), applied_action=applied[k].tolist(),
        ))
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['examples']}, ensure_ascii=False, indent=2))
    for x in result['examples']:
        print(json.dumps({k:v for k,v in x.items() if k not in ['raw_action','applied_action','raw_clipped_torque_Nm','E_command_torque_Nm']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
