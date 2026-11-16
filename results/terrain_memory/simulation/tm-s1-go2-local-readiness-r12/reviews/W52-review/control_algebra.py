"""Pure algebra on saved W52 call128 arrays; no MuJoCo calls or integration."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--originals", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
ROOT = args.originals
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
job = ROOT / "job-20261116T110800Z-w52-d2-eval-PANN_PPO_S5-s1-D8-attempt01"
row = next(json.loads(line) for line in (job / "w51-seeds.jsonl").read_text().splitlines()
           if json.loads(line)["call_id"] == 128)

def arrays(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}

B = arrays(job / row["objects"]["B"]["file"])
S = arrays(job / row["objects"]["S"]["file"])
raw = arrays(job / row["objects"]["IK"]["file"])["kinematics_payload"]
n = int(raw[3])
times = raw[5:5+n]
targets = raw[5+n:5+13*n].reshape(n, 4, 3)[:, [0, 2, 1, 3]]
ik = raw[5+13*n:5+25*n].reshape(n, 12)
delta = raw[5+25*n:5+37*n].reshape(n, 12)
C = arrays(job / "r33-candidates-call000128.npz")
D = arrays(ROOT / "job-20261116T110800Z-w52-d3-eval-PANN_PPO_S5-s1-D8-attempt01/r33-candidates-call000128.npz")
xb, xs = C["candidate000_accepted_states"][:n], C["candidate010_accepted_states"][:n]
ub, us = C["candidate000_accepted_actions"], C["candidate010_accepted_actions"]

def interp(data, grid, tt):
    flat = data.reshape(len(data), -1)
    out = np.stack([np.interp(tt, grid[:len(data)], flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(tt),) + data.shape[1:])

def state_diff(ref, x):
    # Go2 leading free joint; local quaternion difference as in differentiatePos.
    qa = ref[:, [4, 5, 6, 3]]
    qb = x[:, [4, 5, 6, 3]]
    angular = (Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).as_rotvec()
    same = np.all(ref[:, 3:7] == x[:, 3:7], axis=1) | np.all(ref[:, 3:7] == -x[:, 3:7], axis=1)
    angular[same] = 0
    return np.c_[x[:, :3]-ref[:, :3], angular, x[:, 7:19]-ref[:, 7:19], x[:, 19:]-ref[:, 19:]]

def evaluate(p, states):
    ref = interp(p["reference_states"], p["reference_times"], times)
    ref[:, 3:7] /= np.linalg.norm(ref[:, 3:7], axis=1)[:, None]
    K = interp(p["K"][:len(p["reference_actions"])], p["reference_times"], times)
    ff = interp(p["reference_actions"], p["reference_times"], times)
    dx = state_diff(ref, states)
    feedback = np.einsum("tij,tj->ti", K, dx) * p["semantics"][5]
    d = interp(p["delta_values"], p["delta_times"], times) if len(p["delta_times"]) else np.zeros_like(ff)
    return ff + feedback + d, K, dx, feedback

bb, K, dxb, fbb = evaluate(B, xb)
bs, _, dxs, fbs = evaluate(B, xs)
ss, _, _, fss = evaluate(S, xs)
bias = bb - xb[:, 7:19]
feedback_change = bs - bb
groups = {
    "base_position_orientation": slice(0, 6),
    "joint_position": slice(6, 18),
    "base_velocity": slice(18, 24),
    "joint_velocity": slice(24, 36),
}
terms = {name: np.einsum("tij,tj->ti", K[:, :, sl], (dxs-dxb)[:, sl]) * B["semantics"][5]
         for name, sl in groups.items()}
feet = C["candidate010_foot_rows"].reshape(-1, 29)[:, 1:].reshape(-1, 4, 7)
pd = 60 * (ss - xs[:, 7:19]) - 5 * xs[:, 25:37]
limits = np.tile([23.7, 23.7, 45.43], 4)
calf = [2, 5, 8, 11]
samples = []
for t in [2.14, 2.1885, 2.24, 2.3885, 2.4385, 2.488]:
    i = int(np.argmin(abs(times-t)))
    samples.append(dict(
        time_s=float(times[i]),
        target_clearance_mm=((targets[i, :, 2]-.022)*1000).tolist(),
        actual_native_clearance_mm=(feet[i, :, 3]*1000).tolist(),
        actual_native_Fz_N=(feet[i, :, 4]+feet[i, :, 5]).tolist(),
        calf_delta_IK_rad=delta[i, calf].tolist(),
        calf_baseline_support_offset_rad=bias[i, calf].tolist(),
        calf_feedback_response_to_S_rad=feedback_change[i, calf].tolist(),
        calf_actual_action_change_rad=(us[i]-ub[i])[calf].tolist(),
        calf_final_target_minus_IK_rad=(ss[i]-ik[i])[calf].tolist(),
        calf_requested_PD_Nm=pd[i, calf].tolist(),
        feedback_response_components_calf_rad={k:v[i, calf].tolist() for k,v in terms.items()},
        qIK_calf_rad=ik[i, calf].tolist(),
        qS_calf_rad=xs[i, np.array(calf)+7].tolist(),
    ))

result = dict(
    scope="Saved arrays only. No physics, FK, native solver, finite differences, QP rerun, or training.",
    source_call=128,
    source_time_s=float(times[0]),
    delta_identity_max_error_rad=float(np.max(abs(delta-(ik-xb[:, 7:19])))),
    retained_reference_states_equal=bool(np.array_equal(B["reference_states"], S["reference_states"])),
    retained_reference_actions_equal=bool(np.array_equal(B["reference_actions"], S["reference_actions"])),
    retained_K_equal=bool(np.array_equal(B["K"], S["K"])),
    retained_H_equal=bool(np.array_equal(B["H"], S["H"])),
    B_original_delta_count=int(len(B["delta_times"])),
    B_feedback_scale=float(B["semantics"][5]),
    S_feedback_scale=float(S["semantics"][5]),
    S_delta_equals_IK_increment=bool(np.array_equal(S["delta_values"], delta)),
    reconstructed_B_raw_vs_saved_final_action_max_error_rad=float(np.max(abs(bb-ub))),
    reconstructed_S_raw_vs_saved_final_action_max_error_rad=float(np.max(abs(ss-us))),
    decomposition_identity_max_error_rad=float(np.max(abs((us-ub)-delta-feedback_change))),
    native_S_PD_max_abs_Nm=np.max(abs(pd), axis=0).tolist(),
    native_S_PD_min_margin_to_limit_Nm=float(np.min(limits-abs(pd))),
    feedback_response_parts_identity_max_error_rad=float(np.max(abs(sum(terms.values())-feedback_change))),
    full_horizon_delta_max_abs_rad=float(np.max(abs(delta))),
    full_horizon_feedback_response_max_abs_rad=float(np.max(abs(feedback_change))),
    full_horizon_old_support_offset_max_abs_rad=float(np.max(abs(bias))),
    D2_D3_same_S_arrays={
        field:bool(np.array_equal(C["candidate010_"+field], D["candidate019_"+field]))
        for field in ["accepted_states","accepted_actions","reference_states","reference_actions","K","H"]
    },
    D2_D3_all_S_positive_arrays_identical=all(
        np.array_equal(C[f"candidate{11+i:03d}_"+field], D[f"candidate{20+i:03d}_"+field])
        for i in range(9) for field in ["accepted_states","accepted_actions","K","H"]
    ),
    D3_B_vs_S_family_max_abs_action_difference_rad=float(max(
        np.max(abs(D[f"candidate{10+i:03d}_accepted_actions"]-D[f"candidate{20+i:03d}_accepted_actions"]))
        for i in range(9)
    )),
    samples=samples,
)
(OUT / "control-algebra.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="samples"}, indent=2))
for sample in samples:
    if abs(sample["time_s"]-2.188)<1e-6 or abs(sample["time_s"]-2.438)<1e-6:
        print(json.dumps(sample, indent=2))
