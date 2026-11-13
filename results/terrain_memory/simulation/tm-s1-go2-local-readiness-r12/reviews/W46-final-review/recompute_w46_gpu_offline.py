#!/usr/bin/env python3
"""Read existing W46 sources only; no network, physics, GPU or native library calls.

The original paired GPU/native trajectory arrays are not available in this audit.
Values copied from committed compact reports are explicitly labelled as reported.
Only already-present policy/input files are independently hash-checked and reduced.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PREFIX = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
COMMIT = "15ee620b8a909246a12baf999c136c2be299655b"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ma(a):
    return float(np.max(np.abs(a)))


def read_policy(path):
    z = np.frombuffer(path.read_bytes(), dtype="<f8")
    assert np.isfinite(z).all(), path
    h = z[:16]
    T, ns, nu, nx, nr = map(int, h[3:8])
    i, d = 16, {}
    for name, shape in [
        ("states", (T, ns)), ("actions", (T - 1, nu)),
        ("times", (T,)), ("residual", (T, nr)),
        ("K", (T, nu, nx)), ("H", (T, nu, nu)),
        ("base_Qu", (T, nu)), ("base_Qxu", (T, nu, nx)),
        ("base_active", (T, nu)),
    ]:
        n = int(np.prod(shape))
        d[name] = z[i:i+n].reshape(shape)
        i += n
    assert i == len(z), (path, i, len(z))
    d["scale"] = float(h[12])
    d["representation"] = int(h[13])
    assert d["representation"] == 1
    return d


def scalar_feedback(current, reference, ff, K, scale):
    """CPU FP64 transcription of W46 Warp kernel; not a GPU execution check."""
    r, x = reference, current
    qw = r[3]*x[3] + r[4]*x[4] + r[5]*x[5] + r[6]*x[6]
    qx = r[3]*x[4] - r[4]*x[3] - r[5]*x[6] + r[6]*x[5]
    qy = r[3]*x[5] + r[4]*x[6] - r[5]*x[3] - r[6]*x[4]
    qz = r[3]*x[6] - r[4]*x[5] + r[5]*x[4] - r[6]*x[3]
    norm = np.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    sn = np.sqrt(qx*qx+qy*qy+qz*qz)
    angle = np.float64(2.)*np.arctan2(sn, qw)
    if angle > np.float64(np.pi):
        angle -= np.float64(2*np.pi)
    factor = angle/sn if sn > np.float64(1e-15) else np.float64(2.)
    dx = np.r_[x[:3]-r[:3], [qx*factor, qy*factor, qz*factor],
               x[7:19]-r[7:19], x[19:]-r[19:]]
    values = []
    for a in range(12):
        val = np.float64(0.)
        for j in range(36):
            val += K[a,j] * dx[j]
        values.append(ff[a]+scale*val)
    return np.asarray(values)


def perturb_state(state, delta, sign, quatmul):
    q = state["qpos"].copy()
    q[:3] += sign*delta[:3]
    q[7:19] += sign*delta[6:18]
    v = sign*delta[3:6]
    angle = np.linalg.norm(v)
    dq = np.r_[np.cos(angle/2), v*np.sin(angle/2)/angle] if angle > 1e-15 else np.r_[1., v/2]
    q[3:7] = quatmul(q[3:7], dq)
    q[3:7] /= np.linalg.norm(q[3:7])
    return np.r_[q, state["qvel"]+sign*delta[18:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    locations = ap.add_mutually_exclusive_group(required=True)
    locations.add_argument("--existing", type=Path,
                           help="Existing audit layout with a policies/ directory; no copying required")
    locations.add_argument("--policy-root", type=Path,
                           help="Existing server w46-mechanisms directory containing A148/A153/B155/B158")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    policy_root = (args.policy_root if args.policy_root is not None
                   else args.existing / "policies").resolve()
    small = repo/PREFIX/"w46"
    sources = json.loads((small/"delivery/mechanism/sources.json").read_text())
    manifest = {x["source_path"]: x for x in sources}
    spec = importlib.util.spec_from_file_location("policy_arrays", repo/PREFIX/"reviews/W45-final-review/policy_arrays.py")
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    verified, missing_perturbations = [], []
    required = ("input.json", "input-state.npz", "original-baseline.bin", "original-policy.bin")
    for label in ("A148", "A153", "B155", "B158"):
        for name in (*required, "perturbation.npz"):
            p = policy_root / label / name
            if not p.is_file():
                if name == "perturbation.npz":
                    missing_perturbations.append(label)
                    continue
                raise FileNotFoundError(f"Required existing input absent: {p}; no network fallback")
            rel = str(p.relative_to(policy_root))
            key = str(Path("outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46-mechanisms")/rel)
            row = manifest[key]
            b = p.read_bytes()
            assert len(b) == row["size"] and sha(b) == row["sha256"], p
            verified.append(dict(row, existing_relative_path=rel,
                                 independent_existing_file_size_SHA256_verified=True))

    resolved_path = repo/PREFIX/"job-20261112T133729Z-w46-a-eval-PANN_PPO_S5-s1-E30-attempt01-resolved.json"
    resolved = json.loads(resolved_path.read_text())
    bounds = np.asarray(resolved["actuator_ctrlrange"])
    forces = np.asarray(resolved["actuator_forcerange"])
    assert np.all(np.asarray(resolved["actuator_gainprm"])[:,0] == 60)
    assert np.all(np.asarray(resolved["actuator_biasprm"])[:,1:3] == [-60,-5])
    summary, initial_rows, duplicates, models = [], [], [], []
    for label in ["A148", "A153", "B155", "B158"]:
        folder = policy_root/label
        f = json.loads((folder/"input.json").read_text())
        models.append(f["original_model_sha256"])
        with np.load(folder/"input-state.npz") as z:
            state = {k:z[k].copy() for k in z.files}
        perturbation_path = folder/"perturbation.npz"
        delta = np.load(perturbation_path)["delta"] if perturbation_path.is_file() else np.zeros(36)
        signs = [0,1,-1] if perturbation_path.is_file() else [0]
        ps = {}
        for name in ["original-baseline.bin", "original-policy.bin"]:
            p = ps[name] = read_policy(folder/name)
            effective = b"".join(p[k].tobytes() for k in ["states","actions","times","K","H"])+np.float64(p["scale"]).tobytes()
            summary.append(dict(label=label,policy=name,sha256=sha((folder/name).read_bytes()),
                                effective_policy_sha256=sha(effective),scale=p["scale"],
                                representation=p["representation"],array_shapes={k:list(p[k].shape) for k in ["states","actions","times","K","H"]}))
            for sign in signs:
                x = perturb_state(state, delta, sign, pa.quatmul)
                t = float(state["time"])
                ff, fb, dx, K = pa.raw_policy(t,x,p["times"],p["states"],p["actions"],p["K"],p["scale"])
                reference = pa.interp(t,p["times"],p["states"])
                reference[3:7] /= np.linalg.norm(reference[3:7])
                raw = ff+fb
                scalar = scalar_feedback(x,reference,ff,K,p["scale"])
                center = np.clip(raw,bounds[:,0],bounds[:,1])
                cast_x = x.astype(np.float32).astype(np.float64)
                ff_cast,fb_cast,_,_ = pa.raw_policy(t,cast_x,p["times"],p["states"],p["actions"],p["K"],p["scale"])
                lo = np.maximum(bounds[:,0],x[7:19]+(forces[:,0]+5*x[25:37])/60)
                hi = np.minimum(bounds[:,1],x[7:19]+(forces[:,1]+5*x[25:37])/60)
                margin = float(np.min(np.r_[center-lo,hi-center]))
                initial_rows.append(dict(label=label,policy=name,sign=sign,time_s=t,
                    raw_center_CPU_vector_vs_scalar_transcription_max_error_rad=ma(raw-scalar),
                    raw_center_max_abs_rad=ma(raw),raw_feedback_max_abs_rad=ma(fb),
                    raw_center=raw.tolist(),ctrl_clipped_center=center.tolist(),
                    ctrl_clipping_max_change_rad=ma(raw-center),
                    minimum_center_margin_to_R29_PD_and_ctrl_bounds_rad=margin,
                    all_center_components_strictly_interior=bool(margin>1e-12),
                    predicted_pre_forceclip_PD_max_abs_Nm=ma(60*(center-x[7:19])-5*x[25:37]),
                    FP32_initial_state_cast_max_component_error=ma(cast_x-x),
                    isolated_initial_state_cast_raw_center_difference_rad=ma((ff_cast+fb_cast)-raw),
                    initial_center_FP32_control_cast_error_rad=ma(center.astype(np.float32).astype(np.float64)-center),
                    scale_times_K_max_row_L1=float(np.max(np.sum(np.abs(p["scale"]*K),axis=1))),
                    available_perturbation=perturbation_path.is_file()))
        same = {k:bool(np.array_equal(ps["original-baseline.bin"][k],ps["original-policy.bin"][k])) for k in ["states","actions","times","K","H","residual","base_Qu","base_Qxu","base_active"]}
        same["scale"] = ps["original-baseline.bin"]["scale"]==ps["original-policy.bin"]["scale"]
        duplicates.append(dict(label=label,arrays_equal=same,
             effective_policy_equal=all(same[k] for k in ["states","actions","times","K","H","scale"])))

    g = json.loads((small/"gpu.json").read_text())
    phase = json.loads((small/"phase.json").read_text())
    timing = g["timing"]
    timing_sum = sum(timing.values())
    counted = dict(scenarios=4*2*3,holds=175,physics_substeps_per_hold=10,
        GPU_world_steps=175*24*10,native_world_steps=175*24*10,
        GPU_warmup_world_steps_reported=48,
        GPU_executed_warmup_world_steps_from_CUDA_capture_semantics=24,
        GPU_capture_only_world_steps_not_executed_during_capture=24,
        first20ms_is_included_in_total=True,
        feedback_kernel_launches=1+2*175,
        host_HQP_projection_calls_per_hold=2*2*24,
        host_HQP_projection_calls_in_175holds=175*2*2*24,
        host_HQP_projection_calls_initial_and_recenter=2*24+2*24,
        total_host_HQP_projection_calls=175*2*2*24+4*24,
        native_hold_calls=175*24)
    corrected_first = [dict(action_decision_after_origin_s=(int(k)-1)*.002,
        held_interval_end_after_origin_s=int(k)*.002,
        position_max_mixed_qpos_units_reported=v["position_max"],
        velocity_max_mixed_qvel_units_reported=v["velocity_max"],
        held_action_difference_rad_reported=v["action_difference_rad"])
        for k,v in g["first_divergence"].items()]
    jobs = []
    job_paths = []
    for stream,seed in [("a",1),("b",2)]:
        jp = repo/PREFIX/f"job-20261112T133729Z-w46-{stream}-eval-PANN_PPO_S5-s{seed}-E30-attempt01.json"
        job_paths.append(jp)
        j = json.loads(jp.read_text())
        jt = j["timing"]
        jobs.append(dict(stream=stream,wall_s_reported=j["wall_s"],simulation_s_reported=j["simulation_s"],
            wall_per_simulation_second=j["wall_s"]/j["simulation_s"],
            planning_percent_of_wall=100*jt["planning"]["total_s"]/j["wall_s"],
            startup_plus_planning_percent_of_wall=100*(jt["planning"]["total_s"]+jt["startup_planning"]["total_s"])/j["wall_s"],
            timing_reported=jt))
    starts=[v["stages"][0]["started_unix_s"] for v in phase["stream_results"].values()]
    source_paths=[repo/"AGENTS.md",repo/"docs/S1_W46_STABILITY_PLAN.md",
        repo/"scripts/s1_go2_w46_gpu.py",repo/"scripts/s1_go2_w46_native_bridge.cc",
        repo/"scripts/s1_go2_w41_warp_backend.py",repo/"scripts/compare_s1_go2_w46_saved.py",
        repo/PREFIX/"reviews/W45-final-review/policy_arrays.py",small/"gpu.json",small/"phase.json",
        small/"delivery/mechanism/sources.json",resolved_path,
        repo/"experiments/terrain_memory/s1-go2-r11-S5-r29.patch",
        repo/"experiments/terrain_memory/s1-go2-r11-S5-r29-qp1.patch",
        repo/"experiments/terrain_memory/s1-go2-r11-S5-r29-qp2.patch",
        repo/"experiments/terrain_memory/s1-go2-r11-S5-r33.patch"]+job_paths
    source_hashes = []
    for p in source_paths:
        relative = str(p.relative_to(repo))
        # AGENTS is contextual documentation, now intentionally updated by this
        # review. Pin its provenance to the reviewed result commit. All actual
        # computation inputs below retain their explicit working-tree hashes.
        pinned = relative == "AGENTS.md"
        raw = (subprocess.check_output(["git", "-C", str(repo), "show", f"{COMMIT}:{relative}"])
               if pinned else p.read_bytes())
        source_hashes.append(dict(path=relative,size=len(raw),sha256=sha(raw),
                                 read_from_ref=COMMIT if pinned else "working_tree"))
    report=dict(source_commit=COMMIT,method="Existing files only; pure NumPy CPU arithmetic, source inspection and compact-report calculations; no network, physics, new GPU execution or native-library execution.",
        independent_original_files_verified=len(verified),independent_original_bytes_verified=sum(x["size"] for x in verified),
        source_provenance=source_hashes,original_file_provenance=verified,
        GPU_paired_arrays_independently_recomputed=False,
        GPU_raw_array_access="This script never reads paired trajectory/action arrays and has no network fallback. The published audit could not obtain those arrays after automatic-review refusals.",
        GPU_policy_scope="W45 saved original-baseline.bin and original-policy.bin; not W46/r35 constant/load newly selected policies.",
        policy_arrays=summary,policy_pair_comparisons=duplicates,
        unique_effective_policy_count=len({(x["label"],x["effective_policy_sha256"]) for x in summary}),
        distinct_declared_scenarios_if_effective_policies_deduplicated=3*len({(x["label"],x["effective_policy_sha256"]) for x in summary}),
        independent_initial_raw_center_recomputation=dict(count=len(initial_rows),
            missing_perturbation_labels=missing_perturbations,
            missing_initial_scenarios=4*len(missing_perturbations),
            does_not_execute_GPU=True,does_not_execute_HQP=True,
            all_available_centers_strictly_interior=all(x["all_center_components_strictly_interior"] for x in initial_rows),
            maximum_CPU_vector_vs_scalar_transcription_error_rad=max(x["raw_center_CPU_vector_vs_scalar_transcription_max_error_rad"] for x in initial_rows),
            minimum_PD_and_ctrl_bound_margin_rad=min(x["minimum_center_margin_to_R29_PD_and_ctrl_bounds_rad"] for x in initial_rows),
            rows=initial_rows),
        same_original_model_hash_in_all_four_inputs=len(set(models))==1,
        source_counted_work=counted,
        published_GPU_values_not_raw_recomputed={k:g[k] for k in ["status","same_state_action","short20ms","final_state_component_max_error","native_bridge_identity_check","full_native_cost_implemented_on_gpu","equivalent_native_ranking"]},
        timing_reported=dict(wall_s=g["wall_s"],action_setup_s=g["action_setup_s"],compile_capture_s=g["compile_capture_s"],**timing),
        timing_arithmetic=dict(loop_components_sum_s=timing_sum,
            action_mapping_and_host_projection_average_ms_per_24world_hold=timing["action_mapping_and_host_projection_s"]*1000/175,
            device_advance_average_ms_per_24world_hold=timing["device_advance_wall_s"]*1000/175,
            native_confirmation_average_ms_per_24world_hold=timing["native_confirmation_s"]*1000/175,
            total_loop_average_ms_per_24world_hold=timing_sum*1000/175,
            action_mapping_plus_native_confirmation_share_of_loop_percent=100*(timing["action_mapping_and_host_projection_s"]+timing["native_confirmation_s"])/timing_sum,
            setup_and_compile_share_of_wall_percent=100*(g["action_setup_s"]+g["compile_capture_s"])/g["wall_s"],
            remainder_wall_s=g["wall_s"]-g["action_setup_s"]-g["compile_capture_s"]-timing_sum,
            GPU_end_before_earliest_original_task_start_s=min(starts)-g["ended_unix_s"]),
        first_ten_holds_corrected_timestamps_reported_numbers=corrected_first,
        CPU_original_tasks=jobs,actual_resources_reported=phase["actual_resources"],
        source_findings=[
            "raw_center_max_error_rad is computed after bridge.project mutates contiguous raw/cpu arrays in place, so it is not an independent pre-projection measurement.",
            "At k==9 the source only checks trunk boolean mismatch before continuing to350ms; no explanation/quantitative state-or-action-difference or body-fall gate is implemented at that decision.",
            "All reported20ms trunk values are zero; zero disagreement contains no positive trunk-violation coverage.",
            "Action at index k-1 generated the state at index k; published first_divergence uses one time label for both.",
            "Native3.2.3 versus MuJoCo4/MJWarp FP32 conflates version, compiled model and precision; original plant Euler was not rerun.",
            "H-QP is exact native host execution, with synchronous host/device exchange every2ms hold; this is not a resident all-device controller.",
            "same_state_action compares Warp kernel against Python raw_policy and a shared native R29Project; it does not directly invoke the original native Policy::Action. The separate native-bridge identity check covers four selected nominal cases.",
            "Source warmup count48 includes one24-world step inside CUDA graph capture. CUDA capture records without executing; only the preceding24-world step advances warmup physics. Main-loop42000 count is unaffected.",
            "No separate timing for host H-QP alone, uploads, allocation or GPU competition; recorded map+projection timing includes both engine checks.",
            "Published compact numeric trajectory values have not been independently recomputed in this audit."
        ],external_semantics_sources=[
            {"url":"https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/memory_management.html","section":"Graph Allocations","use":"CUDA graph capture records operations; execution starts at capture_launch."},
            {"url":"https://github.com/NVIDIA/warp/discussions/565","date":"2025-03-06","use":"Warp maintainer explicitly distinguishes capture recording from CUDA execution."}
        ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),verified_original_files=len(verified),
        verified_original_bytes=report["independent_original_bytes_verified"],
        unique_effective_policy_count=report["unique_effective_policy_count"],
        independent_initial_raw_center_recomputation={k:v for k,v in report["independent_initial_raw_center_recomputation"].items() if k!="rows"},
        policy_pair_comparisons=duplicates,timing_arithmetic=report["timing_arithmetic"],
        CPU_timing=[{k:v for k,v in x.items() if k!="timing_reported"} for x in jobs]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
