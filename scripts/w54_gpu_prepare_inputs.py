#!/usr/bin/env python3
"""Freeze finite W54 C inputs without stepping or replaying any plant."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from s1_w54_f_backend import assemble_centroidal
from w54_gpu_force_horizon import ROOT, REL, KEYS, sha, write

OLD_NAME = "job-20261117T023610Z-w53-f-a-eval-PANN_PPO_S5-s1-E30-attempt02"
OLD_ROOT = Path("/root/autodl-tmp/MCP-RL-w42/outputs")/REL/OLD_NAME


def confirmed_stand_history(trajectory, cfg):
    """Use exactly the fixed 500 Hz observed-force hysteresis before motion."""
    contact = np.zeros(4, dtype=bool)
    touch = np.zeros(4, dtype=int)
    release = np.zeros(4, dtype=int)
    anchor = trajectory["feet"][0].copy()
    confirmed, anchors = [], []
    events = []
    ec = cfg["contact_events"]
    on_steps = round(ec["touch_confirm_s"]*ec["sample_Hz"])
    off_steps = round(ec["release_confirm_s"]*ec["sample_Hz"])
    for tick,t in enumerate(trajectory["time"]):
        measured = trajectory["contact"][tick]
        force = trajectory["actual_grf"][tick,:,2]
        on = measured & (force >= ec["touch_on_N"])
        off = (~measured) | (force < ec["touch_off_N"])
        touch = np.where(on,touch+1,0)
        release = np.where(off,release+1,0)
        entered = (~contact) & (touch >= on_steps)
        exited = contact & (release >= off_steps)
        anchor[entered] = trajectory["feet"][tick,entered]
        contact[entered] = True
        contact[exited] = False
        for leg in np.flatnonzero(entered):
            events.append(dict(tick=tick,time_s=float(t),leg=int(leg),event="confirmed_touchdown",
                               anchor_m=anchor[leg].tolist()))
        confirmed.append(contact.copy());anchors.append(anchor.copy())
    return np.asarray(confirmed),np.asarray(anchors),events


def legacy(out,cfg):
    source = OLD_ROOT/"w53-f"
    manifest = json.loads((source/"manifest.json").read_text())
    with np.load(OLD_ROOT/"trajectory.npz") as z:
        trajectory = {k:z[k].copy() for k in ("time","phase","contact","actual_grf","feet")}
    chunks = []
    for item in manifest["control_chunks"]:
        with np.load(source/item["path"]) as z:
            chunks.append({k:z[k].copy() for k in ("time","force")})
    control = {k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
    contacts,anchors,events = confirmed_stand_history(trajectory,cfg)
    records = []
    n = cfg["mpc"]["intervals"]
    for number,item in enumerate(manifest["full_mpc_files"]):
        path = source/item["path"]
        t = item["time_s"]
        tick = int(np.argmin(np.abs(trajectory["time"]-t)))
        rec = dict(source_kind="W53_saved_F_plan",source=str(path),source_sha256=sha(path),
                   physical_time_s=t,source_call=item["call"],status="not_reconstructible")
        if trajectory["phase"][tick] != 0:
            rec["reason"] = "W53 moving record lacks repaired load-transfer, committed event and WBC-residual memory; old future contact/feet are not reused"
        elif t < .06-1e-9 or not contacts[tick].all():
            rec["reason"] = "initial settling lacks four confirmed load-bearing anchors at this saved state; no synthetic future touchdown is inserted"
        else:
            with np.load(path) as z:
                x0 = z["state_reference"][0].copy()
                reference = z["state_reference"].copy()
                mass,inertia = float(z["mass"]),z["inertia"].copy()
            last_tick = np.flatnonzero(control["time"] < t-1e-10)[-1]
            last_force = control["force"][last_tick].copy()
            feet = np.tile(anchors[tick],(n+1,1,1))
            blocks = [assemble_centroidal(mass,inertia,reference[k,3:6],feet[k],
                                          x0[:3],x0[6:9],cfg["mpc"]["dt_s"],cfg["friction_coefficient"])
                      for k in range(n)]
            p = dict(stage_A=np.stack([b["A"] for b in blocks]),
                     stage_B=np.stack([b["B"] for b in blocks]),
                     stage_c=np.stack([b["c"] for b in blocks]),
                     x0=x0,reference=reference,last_force=last_force,
                     contact=np.tile(contacts[tick],(n,1)),feet=feet,
                     force_normal_upper=np.full((n,4),np.inf))
            dest = out/("input-%04d.npz"%number)
            np.savez_compressed(dest,**p)
            rec.update(status="ready",input_path=str(dest),input_sha256=sha(dest),
                       reconstruction="stand-only causal contact hysteresis and actual confirmed-touchdown anchors; unchanged stand body reference; repaired F stage builder; immediately previous actual WBC force",
                       trajectory_tick=tick,previous_WBC_time_s=float(control["time"][last_tick]),
                       anchor_confirmation_events=[e for e in events if e["tick"]<=tick])
        records.append(rec)
    failed = source/"failed-mpc-3.120000.npz"
    records.append(dict(source_kind="W53_failed_F_QP",source=str(failed),source_sha256=sha(failed),
                        physical_time_s=3.12,status="not_reconstructible",
                        reason="failed static-torque matrix retained as wrong-model evidence; missing repaired moving event and WBC state prevents a causal repaired horizon; algebraic correction is handled by F"))
    return records,dict(legacy_full_plans=len(manifest["full_mpc_files"]),legacy_failed_matrices=1,
        contact_hysteresis=cfg["contact_events"],
        original_trajectory_sha256=sha(OLD_ROOT/"trajectory.npz"),
        prior_control_sources=[dict(path=str(source/c["path"]),sha256=sha(source/c["path"])) for c in manifest["control_chunks"]])


def natural(out, record_dir, limit, target_times=None):
    manifest = json.loads((record_dir/"manifest.json").read_text())
    available = manifest["full_mpc_files"]
    targets = {}
    missing = []
    if target_times is None:
        frozen = available[:limit]
    else:
        frozen = []
        for target in target_times:
            later = [entry for entry in available if entry["time_s"] >= target-1e-9]
            if not later:
                missing.append(dict(source_kind="W54_natural_F_plan",target_time_s=target,
                    status="unavailable",reason="original task ended before an eligible saved plan; no physics added"))
                continue
            entry=later[0]
            targets.setdefault(entry["path"],[]).append(target)
            if not any(x["path"]==entry["path"] for x in frozen):frozen.append(entry)
    records = []
    for i,entry in enumerate(frozen):
        path=record_dir/entry["path"]
        with np.load(path,allow_pickle=False) as z:
            p={k:z[k].copy() for k in KEYS}
        dest=out/("input-%04d.npz"%i)
        np.savez_compressed(dest,**p)
        records.append(dict(source_kind="W54_natural_F_plan",source=str(path),
            source_sha256=sha(path),source_call=entry["call"],physical_time_s=entry["time_s"],
            status="ready",input_path=str(dest),input_sha256=sha(dest),
            reconstruction="none; corrected causal event/stage contract saved during original F physics",
            requested_times_s=targets.get(entry["path"])))
    records.extend(missing)
    return records,dict(record_dir=str(record_dir),available_completed_plans=len(available),
                        finite_limit=limit if target_times is None else len(target_times),
                        declared_target_times_s=target_times,selected_completed_plans=len(frozen),
                        clock_matching_tolerance_s=1e-9,
                        selection="first completed plan at or after each predeclared time, deduplicated" if target_times is not None else "first completed plan records in causal order, one pass; no new physics to fill missing records")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pass-id",required=True)
    parser.add_argument("--legacy",action="store_true")
    parser.add_argument("--record-dir",type=Path)
    parser.add_argument("--limit",type=int,default=160)
    parser.add_argument("--target-times")
    parser.add_argument("--backend",default="s1_w54_f_backend")
    args=parser.parse_args()
    target_times=[float(x) for x in args.target_times.split(",")] if args.target_times else None
    wall0=time.perf_counter();usage0=resource.getrusage(resource.RUSAGE_SELF)
    out=ROOT/"outputs"/REL/"w54-C"/(args.pass_id+"-inputs")
    out.mkdir(parents=True,exist_ok=False)
    configuration=ROOT/"experiments/terrain_memory/s1-w54-f.json"
    cfg=json.loads(configuration.read_text()) if args.legacy else json.loads((args.record_dir/"manifest.json").read_text())["config"]
    frozen_config=out/"configuration.json";write(frozen_config,cfg)
    records,details=legacy(out,cfg) if args.legacy else natural(out,args.record_dir,args.limit,target_times)
    configuration=frozen_config
    manifest=dict(schema="w54_c_finite_causal_inputs_v1",created_unix_s=time.time(),
                  pass_id=args.pass_id,backend_module=args.backend,configuration=str(configuration),
                  configuration_sha256=sha(configuration),
                  source_stage_builder_sha256=sha(ROOT/"scripts"/(args.backend+".py")),
                  records=records,details=details,
                  eligible=sum(r["status"]=="ready" for r in records),
                  not_reconstructible=sum(r["status"]!="ready" for r in records),
                  new_physics_steps=0,new_native_rollout=0,new_FD=0,new_PPO_samples=0,
                  preparation_timing=dict(wall_s=time.perf_counter()-wall0,
                    user_cpu_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime-usage0.ru_utime,
                    system_cpu_s=resource.getrusage(resource.RUSAGE_SELF).ru_stime-usage0.ru_stime),
                  deadline=None,wall_deadline_enabled=False)
    write(out/"manifest.json",manifest)
    write(ROOT/"results"/REL/"w54"/("task-c-"+args.pass_id+"-inputs.json"),manifest)
    print(json.dumps(dict(manifest=str(out/"manifest.json"),eligible=manifest["eligible"],
                          not_reconstructible=manifest["not_reconstructible"])))


if __name__=="__main__":
    main()
