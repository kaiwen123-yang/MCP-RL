"""Explicit model-only factual fitting; never trains/calls the frozen Go2 policy.

Same initialization, window indices, labels, Adam steps and latent-consistency
weight for ordinary one-step ID and autoregressive multistep ID. Validation
uses separate complete episodes and never selects a checkpoint. No task-self
cost gradient appears. Importing the script performs no fitting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(state):
    hashed=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        hashed.update(name.encode("utf-8"));hashed.update(str(array.dtype).encode("ascii"))
        hashed.update(str(array.shape).encode("ascii"));hashed.update(array.tobytes())
    return hashed.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_episodes(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes, descriptions = {"train":[], "val":[]}, {"train":[], "val":[]}
    for entry in manifest["episodes"]:
        split = "val" if entry["split"] == "validation" else entry["split"]
        if split not in episodes: raise ValueError("Episode split must be train or val")
        path = manifest_path.parent/entry["file"]
        with np.load(path, allow_pickle=False) as saved:
            episode = {key: np.asarray(saved[key], dtype=np.float32) for key in ("history","sensors","context","pose","commands")}
        transitions = len(episode["commands"])
        shapes = {"history":(transitions+1,6,45), "sensors":(transitions+1,42),
                  "context":(transitions+1,6), "pose":(transitions+1,3), "commands":(transitions,3)}
        if any(episode[key].shape != shape or not np.isfinite(episode[key]).all() for key,shape in shapes.items()):
            raise ValueError(f"Invalid finite macro snapshot schema: {entry['file']}")
        if transitions != entry["transitions"]: raise ValueError("Manifest transition count differs from NPZ")
        episodes[split].append(episode)
        descriptions[split].append({**entry, "sha256":digest(path)})
    if sorted(e["seed"] for e in descriptions["train"]) != list(range(101,107)) or sorted(e["seed"] for e in descriptions["val"]) != [201,202]:
        raise ValueError("Frozen split requires six train seeds101..106 and two validation seeds201/202")
    return manifest, episodes, descriptions


def normalization_from_train(episodes, dt):
    inputs, commands, residuals = [], [], []
    for episode in episodes:
        context, pose, command = episode["context"], episode["pose"], episode["commands"]
        inputs.append(np.concatenate((episode["history"].reshape(len(context),-1), episode["sensors"], context), axis=-1))
        commands.append(command)
        delta = pose[1:,:2]-pose[:-1,:2]
        c,s = np.cos(pose[:-1,2]),np.sin(pose[:-1,2])
        local_delta = np.column_stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1]))
        yaw_delta = (pose[1:,2]-pose[:-1,2]+np.pi)%(2*np.pi)-np.pi
        response = np.column_stack((local_delta,yaw_delta,context[1:,:3]))
        residual = response-np.concatenate((dt*command,command),axis=-1)
        residual[:,2] = (residual[:,2]+np.pi)%(2*np.pi)-np.pi
        residuals.append(residual)
    values, cmds, residual = map(lambda parts: np.concatenate(parts,axis=0), (inputs, commands, residuals))
    # Scales are fitted only on train episodes. Residual mean is NOT added to
    # predictions: a zero decoder remains exactly the ideal-command baseline.
    return {"input_mean":values.mean(0).tolist(), "input_scale":np.maximum(values.std(0),1e-3).tolist(),
            "command_mean":cmds.mean(0).tolist(), "command_scale":np.maximum(cmds.std(0),1e-3).tolist(),
            "residual_scale":np.maximum(residual.std(0),1e-3).tolist()}


def windows(episodes, horizon):
    parts = {k:[] for k in ("history","sensors","context","pose","commands","mask")}
    for episode in episodes:
        count = len(episode["commands"])
        for start in range(count):
            valid = min(horizon,count-start)
            for key in ("history","sensors","context","pose"):
                values = episode[key][start:start+valid+1]
                parts[key].append(np.concatenate((values,np.repeat(values[-1:],horizon-valid,axis=0)),axis=0))
            cmds = episode["commands"][start:start+valid]
            parts["commands"].append(np.concatenate((cmds,np.zeros((horizon-valid,3),dtype=np.float32)),axis=0))
            parts["mask"].append(np.arange(horizon)<valid)
    return {key:np.stack(value) for key,value in parts.items()}


def predict_window(model, batch, arm):
    import torch
    from go2_frozen.compact import CompactState
    b,h,_ = batch["commands"].shape
    if arm == "one_step":
        state = model.start(batch["history"][:,:-1].reshape(b*h,6,45), batch["sensors"][:,:-1].reshape(b*h,42),
                            batch["context"][:,:-1].reshape(b*h,6), batch["pose"][:,:-1].reshape(b*h,3))
        predicted = model.step(state,batch["commands"].reshape(b*h,3))
        return CompactState(predicted.pose.reshape(b,h,3),predicted.twist.reshape(b,h,3),predicted.hidden.reshape(b,h,-1))
    state = model.start(batch["history"][:,0],batch["sensors"][:,0],batch["context"][:,0],batch["pose"][:,0])
    poses,twists,hidden = [],[],[]
    for step in range(h):
        proposed = model.step(state,batch["commands"][:,step])
        active = batch["mask"][:,step,None]
        state = CompactState(torch.where(active,proposed.pose,state.pose),
                             torch.where(active,proposed.twist,state.twist),
                             torch.where(active,proposed.hidden,state.hidden))
        poses.append(state.pose);twists.append(state.twist);hidden.append(state.hidden)
    return CompactState(torch.stack(poses,1),torch.stack(twists,1),torch.stack(hidden,1))


def factual_objective(model,batch,arm,latent_weight):
    import torch
    from go2_frozen.compact import wrap_angle
    predicted = predict_window(model,batch,arm)
    delta = predicted.pose[...,:2]-batch["pose"][:,1:,:2]
    reference_yaw = batch["pose"][:,:-1,2]
    c,s = torch.cos(reference_yaw),torch.sin(reference_yaw)
    local_error = torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
    errors = torch.cat((local_error,wrap_angle(predicted.pose[...,2]-batch["pose"][:,1:,2]).unsqueeze(-1),
                        predicted.twist-batch["context"][:,1:,:3]),-1)/model.residual_scale
    mask = batch["mask"].to(errors.dtype)
    factual = (errors.square().mean(-1)*mask).sum()/mask.sum()
    b,h = mask.shape
    with torch.no_grad():
        target_hidden = model.encode(batch["history"][:,1:].reshape(b*h,6,45),
                                     batch["sensors"][:,1:].reshape(b*h,42),
                                     batch["context"][:,1:].reshape(b*h,6)).reshape(b,h,-1)
    latent = ((predicted.hidden-target_hidden).square().mean(-1)*mask).sum()/mask.sum()
    return factual+latent_weight*latent,{"factual_loss":factual,"latent_consistency_loss":latent}


def validate(model,data):
    import torch
    from go2_frozen.compact import wrap_angle
    result={}
    work={"encoder_sample_calls":0,"response_sample_steps":0}
    for arm,label in (("one_step","one_step"),("multistep","autoregressive_h5")):
        pose_errors=[];twist_errors=[];masks=[]
        with torch.inference_mode():
            for begin in range(0,len(data["commands"]),128):
                batch={k:v[begin:begin+128] for k,v in data.items()}
                prediction=predict_window(model,batch,arm)
                error=prediction.pose-batch["pose"][:,1:]
                error[...,2]=wrap_angle(error[...,2])
                pose_errors.append(error.cpu().numpy());twist_errors.append((prediction.twist-batch["context"][:,1:,:3]).cpu().numpy());masks.append(batch["mask"].cpu().numpy())
                b,h=batch["mask"].shape
                work["encoder_sample_calls"]+=b*h if arm=="one_step" else b
                work["response_sample_steps"]+=b*h
        pe,ve,mask=np.concatenate(pose_errors),np.concatenate(twist_errors),np.concatenate(masks)
        def metrics(p,v,valid):
            p,v=p[valid],v[valid]
            return {"labels":len(p),"position_rmse_m":float(np.sqrt(np.mean(np.sum(p[:,:2]**2,axis=-1)))),
                    "yaw_rmse_rad":float(np.sqrt(np.mean(p[:,2]**2))),
                    "body_velocity_rmse_m_s":float(np.sqrt(np.mean(np.sum(v[:,:2]**2,axis=-1)))),
                    "yaw_rate_rmse_rad_s":float(np.sqrt(np.mean(v[:,2]**2)))}
        result[label]={**metrics(pe,ve,mask),"per_horizon":[metrics(pe[:,i],ve[:,i],mask[:,i]) for i in range(pe.shape[1])]}
    return {"windows":len(data["commands"]),**result},work


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/go2_compact_fit.json"))
    args=parser.parse_args()
    protocol=json.loads(args.config.read_text(encoding="utf-8"))
    fitting=protocol["fit"]
    args.steps=fitting["updates_per_model"];args.batch_size=fitting["batch_size"]
    args.horizon=fitting["horizon"];args.seed=fitting["initialization_seed"]
    args.batch_seed=fitting["batch_seed"];args.lr=fitting["learning_rate"]
    args.latent_consistency_weight=fitting["latent_consistency_weight"]
    args.threads=protocol["resource_budget"]["torch_threads"]
    if args.dataset.is_dir(): args.dataset=args.dataset/"dataset.json"
    if args.threads!=1: raise ValueError("This frozen fitting protocol uses one CPU thread")
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
    resources=protocol["resource_budget"]
    if hasattr(os,"sched_setaffinity"): os.sched_setaffinity(0,set(resources["cpu_affinity"]))
    if hasattr(os,"getpriority"):
        current_nice=os.getpriority(os.PRIO_PROCESS,0)
        if current_nice<resources["nice"]: os.nice(resources["nice"]-current_nice)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/"src"))
    from go2_frozen.compact import CompactResponse
    start=time.perf_counter()
    manifest,episodes,descriptions=load_episodes(args.dataset)
    dt=float(manifest.get("macro_dt",.2))
    if not np.isclose(dt,.2): raise ValueError("Frozen macro response period is .2 seconds")
    normalization=normalization_from_train(episodes["train"],dt)
    arrays={split:windows(items,args.horizon) for split,items in episodes.items()}
    data={split:{k:torch.from_numpy(v) for k,v in items.items()} for split,items in arrays.items()}
    rng=np.random.default_rng(args.batch_seed)
    indices=rng.integers(0,len(arrays["train"]["commands"]),size=(args.steps,args.batch_size),dtype=np.int64)
    torch.manual_seed(args.seed)
    initial=CompactResponse(normalization,macro_dt=dt)
    initial_state=copy.deepcopy(initial.state_dict())
    cfg={"seed":args.seed,"batch_seed":args.batch_seed,"updates":args.steps,"batch_size":args.batch_size,"horizon":args.horizon,
         "learning_rate":args.lr,"threads":args.threads,"latent_consistency_weight":args.latent_consistency_weight}
    dataset={"manifest_sha256":digest(args.dataset),"training_episodes":descriptions["train"],"validation_episodes":descriptions["val"],
             "training_transitions":sum(e["transitions"] for e in descriptions["train"]),
             "validation_transitions":sum(e["transitions"] for e in descriptions["val"]),
             "policy_sha256":manifest.get("policy_sha256_before"),"metadata":manifest}
    source={name:digest(ROOT/name) for name in ("src/go2_frozen/compact.py","scripts/fit_go2_compact.py")}
    summary={"status":"RUNNING","config":cfg,"protocol":protocol,"protocol_sha256":digest(args.config),"dataset":dataset,"source_identity":source,
             "resource_settings":{"torch_threads":torch.get_num_threads(),"gpu_used":False,
                                  "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                                  "nice":os.getpriority(os.PRIO_PROCESS,0) if hasattr(os,"getpriority") else None},
             "shared_initial_state_sha256":state_digest(initial_state),
             "parameter_count":sum(p.numel() for p in initial.parameters()),
             "zero_head_prior":"Both models use Euler body displacement=dt*command rotated by the current yaw, with endpoint twist=command; this is not identical to the older midpoint-yaw kinematic_step",
             "shared_batch_indices_sha256":hashlib.sha256(indices.tobytes()).hexdigest(),
             "normalization_source":"training episodes only; output residual is scaled but not mean-shifted",
             "arms":{},"total_policy_optimizer_steps":0,"new_environment_steps":0,"policy_inferences":0,
             "interpretation":"Ordinary one-step versus multistep factual identification with common latent consistency; not a new policy-learning method or established original contribution"}
    write_json(output/"fit-summary.json",summary)
    for method,arm in (("id_one_step","one_step"),("compact_multistep","multistep")):
        arm_start=time.perf_counter();model=CompactResponse(normalization,macro_dt=dt);model.load_state_dict(initial_state)
        initial_hash=state_digest(model.state_dict())
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        records=[];valid_labels=0;train_start=time.perf_counter()
        for step,index in enumerate(indices,1):
            batch={k:v[index] for k,v in data["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss,parts=factual_objective(model,batch,arm,args.latent_consistency_weight)
            loss.backward();optimizer.step()
            valid_labels+=int(batch["mask"].sum())
            records.append({"step":step,"loss":float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}})
        fit_wall=time.perf_counter()-train_start
        validation_start=time.perf_counter();validation,val_work=validate(model,data["val"])
        val_wall=time.perf_counter()-validation_start
        checkpoint=output/f"{method}.pt"
        torch.save({"format":"go2-compact-factual-v1","arm":method,"architecture":{"macro_dt":dt,"hidden_size":32,"encoder_width":64},
                    "state_dict":model.state_dict(),"fitting_config":cfg,"dataset_manifest_sha256":dataset["manifest_sha256"],
                    "policy_sha256":dataset["policy_sha256"],"source_identity":source,
                    "input_contract":"history newest-first6x45; sensors bodygyro*.25,gravity,q,dq*.05,previous_action; context bodyvx/bodyvy/world-angular-z/roll/pitch/height",
                    "output_contract":"body displacement/world yaw change and next bodyvx/bodyvy/world angular z; no future contact/height/tilt",
                    "checkpoint_selection":"fixed final step; no validation selection","policy_optimizer_steps":0},checkpoint)
        write_json(output/f"{method}-training.json",records)
        bs,h,updates=args.batch_size,args.horizon,args.steps
        work={"encoder_sample_calls":updates*bs*(2*h if arm=="one_step" else h+1),
              "response_sample_steps":updates*bs*h,"reverse_response_sample_steps":updates*bs*h,
              "detached_target_encoder_sample_calls":updates*bs*h,"backward_updates":updates,
              "actual_valid_supervised_macro_labels":valid_labels,"padded_forward_steps_counted":True,
              "validation":val_work}
        summary["arms"][method]={"checkpoint":checkpoint.name,"checkpoint_sha256":digest(checkpoint),
                              "checkpoint_bytes":checkpoint.stat().st_size,"parameter_count":sum(p.numel() for p in model.parameters()),
                              "initial_state_sha256":initial_hash,"final_state_sha256":state_digest(model.state_dict()),
                              "model_optimizer_steps":args.steps,"policy_optimizer_steps":0,
                              "fit_wall_seconds":fit_wall,"validation_wall_seconds":val_wall,"arm_wall_seconds":time.perf_counter()-arm_start,
                              "validation":validation,"work":work,"last_train_loss":records[-1]}
        write_json(output/"fit-summary.json",summary)
    summary.update(status="COMPLETED_CONFIGURED_MODEL_FIT",total_model_optimizer_steps=2*args.steps,total_fit_wall_seconds=time.perf_counter()-start)
    write_json(output/"fit-summary.json",summary)
    print(json.dumps({"status":summary["status"],"model_updates":summary["total_model_optimizer_steps"],"policy_updates":0,"new_environment_steps":0,"wall_seconds":summary["total_fit_wall_seconds"]}))


if __name__=="__main__":
    main()
