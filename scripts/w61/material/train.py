"""Train/evaluate the fixed W61-A coupon family and matched adaptation baselines on D57."""
import json
import os
import platform
import resource
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy.optimize import lsq_linear
import torch
from coupons import CONFIG, COMPACT, ROOT, WORK, create, digest, unlabeled_probes
from tm02.w61_material import Family, HistoryGRU, MaterialResponse, features, sufficient_statistics

TZ = ZoneInfo('Asia/Shanghai')


def stamp():
    return datetime.now(TZ).isoformat(timespec='seconds')


def dump(name, data):
    (COMPACT / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def environment():
    import importlib.metadata as meta
    version = lambda pkg: meta.version(pkg)
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        scipy=scipy.__version__, jax=version('jax'), jaxlib=version('jaxlib'), device='cpu',
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        gpu_memory_allocated_bytes=0, omp=os.environ.get('OMP_NUM_THREADS'),
        solver_gpu_utilization='No GPU calls; system-wide shared-card load is outside this measurement.')


def load_split(raw, split, cfg):
    strain, dt, obs, target = [raw[f'{split}_{k}'] for k in ('strain','dt','observed','reference')]
    tensor = lambda a: torch.tensor(a,dtype=torch.float32)
    x = tensor(features(strain,dt,cfg))
    dx = np.diff(strain,axis=2,prepend=strain[:,:,:1])
    inp = tensor(np.concatenate([strain/.01, dx/.003, dt[...,None]/.05], -1))
    support_observed = tensor(obs[:,:4]/cfg['stress_scale_MPa'])
    target = tensor(target[:,4:]/cfg['stress_scale_MPa'])
    # Holdout evaluations only consume clean query stress after all predictions.
    return dict(sx=x[:,:4], sy=support_observed, qx=x[:,4:], qi=inp[:,4:], qy=target)


def batch(data, ids, supports):
    x = data['sx'][ids][:,supports]
    y = data['sy'][ids][:,supports]
    return *sufficient_statistics(x,y), data['qx'][ids], data['qi'][ids]


def predict(net, stats, probe):
    gram, rhs, x, inp = stats
    if isinstance(net,HistoryGRU):
        return net(gram,rhs,x,inp)
    weights = net.adapt(gram,rhs,probe)
    return torch.einsum('bptck,bk->bptc',x,weights)


def train_model(method, seed, data, probe, cfg):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HistoryGRU() if method=='support_gru' else Family(cfg,method)
    opt = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    curve = []
    wall = time.perf_counter()
    cpu = time.process_time()
    for it in range(cfg['updates']):
        ids = rng.choice(data['qx'].shape[0],cfg['batch_materials'],replace=False)
        supports = [rng.integers(4)] if it%2 == 0 else list(range(4))
        prediction = predict(model,batch(data, ids, supports),probe)
        error = prediction[:,:,1:] - data['qy'][ids,:,1:]
        loss = error.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    elapsed = time.perf_counter()-wall
    return model, np.array(curve), dict(wall_s=elapsed, cpu_core_s=time.process_time()-cpu,
        material_episodes=cfg['updates']*cfg['batch_materials'],
        query_response_points_presented=cfg['updates']*cfg['batch_materials']*4*(cfg['length']-1),
        query_points_per_s=cfg['updates']*cfg['batch_materials']*4*(cfg['length']-1)/elapsed,
        trainable_parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=cfg['updates'])


def physical_coupon_baseline(data, count):
    fits = []
    for x,y in zip(data['sx'][:,:count], data['sy'][:,:count]):
        fit = lsq_linear(x.numpy().reshape(-1,16).astype(float), y.numpy().ravel().astype(float), bounds=(0,np.inf), method="bvls").x
        fits.append(fit)
    return torch.einsum('bptck,bk->bptc',data['qx'],torch.tensor(np.stack(fits),dtype=torch.float32))


def frozen_coupon_baseline(raw, split, cfg):
    import jax
    jax.config.update('jax_enable_x64', True)
    from tm02.material import load_weights
    from tm02.evaluate import integrate
    settings = json.loads((ROOT/cfg['reference_config']).read_text())['numerics']
    w, norm = load_weights(cfg['frozen_pann'])
    values = []
    for e, ds in zip(raw[split+'_strain'][:,4:], raw[split+'_dt'][:,4:]):
        paths = []
        for path, step in zip(e, ds):
            row, failure = integrate(w,norm,step.cumsum(),path,settings)
            paths.append(row['stress'])
        values.append(paths)
    return np.asarray(values)


def score(split, method, seed, count, forecast, raw, cfg):
    truth = raw[split+'_reference'][:,4:]
    strain = raw[split+'_strain'][:,4:]
    rows = []
    for i, sid in enumerate(range(cfg['material_ids'][split][0],cfg['material_ids'][split][1]+1)):
        for j, name in enumerate(cfg['query_paths']):
            delta = forecast[i,j,1:] - truth[i,j,1:]
            complete = bool(np.isfinite(delta).all())
            release = np.max(np.abs(strain[i,j,1:]),axis=-1)<1.e-10
            rows.append(dict(split=split,method=method,seed=seed,support_count=count,specimen=sid,
                path=name,complete=complete,scored_tensor_times=len(delta) if complete else 0,
                pressure_release_tensor_times=int(release.sum()),
                rmse_MPa=float(np.sqrt(np.mean(delta**2))) if complete else None,
                zero_strain_stress_rmse_MPa=float(np.sqrt(np.mean(delta[release]**2))) if complete and release.any() else None))
    return rows


def timing(model, probe, raw, cfg):
    adaptation, integration = [], []
    indicators = []
    # Whole measured publication includes support replay, contractions, solve and conversion.
    for i, sid in enumerate(range(cfg['material_ids']['holdout'][0],cfg['material_ids']['holdout'][1]+1)):
        e, dt = raw['holdout_strain'][i], raw['holdout_dt'][i]
        observations = raw['holdout_observed'][i]
        for _ in range(5):
            st=time.perf_counter_ns()
            response = model.infer(e[:4],dt[:4],observations[:4],probe)
            adaptation.append((time.perf_counter_ns()-st)/1.e6)
        row = dict(specimen=sid, **response.indicators())
        indicators.append(row)
        for page, ticks in zip(e[4:], dt[4:]):
            state=response.virgin()
            for strain, tick in zip(page[1:],ticks[1:]):
                st=time.perf_counter_ns()
                _,state=response.step(strain,state,tick)
                integration.append((time.perf_counter_ns()-st)/1.e6)
    summarize = lambda a: dict(n=len(a),p50_p95_p99_ms=np.percentile(a,[50,95,99]).tolist(),
        mean_rate_per_s=1000/np.mean(a))
    return dict(complete_support4_publication=summarize(adaptation),constitutive_step_one_coupon=summarize(integration),
        shared_host=True,robot_chain_latency=None,onboard_deadline_miss_rate=None), indicators


def main():
    cfg=json.loads(CONFIG.read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start, cpu = time.perf_counter(), time.process_time()
    identity=dict(started_shanghai=stamp(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=digest(CONFIG),
        environment=environment(),deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    create(cfg)
    raw = np.load(WORK/'coupons.npz')
    splits = {s: load_split(raw,s,cfg) for s in cfg['material_ids']}
    probe_e, probe_dt = map(np.stack,unlabeled_probes(cfg))
    p = torch.tensor(features(probe_e,probe_dt,cfg)).reshape(-1,16)
    probe = p.T @ p / len(p)
    predictions, curves, trained, models = {}, {}, {}, {}
    scores, costs, indicators = [], {}, {}
    for seed in cfg['train_seeds']:
        for method in cfg['models'][:5]:
            label = f'{method}_{seed}'
            model, loss, info = train_model(method,seed,splits['train'],probe,cfg)
            models[label] = model.state_dict()
            curves[label], trained[label] = loss, info
            for split in ('validation','holdout'):
                data = splits[split]
                for count in cfg['support_counts']:
                    with torch.no_grad():
                        stats=batch(data,list(range(len(data['qx']))),list(range(count)))
                        output=predict(model,stats,probe).numpy()*cfg['stress_scale_MPa']
                    predictions[f'{split}_{label}_{count}']=output
                    scores.extend(score(split,method,seed,count,output,raw,cfg))
            if method=='lowrank_response':
                costs[label],indicators[label]=timing(model,probe,raw,cfg)
            print(label, f'{info["wall_s"]:.2f}s', flush=True)
    for split in ('validation','holdout'):
        for count in cfg['support_counts']:
            output=physical_coupon_baseline(splits[split],count).numpy()*cfg['stress_scale_MPa']
            predictions[f'{split}_spectrum_nnls_{count}']=output
            scores.extend(score(split,'spectrum_nnls',None,count,output,raw,cfg))
        output=frozen_coupon_baseline(raw,split,cfg)
        predictions[f'{split}_frozen_tm02_pann']=output
        for count in cfg['support_counts']:
            scores.extend(score(split,'frozen_tm02_pann',None,count,output,raw,cfg))
    # Only after the last fixed training block do we aggregate test scores.
    summary=[]
    for method in cfg['models']:
        for n in cfg['support_counts']:
            rows=[r for r in scores if r['method']==method and r['split']=='holdout' and r['support_count']==n]
            rmse=[r['rmse_MPa'] for r in rows if r['complete']]
            unloaded=[r['zero_strain_stress_rmse_MPa'] for r in rows if r['zero_strain_stress_rmse_MPa'] is not None]
            summary.append(dict(method=method,support_count=n,complete=len(rmse),registered=len(rows),
                mean_path_rmse_MPa=np.mean(rmse) if rmse else None,
                mean_path_zero_strain_stress_rmse_MPa=np.mean(unloaded) if unloaded else None))
    torch.save(dict(state_dicts=models,probe_gram=probe,config=cfg),WORK/'weights.pt')
    np.savez_compressed(WORK/'predictions.npz',**predictions)
    np.savez_compressed(WORK/'curves.npz',**curves)
    np.savez_compressed(COMPACT/'query-paths.npz',rmse_MPa=np.array([r['rmse_MPa'] if r['complete'] else np.nan for r in scores]),
        recovery_rmse_MPa=np.array([r['zero_strain_stress_rmse_MPa'] if r['zero_strain_stress_rmse_MPa'] is not None else np.nan for r in scores]),
        split=[r['split'] for r in scores],model=[r['method'] for r in scores],seed=[r['seed'] or -1 for r in scores],
        support_count=[r['support_count'] for r in scores],material=[r['specimen'] for r in scores],path=[r['path'] for r in scores])
    dump('result.json',dict(**identity,ended_shanghai=stamp(),selection=cfg['selection'],
        experiment='isolated six-component virgin coupon adaptation, matched spectra and BE history integrator',
        summary=summary,training=trained,latency=costs,material_indicators=indicators,
        full_run_wall_s=time.perf_counter()-start,cpu_core_h=(time.process_time()-cpu)/3600,
        peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        original_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in
                        [WORK/'weights.pt',WORK/'predictions.npz',WORK/'coupons.npz',WORK/'curves.npz']},
        old_pann_sha256=digest(Path(cfg['frozen_pann'])),robot_episodes=0,hardware_episodes=0))


if __name__=='__main__':
    from pathlib import Path
    main()
