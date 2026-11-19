"""Reconstruct W55 material histories from actually applied, world-keyed boundaries."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R12 = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
RAW = ROOT/'outputs'/R12/'w55-c'
COMPACT = ROOT/'results'/R12/'w55/c'
LEGACY = ROOT.parent/'MCP-RL-w54'/'outputs'/R12
OLDER = ROOT.parent/'MCP-RL-w42'/'outputs'/R12
CASES = {
    'N44': (LEGACY, '*-w54-n-a-r44-eval*', 'training'),
    'F51': (LEGACY, '*-w54-f-a-r51-eval*', 'validation'),
    'N48': (LEGACY, '*-w54-n-a-r48-eval*', 'known_diagnostic'),
    'F52': (LEGACY, '*-w54-f-a-r52-eval*', 'known_diagnostic'),
    'B46': (OLDER, '*-w46-b-eval*-s2-E30-*', 'held_out_seed2'),
}
REVIEW = ROOT/'results'/R12/'reviews/W54-review'
WEIGHTS = ROOT/'work/assets/B_random1000--ideal-potentials.npz'
H = 0.05
FORCE_SCALE = 25000.
DT = 0.0002


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def make_reference():
    from tm02.s1_material import ReferenceColumns
    params = json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    return ReferenceColumns(params)


def geometry(bottom, keys):
    """Map the entire finite sole to exact overlap areas and the lowest owner."""
    half = np.sqrt(.001)/2
    soft = ((bottom[:,:,0] >= .8+half) & (bottom[:,:,0] <= 2.2-half)
            & (np.abs(bottom[:,:,1]) <= .6-half))
    T, K = len(bottom), len(keys)
    normal = np.full((T,K), np.inf)
    area = np.zeros((T,K))
    owner = np.full((T,K), -1, np.int8)
    row = np.arange(T)
    for j,key in enumerate(keys):
        lo = np.maximum(bottom[:,:,:2]-half, key*.02)
        hi = np.minimum(bottom[:,:,:2]+half, (key+1)*.02)
        aa = np.where(soft,np.prod(np.maximum(hi-lo,0),axis=2),0.)
        zz = np.where(aa>0,bottom[:,:,2],np.inf)
        leg = zz.argmin(axis=1)
        valid = np.isfinite(zz[row,leg])
        normal[:,j] = zz[row,leg]/H
        area[:,j] = aa[row,leg]
        owner[:,j] = np.where(valid,leg,-1)
    return normal, area, owner


def reference_run(normal, dt, endpoint_repeat=1):
    law = make_reference()
    e = np.zeros((normal.shape[1],2)); q = e.copy()
    out = np.empty((*normal.shape,6))
    for k, n in enumerate(normal):
        for _ in range(endpoint_repeat):
            fe,fq,fs,_ = law.step(e,q,np.zeros(len(n)),dt,np.zeros(len(n),bool))
            ii = np.flatnonzero(n <= fe[:,0])
            if len(ii):
                ce,cq,cs,_ = law.step(e[ii],q[ii],n[ii],dt,np.ones(len(ii),bool))
                press = cs[:,0] < 0
                at = ii[press]
                fe[at],fq[at],fs[at] = ce[press],cq[press],cs[press]
            e,q = fe,fq
        out[k] = np.c_[e,q,fs]
    return out


def material_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def prepare_case(name, old_metrics):
    start = time.perf_counter()
    where, pattern, role = CASES[name]
    directory, = where.glob(pattern)
    final = np.load(directory/'material-final.npz')
    all_keys = final['keys'][final['ever_loaded'].astype(bool)]
    known = {tuple(k) for k in old_metrics.get(name,[])}
    keys = np.asarray([k for k in all_keys if tuple(k) not in known])
    blocks = sorted(directory.glob('material-[0-9]*.npz'))
    foot = np.concatenate([np.load(f)['foot'] for f in blocks])
    failures = material_events(directory/'material-failures.jsonl')
    if failures:
        bad_tick = min(round(e['time_s']/DT) for e in failures)
        foot = foot[np.rint(foot[:,1]/DT).astype(int) < bad_tick]
    t = np.rint(foot[:,0]/DT).astype(np.int64)
    bottom = foot[:,6:18].reshape(-1,4,3)
    eligible = (bottom[:,:,0]>=.8+np.sqrt(.001)/2) & (bottom[:,:,2]<=0)
    first = max(0, (int(np.flatnonzero(eligible.any(axis=1))[0])//10-1)*10)
    normal,area,owner = geometry(bottom[first:],keys)
    t = t[first:]
    coarse = np.flatnonzero((t%10)==0)
    fine = reference_run(normal,DT)
    observed = reference_run(normal[coarse],.002)
    # One additional counterfactual uses the very same endpoint/area as the
    # observer throughout each 2ms window, retaining 10 backward-Euler steps.
    held = reference_run(normal[coarse],DT,10)
    key_index = {tuple(k):j for j,k in enumerate(keys)}
    aligned = np.zeros(8)
    rows_checked = 0
    for block in blocks:
        cells = np.load(block)['cells']
        if not len(cells):
            continue
        samples = np.rint(cells[:,0]/DT).astype(int)-t[0]-1
        col = np.fromiter((key_index.get(tuple(x),-1) for x in cells[:,1:3].astype(int)),int,count=len(cells))
        keep = (samples>=0) & (samples<len(t)) & (col>=0)
        i,j = samples[keep],col[keep]
        if not len(i):
            continue
        state = fine[i,j]
        force = np.maximum(-state[:,4],0)*area[i,j]*FORCE_SCALE
        loaded = force>1e-8
        target = cells[keep]
        pred = np.c_[state,loaded,force]
        truth = np.c_[target[:,3:10],target[:,12]]
        aligned = np.maximum(aligned, np.max(np.abs(pred-truth),axis=0))
        rows_checked += len(i)
    if aligned[:4].max()>1e-9 or aligned[4:6].max()>1e-6 or aligned[6] or aligned[7]>1e-6:
        raise ValueError(f'{name}: reference and published cell histories differ {aligned}')
    logged = material_events(directory/'material-events.jsonl')
    events={}
    for k in keys:
        obs = [e for e in logged if e.get('world_key')==k.tolist() and round(e['time_s']/DT)<=int(t[-1])+1]
        events[','.join(map(str,k))]=[(e.get('event_id'),e['type'],e.get('leg'),round(e['time_s'],6),bool(e.get('cross_leg'))) for e in obs]
    # Exact complete constitutive inputs, excluding sole area from the strain
    # boundary; area is retained separately when calculating every cell force.
    signatures = [hashlib.sha256(np.ascontiguousarray(normal[:,i]).tobytes()).hexdigest() for i in range(len(keys))]
    np.savez_compressed(RAW/f'{name}-histories.npz',ticks=t,coarse=coarse,keys=keys,normal=normal,area=area,owner=owner,
                        reference_fine=fine,reference_observed=observed,reference_endpoint_fine=held)
    rec = dict(job=directory.name,source=str(directory),role=role,ever_loaded=len(all_keys),new_keys=len(keys),
               previously_audited=[list(k) for k in sorted(known)],begin_tick=int(t[0]),final_applied_tick=int(t[-1])+1,
               excluded_terminal_candidates=failures,regular_observer_end_tick=int(t[coarse[-1]]),
               fine_steps=len(t),observer_steps=len(coarse),original_rows_checked=rows_checked,
               max_original_e_q_stress_loaded_force_error=aligned.tolist(),unique_full_boundaries=len(set(signatures)),
               full_boundary_sha256=dict(zip([','.join(map(str,k)) for k in keys],signatures)),
               events=events,reference_and_assembly_wall_s=time.perf_counter()-start)
    write(COMPACT/f'{name}-inventory.json',rec)
    print(name, len(keys),len(set(signatures)),len(t),round(rec['reference_and_assembly_wall_s'],2),flush=True)


if __name__=='__main__':
    import sys
    RAW.mkdir(parents=True,exist_ok=True);COMPACT.mkdir(parents=True,exist_ok=True)
    review=json.loads((REVIEW/'pann-replay-summary.json').read_text())
    known={'N48':review['cases']['r48']['keys'],'F52':review['cases']['r52']['keys']}
    for name in (sys.argv[1:] or CASES):
        prepare_case(name,known)
