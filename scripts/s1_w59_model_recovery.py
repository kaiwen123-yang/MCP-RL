"""FP64 whole-foot fit of only observable low-dimensional release/recovery corrections."""
import gzip
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from s1_w55_c_data import DT, R12, WEIGHTS, geometry
from s1_w60_cpu_observer import CPUColumns
from score_s1_w57_material import material_response, truth_at
from s1_w59_model_fields import BG
from s1_w59_model_bank import OUT, ROOT, WORK, write

M2=ROOT.parent/'MCP-RL-w58-m2'
W57=ROOT.parent/'MCP-RL-w57'
SPEC=json.loads((ROOT/'experiments/terrain_memory/w58/material.json').read_text())['recorded_episodes']


def read(p): return json.load(gzip.open(p,'rt'))


def episodes():
    for case,ep in zip(('a','b'),SPEC['development']):
        a=W57/'outputs'/R12/'w57/m'
        b=W57/'results'/R12/'w57/m'
        yield ep, ('fit' if case=='a' else 'development_validation'),ROOT.parent/'MCP-RL-w56-r60'/'outputs'/R12/ep, a/(case+'-M0-starts.json.gz'),a/(case+'-M1-starts.json.gz'), b/(case+'-held-forecasts.json.gz')
    for ep in SPEC['calibration']+SPEC['retained']:
        a=M2/'work/w58-m2'; b=M2/'results'/R12/'w58/m2'
        role=('fit' if ep in SPEC['calibration'] else 'development_validation' if '-b-' in ep else 'same_case_diagnostic')
        yield ep,role,W57/'outputs'/R12/ep,a/(ep+'-M0-starts.json.gz'),a/(ep+'-M1-starts.json.gz'),b/(ep+'-forecasts.json.gz')


def propagated(law,x,normal):
    x=x.copy()
    for boundary in normal:
        active=np.isfinite(boundary)
        free=law.step(x[:,:2],x[:,2:4],np.zeros(len(x)),DT,np.zeros(len(x),bool))
        contact=active&(np.where(active,boundary,0)<=free[0][:,0])
        if contact.any():
            load=law.step(x[contact,:2],x[contact,2:4],boundary[contact],DT,np.ones(contact.sum(),bool))
            free[0][contact],free[1][contact]=load[0],load[1]
        x=np.column_stack(free[:2])
    return x


def responses(law,x,z,areas):
    free=law.step(x[:,:2],x[:,2:4],np.zeros(len(x)),DT,np.zeros(len(x),bool))[0][:,0]
    stress=law.step(x[:,:2],x[:,2:4],z/.05,DT,np.ones(len(x),bool))[2][:,0]
    return 25000.*areas*np.where(z/.05<=free,np.maximum(-stress,0.),0.)


def prepare(law,entry):
    episode,role,raw,p0,p1,scores=entry
    all0={q['event_id']:q for q in read(p0)}
    all1=read(p1);old=read(scores)
    forward=old.get('forward',old.get('rows',[]))
    old_short={v['event_id']:v for v in forward if v['mode']=='M1'}
    old_zero={v['event_id']:v for v in forward if v['mode']=='M0'}
    old_reload={v['event_id']:v for v in old['release_to_reload'] if v['mode']=='M1'}
    blocks=sorted(raw.glob('material-[0-9]*.npz'))
    foot=np.concatenate([np.load(f)['foot'] for f in blocks])
    bottom=foot[:,6:18].reshape(-1,4,3)
    eligible=[q for q in all1 if 'release_state' in q and q['event']['cross_leg']]
    # Saved short-horizon queries were keyed at the later evaluation time.
    # Restore any omitted *current* geometric cells from a measured-only M1 replay.
    from collect_s1_w57_material import geometry_keys, PARAMETERS, state_at
    from s1_w57_material import W57MaterialObserver
    missing={}
    for q in eligible:
        active=set(geometry_keys(bottom[q['event_tick'],q['leg'],:2]))
        need=sorted(active-set(map(tuple,q['keys'])))
        if need:missing.setdefault(q['start'],[]).append((q,need))
    if missing:
        obs=W57MaterialObserver(PARAMETERS,WEIGHTS,mode='M1')
        for tick in range(0,max(missing)+1,10):
            for q,need in missing.get(tick,[]):
                q['observed_state'].extend(state_at(obs.ground,need));q['keys'].extend([list(k) for k in need])
            if tick==max(missing):break
            for v in foot[tick:tick+10]:
                obs.completed(v[0],v[1],v[6:18].reshape(4,3),v[18:30].reshape(4,3),v[32:42:3])
            obs.observe(v[1],None,None)
    truth=truth_at(blocks,[dict(event_id=q['event_id'],target=q['release_publication_tick'],keys=[q['key']]) for q in eligible])
    rows=[]
    for q in eligible:
        eid=q['event_id'];k=q['event_tick'];src=q['release_publication_tick'];leg=q['leg'];indices=q['keys']
        a=indices.index(q['key']);z=bottom[k,leg,2]
        normal,area,owner=geometry(bottom[q['start']:q['target']+1],np.array(indices))
        ev=k-q['start'];state=propagated(law,np.asarray(q['observed_state'])[:,:4],normal[:ev])
        weights=np.where(owner[ev]==leg,area[ev],0.)
        if abs(weights.sum()-.001)>1e-12:raise ValueError(f'Incomplete source footprint at {episode}/{eid}')
        current=responses(law,state,np.full(len(state),z),weights)
        m1_cell=q['event']['normal_N']+old_reload[eid]['errors']['next_cell_force_N']
        # Aggregate measured foot label. Only the available anchor is replaced.
        # Remaining cells are carried by the contemporaneous M1 observer.
        rows.append(dict(id=eid,key=q['key'],gap=(k-src)*DT,z=z,area=weights[a],
            aggregate_measured=foot[k,32+3*leg],other=current.sum()-current[a],m1=m1_cell,
            m0_release=all0[eid]['release_state'][:4],m1_release=q['release_state'][:4],
            truth_release=truth[eid][0,:4],truth_pre=np.asarray(q['event']['e_before_step']+q['event']['q_before_step']),
            truth_cell=q['event']['normal_N'],m1_reload=old_reload[eid],short=old_short[eid],idx=a,
            continuation=normal[ev:-1,a].copy(),end_z=bottom[q['target'],leg,2],
            end_area=area[-1,a] if owner[-1,a]==leg else 0.,m1_event_start=state[a].copy(),
            family=q['family'],role=role,episode=episode,release_source_id=q['release']['event_id']))
    return rows,[(v,old_zero[k]) for k,v in old_short.items()]


def candidate(params,x0,x1,elapsed):
    blend,rate,_=params
    return (x1+blend*(x0-x1))*np.exp(-rate*elapsed[:,None])


def stats(x):
    x=np.asarray(x);return dict(n=len(x),bias=float(x.mean()),MAE=float(np.mean(abs(x))),p95=float(np.quantile(abs(x),.95)))


def main():
    started=time.time();law=CPUColumns(WEIGHTS);allrows=[];short={}
    for entry in episodes():
        rows,s=prepare(law,entry);allrows.extend(rows);short[entry[0]]=s
        print(entry[0],len(rows),flush=True)
    def arr(k):return np.asarray([r[k] for r in allrows],float)
    x0,x1,delay,z,area=(arr(k) for k in ('m0_release','m1_release','gap','z','area'))
    measured,other=arr('aggregate_measured'),arr('other')
    train=np.array([r['role']=='fit' for r in allrows])
    def predict(params):
        x=candidate(params,x0,x1,delay)
        return np.exp(params[2])*responses(law,x,z,area)
    def error(p):return (other+predict(p)-measured)[train]
    fit=least_squares(error,[0.,1.,0.],bounds=([0.,.25,-.15],[1.,10.,.15]),loss='soft_l1',f_scale=.1,
                      max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    p=fit.x;x=candidate(p,x0,x1,delay);obs=predict(p)
    true0=arr('truth_release');truepre=arr('truth_pre');truthforce=arr('truth_cell')
    ftrue_src=np.exp(p[2])*responses(law,candidate(p,true0,true0,delay),z,area)
    ftrue_pre=np.exp(p[2])*responses(law,truepre,z,area)
    details=[];per={}
    for j,r in enumerate(allrows):
        e=r['episode'];at20=propagated(law,x[j:j+1],r['continuation'][:,None])
        at20m1=propagated(law,r['m1_event_start'][None,:],r['continuation'][:,None])
        df=responses(law,at20,np.array([r['end_z']]),np.array([r['end_area']]))[0]-responses(law,at20m1,np.array([r['end_z']]),np.array([r['end_area']]))[0]
        details.append(dict(episode=e,event_id=r['id'],world_key=r['key'],source_release_id=r['release_source_id'],gap_s=r['gap'],
            source_depth_error_m=float(-.05*((x1[j]+p[0]*(x0[j]-x1[j]))[0]-true0[j,0])),
            m3_preload_depth_error_m=float(-.05*(x[j,0]-truepre[j,0])),
            m1_preload_depth_error_m=r['m1_reload']['errors']['preload_depth_m'],
            m1_foot_force_error_N=r['other']+r['m1']-r['aggregate_measured'],
            m3_foot_force_error_N=r['other']+obs[j]-r['aggregate_measured'],
            m1_cell_force_error_N=r['m1']-r['truth_cell'],m3_cell_force_error_N=obs[j]-r['truth_cell'],
            source_component_N=obs[j]-ftrue_src[j],free_component_N=ftrue_src[j]-ftrue_pre[j],
            reload_component_N=ftrue_pre[j]-truthforce[j],
            short_change_N=df))
    for ep,role,*_ in episodes():
        grp=[r for r in details if r['episode']==ep];change={r['event_id']:r['short_change_N'] for r in grp}
        shorts=[r for r,a in short[ep] if r['family']!='release']
        short0=[a for r,a in short[ep] if r['family']!='release']
        per[ep]=dict(role=role,events=len(grp),**{k:stats([r[k] for r in grp]) for k in (
            'source_depth_error_m','m1_preload_depth_error_m','m3_preload_depth_error_m','m1_foot_force_error_N','m3_foot_force_error_N','m1_cell_force_error_N','m3_cell_force_error_N','source_component_N','free_component_N','reload_component_N')},
            short_M0=stats([r['errors']['future_force_N'] for r in short0]),
            short_M1=stats([r['errors']['future_force_N'] for r in shorts]),
            short_M3=stats([r['errors']['future_force_N']+change.get(r['event_id'],0.) for r in shorts]),
            opposite_initial_vs_free=int(sum(r['source_component_N']*r['free_component_N']<0 for r in grp)))
    val=[v for v in per.values() if v['role']=='development_validation']
    # Exact prior adoption criterion is evaluated per whole validation record.
    # The strict short-horizon preservation below neither widens it nor reads W59 sealed states.
    adopt=all(v['m3_cell_force_error_N']['MAE']<v['m1_cell_force_error_N']['MAE'] and
              v['m3_foot_force_error_N']['MAE']<v['m1_foot_force_error_N']['MAE'] and
              v['m3_preload_depth_error_m']['MAE']<=v['m1_preload_depth_error_m']['MAE'] and
              v['short_M3']['MAE']<=v['short_M1']['MAE']+.1*abs(v['short_M0']['MAE']-v['short_M1']['MAE']) for v in val)
    target=WORK/'material-events.json.gz'
    with gzip.open(target,'wt') as f:json.dump(details,f,allow_nan=False)
    write(OUT/'recovery.json',dict(schema='W59-first-cross-foot-reload-FP64',fitted=bool(fit.success),fit_rows=int(train.sum()),
        parameters=dict(release_prior_blend=float(p[0]),positive_recovery_rate_s_inv=float(p[1]),contact_gain=float(np.exp(p[2]))),
        bounds=[[0.,1.],[.25,10.],[float(np.exp(-.15)),float(np.exp(.15))]],law='interpolate two causal released estimates; positive common free-mode rate; unmodified PANN unilateral solve with bounded positive contact gain at first cross-foot reload',
        fit_target='measured whole-foot resultant, other footprint cells predicted by contemporaneous M1; no true cell force or true internal strain in fit',
        scope='One identified released anchor per saved footprint; short prediction explicitly rescored through the remaining measured 5kHz boundary. Non-cross-foot and loaded-only paths retain M1.',
        diagnostics_truth_only=['source and free initial state','preload depth','single cell normal component'],
        fit_nfev=fit.nfev,fit_jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist(),per_episode=per,
        acceptance='per B recording: improve cross-foot whole force and cell force, preserve preload depth and original short-force allowance 0.1*abs(M0-M1)',
        offline_candidate_accepted=adopt,accepted_physical=False,elapsed_s=time.time()-started,**BG))
    print('material done',p.tolist(),adopt,flush=True)

if __name__=='__main__':main()
