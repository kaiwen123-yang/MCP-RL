"""Held-out physics-conditioned material forecasts on complete W56 A/B recordings."""
import argparse
import ctypes as ct
from collections import defaultdict
import gzip
import json
import os
import resource
import time
from types import SimpleNamespace

import numpy as np

from collect_s1_w57_material import OUT, ART, DT, PARAMETERS, WEIGHTS, ROOT, geometry_keys, load_episode
from s1_w55_c_data import geometry
from s1_w60_cpu_observer import CPUColumns
from tm02.s1_material import ReferenceColumns
from tm02.s1_material_prediction import inverse


def read_gzip(p):
    with gzip.open(p,'rt') as f:
        return json.load(f)


def batch(model, initial, boundary, lengths, free_only, label, model_cpu_subset):
    import torch
    arrays = [np.ascontiguousarray(a,dtype=d) for a,d in zip(
        (model,initial,boundary,lengths),(float,float,float,np.int32))]
    n = len(lengths)
    p = ct.c_void_p
    fn = ct.CDLL(str(ROOT/'work/w57-m/forecast_gpu.so')).w57_forecast_gpu
    fn.argtypes = [p,p,p,p,ct.c_int,ct.c_double,ct.c_int,p,p]
    first = time.perf_counter()
    cpu0 = time.process_time()
    torch.cuda.reset_peak_memory_stats()
    ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    device=[torch.from_numpy(a).cuda() for a in arrays]
    output=torch.empty((n,6),dtype=torch.float64,device='cuda')
    ev[1].record()
    status=fn(*(a.data_ptr() for a in device),n,DT,free_only,output.data_ptr(),torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'{label} CUDA returned {status}')
    ev[2].record()
    values=output.cpu().numpy()
    ev[3].record()
    ev[3].synchronize()
    elapsed=time.perf_counter()-first
    cpu_host=time.process_time()-cpu0
    parts=[ev[i].elapsed_time(ev[i+1])*.001 for i in range(3)]
    peak=int(torch.cuda.max_memory_allocated())
    ca=np.asarray(model_cpu_subset,dtype=int)
    sub=[arrays[0],np.ascontiguousarray(arrays[1][ca]),
         np.zeros(1) if free_only else np.ascontiguousarray(arrays[2][:,ca]),np.ascontiguousarray(arrays[3][ca])]
    out=np.empty((len(ca),6))
    host=ct.CDLL(str(ROOT/'work/w57-m/forecast_cpu.so')).w57_forecast_cpu
    host.argtypes=[p,p,p,p,ct.c_int,ct.c_double,ct.c_int,p]
    host.restype=None
    tick=time.perf_counter();host_cpu=time.process_time()
    host(*(a.ctypes.data for a in sub),len(ca),DT,free_only,out.ctypes.data)
    sec=time.perf_counter()-tick
    if not np.isfinite(values).all() or max(np.max(values[:,4]),np.max(out[:,4]))>1e-7:
        raise RuntimeError(f'{label}: inadmissible numerical residual')
    return values,dict(label=label,device=torch.cuda.get_device_name(),paths=n,
       logical_column_steps=int(np.sum(lengths,dtype=np.int64)),max_observed_abs_strain=float(values[:,5].max()),
       max_newton_residual_MPa=float(values[:,4].max()),gpu_allocation_H2D_s=parts[0],gpu_kernel_s=parts[1],
       D2H_s=parts[2],gpu_end_to_end_after_host_assembly_s=elapsed,host_cpu_core_s=cpu_host,
       torch_peak_allocated_bytes=peak,gpu_paths_per_wall_s=n/elapsed,cpu_comparison_paths=len(ca),
       cpu_comparison_wall_s=sec,cpu_comparison_core_s=time.process_time()-host_cpu,
       cpu_gpu_max_abs_e_q=float(np.max(np.abs(out[:,:4]-values[ca,:4]))))


def material_response(law, state, normal, areas):
    e,q=state[:,:2],state[:,2:4]
    n=len(e)
    free=law.step(e,q,np.zeros(n),DT,np.zeros(n,bool))[0][:,0]
    def force(z):
        stress=law.step(e,q,np.full(n,z/.05),DT,np.ones(n,bool))[2][:,0]
        return float(np.dot(areas,np.where(z/.05<=free,np.maximum(-stress,0.),0.))*25000.)
    now=force(normal)
    tangent=(force(normal-1e-9)-now)/1e-9
    return now,tangent


def force_observability(law, state, z, areas):
    # Conditional local Jacobian of one measured resultant with respect to
    # the two prior hidden strains in each owned world cell, at fixed kinematics.
    e,q=state[:,:2],state[:,2:4]
    size=len(e)
    fixed=np.ones(size,bool)
    def cell_forces(hidden):
        free=law.step(e,hidden,np.zeros(size),DT,~fixed)[0][:,0]
        stress=law.step(e,hidden,np.full(size,z/.05),DT,fixed)[2][:,0]
        return areas*25000.*np.where(z/.05<=free,np.maximum(-stress,0.),0.)
    h=1e-7
    columns=[]
    for axis in range(2):
        perturbation=np.zeros_like(q)
        perturbation[:,axis]=h
        columns.append((cell_forces(q+perturbation)-cell_forces(q-perturbation))/(2*h))
    s=float(np.linalg.norm(columns))
    force_resolution=1e-7*25000.*areas.sum()
    rank=int(s*h>force_resolution)
    return dict(owned_cells=size,prior_q_components=2*size,local_force_rank=rank,
        local_nullity=2*size-rank,singular_value_N_per_unit_internal_strain=s,
        finite_difference_internal_strain=h,solver_force_resolution_N=float(force_resolution))


def invert(law, state, areas, load, dt):
    g=SimpleNamespace(e=state[:,:2],q=state[:,2:4],law=law,h=.05,gamma=.025)
    return inverse(g,(np.arange(len(state)),areas),float(load),dt)[0]


def truth_at(blocks, queries):
    by_block=defaultdict(list)
    result={}
    for q in queries:
        if q['target']:
            by_block[(q['target']-1)//5000].append(q)
    for block,requests in by_block.items():
        with np.load(blocks[block]) as z:
            rows=z['cells']
        ts=np.rint(rows[:,0]/DT).astype(int) if len(rows) else np.zeros(0,int)
        for q in requests:
            tick=q['target']
            at=rows[ts.searchsorted(tick):ts.searchsorted(tick,'right')]
            known={tuple(v[1:3].astype(int)):v[3:9] for v in at}
            result[q['event_id']]=np.array([known.get(tuple(key),np.zeros(6)) for key in q['keys']])
    return result


def execution_targets(base):
    out={}
    for line in (base/'w55-f/normal-execution.jsonl').open():
        d=json.loads(line)
        out[round(d['consume_s']/DT),d['leg']]=d
    return out


def forward_bootstrap(case, queries_by_mode, foot, blocks, base, law):
    bottom=foot[:,6:18].reshape(-1,4,3)
    truth=truth_at(blocks,queries_by_mode['M0'])
    records=execution_targets(base)
    init=[]; boundaries=[]; offsets={}; last_area={};previous_area={};assembly=time.perf_counter()
    for mode,queries in queries_by_mode.items():
        for q in queries:
            start,end=q['start'],q['target']
            times=bottom[start:end+1]
            n,a,owner=geometry(times,np.array(q['keys']))
            i=len(init)
            init.extend(np.array(q['observed_state'])[:,:4])
            boundaries.append(n[:-1])
            ei=q['event_id']
            offsets[mode,ei]=slice(i,len(init))
            last_area[ei]=np.where(owner[-1]==q['leg'],a[-1],0.)
            previous_area[ei]=np.where(owner[-2]==q['leg'],a[-2],0.)
    n=np.column_stack(boundaries)
    host_assembly=time.perf_counter()-assembly
    m=np.array(init)
    all_rows=np.arange(len(init))
    x,timing=batch(law.model,m,n,np.full(len(m),100),False,
                   f'{case}/20ms for both M states',all_rows)
    timing['boundary_and_array_assembly_s']=host_assembly
    reference=ReferenceColumns(PARAMETERS)
    last_release={}
    next_load_after_release={}
    for line in (base/'material-events.jsonl').open():
        event=json.loads(line)
        key=tuple(event['world_key'])
        if event['type']=='release':
            last_release[key]=event['event_id']
        elif key in last_release:
            next_load_after_release[last_release.pop(key)]=round(event['time_s']/DT)
    result=[]
    for mode,queries in queries_by_mode.items():
        for q in queries:
            ei=q['event_id'];leg=q['leg'];end=q['target'];keys=q['keys']
            anchor=keys.index(q['key'])
            state=x[offsets[mode,ei],:4]
            true_state=truth[ei][:,:4]
            a=last_area[ei];owned=a>0
            f_actual=float(foot[end,32+leg*3])
            error={'anchor_depth_m':float(-.05*(state[anchor,0]-true_state[anchor,0])),
                   'anchor_internal_q11':float(state[anchor,2]-true_state[anchor,2])}
            uninterrupted_release=(q['family']=='release' and
                next_load_after_release.get(ei, np.inf)>end)
            if uninterrupted_release:
                error['continuously_unloaded_anchor_depth_m']=error['anchor_depth_m']
            forecast_force=0.
            identifiable=None
            if owned.any():
                z=bottom[end,leg,2]
                forecast_force,k=material_response(law,state[owned],z,a[owned])
                reference_force,kref=material_response(reference,true_state[owned],z,a[owned])
                error['tangent_N_per_m']=k-kref
                error['reference_vs_recorded_force_N']=reference_force-f_actual
                resolved_load=f_actual>1e-7*25000.*a[owned].sum()
                if resolved_load and mode=='M0':
                    identifiable=force_observability(law,state[owned],z,a[owned])
                if resolved_load:
                    delta=invert(law,state[owned],a[owned],f_actual,DT)
                    error['common_sole_sink_m']=float(delta+z)
            error['future_force_N']=forecast_force-f_actual
            wbc=records.get((end,leg))
            dref=None
            if wbc:
                pa=previous_area[ei];keep=pa>0
                # Match archived total owned area; an unrepresented border cell invalidates the WBC check.
                if keep.any() and abs(pa.sum()-wbc['owned_area_m2'])<1e-14:
                    delta0=invert(law,state[keep],pa[keep],wbc['measured_N'],.002)
                    delta1=invert(law,state[keep],pa[keep],wbc['target_N'],.002)
                    dref=dict(measured_N=wbc['measured_N'],target_N=wbc['target_N'],
                         predicted_increment_m=delta0-delta1,archived_increment_m=wbc['measured_delta_m']-wbc['target_delta_m'])
            result.append(dict(mode=mode,case=case,event_id=ei,leg=leg,
                 base_foot_interval_tick=q['foot_interval_onset'],family=q['family'],world_key=q['key'],
                 source_publication_id=q['publication_id'],source_publication_s=q['publication_s'],
                 target_left_s=float(foot[end,0]),target_end_s=float(foot[end,1]),
                 measured_force_N=f_actual,predicted_force_N=forecast_force,
                 continuously_unloaded_after_actual_release=bool(uninterrupted_release),
                 aggregate_force_observability=identifiable,errors=error,wbc=dref))
    return result,timing


def reload(case, queries_by_mode, foot, law):
    rows=[];lengths=[];initial=[]
    for mode,queries in queries_by_mode.items():
        for q in queries:
            if 'release_state' in q:
                lengths.append(q['event_tick']-q['release_publication_tick'])
                initial.append(q['release_state'][:4])
                rows.append((mode,q))
    # Full original .2 ms step; a bit-identical free state can remain stationary.
    counts=np.array(lengths,dtype=np.int32)
    cpu_ids=[]
    for mode in ('M0','M1'):
        for family in ('same_foot_reload','cross_foot_revisit','long_recovery'):
            family_rows=[j for j,(m,q) in enumerate(rows) if m==mode and q['family']==family]
            if family_rows:
                cpu_ids += [family_rows[0],max(family_rows,key=lambda j:counts[j])]
    values,timing=batch(law.model,np.array(initial),np.zeros(1),counts,True,
                         f'{case}/release-to-reload for both M states',np.unique(cpu_ids))
    bottom=foot[:,6:18].reshape(-1,4,3)
    scores=[]
    for j,(mode,q) in enumerate(rows):
        e=q['event'];leg=q['leg'];k=q['event_tick']
        state=values[j:j+1,:4]
        area=float(e['area_m2'])
        z=bottom[k,leg,2]
        f,stiff=material_response(law,state,z,np.array([area]))
        true_state=np.array([e['e_before_step']+e['q_before_step']])
        _,ref_stiff=material_response(ReferenceColumns(PARAMETERS),true_state,z,np.array([area]))
        stress=-float(e['normal_N'])/(25000.*area)
        pred_e=law.step(state[:,:2],state[:,2:4],np.array([stress]),DT,np.zeros(1,bool))[0][0,0]
        scores.append(dict(mode=mode,case=case,event_id=q['event_id'],source_plant_release_event_id=q['release']['event_id'],
            observer_history_at_source=q['source_observed_history'],world_key=q['key'],leg=leg,family=q['family'],
            source_publication_s=q['release_publication_tick']*DT,target_end_s=e['time_s'],gap_steps=int(counts[j]),
            errors=dict(preload_depth_m=float(-.05*(state[0,0]-true_state[0,0])),
                       preload_internal_q11=float(state[0,2]-true_state[0,2]),next_cell_force_N=float(f-e['normal_N']),
                       next_cell_conditional_sink_m=float(-.05*pred_e+z),next_cell_tangent_N_per_m=float(stiff-ref_stiff))))
    return scores,timing


def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['a','b']);a=p.parse_args()
    start=time.time();cpu0=time.process_time()
    base,blocks,foot=load_episode(a.case)
    qs={m:read_gzip(ART/f'{a.case}-{m}-starts.json.gz') for m in ('M0','M1')}
    law=CPUColumns(WEIGHTS)
    rows,ft=forward_bootstrap(a.case,qs,foot,blocks,base,law)
    later,rt=reload(a.case,qs,foot,law)
    path=OUT/f'{a.case}-held-forecasts.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as fp:
        json.dump(dict(episode=base.name,rows=rows,release_to_reload=later),fp,allow_nan=False,separators=(',',':'))
    j=dict(case=a.case,source=base.name,pid=os.getpid(),started_unix_s=start,ended_unix_s=time.time(),
        host_cpu_core_s=time.process_time()-cpu0,process_max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        unique_publications=len(set(r['source_publication_id'] for r in rows)),forward_gpu_cpu=ft,free_gpu_cpu=rt,
        deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False,new_robot_episodes=0,new_policy_transitions=0)
    (OUT/f'{a.case}-forecast-resources.json').write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':a.case,'elapsed_s':j['ended_unix_s']-start,'forward':len(rows),'release':len(later)}),flush=True)


if __name__=='__main__':
    main()
