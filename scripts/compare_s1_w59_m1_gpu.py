"""Compare complete 500 Hz M1 observation on recorded W59 5 kHz inputs.

Both observers start from a virgin reference or the same registered pre-control
snapshot. Their states advance independently; no simulated physics is stepped.
"""
import argparse
import faulthandler
import signal
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from zoneinfo import ZoneInfo

import numpy as np

from s1_w57_material import W57MaterialObserver
from s1_w60_cpu_observer import CallMarker
from s1_w59_m1_evidence import load_queries
from s1_w59_m1_hybrid import HybridObserver


ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
RAW = ROOT/'outputs'/REL
COMPACT = ROOT/'results'/REL/'w59'
WEIGHTS = ROOT/'work/assets/B_random1000--ideal-potentials.npz'
PARAMETERS = ROOT/'experiments/tm02/r2-multihistory.json'
DEFAULT_OUTPUT = ROOT/'work/w59/m1-gpu-study/replay'
LEGS = ('FL','FR','RL','RR')
STATS = ('max_strain','max_stress','max_rate','max_residual')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def max_abs(x):
    return float(np.max(np.abs(x), initial=0.))


def signature(event):
    return (event['event_id'],event['kind'],tuple(event['world_key']),
            event['time_s'],event['owner'],event['previous_load_event_id'])


def open_observers(propagation_library=None):
    par = read(PARAMETERS)['reference_material']['parameters']
    start = time.perf_counter()
    cpu = W57MaterialObserver(par, WEIGHTS, mode='M1')
    cpu_init = time.perf_counter()-start
    start = time.perf_counter()
    gpu = HybridObserver(par, WEIGHTS, mode='M1', propagation_library=propagation_library)
    return (cpu,gpu), [cpu_init,time.perf_counter()-start]


class Trace:
    """Observe the arrays that the production ledger consumes, without copying in the timer."""
    def __init__(self, observer):
        self.original = observer._ledger
        observer._ledger = self.record
        self.clear()

    def clear(self):
        self.ids = np.empty(0,dtype=int)
        self.owners = np.empty((10,0),dtype=np.int32)
        self.active = np.empty((10,0),dtype=np.uint8)

    def record(self, ids, owners, active, ends):
        self.ids, self.owners, self.active = ids,owners,active
        self.original(ids,owners,active,ends)


def publish(obs, starts, ends, feet, speed, fz):
    begin = time.perf_counter_ns()
    for row in range(10):
        obs.completed(starts[row],ends[row],feet[row],speed[row],fz[row])
    obs.observe(ends[-1],feet[-1],speed[-1])
    return (time.perf_counter_ns()-begin)*1.e-6


def trace_diff(a,b):
    both = np.union1d(a.ids,b.ids)
    if np.array_equal(a.ids,b.ids):
        aa,bb = a.active,b.active
        oa,ob = a.owners,b.owners
    else:
        aa,bb = np.zeros((2,10,len(both)),dtype=np.uint8)
        oa,ob = np.full((2,10,len(both)),-1,dtype=np.int32)
        ai,bi = np.searchsorted(both,a.ids),np.searchsorted(both,b.ids)
        aa[:,ai],bb[:,bi] = a.active,b.active
        oa[:,ai],ob[:,bi] = a.owners,b.owners
    mismatch = (aa!=bb) | ((aa!=0) & (bb!=0) & (oa!=ob))
    return int(mismatch.sum()),int(np.count_nonzero(aa)),int(np.count_nonzero(bb)),10*len(both)


def extract_packet(mat,phys,force_indices):
    if not np.array_equal(mat[:,:2],phys[:,:2]):
        raise ValueError('Material and force rows do not have the same 5 kHz clocks')
    return (phys[:,0],phys[:,1],mat[:,6:18].reshape(-1,4,3),
            mat[:,18:30].reshape(-1,4,3),phys[:,force_indices])


def anchors():
    reg = read(COMPACT/'action/registry-v2.json')
    yield from reg['states']['development']


def snapshot_comparison(dest, propagation_library=None):
    obs,init = open_observers(propagation_library)
    traces = [Trace(x) for x in obs]
    rows = []
    for order,source in enumerate(anchors()):
        blob = Path(source['snapshot']).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=source['sha256']:
            raise ValueError('Snapshot content has changed')
        record = dict(id=source['snapshot_id'],stratum=source['stratum'],times_ms=[])
        for rep in range(6):
            for o in obs:
                state = pickle.loads(gzip.decompress(blob))
                vars(o.ground).update(state['belief'])
                vars(o).update(state['observer'])
            packet=obs[0].packet
            inputs = tuple(np.asarray([r[j] for r in packet]) for j in range(4)) + (np.asarray(obs[0].force_measurements),)
            for o in obs:
                o.packet=[];o.force_measurements=[]
            times = [None,None]
            for i in ((0,1) if (order+rep)%2==0 else (1,0)):
                traces[i].clear()
                times[i] = publish(obs[i],*inputs)
            record['times_ms'].append(times)
        a,b = obs
        differences=trace_diff(*traces)
        record.update(columns=[len(tr.ids) for tr in traces],force_max_abs_N=max_abs(a.last_forces-b.last_forces),
            prior_force_max_abs_N=max_abs(a.last_prior_forces-b.last_prior_forces),
            e_max_abs=max_abs(a.ground.e-b.ground.e),q_max_abs=max_abs(a.ground.q-b.ground.q),
            stress_max_abs_MPa=max_abs(a.ground.stress-b.ground.stress),cell_trace_differences=differences[0],
            cell_tick_denominator=differences[3],events_cpu=[signature(e) for e in a.last_packet_events],
            events_gpu=[signature(e) for e in b.last_packet_events],
            equal_event_chain=a.last_packet_events==b.last_packet_events,
            warm_median_ms=np.median(np.asarray(record['times_ms'])[1:],axis=0).tolist())
        rows.append(record)
    obs[1].close()
    med=np.array([x['warm_median_ms'] for x in rows])
    write(dest/'snapshots.json',dict(startup_s=init,registered_states=len(rows),
        nonempty_states=sum(x['columns'][0]>0 for x in rows),time_ratio=float(med[:,0].sum()/med[:,1].sum()),
        time_scope='Ten native completed inputs plus full published observe call; includes upload, download and stream synchronization. Median of five paired calls per restored state after one initial pass.',
        rows=rows,new_physics_steps=0,clock_shanghai=now()))
    print('Snapshot comparison finished',len(rows),flush=True)


def time_stats(values):
    if not values.size:
        return dict(n=0)
    total=values.sum(axis=0)
    return dict(n=len(values),cpu_s=float(total[0]/1000),hybrid_s=float(total[1]/1000),
        speedup=float(total[0]/total[1]),median_ms=np.median(values,axis=0).tolist(),
        p95_ms=np.percentile(values,95,axis=0).tolist())



class NativeTimer:
    """Time the same native ABI boundary on both observers, retaining backend-specific diagnostics."""
    def __init__(self, original):
        self.original = original
        self.packet = original.w57_packet
        self.packet_wall_s = 0.
        self.packet_calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def w57_packet(self, *args):
        begin = time.perf_counter()
        result = self.packet(*args)
        self.packet_wall_s += time.perf_counter() - begin
        self.packet_calls += 1
        return result


def instrument(case, dest, observers):
    for backend, observer in zip(('cpu', 'hybrid'), observers):
        directory = dest/(case+'-'+backend+'-marker')
        directory.mkdir(exist_ok=True)
        observer.marker = CallMarker(directory)
        observer.native = NativeTimer(observer.native)


def release_markers(observers):
    faulthandler.unregister(signal.SIGUSR1)
    for observer in observers:
        marker = observer.marker
        marker.page.close()
        marker.file.close()
        marker.stacks.close()


def query_pair(observers, packet, physical_t, registered, sequence):
    t, xy, recorded = registered
    if abs(t-physical_t)>1.e-9:
        raise ValueError(f'Recovery query and M1 publication clocks differ: {t}, {physical_t}')
    predictions, masks, costs = [None,None], [None,None], [0.,0.]
    for i in ((0,1) if sequence%2==0 else (1,0)):
        begin = time.perf_counter_ns()
        predicted, mask = observers[i].recovery(xy)
        costs[i] = (time.perf_counter_ns()-begin)*1.e-6
        predictions[i] = predicted.reshape(12)
        masks[i] = mask
    return dict(packet=packet, time_s=float(t), xy_m=xy.tolist(), recorded_m=recorded.tolist(),
                cpu_m=predictions[0].tolist(), hybrid_m=predictions[1].tolist(),
                masks=[x.tolist() for x in masks], times_ms=costs,
                backend_max_abs_m=max_abs(predictions[0]-predictions[1]),
                cpu_vs_recorded_max_abs_m=max_abs(predictions[0]-recorded),
                hybrid_vs_recorded_max_abs_m=max_abs(predictions[1]-recorded),
                differing_foot_masks=int(np.count_nonzero(masks[0]!=masks[1])))


def query_summary(rows, times, registered):
    elapsed = np.asarray([r['times_ms'] for r in rows]).reshape(-1,2)
    complete = times.copy()
    for row in rows:
        complete[row['packet']-1] += row['times_ms']
    return dict(registered_queries=registered,completed_paired_queries=len(rows),
                compared_height_values=12*len(rows),compared_foot_masks=4*len(rows),
                differing_foot_masks=sum(r['differing_foot_masks'] for r in rows),
                max_abs_cpu_hybrid_m=max((r['backend_max_abs_m'] for r in rows),default=None),
                max_abs_cpu_recorded_m=max((r['cpu_vs_recorded_max_abs_m'] for r in rows),default=None),
                max_abs_hybrid_recorded_m=max((r['hybrid_vs_recorded_max_abs_m'] for r in rows),default=None),
                separate_query_timing=time_stats(elapsed),complete_M1_including_queries=time_stats(complete),
                query_timing_scope='Each original post-publication recovery call with the actual planned xy, its full host lookup, native FP64 recovery, mask, markers and return. Query reconstruction and numerical comparison excluded.')


def replay_case(case,dest,with_queries=False,propagation_library=None):
    name=f'w59-r3-5902-ppo-full-fallback-{case.lower()}-e60-0021'
    raw=RAW/name
    source=read(COMPACT/'episodes'/(name+'.json'))
    expected=source['physical_samples']//10
    if source['stage_status']!='passed' or source['physical_samples']!=300000:
        raise ValueError('The selected original E60 is not a closed full run')
    force_indices=[source['physical_columns'].index(f'grf_{leg}_z') for leg in LEGS]
    cfg=read(raw/'w55-f/manifest.json')
    gate=cfg['config']['contact_events']
    thresholds=np.array([gate['touch_off_N'],gate['touch_on_N'],gate['load_on_bodyweight_fraction']*cfg['model_mass_kg']*9.81])
    queries=load_queries(raw) if with_queries else {}
    if with_queries and len(queries)!=source['policy']['observer']['query_calls']:
        raise ValueError('Registered recovery query count differs from the source episode')
    query_rows=[]
    obs,init=open_observers(propagation_library)
    if with_queries:
        instrument(case,dest,obs)
    traces=[Trace(o) for o in obs]
    times=np.zeros((expected,2),dtype=np.float64)
    sizes=np.zeros((expected,2),dtype=np.int32)
    diffs=np.zeros((expected,10),dtype=np.float64)
    force=np.zeros((expected,2,10,4),dtype=np.float64)
    measured=np.zeros((expected,10,4),dtype=np.float64)
    soft=np.zeros((expected,10,4),dtype=bool)
    clock=np.zeros(expected)
    phases=np.zeros(expected,dtype=np.int8)
    counts=Counter()
    events=[Counter(),Counter()]
    first=None
    failed=None
    switch_count=np.zeros((2,3),int)
    switch_mismatch=np.zeros(3,int)
    mask_mismatch=np.zeros(3,int)
    last_mask=np.zeros((2,1,4,3),bool)
    chunks=sorted(raw.glob('material-[0-9][0-9][0-9][0-9].npz'))
    with np.load(raw/'trajectory.npz') as z:
        ctrl_time=z['time'];ctrl_phase=z['phase']
    phy=None;current=-1;processed=0;started=now()
    wall_begin=time.perf_counter()
    with gzip.open(dest/(case+'-events.jsonl.gz'),'wt') as log:
        for chunk in chunks:
            with np.load(chunk) as z:
                mat=z['foot']
            if not len(mat):
                continue
            tick=round(mat[0,0]/.0002)
            part=tick//150000
            if part!=current:
                current=part
                with np.load(raw/f'physical-{part:03d}.npz') as z:
                    phy=z['physical']
            offset=tick-round(phy[0,0]/.0002)
            phys=phy[offset:offset+len(mat)]
            inputs=extract_packet(mat,phys,force_indices)
            if tick!=processed*10 or len(mat)%10:
                raise ValueError('A recorded complete packet is missing')
            for j in range(0,len(mat),10):
                k=processed
                segment=slice(j,j+10)
                t=inputs[1][j+9]
                sub=tuple(x[segment] for x in inputs)
                # Alternate which backend sees the host inputs first.
                for i in ((0,1) if k%2==0 else (1,0)):
                    traces[i].clear()
                    try:
                        times[k,i]=publish(obs[i],*sub)
                    except (RuntimeError,ValueError,FloatingPointError) as err:
                        failed=dict(backend=('cpu','hybrid')[i],time_s=float(t),
                                    packet=k,exception=type(err).__name__,message=str(err))
                        break
                if failed:break
                a,b=obs
                if len(a.ground.ids)!=len(b.ground.ids) or (k==0 or len(a.ground.ids)!=previous_size) and a.ground.ids!=b.ground.ids:
                    raise ValueError('World-cell identities differ under identical measured geometry')
                previous_size=len(a.ground.ids)
                sizes[k]=[len(tap.ids) for tap in traces]
                clock[k]=t
                phases[k]=ctrl_phase[np.searchsorted(ctrl_time,t+1.e-9,side='right')-1]
                force[k]=[a.last_forces,b.last_forces]
                soft[k]=mat[segment,2:6].astype(bool)
                measured[k]=sub[-1]
                trace_errors,active_a,active_b,denom=trace_diff(*traces)
                counts.update(trace_different=trace_errors,active_cpu=active_a,active_gpu=active_b,trace_denominator=denom)
                diff=a.last_forces-b.last_forces
                prior=a.last_prior_forces-b.last_prior_forces
                diffs[k]=[max_abs(diff),max_abs(prior),max_abs(a.ground.e-b.ground.e),max_abs(a.ground.q-b.ground.q),
                    max_abs(a.ground.stress-b.ground.stress),max_abs(a.last_correction-b.last_correction),
                    len(a.relaxing.symmetric_difference(b.relaxing)),trace_errors,
                    max_abs(a.last_correction[:,4]-b.last_correction[:,4]),
                    max_abs(np.array([getattr(a.ground,s)-getattr(b.ground,s) for s in STATS]))]
                em=[signature(v) for v in a.last_packet_events],[signature(v) for v in b.last_packet_events]
                if em[0]!=em[1]:
                    counts['packet_event_chain_mismatch']+=1
                    if first is None:first=dict(time_s=t,cpu=em[0],hybrid=em[1])
                for i,batch in enumerate(em):
                    events[i].update(e[1] for e in batch)
                    for e in batch:
                        log.write(json.dumps([i,k,*e],separators=(',',':'))+'\n')
                # The gate inputs here are observer normal forces at the three original thresholds.
                # Actual plant GRFs are fixed inputs in this saved-motion replay.
                mask=force[k,:,:,:,None]>thresholds
                edge=np.diff(np.concatenate((last_mask,mask),axis=1).astype(np.int8),axis=1)
                last_mask=mask[:,-1:].copy()
                switch_count+=np.count_nonzero(edge,axis=(1,2))
                switch_mismatch+=np.count_nonzero(edge[0]!=edge[1],axis=(0,1))
                mask_mismatch+=np.count_nonzero(mask[0]!=mask[1],axis=(0,1))
                processed+=1
                if processed in queries:
                    try:
                        query_rows.append(query_pair(obs,processed,t,queries[processed],len(query_rows)))
                    except (RuntimeError,ValueError,FloatingPointError) as err:
                        failed=dict(stage='recovery',time_s=float(t),packet=processed,
                                    exception=type(err).__name__,message=str(err))
                        break
            print(case,'physical_s',round(float(clock[processed-1]),3) if processed else 0,'packets',processed,flush=True)
            write(dest/(case+'-live.json'),dict(at=now(),processed_packets=processed,registered_packets=expected,
                physical_s=float(clock[processed-1]) if processed else 0,wall_s=time.perf_counter()-wall_begin,
                max_abs_foot_force_N=float(diffs[:processed,0].max(initial=0)),event_chain_differences=counts['packet_event_chain_mismatch']))
            if failed:break
    a,b=obs
    ws=slice(0,processed)
    compact_names=['force_N','prior_force_N','strain','q','stress_MPa','correction_mixed_units','relaxing_members',
        'trace_entries','chosen_tau_s','stored_maxima_mixed_units']
    data=dict(time=clock[ws],phase=phases[ws],columns=sizes[ws],times_ms=times[ws],diffs=diffs[ws],
        diff_names=np.array(compact_names),observer_Fz=force[ws],actual_Fz=measured[ws],is_material=soft[ws])
    np.savez_compressed(dest/(case+'-packets.npz'),**data)
    pred=force[ws,0]-force[ws,1]
    native=source['policy']['observer']
    result=dict(episode=name,started=started,finished=now(),backend_order=['cpu','hybrid'],
        registered_packets=expected,completed_paired_packets=processed,original_5kHz_rows=source['physical_samples'],
        complete=processed==expected and failed is None,failure=failed,init_wall_s=init,
        timing_scope='Ten completed inputs plus one full M1 observe call. FP64 geometry, input packing, host/device copies, GPU synchronization, native/CPU final correction, world cell ledger and host observation publication included. File loading, comparison, logging and WBC/plant excluded.',
        input_scope='Exactly the stored left-boundary foot bottom, rolling velocity and actual rigid/TM02 total normal GRF; observer state initialized virgin, posterior carried across all packets.',
        totals=time_stats(times[ws]),by_columns={tag:time_stats(times[ws][mask]) for tag,mask in [('CPU_empty',sizes[ws,0]==0),('CPU_nonempty',sizes[ws,0]>0)]},
        by_phase={str(int(ph)):time_stats(times[ws][phases[ws]==ph]) for ph in np.unique(phases[ws])},
        exact_5kHz_cell_trace=dict(different=counts['trace_different'],tested_cell_ticks=counts['trace_denominator'],
            cpu_loaded=counts['active_cpu'],hybrid_loaded=counts['active_gpu']),
        force_max_abs_N=float(np.abs(pred).max(initial=0)),force_component_count=int(pred.size),
        force_RMSE_N=float(np.sqrt(np.mean(pred**2))) if pred.size else None,
        per_foot_max_abs_N=np.abs(pred).max(axis=(0,1),initial=0).tolist(),
        material_force_measured_MAE_N=[float(np.abs(force[ws,i]-measured[ws])[soft[ws]].mean()) if soft[ws].any() else None for i in (0,1)],
        peak_output_deviation={k:float(diffs[ws,i].max(initial=0)) for i,k in enumerate(compact_names)},
        threshold_N=thresholds.tolist(),threshold_opportunities_per_level=processed*10*4,
        threshold_mask_mismatches=mask_mismatch.tolist(),threshold_signed_edge_mismatches=switch_mismatch.tolist(),
        threshold_switch_counts_by_backend=switch_count.tolist(),
        event_counts={n:dict(x) for n,x in zip(('cpu','hybrid'),events)},event_chain_differences=counts['packet_event_chain_mismatch'],
        first_event_difference=first,domain_exit=[o.ground.domain_exit for o in obs],
        original_reference=dict(cells=[native['world_cells'],len(a.ground.ids),len(b.ground.ids)],
            windows=[native['complete_windows'],a.windows,b.windows],
            column_steps=[native['sparse_global_column_steps'],a.integration_column_steps,b.integration_column_steps],
            events=[native['publication_event_count'],a.ground.event_serial,b.ground.event_serial],
            archived_observer_wall_s=native['observation_wall_s'],archived_E60_wall_s=source['wall_s']),
        raw=[case+'-packets.npz',case+'-events.jsonl.gz'],cpu_affinity=sorted(os.sched_getaffinity(0)),
        new_plant_steps=0,online_changed=False)
    library=propagation_library or ROOT/'work/w59/m1-gpu-study/propagate.so'
    result['propagation_library']=dict(path=str(library),sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
    result['query_workload_included']=with_queries
    if with_queries:
        result['recovery']=query_summary(query_rows,times[ws],len(queries))
        result['complete'] &= len(query_rows)==len(queries)
        result['original_reference']['query_calls']=native['query_calls']
        result['original_reference']['archived_query_wall_s']=native['query_wall_s']
        result['native_timing']={name:dict(
            observer=o.timing(),fused_packet_calls=o.native.packet_calls,
            fused_packet_wall_s=o.native.packet_wall_s,
            host_observation_excluding_native_packet_s=result['totals'][key]-o.native.packet_wall_s,
            marker_writes=o.marker.generation)
            for name,key,o in zip(('cpu','hybrid'),('cpu_s','hybrid_s'),obs)}
        result['native_timing']['CPU_correction_scope']='The original CPU ABI fuses propagation and force correction. Only their joint elapsed time is measured on CPU. The hybrid ABI separately times synchronous propagation and host force correction.'
        result['raw'].append(case+'-queries.json')
        write(dest/(case+'-queries.json'),query_rows)
        release_markers(obs)
    write(dest/(case+'-summary.json'),result)
    b.close()
    print(case,'complete',result['complete'],'ratio',result['totals'].get('speedup'),
          'events',result['event_counts'],flush=True)



def force_error(predicted, measured, selected):
    error = predicted[selected] - measured[selected]
    if not len(error):
        return dict(foot_ticks=0)
    return dict(foot_ticks=len(error), MAE_N=float(np.abs(error).mean()),
                RMSE_N=float(np.sqrt(np.mean(error**2))), max_abs_N=max_abs(error))


def compare_edges(first, second, selected):
    # Never synthesize an initial edge or cross the edge of a selected footprint.
    pairs = selected[:-1] & selected[1:]
    s1 = np.diff(first.astype(np.int8),axis=0)[pairs]
    s2 = np.diff(second.astype(np.int8),axis=0)[pairs]
    return dict(
        foot_ticks=int(np.count_nonzero(selected)),
        state_mismatch=int(np.count_nonzero((first!=second)&selected)),
        adjacent_foot_tick_pairs=int(np.count_nonzero(pairs)),
        first_rising=int(np.count_nonzero(s1==1)), first_falling=int(np.count_nonzero(s1==-1)),
        second_rising=int(np.count_nonzero(s2==1)), second_falling=int(np.count_nonzero(s2==-1)),
        matched_rising_same_tick=int(np.count_nonzero((s1==1)&(s2==1))),
        matched_falling_same_tick=int(np.count_nonzero((s1==-1)&(s2==-1))),
        signed_edge_mismatch=int(np.count_nonzero(s1!=s2)))


def sensor_comparison(case,dest):
    result = read(dest/(case+'-summary.json'))
    raw = RAW/result['episode']
    source = read(COMPACT/'episodes'/(result['episode']+'.json'))
    columns = source['physical_columns']
    contact_idx = [columns.index('contact_'+leg) for leg in LEGS]
    force_idx = [columns.index('grf_'+leg+'_z') for leg in LEGS]
    with np.load(dest/(case+'-packets.npz')) as packet:
        feet = packet['observer_Fz']
        outputs = [feet[:,i].reshape(-1,4) for i in range(2)]
        sensed = packet['actual_Fz'].reshape(-1,4)
        soft = packet['is_material'].reshape(-1,4)
        clock = packet['time']
    contact = np.empty_like(soft)
    index = 0
    for src in sorted(raw.glob('physical-[0-9][0-9][0-9].npz')):
        if index==len(sensed):
            break
        with np.load(src) as physical:
            rows = physical['physical'][:len(sensed)-index]
        n = len(rows)
        if (not np.array_equal(rows[:,force_idx],sensed[index:index+n])
            or not np.array_equal(rows[9::10,1],clock[index//10:(index+n)//10])):
            raise ValueError('Archived plant contact misaligned with replayed forces')
        contact[index:index+n] = rows[:,contact_idx].astype(bool)
        index += n
    if index!=len(sensed):
        raise ValueError('Recorded physical contact is incomplete')
    all_ticks = np.ones_like(soft)
    final = (np.arange(len(soft))%10==9)[:,None]
    domains = dict(all=all_ticks, material=soft)
    force_domains = dict(domains, material_contact=soft&contact,
                         material_final_corrected_boundary=soft&final,
                         material_other_nine_boundaries=soft&~final)
    error = {domain:{backend:force_error(pred,sensed,mask)
                     for backend,pred in zip(('cpu','hybrid'),outputs)}
             for domain,mask in force_domains.items()}
    error['material_per_foot'] = {leg:{backend:force_error(pred[:,i],sensed[:,i],soft[:,i])
                                       for backend,pred in zip(('cpu','hybrid'),outputs)}
                                  for i,leg in enumerate(LEGS)}

    # Recorded physical contact flags and instantaneous threshold predicates are
    # separate quantities. Controller confirmation has its own gait, dwell and WBC state.
    off,on,load = result['threshold_N']
    predicates = dict(positive_force_vs_contact_flag=([x>0 for x in outputs],contact),
                      force_over_touch_off=([x>off for x in outputs],sensed>off),
                      force_at_touch_on=([x>=on for x in outputs],sensed>=on),
                      force_at_load_on=([x>=load for x in outputs],sensed>=load))
    edges = {}
    for name,(output_mask,reference) in predicates.items():
        a,b = output_mask
        edges[name] = {domain:dict(
            cpu_vs_hybrid=compare_edges(a,b,mask),
            cpu_vs_recorded=compare_edges(a,reference,mask),
            hybrid_vs_recorded=compare_edges(b,reference,mask))
            for domain,mask in domains.items()}
    controller_events = Counter()
    with (raw/'w55-f/contact-events.jsonl').open() as ledger:
        for line in ledger:
            controller_events[json.loads(line)['kind']] += 1
    observer = source['policy']['observer']
    threshold = dict(touch_off_N=off,touch_on_N=on,load_on_N=load)
    write(dest/(case+'-sensors.json'),dict(episode=result['episode'],created=now(),
        original_plant_record=True,raw_input_rows=len(sensed),feet=list(LEGS),threshold_N=threshold,
        definition='M1 reports material force. All spans both rigid and material feet. Material includes all ticks with sole on TM02. Material contact also requires the recorded physical contact flag. Every signed edge uses two consecutive eligible foot-ticks, with no invented initial edge.',
        force_error_vs_actual=error,instantaneous_contact=edges,
        recorded_closed_loop_controller_events=dict(controller_events),
        original_recovery_queries=dict(count=observer['query_calls'],cpu_wall_s=observer['query_wall_s'],
            included_in_publication_only_timing=False,
            included_in_reported_complete_M1_timing=bool(result.get('query_workload_included', False)),
            complementary_timing_path=('recovery.complete_M1_including_queries'
                if result.get('query_workload_included', False) else None)),
        paired_timing_instrumentation='Both backend ledgers receive the same single Python wrapper and three reference assignments. Tap clearing, data loading, and validation run outside the measured interval. The wrapper copies no arrays.',
        new_plant_steps=0))
    print(case,'sensor comparison complete',len(sensed),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('snapshots','episodes','sensors'),required=True)
    parser.add_argument('--cases',nargs='+',choices=('A','B'),default=['A','B'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--with-queries',action='store_true')
    parser.add_argument('--propagation-library',type=Path)
    args=parser.parse_args()
    library=args.propagation_library.resolve() if args.propagation_library else None
    if args.output is None:
        suffix = ((('-'+library.stem) if library else '')+('-queries' if args.with_queries else ''))
        args.output=DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name+suffix)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='snapshots':
        snapshot_comparison(args.output,library)
    else:
        for case in args.cases:
            if args.mode=='sensors':
                sensor_comparison(case,args.output)
            else:
                replay_case(case,args.output,args.with_queries,library)


if __name__=='__main__':
    main()
