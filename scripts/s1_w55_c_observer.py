"""Causal 2 ms publication from an available sequence of 0.2 ms foot observations."""
import json
import time
import numpy as np
from s1_w55_c_data import COMPACT,RAW,ROOT,WEIGHTS,DT,write
from s1_w55_c_replay import interclock


def observe_window(observer, boundary_times, material_times, measured_bottom, measured_velocity, publish_time):
    """Advance using foot measurements at each interval's left edge. Publish
    after the last interval. The inputs have no ground force or true state."""
    starts=np.asarray(boundary_times)
    ends=np.asarray(material_times)
    prior=np.r_[observer.time,ends[:-1]]
    if (len(ends)!=10 or not np.allclose(starts,prior,rtol=0,atol=1e-9)
            or not np.allclose(ends-starts,DT,rtol=0,atol=1e-9)
            or abs(ends[-1]-publish_time)>1e-9):
        raise ValueError('W55 500 Hz publication requires the ten contiguous preceding 0.2 ms intervals.')
    for end,bottom,velocity in zip(ends,measured_bottom,measured_velocity,strict=True):
        observer.observe(float(end),bottom,velocity)


def deployed_window_check():
    import jax
    from tm02.s1_material import LearnedColumns
    from s1_go2_r10_observer import MaterialObserver
    jax.config.update('jax_enable_x64',True)
    inventory=json.loads((COMPACT/'N44-inventory.json').read_text())
    from pathlib import Path
    directory=Path(inventory['source'])
    foot=np.concatenate([np.load(f)['foot'][:,:30] for f in sorted(directory.glob('material-[0-9]*.npz'))])
    with np.load(RAW/'N44-histories.npz') as data,np.load(RAW/'frozen-N44-model.npz') as saved:
        keys=data['keys'];ticks=data['ticks'];fine=saved['fine'];inv=saved['inverse']
        loaded=np.flatnonzero(np.any(fine[:,:,4]<-1e-7,axis=1))
        start=int(ticks[loaded[0]]//10*10)
        start_index=start-int(ticks[0])
        if start_index and np.any(fine[start_index-1,:,:4]!=0):
            raise ValueError('First checked packet does not start at virgin state')
        params=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
        observer=MaterialObserver(params,None,law=LearnedColumns(WEIGHTS))
        observer.time=float(foot[start,0])
        err=np.zeros(4);checked=0
        for s in range(start,start+30,10):
            rows=foot[s:s+10]
            observe_window(observer,rows[:,0],rows[:,1],rows[:,6:18].reshape(-1,4,3),rows[:,18:30].reshape(-1,4,3),rows[-1,1])
            g=observer.ground
            state=np.array([np.r_[g.e[g.ids[tuple(k)]],g.q[g.ids[tuple(k)]]] if tuple(k) in g.ids else np.zeros(4) for k in keys])
            expected=fine[s+9-int(ticks[0]),inv,:4]
            err=np.maximum(err,np.max(np.abs(state-expected),axis=0));checked+=len(keys)
        if err.max()>1e-9:raise ValueError(f'Causal packet disagrees with continuous independent PANN histories {err}')
    return dict(episode=directory.name,start_tick=start,packets=3,mixed_updates=30,cell_state_comparisons=checked,
                max_e11_e22_q11_q22_errors=err.tolist(),injection='Virginal model state, prefix before first applied load. Only foot poses, velocities and observed times supplied to deployed MaterialObserver.')


def main():
    start=time.time();cpu=time.process_time();numbers={}
    for name in ('N44','F51','N48','F52','B46','F53'):
        with np.load(RAW/f'{name}-histories.npz') as data,np.load(RAW/f'frozen-{name}-model.npz') as frozen:
            idx=data['coarse'];mask=idx>0;ii=idx[mask];truth=data['reference_fine'][ii-1];area=data['area'];inv=frozen['inverse']
            win=frozen['fine'][ii-1][:,inv];endpoint=frozen['observer'][mask][:,inv]
            numbers[name]=dict(common_publications=len(ii),world_cells=len(inv),
                endpoint_observer_vs_physical=interclock(endpoint,truth,area[ii],area[ii-1]),
                substep_window_observer_vs_physical=interclock(win,truth,area[ii-1],area[ii-1]))
    check=deployed_window_check()
    write(COMPACT/'causal-observer-window.json',dict(untouched_robot_controllers=True,source='Existing complete independently integrated frozen PANN and analytic histories. Fine values are read only at the right edge of each closed ten-substep window. Source availability requires the ten time-stamped foot boundaries; 2 ms endpoints alone cannot reconstruct them.',
        deployed_packet_check=check,whole_histories_at_common_2ms_times=numbers,start_unix=start,end_unix=time.time(),cpu_s=time.process_time()-cpu))
    print({n:[round(v[k]['max_force_N'],5) for k in ('endpoint_observer_vs_physical','substep_window_observer_vs_physical')] for n,v in numbers.items()},flush=True)


if __name__=='__main__':main()
