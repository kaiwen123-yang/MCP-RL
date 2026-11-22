"""Optional low-frequency P2 prototype. The inner r60/P1 horizon remains 0.35 s."""
from copy import copy
import numpy as np

from s1_w57_planning import Forecast, footprint


HORIZON = 2.0  # .25s source support + longest observed 1.7472s short cross-leg release/reload gap.
OFFSETS = (0., -.05, .05)
PROBE = .002
SUBSTEP = .0002


def intersection(g, source, target):
    lo=np.maximum(source-g.half,target-g.half);hi=np.minimum(source+g.half,target+g.half)
    keys={}
    if np.any(hi<=lo):return keys
    for key in footprint(g,source).keys() & footprint(g,target).keys():
        lower=np.maximum(lo,np.array(key)*g.cell)
        upper=np.minimum(hi,(np.array(key)+1)*g.cell)
        area=float(np.prod(np.maximum(upper-lower,0.)))
        if area>0.:keys[key]=area
    return keys


def schedule(controller,t,mission,mq,observer,grf,due):
    probe=copy(controller)
    probe.next_liftoff=due
    events=probe._forecast_events(t,t+HORIZON,mq,mission)
    pairs={}
    g=observer.ground
    bottom=observer.published_bottom
    if bottom is None:return probe,events,pairs
    for src in controller.next_pair:
        if not controller.measured[src] or grf[src,2]<=1. or not g.interior(bottom[src,:2]):
            continue
        release=next((a for a,b,e in events[src] if a>=t and b>a),None)
        if release is None:continue
        for leg,seq in enumerate(events):
            if leg==src:continue
            for j,(a,b,event) in enumerate(seq):
                if b<=release or b+PROBE>t+HORIZON:continue
                target=event['landing'][:2]
                overlap=intersection(g,bottom[src,:2],target)
                if overlap:
                    pairs[(int(src),leg,j)]=dict(release_s=release,touch_s=b,xy=target,
                                                 surface_z=float(event['landing'][2]-controller.radius[leg]),
                                                 overlap=overlap)
    return probe,events,pairs


def shared_targets(candidates):
    common=set.intersection(*(set(c[2]) for c in candidates))
    matches={}
    for key in sorted(common):
        cells=set.intersection(*(set(c[2][key]['overlap']) for c in candidates))
        if cells:matches[key]=sorted(cells)
    return matches


def score(c,t,mission,mq,observer,grf,matches,history):
    controller,events,target=c
    # The event grid is constructed from the published gait and public geometry. No future
    # measured contact/force is consulted. The 2ms test force is identical across the arms.
    chosen=[(key,target[key]) for key in matches]
    limit=max(x['touch_s']+PROBE for _,x in chosen)
    breaks=[t,limit]
    for sequence in events:
        breaks.extend(float(x) for a,b,_ in sequence for x in (a,b) if t+1e-9<x<limit-1e-9)
    for _,x in chosen:
        breaks.extend([x['touch_s'],x['touch_s']+PROBE])
    grid=np.array(sorted(set(round(float(x)-t,10) for x in breaks)))+t
    mids=.5*(grid[1:]+grid[:-1])
    contact,feet=controller._future_contact_feet(t,mids,mq,mission)
    loads=np.zeros((len(mids),4,3));mass_weight=controller.mass*9.81
    last=controller.last_plan
    for j,when in enumerate(mids):
        legs=np.flatnonzero(contact[j])
        if not len(legs):continue
        if last is not None and last['times'][0]<=when<last['times'][-1]:
            k=np.searchsorted(last['times'][1:],when)
            values=np.maximum(last['forces'][k,legs,2],0.)
        else:
            values=np.ones(len(legs))
        # Positive normal requests on the available support feet; outside the last published
        # horizon this is an explicit equal-share loading scenario, not a solved 2 s MPC.
        if np.sum(values)>0.:loads[j,legs,2]=mass_weight*values/values.sum()
        for (src,dst,ordinal),x in chosen:
            if x['touch_s']<=when<x['touch_s']+PROBE:
                loads[j,dst,2]=mass_weight/2
    # Independent columns cannot couple between spatially disjoint footprint components.
    # Keep the complete transitive footprint group of every probed leg, including all
    # overlaps that could transmit prior load to that future footprint.
    g=observer.ground
    maps=[[set(footprint(g,p[:2])) if flag else set() for p,flag in zip(row,ct)]
          for row,ct in zip(feet,contact)]
    connected=set().union(*(set(footprint(g,x['xy'])) for _,x in chosen))
    allmaps=[m for row in maps for m in row if m]
    while True:
        ext=connected.union(*(m for m in allmaps if m & connected))
        if ext==connected:break
        connected=ext
    included=np.array([[bool(m&connected) for m in row] for row in maps])
    forecast=Forecast(g,grid,grid,feet[:,:,:2],contact & included,observer.published_bottom,controller.measured)
    result=forecast.propagate(loads,SUBSTEP,history)
    distances=[]
    for (src,dst,ordinal),x in chosen:
        end=int(np.argmin(abs(grid[1:]-(x['touch_s']+PROBE))))
        z=float(result['z'][end,dst])
        dz=abs(z-x['surface_z']);distances.append(dz)
    return dict(score_m=max(distances),travel_per_target_m=distances,substeps=result['steps'],
                world_cells=len(forecast.keys),protected=int(forecast.protected.sum()),
                resettable=int(forecast.reset.sum()))


def compare(controller,t,mission,mq,observer,grf,due,also_reset=False):
    candidates=[schedule(controller,t,mission,mq,observer,grf,due+offset) for offset in OFFSETS]
    matches=shared_targets(candidates)
    result=dict(nominal_due_s=due,offsets_s=list(OFFSETS),horizon_s=HORIZON,
                common_targets=[dict(source=k[0],target=k[1],target_ordinal=k[2],world_keys=v)
                                for k,v in matches.items()],
                matched_plans=[[dict(source=k[0],target=k[1],release_s=c[2][k]['release_s'],
                                     touch_s=c[2][k]['touch_s'],xy=c[2][k]['xy'].tolist(),
                                     overlap_m2=sum(c[2][k]['overlap'][i] for i in shared))
                                for k,shared in matches.items()] for c in candidates])
    if not matches:
        result.update(selected=0,requested_due_s=due,reason='no_common_crossfoot_revisit')
        return result
    for history in (('full','private_reset') if also_reset else ('full',)):
        scores=[score(c,t,mission,mq,observer,grf,matches,history) for c in candidates]
        z=[r['score_m'] for r in scores]
        # Tie tolerance below one picometre, well below the validated material root precision.
        best=0
        for i in (1,2):
            if z[i] < z[best]-1.e-12:best=i
        result[history]=dict(scores=scores,winner=best)
    best=result['full']['winner']
    result.update(selected=best,requested_due_s=due+OFFSETS[best],reason='loaded_sole_travel_at_equal_force')
    return result
