"""Resource-constrained projection of the registered W61 head, curve and frozen tasks."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from heapq import heappop, heappush


@dataclass(frozen=True)
class Task:
    kind: str
    head: str
    name: str
    slots: int
    wall: float
    physical: float = 0.
    render: float = 0.
    upload: float = 0.
    case: str = ""
    stage: str = ""


def simulate(*, now, width, heads, seats, versions, nodes, last, batch_wall, in_flight,
             curves_done, curve_wall, native_jobs, done_native, native_wall, render_rate,
             asset_bytes, upload_rate, long_rate, nominal_render_backlog=0., upload_backlog=0., curve_bytes_per_s=0., initial_ready=None):
    """Submit head continuations, one paired curve and independent single native episodes.

    The widths are scenarios for the capacity available to this E study. Existing
    plant failures are retained. For future tasks assume full windows and passing E30.
    The pool grants an earlier atomic pair before a newly arriving single.
    """
    v=dict(versions)
    primary=[h for h in heads if not h.startswith("e-fisher-")]
    logical=[h for h in seats if h in v and v[h]<last]
    clock=0.; serial=0; free=width
    queue=deque(); running=[]; native_ready={}; done=set(done_native);native_enqueued=set()
    curve_ready=deque(); curves=set(curves_done); trains=set(); curves_out=0
    eligible={h:0. for h in heads if v[h]>=last}; primary_end=None
    render_end=nominal_render_backlog;upload_end=upload_backlog;latest_long_end=0.
    native_by_id={j["id"]:j for j in native_jobs}
    assigned=Counter(j["head"] for j in native_jobs if j["id"] in done)
    first_long_offered={j["head"] for j in native_jobs if j["id"] in done and j["stage"] in ("short","recovered")}
    physics_started={}; last_curve_end=0.; counts=Counter(); slot_seconds=0.
    required=dict(train=sum(last-v[h] for h in heads),curve=sum((h,n) not in curves_done for h in heads for n in nodes),
                  native=sum(j["id"] not in done_native for j in native_jobs))

    def scan_native():
        for idx,j in enumerate(native_jobs):
            if j["id"] in done or j["id"] in native_enqueued:continue
            if j["version"] and v[j["head"]]<last:continue
            if j.get("predecessor") and j["predecessor"] not in done:continue
            native_ready.setdefault(j["id"],(clock,idx))

    def scan_curves(h):
        for node in nodes:
            key=(h,node)
            if v[h]>=node and key not in curves:
                curves.add(key);curve_ready.append(key)

    ready_at={h:(initial_ready or {}).get(h,0.) for h in heads}
    # Already running paired leases occupy physical slots at the snapshot.
    # When projecting a smaller effective width, start its oldest running pairs first.
    running_heads=[h for h,version in in_flight if h in v and v[h]==version]
    running_heads.sort(key=lambda h:-in_flight[h,v[h]])
    for h in running_heads[:width//2]:
        node=v[h];remaining=max(1.,batch_wall[h]-in_flight.pop((h,node)))
        queue.append(Task("train",h,str(node),2,remaining));trains.add(h)
        if h not in logical:logical.append(h)
    for h in heads:scan_curves(h)
    scan_native()
    while True:
        if primary_end is None and all(v[h]>=last for h in primary):primary_end=clock
        outstanding=list(queue)+[t[2] for t in running]
        nontrain=bool(native_ready or curve_ready or any(x.kind!="train" for x in outstanding))
        pairs=min(width//2, 2 if nontrain else 3)
        logical=[h for h in logical if v[h]<last]
        for h in heads:
            if len(logical)>=pairs:break
            if h not in logical and v[h]<last:logical.append(h)
        for h in sorted(logical,key=lambda h:(ready_at[h],logical.index(h))):
            if len(trains)>=pairs:break
            if h in trains:continue
            node=v[h]
            elapsed=in_flight.pop((h,node),0.)
            wall=max(1.,batch_wall[h]-elapsed)
            queue.append(Task("train",h,str(node),2,wall));trains.add(h)
        if curve_ready and not curves_out:
            h,node=curve_ready.popleft()
            queue.append(Task("curve",h,str(node),2,curve_wall,120.,120.*render_rate["curve"],120.*curve_bytes_per_s/upload_rate))
            curves_out=1
        queued_native=sum(x.kind=="native" for x in queue)
        earlier=sum(x.slots for x in queue if x.kind!="native")
        offers=max(0,max(1,free-earlier)-queued_native)
        for _ in range(min(offers,len(native_ready))):
            def order(k):
                j=native_by_id[k];new_long=j["stage"] in ("short","recovered") and j["head"] not in first_long_offered
                return (assigned[j["head"]],0 if new_long else 1,native_ready[k])
            name=min(native_ready,key=order)
            t=native_by_id[name];del native_ready[name];native_enqueued.add(name)
            if t["stage"] in ("short","recovered"):first_long_offered.add(t["head"])
            h,stage=t["head"],t["stage"];physical=t["duration"]
            long=stage in ("short","recovered")
            wall=physical*width/long_rate if long else native_wall[stage]*(physical/int(stage[1:]))
            queue.append(Task("native",h,name,1,wall,physical,t.get("recorded_horizon",physical)*render_rate["native"],
                              asset_bytes.get(stage,0.)/upload_rate,t["case"],stage))
            assigned[h]+=1
        while queue and queue[0].slots<=free:
            task=queue.popleft();free-=task.slots;serial+=1
            if task.stage in ("short","recovered"):physics_started.setdefault(task.head,clock)
            heappush(running,(clock+task.wall,serial,task))
        if not running:
            if queue or curve_ready or native_ready or any(v[h]<last for h in heads):
                raise RuntimeError("Unserviceable projection workload")
            break
        clock,_,task=heappop(running);free+=task.slots;counts[task.kind]+=1;slot_seconds+=task.wall*task.slots
        if task.kind=="train":
            h=task.head;v[h]+=1;trains.remove(h);ready_at[h]=clock;scan_curves(h)
            if v[h]>=last:eligible[h]=clock
        elif task.kind=="curve":
            curves_out=0;last_curve_end=clock
        else:
            done.add(task.name)
            if task.stage in ("short","recovered"):latest_long_end=clock
        if task.render or task.upload:
            render_end=max(render_end,clock)+task.render
            upload_end=max(upload_end,render_end)+task.upload
        scan_native()
    if any(j["id"] not in done for j in native_jobs) or any(counts[k]!=n for k,n in required.items()):
        raise RuntimeError("The projection did not charge every remaining registered task exactly once")
    def date(s):
        return (now+timedelta(seconds=s)).isoformat(timespec="minutes")
    first=min(eligible.values()) if eligible else None
    total=max(clock,render_end,upload_end)
    return dict(effective_slots=width,modeled_physical_work_items=required,charged_slot_hours=slot_seconds/3600,
                mean_physical_slots=slot_seconds/clock if clock else 0.,first_8192=date(first),first_long_start=date(min(physics_started.values())) if physics_started else None,
                first_heads=sorted(h for h,s in eligible.items() if s==first),selected_heads=len(heads),selected_primary_heads=len(primary),
                all_selected_primary_8192=date(primary_end),all_selected_training_8192=date(max(eligible.values())),
                all_primary_8192=date(primary_end) if len(primary)==9 else None,
                all_12_training_8192=date(max(eligible.values())) if len(heads)==12 else None,all_long_physics=date(latest_long_end) if latest_long_end else None,
                all_physics=date(clock),last_curve_physics=date(last_curve_end) if last_curve_end else None,all_required_media=date(render_end),
                modeled_finish=date(total),hours_remaining=total/3600,
                media_and_upload_tail_h=max(0.,total-clock)/3600,
                head_freeze_dates={h:date(s) for h,s in eligible.items()})
