"""Opt-in passive request recorder for an unstarted, already-qualified frozen W59 process."""
import os
import sys
from pathlib import Path


def enable():
    entry=Path(sys.argv[0]).resolve()
    if entry.name!='run_s1_go2_r12.py':
        return
    import hashlib
    import inspect
    import json
    sys.path.insert(0,str(entry.parent))
    try:
        from s1_go2_r11_runtime import run
        from s1_w59_snapshots import pre_tick
    finally:
        sys.path.pop(0)
    source,first=inspect.getsourcelines(run)
    start=next(i for i,line in enumerate(source) if 'for step in range(first_step,stop_step+1)' in line)
    line_no=first+next(i for i in range(start,len(source)) if source[i].strip()=='t = float(data.time)')
    clock=sys.monitoring
    ident=clock.PROFILER_ID
    clock.use_tool_id(ident,'W59 pre-control scientific recording')
    def on_line(code,line):
        if line!=line_no:return clock.DISABLE
        values=sys._getframe(1).f_locals
        # Only a fresh registered scientific episode can opt into this recording.
        if (not values['w59'] or values['w59_branch'] or values['capture_requests'] or
                values['w59'].get('learner_config') or values['w57']['task_mode'] not in ('short','recovered','original')):
            return clock.DISABLE
        if values['data'].time>60.:return clock.DISABLE
        if values['step']==values['first_step']:
            receipt=dict(recorder='CPython 3.12 local LINE event at original pre-control tick',
                path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_runtime=str(Path(inspect.getfile(run)).resolve()),
                attempt=values['args'].attempt,physics_source_modified=False,
                effect='Serializes current state to the already-owned snapshot_buffer consumed by the unchanged legal-request publisher')
            (values['out']/'w59-passive-recorder.json').write_text(json.dumps(receipt,indent=2)+'\n')
        pre_tick(values['agent'],values['joint'],values['mission'],values['step'],
                 values['previous_phase'],values['first_completion'],values['evaluation_legal_route'])
    clock.register_callback(ident,clock.events.LINE,on_line)
    clock.set_local_events(ident,run.__code__,clock.events.LINE)


if os.environ.get('W59_RECORD_FROZEN_LEGAL_REQUESTS')=='1':
    enable()
