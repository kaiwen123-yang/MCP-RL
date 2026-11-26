from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import time
from compare_s1_w58 import R, metrics
from inspect_s1_w58_predecessor import read
from publish_s1_go2_w54 import write

SH = ZoneInfo('Asia/Shanghai')
WORK = Path('work/w58').resolve()
def now():
    return datetime.now(SH).isoformat(timespec='seconds')

def score(name):
    started = time.monotonic()
    item = metrics(name)
    if not item.get('full_window') or len(item['targets']) != 12:
        raise ValueError('Incomplete terminal target window '+name)
    return dict(attempt=name, status=item['status'], targets=len(item['targets']),
                realized=sum(t['realized'] for t in item['targets']),
                cpu_worker=os.getpid(), processing_s=round(time.monotonic()-started,3), finished=now())

if __name__ == '__main__':
    stage=read(WORK/'stage.json')
    if stage['stage']!='independent_validation' or read(R/'development-decision.json')['joint_eligible']:
        raise RuntimeError('Expected the frozen four-arm independent validation batch')
    initial=[]
    for record in sorted((R/'attempts').glob('*-validation-*.json')):
        row=read(record)
        if 'ended_unix_s' in row and row.get('status') in ('passed','mechanism_criteria_unmet'):
            initial.append(record.stem)
    receipt=dict(started=now(),split='validation',source='compare_s1_w58.metrics',rows=[],
                 deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    path=WORK/'closed-validation-scoring-20261126.json'
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(score,initial):
            receipt['rows'].append(result)
            write(path,receipt)
            print(result,flush=True)
    receipt['initial_complete']=now()
    write(path,receipt)
    waiting='w58-g0-m1-fixed-wait-a-validation-recovered-a01'
    last='w58-g0-m1-fixed-wait-b-validation-recovered-a01'
    while waiting not in initial:
        dest=R/'attempts'/(waiting+'.json')
        if 'ended_unix_s' in read(R/'attempts'/(last+'.json')):break
        if dest.is_file() and read(dest).get('status') in ('passed','mechanism_criteria_unmet'):
            result=score(waiting)
            receipt['rows'].append(result)
            print(result,flush=True)
            break
        time.sleep(15)
    receipt['ended']=now()
    write(path,receipt)
