"""Check CPU final-force assimilation split against the frozen M1 on the registered W59 packets."""
import ctypes as ct
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
from s1_w57_material import W57MaterialObserver
from s1_w60_cpu_observer import ARRAY, BYTE


def main():
    work=ROOT/'work/w59/m1-gpu-study'
    registry=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/action/registry-v2.json'
    pars=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    last=ct.CDLL(str(work/'correction.so')).w59_correct_last
    last.restype=None
    rows=[]
    for request in json.loads(registry.read_text())['states']['development']:
        data=Path(request['snapshot']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=request['sha256']:
            raise ValueError('Registered snapshot differs')
        snap=pickle.loads(gzip.decompress(data))
        observer=W57MaterialObserver(pars,ROOT/'work/assets/B_random1000--ideal-potentials.npz',mode='M1')
        vars(observer.ground).update(snap['belief'])
        vars(observer).update(snap['observer'])
        frozen=observer.native.w57_packet
        last.argtypes=[*frozen.argtypes,ARRAY,BYTE]
        nominal=observer.ground.law.lib.w60_packet
        row=dict(snapshot_id=request['snapshot_id'],columns=0)
        class Compare:
            def w57_packet(self,*packed):
                a=list(packed)
                for j in (8,9,10,11,12,13,16):a[j]=a[j].copy()
                m,n,ticks,dt,normal,areas,owners,gamma,state,settled,loaded,forces,stats,trace,correct,measured,diagnostics=a
                for k in range(ticks):
                    if k==ticks-1:
                        before,was_loaded=state.copy(),loaded.copy()
                    bad=nominal(m,n,1,dt[k:k+1],normal[k],areas[k],owners[k],gamma,state,settled,loaded,forces[k],stats[k])
                    if bad>=0:raise ValueError(f'Frozen nominal failed at {k},{bad}')
                    trace[k]=loaded
                last(*a,before,was_loaded)
                bad=frozen(*packed)
                if bad>=0:raise ValueError(f'Frozen M1 failed at {bad}')
                fields=('state','settled','loaded','forces','stats','active_trace','diagnostics')
                equal={key:np.array_equal(a[j].view('u1'),packed[j].view('u1')) for key,j in zip(fields,(8,9,10,11,12,13,16))}
                row.update(columns=n,bitwise_equal=equal,corrected_legs=int(np.count_nonzero(diagnostics[:,4])))
                return bad
        observer.native=Compare()
        v=observer.packet[-1];observer.observe(float(v[1]),v[2],v[3])
        rows.append(row)
    active=[r for r in rows if r['columns']]
    verdict=all(all(r['bitwise_equal'].values()) for r in active)
    report=dict(registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),registered=len(rows),
        nonempty=len(active),corrected_legs=sum(r['corrected_legs'] for r in active),all_outputs_bitwise_equal=verdict,
        scope='Same-input CPU nominal propagation (ten original w60 ticks) plus standalone final assimilation versus frozen full CPU M1.',
        results=rows,new_robot_steps=0)
    (work/'correction-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['registered'],report['nonempty'],report['corrected_legs'],verdict)
    if not verdict:raise ValueError('Standalone CPU correction did not reproduce the reference')

if __name__=='__main__':
    main()
