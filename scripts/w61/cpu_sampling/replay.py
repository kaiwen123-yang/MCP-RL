"""Evaluate the frozen C ABI on the recorded inputs/outputs of independent full robot collectors."""
import ctypes as ct
import json,sys,time,statistics
from pathlib import Path
import numpy as np
from build import ROOT,WORK,digest

sys.path.insert(0,str(ROOT/'scripts'))
from s1_w60_cpu_observer import ARRAY,BYTE,INDEX
OLD=Path('/root/autodl-tmp/MCP-RL-w59/work/w57-m/material.so')
DATA=Path('/root/autodl-tmp/MCP-RL-w61-material/work/w61/material/w61-e-packet-throughput-v1')
SIG=[ARRAY,ct.c_int,ct.c_int,ARRAY,ARRAY,ARRAY,INDEX,ct.c_double,ARRAY,BYTE,BYTE,ARRAY,ARRAY,BYTE,ct.c_int,ARRAY,ARRAY]

def execute(lib,model,p):
    n,ticks=int(p['n']),int(p['ticks'])
    s=p['start_state'].copy();settled=p['start_settled'].copy();loaded=p['start_loaded'].copy()
    f=np.zeros((ticks,4));st=f.copy();tr=np.empty((ticks,n),'u1');diag=np.zeros((4,8))
    start=time.perf_counter_ns()
    bad=lib.w57_packet(model,n,ticks,p['dt'],p['normal'],p['area'],p['owners'],float(p['gamma']),
                       s,settled,loaded,f,st,tr,int(p['correct']),p['measured'],diag)
    elapsed=(time.perf_counter_ns()-start)/1e6
    return (bad,s,settled,loaded,f,st,tr,diag),elapsed

def main():
    original=ct.CDLL(str(OLD));candidate=ct.CDLL(str(WORK/'build/exact-material.so'))
    original.w57_packet.argtypes=candidate.w57_packet.argtypes=SIG
    rows=[]
    for run in ('prefix12-cpu-a02','full0-cpu','full1-cpu'):
        for case in ('A','B'):
            src=DATA/run/case/'parity-samples.npz'
            with np.load(src) as z:
                model=z['model']
                for seq in sorted(int(k.split('_')[0][1:]) for k in z.files if k.endswith('_n')):
                    prefix=f'p{seq}_';fields={k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
                    a,ta=execute(original,model,fields);b,tb=execute(candidate,model,fields)
                    matches=[a[0]==b[0],*[x.tobytes()==y.tobytes() for x,y in zip(a[1:],b[1:])]]
                    frozen=[fields[k] for k in ('end_state','end_settled','end_loaded','forces','stats','trace','diagnostics')]
                    saved=all(x.tobytes()==y.tobytes() for x,y in zip(a[1:],frozen))
                    h=ct.c_uint64();m=ct.c_uint64();candidate.w61_packet_counts(ct.byref(h),ct.byref(m))
                    # Alternate order on the same real input. Timers exclude construction of the fresh state.
                    bt=[tb];at=[ta]
                    for k in range(3):
                        for target,vals in ((candidate,bt),(original,at)) if k%2==0 else ((original,at),(candidate,bt)):
                            _,duration=execute(target,model,fields);vals.append(duration)
                    rows.append(dict(run=run,case=case,seq=seq,columns=int(fields['n']),identical=all(matches),field_equal=matches,
                                     reference_matches_saved=saved,hits=h.value,misses=m.value,original_ms=statistics.median(at),candidate_ms=statistics.median(bt)))
    report=dict(original=str(OLD),source_sha256=digest(OLD),candidate_sha256=digest(WORK/'build/exact-material.so'),
                rows=rows,all_bitwise_equal=all(r['identical'] and r['reference_matches_saved'] for r in rows),
                summed_median_speedup=sum(r['original_ms'] for r in rows)/sum(r['candidate_ms'] for r in rows),
                hits=sum(r['hits'] for r in rows),misses=sum(r['misses'] for r in rows))
    (WORK/'packet-contrast.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),len(rows))
    if not report['all_bitwise_equal']:print('differences',[(r['run'],r['case'],r['seq'],r['field_equal']) for r in rows if not r['identical']][:8])
if __name__=='__main__':main()
