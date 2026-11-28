"""Verify every retained real prefix packet in paired GPU launches; exclude replay from throughput."""
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
from paired import RAW,DEST,SPEC,write
from transport import IN_BYTES,OUT_BYTES,MAX_CELLS,TICKS,input_views,output_views
W59=Path('/root/autodl-tmp/MCP-RL-w59')
sys.path.insert(0,str(W59/'scripts'))
from s1_w60_cpu_observer import ARRAY,BYTE,INDEX


def main():
    spec=json.loads(SPEC.read_text())
    cpu=json.loads((DEST/'prefix12-cpu-a02.json').read_text())
    samples=[np.load(Path(cpu['raw_directory'])/case/'parity-samples.npz') for case in ('A','B')]
    common=sorted({int(k[1:k.index('_')]) for k in samples[0].files if k.endswith('_n')} &
                  {int(k[1:k.index('_')]) for k in samples[1].files if k.endswith('_n')})
    model=samples[0]['model'];assert np.array_equal(model,samples[1]['model'])
    lib=ct.CDLL(spec['fused_binary']);ptr=ct.c_void_p
    lib.w61_create.argtypes=[ptr,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ptr)]
    lib.w61_dispatch.argtypes=[ptr,ct.c_int,ptr,ptr,ptr,ptr]
    lib.w61_destroy.argtypes=[ptr];lib.w61_destroy.restype=None
    correction=ct.CDLL(str(W59/'work/w59/m1-gpu-study/correction.so')).w59_correct_last
    correction.argtypes=[ARRAY,ct.c_int,ct.c_int,ARRAY,ARRAY,ARRAY,INDEX,ct.c_double,ARRAY,BYTE,BYTE,ARRAY,ARRAY,BYTE,ct.c_int,ARRAY,ARRAY,ARRAY,BYTE]
    correction.restype=None
    handle=ptr();assert lib.w61_create(model.ctypes.data,2,MAX_CELLS,TICKS,ct.byref(handle))==0
    bufs=[np.zeros(IN_BYTES+OUT_BYTES,'u1') for _ in samples]
    ips=np.array([b.ctypes.data for b in bufs],dtype=np.uintp);ops=ips+np.uintp(IN_BYTES)
    records=[]
    try:
        for seq in common:
            inp=[{k.removeprefix('p'+str(seq)+'_'):z[k] for k in z.files if k.startswith('p'+str(seq)+'_')} for z in samples]
            ns=np.array([v['n'] for v in inp],dtype='i4')
            gam=np.array([v['gamma'] for v in inp],dtype='f8')
            for n,d,b in zip(ns,inp,bufs):
                for dest,name in zip(input_views(b,int(n),TICKS),('dt','normal','area','start_state','owners','start_settled','start_loaded')):
                    np.copyto(dest,d[name])
            assert lib.w61_dispatch(handle,2,ns.ctypes.data,gam.ctypes.data,ips.ctypes.data,ops.ctypes.data)==0
            for case,n,d,b in zip(('A','B'),ns,inp,bufs):
                state,before,force,stats,bad,settled,loaded,previous,trace=output_views(b,int(n),TICKS)
                diag=np.zeros((4,8),'f8')
                if bad[0]<0:
                    correction(model,int(n),TICKS,d['dt'],d['normal'],d['area'],d['owners'],float(d['gamma']),state,
                              settled,loaded,force,stats,trace,int(d['correct']),d['measured'],diag,before,previous)
                records.append(dict(seq=seq,case=case,n=int(n),failure_matches=int(bad[0])==int(d['bad']),
                      mask_matches=bool(np.array_equal(settled,d['end_settled']) and np.array_equal(loaded,d['end_loaded']) and np.array_equal(trace,d['trace'])),
                      max_abs_force_N=float(abs(force-d['forces']).max()),max_abs_state=float(abs(state-d['end_state']).max()),
                      max_abs_diagnostics=float(abs(diag-d['diagnostics']).max())))
    finally:lib.w61_destroy(handle)
    write(DEST/'prefix12-saved-input-equivalence.json',dict(source='independent live A/B CPU streams in corrected 12s prefix',
        purpose='exact-input saved-packet numerical comparison only',included_in_live_throughput=False,
        mixed_saved_state_launches=len(common),independent_case_packets=len(records),items=records,
        maximum_force_difference_N=max(r['max_abs_force_N'] for r in records),maximum_state_difference=max(r['max_abs_state'] for r in records),
        all_masks_and_failure_statuses_equal=all(r['mask_matches'] and r['failure_matches'] for r in records)))
    print(len(records),max(r['max_abs_force_N'] for r in records),max(r['max_abs_state'] for r in records),flush=True)

if __name__=='__main__':main()
