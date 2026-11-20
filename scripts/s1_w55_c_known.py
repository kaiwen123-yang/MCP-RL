"""Apply the new candidate to the nine original diagnostic histories.
Reuse the frozen scores from W54, retaining the recorded time convention."""
import hashlib
import json
import time
import numpy as np
from s1_w55_c_data import RAW,COMPACT,REVIEW,LEGACY,DT,geometry,reference_run,write
from s1_w55_c_replay import physical_metrics,weights,run


def main():
    old=json.loads((REVIEW/'pann-replay-summary.json').read_text())
    trained=weights(RAW/'W55-short-load-ideal-potentials.npz')
    paired={};t0=time.time();c0=time.process_time()
    for case,record in old['cases'].items():
        d=LEGACY/record['job'];keys=np.array(record['keys'])
        foot=np.concatenate([np.load(f)['foot'] for f in sorted(d.glob('material-[0-9]*.npz'))])
        normal,area,_=geometry(foot[:,6:18].reshape(-1,4,3),keys)
        first=np.flatnonzero(np.isfinite(normal).any(axis=1))[0]//10*10
        normal,area,foot=normal[first:],area[first:],foot[first:]
        paired[case]={}
        for label in ('plant02ms','observer2ms'):
            if label=='plant02ms':
                steps=np.arange(len(foot));dt=DT
            else:
                steps=np.flatnonzero(np.isclose(foot[:,0]/.002,np.rint(foot[:,0]/.002),atol=1e-5,rtol=0));dt=.002
            target=reference_run(normal[steps],dt)
            pred,timing=run(trained,normal[steps],dt,'cpu')
            np.savez_compressed(RAW/f'known-nine-{case}-{label}.npz',reference=target,candidate=pred,keys=keys,
                                times=foot[steps,1 if label=='plant02ms' else 0],normal=normal[steps],areas=area[steps])
            new=physical_metrics(pred,target,area[steps],keys)
            rr=[]
            for before,after in zip(record[label]['pann_vs_reference_same_boundary_and_dt'],new):
                before_peak=before['unloaded_reference_peak_compression_um']
                rr.append(dict(key=after['key'],frozen_free_error_um=before['unloaded_recovery_abs_max_um'],
                         candidate_free_error_um=after['unloaded_recovery_error_um'],frozen_relative_peak_error=
                         before['unloaded_recovery_abs_max_um']/before_peak if before_peak else None,
                         candidate_relative_peak_error=after['recovery_error_over_peak'],
                         reference_free_peak_um=before_peak,candidate_force_max_error_N=after['normal_force_abs_max_N'],
                         frozen_force_max_error_N=before['normal_force_abs_max_N']))
            paired[case][label]=dict(rows=rr,execution=timing)
    write(COMPACT/'known-nine-candidate.json',dict(paired=paired,old_source='reviews/W54-review/pann-replay-summary.json',
        frozen_learned_integrations_repeated=0,reference_state='Reconstructed from the original complete measured boundary to score the candidate. Archived frozen model scores are used directly.',
        start_unix=t0,end_unix=time.time(),cpu_s=time.process_time()-c0))
    print('Candidate scored on nine known diagnostic keys.',flush=True)


if __name__=='__main__':main()
