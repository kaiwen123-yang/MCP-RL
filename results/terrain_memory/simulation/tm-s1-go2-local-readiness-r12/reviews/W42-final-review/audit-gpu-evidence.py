#!/usr/bin/env python3
"""Audit delivered W41 GPU ZIPs; no simulator or training is run."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[6]
BASE=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'

def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),sum=float(a.sum()),min=float(a.min()),max=float(a.max()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('gpu-audit.json'));args=p.parse_args()
    report=json.loads((BASE/'gpu/w41-gpu/result.json').read_text())
    delivery=json.loads((BASE/'gpu/w41-gpu/delivery.json').read_text())
    g1={r['fixture']:r for r in report['stages']['G1']};g2={r['node']:r for r in report['stages']['G2']['node_blocks']}
    archives=[];read=[];nodes=[];events=[];logs={}
    for asset in delivery['assets']:
        path=args.raw_root/asset['name'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_size==asset['size'] and 'sha256:'+digest==asset['digest']
        with zipfile.ZipFile(path) as z:
            archives.append(dict(name=path.name,bytes=path.stat().st_size,sha256=digest,members=len(z.infolist())))
            for member in z.infolist():
                name=member.filename;short=Path(name).name
                if short.endswith('.log'):
                    value=z.read(member);logs[short]=value.decode(errors='replace')
                elif short.startswith('G2-node') and short.endswith('.npz'):
                    value=z.read(member)
                    with np.load(io.BytesIO(value)) as a:
                        index=int(a['node_index']);assert float(a['wall_s'])==g2[index]['wall_s']
                        numeric={k:a[k] for k in a.files if a[k].dtype.kind in 'fc'}
                        nodes.append(dict(node=index,wall_s=float(a['wall_s']),all_floats_finite=all(np.isfinite(v).all() for v in numeric.values()),
                            states_shape=list(a['states'].shape),AB_shape=list(a['AB'].shape),contact_slots=int(a['contact_dist'].shape[1]),
                            constraint_slots=int(a['efc_force'].shape[1]),penetrating_contact_slots_per_step=(a['contact_dist']<0).sum(axis=1).tolist(),
                            nonzero_constraint_forces_per_step=np.count_nonzero(a['efc_force'],axis=1).tolist(),
                            reported_solver_niter=np.unique(a['solver_niter']).tolist(),AB_max=float(abs(a['AB']).max()),dtype=str(a['AB'].dtype)))
                elif short.endswith('-gpu.npz'):
                    value=z.read(member);fixture=short[:-8]
                    if fixture not in g1:raise ValueError('Unknown G1 fixture '+fixture)
                    with np.load(io.BytesIO(value)) as a:
                        er=a['state_error'];ab=a['AB_error']
                        native_name='outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w41-dynamics/native-v2/'+fixture+'.npz'
                        native_member=z.getinfo(native_name)
                        with np.load(io.BytesIO(z.read(native_member))) as native:
                            ab_difference=abs(np.concatenate([native['A'],native['B']],axis=2)-a['AB'])
                            ab_reconstruction_error=float(abs(ab_difference-ab).max())
                            cpu=native['local_FD_endpoints'];gpu=a['states']
                            nonquat=np.concatenate([gpu[:,:,:3]-cpu[:,:,:3],gpu[:,:,7:]-cpu[:,:,7:]],axis=2)
                            recorded_nonquat=np.concatenate([er[:,:,:3],er[:,:,6:]],axis=2)
                            state_reconstruction_error=float(abs(nonquat-recorded_nonquat).max())
                            assert ab_reconstruction_error==0 and state_reconstruction_error==0
                        read.append(dict(archive=path.name,member=native_name,bytes=native_member.file_size,zip_crc32=native_member.CRC))
                        values=dict(position_max=float(abs(er[:,:,:18]).max()),velocity_max=float(abs(er[:,:,18:]).max()),AB_max=float(abs(ab).max()))
                        for k,v in values.items():assert np.isclose(v,g1[fixture][k],rtol=1e-12,atol=1e-15),(fixture,k)
                        events.append(dict(fixture=fixture,**values,aligned=g1[fixture]['aligned'],
                            AB_error_reconstruction_max=ab_reconstruction_error,nonquaternion_state_error_reconstruction_max=state_reconstruction_error,
                            all_floats_finite=all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fc')))
                else:continue
                read.append(dict(archive=path.name,member=name,bytes=member.file_size,zip_crc32=member.CRC))
    assert len(nodes)==176 and sorted(r['node'] for r in nodes)==list(range(176)) and len(events)==32
    timing=stats([g2[i]['wall_s'] for i in sorted(g2)])
    result=dict(baseline_commit='c4b3f9adc698b15de5c94ef952e2884f3abfec62',archives=archives,complete_members_read=read,
        G1=dict(count=len(events),aligned=sum(r['aligned'] for r in events),all_finite=all(r['all_floats_finite'] for r in events),events=events),
        G2=dict(count=len(nodes),timing_s=timing,total_hours=timing['sum']/3600,first_node_wall_s=g2[0]['wall_s'],
            other_nodes_mean_s=float(np.mean([g2[i]['wall_s'] for i in range(1,176)])),
            logical_FD_inputs=176*49,actual_substep_state_updates=176*49*10,
            all_finite=all(r['all_floats_finite'] for r in nodes),nodes=nodes,
            implementation='Host loop over 176 nodes; per-node 49 inputs with 10-step device scan; one initialized jit function, fixed shapes; no per-node XLA profile retained'),
        G4=dict(attempts=report['stages']['G4'],all_logs_confirm_missing_start_time=all('needs --attempt and --evaluation-start-unix' in logs['w41-gpu-'+b+'-PANN_PPO_S5-s2-update37-E30.log'] for b in ['native','mjx']),physical_evaluations=0),
        G5=dict(status=report['stages']['G5']['status'],evaluated_fixtures=len(report['stages']['G5']['result']['stages']),
            physical_evaluations=0,log_confirms_option_type_error="argument 'opt_tolerance' expects an array" in logs['mjwarp.log']),
        misleading_labels=dict(top=report['status'],G3=report['stages']['G3']['status']),
        memory=report['device_memory_stats'],scope='Delivered-array/log audit. No GPU or physical experiment run; no GPU resource availability inferred.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['archives','misleading_labels','scope']}))
    print(json.dumps(dict(G1=len(events),aligned=sum(x['aligned'] for x in events),G2=len(nodes),G2_timing=timing,read_members=len(read),read_bytes=sum(x['bytes'] for x in read))))
if __name__=='__main__':main()
