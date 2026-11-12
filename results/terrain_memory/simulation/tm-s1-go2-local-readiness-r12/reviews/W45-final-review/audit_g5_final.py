#!/usr/bin/env python3
"""Verify the existing G5 final ZIP using frozen review metadata.

Requires Python 3.11+, NumPy, the ZIP, and g5-final-audit.json. No network,
repository checkout, extracted originals, simulator, or GPU is required.
"""
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import collections
import hashlib
import json
import zipfile
import numpy as np

PREFIX='outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'

def read_json_member(archive, member):
    return json.loads(archive.read(str(member)).decode('utf-8'))

def npz_member(archive, member):
    return np.load(BytesIO(archive.read(str(member))), allow_pickle=False)

def jsonl_member(archive, member):
    return [json.loads(s) for s in archive.read(str(member)).decode('utf-8').splitlines() if s.strip()]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=Path(__file__).resolve().parent,
                        help='Directory containing the verified G5 ZIP; default: this script directory.')
    parser.add_argument('--metadata',type=Path,
                        help='Frozen g5-final-audit.json; default: audit-dir, then this script directory.')
    parser.add_argument('--output',type=Path,
                        help='Output JSON; default: AUDIT_DIR/g5-final-recomputed.json (preserves input metadata).')
    args=parser.parse_args()
    if not __debug__:
        parser.error('Run without -O: verification uses assertions.')
    root=args.audit_dir.resolve()
    metadata_path=args.metadata
    if metadata_path is None:
        metadata_path=root/'g5-final-audit.json'
        if not metadata_path.exists():
            metadata_path=Path(__file__).resolve().parent/'g5-final-audit.json'
    metadata_bytes=metadata_path.read_bytes()
    metadata=json.loads(metadata_bytes)
    asset=metadata['release'];name=asset['asset_name'];attempt=metadata['final']['attempt']
    result_member=metadata['final']['result_member']
    continuation_result_member=metadata['final']['continuation_result_member']
    path=root/name
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert path.stat().st_size==asset['size'] and digest==asset['sha256']
    assert 'sha256:'+digest==asset['remote_digest']
    z=zipfile.ZipFile(path);manifest_member='manifests/'+name[:-4]+'.json'
    manifest=read_json_member(z,manifest_member)
    coverage=[]
    for m in manifest:
        b=z.read(m['member'])
        assert len(b)==m['size']==m['source_size'] and m['offset']==0
        coverage.append(dict(path=m['member'],bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    result=read_json_member(z,result_member)
    out=PurePosixPath(PREFIX)/attempt
    continuation=PurePosixPath(continuation_result_member).parent
    completed=read_json_member(z,continuation_result_member)
    claims=[]
    for d in [result['trajectory']]+result['chunks']:
        member=out/PurePosixPath(d['path']).name;b=z.read(str(member));h=hashlib.sha256(b).hexdigest()
        assert len(b)==d['size'] and h==d['sha256']
        claims.append(dict(name=member.name,bytes=len(b),sha256=h))
    calls=jsonl_member(z,out/'planner-calls.jsonl')
    execution=jsonl_member(z,out/'action-execution.jsonl')
    fixture=[]
    for i,stage in zip([0,4,11,20,23,27],completed['stages']):
        arrays=[];maxima=[];finite=True
        for eps in completed['fd_steps']:
            with npz_member(z,continuation/f'mjwarp-{i:02d}-eps{eps:g}.npz') as a:
                assert a['AB'].shape==(10,36,48) and a['states'].shape==(10,37)
                finite=finite and bool(np.isfinite(a['AB']).all() and np.isfinite(a['states']).all())
                arrays.append(a['AB']);maxima.append(float(np.max(abs(a['AB']))))
        rel=[float(np.linalg.norm(arrays[j]-arrays[j+1])/max(np.linalg.norm(arrays[j+1]),1e-12)) for j in [0,1]]
        assert np.array_equal(maxima,[x['AB_max'] for x in stage['step_sensitivity']])
        relative_recompute_error=float(np.max(np.abs(np.asarray(rel)-stage['adjacent_relative_AB_changes'])))
        # FP64 norm reduction may differ by a few ulps across BLAS runtimes.
        assert relative_recompute_error <= 32*np.finfo(float).eps*max(1.,max(abs(x) for x in rel))
        fixture.append({'fixture':stage['fixture'],'all_finite':finite,'AB_absolute_maxima':maxima,
                        'adjacent_relative_AB_changes':rel,'relative_recompute_max_error':relative_recompute_error})
    with npz_member(z,out/'trajectory.npz') as a:
        tr={'samples':len(a['time']),'first_time_s':float(a['time'][0]),'last_time_s':float(a['time'][-1]),
            'max_body_x_m':float(np.max(a['qpos'][:,0])),'min_body_z_m':float(np.min(a['qpos'][:,2])),
            'final_body_xyz_m':a['qpos'][-1,:3].tolist(),
            'nonfoot_contact_rows':int(np.count_nonzero(a['nonfoot_ground_contacts'])),
            'all_states_finite':bool(np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all())}
        assert tr['max_body_x_m']==result['max_x_m'] and tr['last_time_s']==result['simulation_s']
    with npz_member(z,out/'physical-000.npz') as a:
        phy=a['physical'];pc=result['physical_columns'];col={n:i for i,n in enumerate(pc)}
        physical={'rows':len(phy),'columns':len(pc),'last_post_integration_s':float(phy[-1,col['post_integration_time_s']]),
                  'nonfoot_contact_rows':int(np.count_nonzero(phy[:,col['nonfoot_ground_contacts']])),
                  'all_finite':bool(np.isfinite(phy).all())}
    with npz_member(z,out/'prediction-000.npz') as a:
        pred={'origins':len(a['plan_origin']),'states_shape':list(a['plan_states'].shape),
              'actions_shape':list(a['plan_actions'].shape),'times_shape':list(a['plan_times'].shape)}
    requests=completed['evaluations'][0]['requests']
    audit={
      'audit_source_commit':metadata['audit_source_commit'],
      'scope':'Existing G5 final published originals recovered and verified; no server execution, physics or training.',
      'verification_inputs':{'metadata_file':metadata_path.name,
        'metadata_sha256':hashlib.sha256(metadata_bytes).hexdigest(),
        'remote_metadata_scope':'Release URL, size and digest are frozen review metadata; this run makes no network request.',
        'historical_git_context_scope':'The historical Git mismatch is carried from the frozen audit, not re-queried or used as numerical evidence.'},
      'release':dict(asset,sha256=digest,download_size_and_sha256_verified=True),
      'manifest_coverage':{'zip_entries':len(z.infolist()),'source_entries':len(manifest),
        'source_bytes':sum(x['bytes'] for x in coverage),'manifest_bytes':len(z.read(manifest_member)),
        'all_sources_read_and_zip_crc_verified':True,'all_sizes_match':True,'all_offsets_zero_full_sources':True,
        'source_hashes_computed':len(coverage),'source_hash_note':'Manifest does not publish per-source SHA256. Whole-asset remote SHA256 matches; independently computed per-source SHA256 retained.',
        'result_embedded_sha256_claims_verified':claims},
      'current_git_mismatch':metadata.get('current_git_mismatch'),
      'final':{'status':completed['status'],'first_error':completed['first_error'],'exit_code':completed['evaluations'][0]['exit_code'],
        'attempt':attempt,'result_member':result_member,'continuation_result_member':continuation_result_member,
        'source_commit':result['source_commit'],'controller_build':result['case_identity']['controller_build'],
        'controller_artifact':result['case_identity']['controller_artifact'],'policy_samples':result['policy']['total_transitions'],
        'frozen_policy':'W40-B checkpoint009617/update37','training':result['policy']['training'],
        'success':result['success'],'reason':result['reason'],'stage_status':result['stage_status'],
        'simulation_s':result['simulation_s'],'wall_s':result['wall_s'],'wall_hours':result['wall_s']/3600,
        'actual_material_loaded':result['stage_progress']['actual_material_loaded'],'material_world_cells':result['material']['world_cells'],
        'legal_route':result['legal_route'],'deadline_unix_s':completed['deadline_unix_s'],
        'predictor_only':completed['predictor_only'],'original_cpu_plant_unchanged':completed['original_cpu_plant_unchanged'],
        'versions':completed['versions'],'dtype':completed['dtype']},
      'calls':{'planner_records':len(calls),'first_call_id':calls[0]['call_id'],'last_call_id':calls[-1]['call_id'],
        'return_reason_counts':dict(collections.Counter(x['return_reason'] for x in calls)),
        'all_metric_valid':all(x['metric_valid'] for x in calls),'gpu_request_count':requests,
        'aggregate_requests_per_planner_record':requests/len(calls),
        'action_execution_records':len(execution),'action_query_kind_counts':dict(collections.Counter(x['query_kind'] for x in execution)),
        'reported_timing':result['timing'],'planner_record_wall_sum_s':sum(x['wall_s'] for x in calls)},
      'trajectory_recomputed':tr,'physical_recomputed':physical,'prediction_shapes':pred,
      'fd_fixtures_recomputed':fixture,'selected_fd_step':completed['selected_fd_step'],
      'conclusion':'G5 finished the full 30 s observation without task progress. It is a scientific failure, not still running or the older Python-float error. Do not rerun to collect files; final originals are already on GitHub.'}
    z.close()
    output=args.output or root/'g5-final-recomputed.json'
    if output.resolve()==metadata_path.resolve():
        parser.error('--output must differ from the frozen input metadata file.')
    output.parent.mkdir(parents=True,exist_ok=True)
    hashes_output=output.with_name('g5-source-hashes-recomputed.json')
    hashes_output.write_text(json.dumps(coverage,indent=2)+'\n',encoding='utf-8')
    output.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'source_hashes_output':str(hashes_output),
        'verified_sources':len(coverage),'planner_records':len(calls),'gpu_requests':requests,
        'simulation_s':result['simulation_s'],'reason':result['reason']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
