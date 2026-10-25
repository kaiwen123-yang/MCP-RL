#!/usr/bin/env python3
"""Execute complete Go2 missions from the explicit S1 phase configuration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def read_index(config):
    path=ROOT/config['outputs']['compact_root']/'index.json'
    return path,json.loads(path.read_text()) if path.exists() else dict(
        run=config['run_id'],status='running',attempts=[],reused_records=[],
        new_training_runs=0,hardware_activated=False,NX_timing_measured=False)


def register_start(config,args,seed,fingerprint):
    """Count each physical reset, including an attempt that subsequently fails."""
    if 'phase_A' not in config:
        return
    path,index=read_index(config)
    attempts=index['attempts']
    development=args.development or args.cohort.startswith('development-')
    budget=config['budget']
    if development and seed!=config['execution']['development_seed']:
        raise ValueError('Development missions use the configured seed zero.')
    if len(attempts)>=budget['maximum_new_mission_attempts_including_development_failures_and_superseded_runs']:
        raise RuntimeError('The authorized new-mission budget is exhausted.')
    if development and sum(a['kind']=='development' for a in attempts)>=budget['maximum_development_missions']:
        raise RuntimeError('The authorized development-mission budget is exhausted.')
    if not development and sum(a['kind']=='formal' for a in attempts)>=budget['maximum_new_formal_episodes_if_affected_B0_must_be_recomputed']:
        raise RuntimeError('The authorized formal-mission budget is exhausted.')
    attempts.append(dict(attempt_id=len(attempts)+1,kind='development' if development else 'formal',
        cohort=args.cohort,terrain=args.terrain,controller=args.controller,seed=seed,
        started_utc=datetime.now(timezone.utc).isoformat(),status='started',
        command=[sys.executable,*sys.argv],execution_fingerprint=fingerprint))
    index['budget']=budget
    index['new_mission_starts']=len(attempts)
    index['development_mission_starts']=sum(a['kind']=='development' for a in attempts)
    save_json(path,index)


def register_result(config,args,seed,record_path):
    if 'phase_A' not in config:
        return
    path,index=read_index(config)
    row=json.loads(Path(record_path).read_text())
    attempt=next(a for a in reversed(index['attempts']) if a['cohort']==args.cohort and
        a['terrain']==args.terrain and a['controller']==args.controller and a['seed']==seed and a['status']=='started')
    attempt.update(status='finished',record=str(Path(record_path).relative_to(ROOT)),
        success=row['success'],termination=row['termination'],duration_s=row['duration_s'],
        trajectory=row['trajectory'],sha256=row['sha256'])
    save_json(path,index)


def reuse_B0(config,args):
    source_root=Path(args.reuse_source_root).resolve()
    old=config['frozen_first_S1']
    old_index_path=source_root/old['result_index']
    old_index=json.loads(old_index_path.read_text())
    source_cohort=old_index['cohort']
    source_small=old_index_path.parent
    destination=ROOT/config['outputs']['compact_root']/args.cohort
    path,index=read_index(config)
    for terrain in config['phase_A']['terrain_ids']:
        for seed in config['phase_A']['paired_optimizer_seeds']:
            source=source_small/source_cohort/f'{terrain}-C0-seed-{seed}.json'
            target=destination/f'{terrain}-B0-seed-{seed}.json'
            if target.exists():
                continue
            row=json.loads(source.read_text())
            provenance=dict(record=str(source),record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_commit=config['reviewed_source_commit'],source_controller='C0',source_cohort=source_cohort,
                trajectory_sha256=row['sha256'],material_sha256=row['material_sha256'],release=old['release'],
                physics_control_commit=old_index.get('physics_control_commit'),
                assets=[a for a in old_index['publication']['assets'] if
                    f'{terrain}-C0-' in a['filename']],
                equivalence='B0 environment, control path and random draws unchanged; logging-only additions do not require new physics.')
            for key in ('trajectory','material_trajectory'):
                row[key]=str(source_root/row[key])
            source_out=Path(row['trajectory']).parent
            row.update(controller='B0',reused_from=provenance,
                events_path=str(source_out/f'{terrain}-C0-seed-{seed}-events.json'),model_path=str(source_out/'go2.mjb'),
                unavailable_diagnostics=['aligned_nominal_forecast','preload_state','unique_actual_event_id',
                    'continuous_loaded_strain_rate','activation_release_strain_jump','actual_foot_vz_over_H','internal_qdot'])
            save_json(target,row)
            index['reused_records'].append(dict(record=str(target.relative_to(ROOT)),**provenance))
    index['B0_reuse_source_index']=str(old_index_path)
    index['B0_reuse_source_release']=old['release']
    save_json(path,index)


def phase_B_readiness(config,args):
    small=ROOT/config['outputs']['compact_root']
    rows=[json.loads(p.read_text()) for p in (small/args.cohort).glob('*-B1-seed-*.json')]
    completed={terrain:sum(r['terrain']==terrain and r['success'] for r in rows)
               for terrain in config['phase_B']['terrain_ids']}
    evidence_path=ROOT/args.forecast_evidence if args.forecast_evidence else None
    evidence=bool(args.forecast_readiness_confirmed and evidence_path and evidence_path.is_file())
    required=config['phase_B']['readiness']
    ready=(completed.get('rigid',0)==required['B1_rigid_completed_of_5'] and
        completed.get('kv_moderate',0)>=required['B1_kv_moderate_minimum_completed_of_5'] and evidence)
    path,index=read_index(config)
    index['phase_B_readiness']=dict(ready=ready,B1_completions=completed,
        confirmed_prefix_repairs_with_same_time_evidence=evidence,
        evidence_file=str(evidence_path) if evidence_path else None,
        evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest() if evidence else None,
        decision='execute authorized phase B' if ready else 'phase B not run; readiness not met')
    save_json(path,index)
    print(json.dumps(index['phase_B_readiness']),flush=True)
    return ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/terrain_memory/s1-control-repair.json')
    parser.add_argument('--terrain',choices=['rigid','tm02_anchor','kv_soft','kv_moderate'])
    parser.add_argument('--controller',choices=['B0','B1','B2','B3','C0','C1','C2','C3','C4'])
    parser.add_argument('--matrix',action='store_true',help='Read terrain, controller and seed cells from the configured phase.')
    parser.add_argument('--phase',choices=['A','B','all'],default='all')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    parser.add_argument('--cohort',default='development-v1')
    parser.add_argument('--development',action='store_true',help='Count a seed-zero mission in the development budget even in the final cohort.')
    parser.add_argument('--reuse-B0',action='store_true',help='Reference old C0 records after confirming unchanged B0 physics/control/random draws.')
    parser.add_argument('--reuse-source-root',default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-s1')
    parser.add_argument('--forecast-readiness-confirmed',action='store_true')
    parser.add_argument('--forecast-evidence',help='Existing same-time forecast/actual evidence supporting the confirmed prefix repairs.')
    parser.add_argument('--source-output-root',default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-study/outputs/terrain_memory/reproduction/material/tm02-memory-study-20261023')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    if args.reuse_B0:
        reuse_B0(config,args)
    if args.matrix:
        phases=['A','B'] if args.phase=='all' else [args.phase]
        for phase in phases:
            if phase=='B' and not phase_B_readiness(config,args):
                continue
            matrix=config['phase_'+phase]
            for terrain in matrix['terrain_ids']:
                for method in matrix['controller_ids']:
                    for seed in matrix['paired_optimizer_seeds']:
                        record=ROOT/config['outputs']['compact_root']/args.cohort/f'{terrain}-{method}-seed-{seed}.json'
                        if record.exists() and json.loads(record.read_text()).get('reused_from'):
                            continue
                        subprocess.run([sys.executable,str(Path(__file__)),'--config',args.config,
                            '--terrain',terrain,'--controller',method,'--seeds',str(seed),
                            '--cohort',args.cohort,'--source-output-root',args.source_output_root,
                            *(['--development'] if args.development else [])],check=True)
        return
    if args.terrain is None or args.controller is None:
        parser.error('Specify --matrix, or both --terrain and --controller.')
    if 'phase_A' in config and args.controller not in ('B0','B1','B2','B3'):
        parser.error('The repair round contains B0/B1/B2/B3 only.')
    if args.controller in ('B2','B3'):
        path,index=read_index(config)
        if not index.get('phase_B_readiness',{}).get('ready'):
            parser.error('Phase B requires the recorded complete-B1 readiness decision.')
    args.register_start=lambda seed,fingerprint:register_start(config,args,seed,fingerprint)
    args.register_result=lambda seed,record_path:register_result(config,args,seed,record_path)
    from tm02.s1_robot import run
    run(ROOT,args)


if __name__ == '__main__':
    main()
