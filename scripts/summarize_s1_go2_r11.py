#!/usr/bin/env python3
"""Derive R11 coverage and totals from actual unique attempt records."""
import csv,json,time
from pathlib import Path
from s1_go2_r11_protocol import ROOT,CONTRACT,LEVELS,CONDITIONS
from s1_go2_r11_runtime import index_file
SMALL=ROOT/CONTRACT['outputs']['compact_root']
def main():
    with index_file(SMALL,CONTRACT) as index:
        attempts=index['attempts'];physical=[a for a in attempts if a.get('physical_started')]
        records=[]
        for attempt in attempts:
            if not attempt.get('result'):continue
            record=json.loads((SMALL/attempt['result']).read_text())
            resolved=json.loads((SMALL/(attempt['attempt']+'-resolved.json')).read_text())
            record['controller_build']=resolved['r11_configuration']['controller_build']
            record['qualification_usable']=attempt.get('qualification_usable',True)
            record['declared_parameter_mismatch']=attempt.get('declared_parameter_mismatch')
            records.append(record)
        index['actual_summary']=dict(total_launches=len(attempts),physical_starts=len(physical),
            zero_step_failures=sum(a.get('status')=='initialization_failed' and not a.get('physical_started') for a in attempts),
            closed_record_simulation_s=sum(r['simulation_s'] for r in records),
            closed_record_physical_steps=sum(r['physical_samples'] for r in records),
            completed_stage_records=sum(r.get('stage_status') in ('passed','physical_material_failure','task_criteria_unmet') for r in records),
            technical_or_budget_partial=sum(r.get('stage_status') in ('technical_interruption','budget_truncated') for r in records),
            active_attempts=[a['attempt'] for a in attempts if a['status'] in ('running','initializing')],
            elapsed_campaign_h=(time.time()-index['campaign_started_unix_s'])/3600 if 'campaign_started_unix_s' in index else None)
        frozen=index.get('frozen_controller')
        accepted_common=(frozen or {}).get('case_identity',{}).get('common_configuration_sha256')
        case_rows=[];stage_rows=[]
        identities=[('FIXED_S5',None,c) for c in CONDITIONS]+[(g,s,c) for s in range(3) for c in CONDITIONS for g in ['PPO_RAW_S5','PANN_PPO_S5']]
        for group,seed,condition in identities:
            candidates=[r for r in records if r.get('purpose')!='training' and r.get('case_identity') and
                (r['case_identity']['group'],r['case_identity']['seed'],r['condition'])==(group,seed,condition) and
                r['case_identity']['terrain']=='tm02_anchor']
            # Keep each controller/case version separate. Only a frozen common version is formal.
            if frozen:
                candidates=[r for r in candidates if r['case_identity']['binary_sha256']==frozen['case_identity']['binary_sha256']]
                if group!='FIXED_S5':
                    primary=index.get('primary_checkpoints',{}).get(f'{group}-seed{seed}')
                    if primary:
                        from s1_go2_r11_protocol import digest
                        candidates=[r for r in candidates if r['case_identity']['checkpoint_sha256']==digest(primary)]
                    else:candidates=[]
            elif candidates:
                latest=candidates[-1]['case_key'];candidates=[r for r in candidates if r['case_key']==latest]
            highest=None;blocker=None
            for level in LEVELS:
                runs=[r for r in candidates if r['evaluation_stage']['level']==level]
                r=runs[-1] if runs else None
                if r:
                    status=r['stage_status'];blocker=r['attempt'] if status!='passed' else None
                    if status=='passed':highest=level
                    reason=r['reason']
                elif blocker:
                    status='blocked_by_predecessor';reason=blocker
                elif not frozen:
                    status='planned';reason='S5 nominal E60 qualification not established'
                elif group!='FIXED_S5' and not index.get('primary_checkpoints'):
                    status='planned';reason='No common frozen trained checkpoint'
                else:
                    status='eligible' if level=='E30' or highest==LEVELS[LEVELS.index(level)-1] else 'blocked_by_predecessor'
                    reason='Not yet executed' if status=='eligible' else 'Independent predecessor not passed'
                    blocker='unexecuted '+level
                stage_rows.append(dict(group=group,seed=seed,condition=condition,level=level,status=status,
                    attempt=r['attempt'] if r else None,case_key=r['case_key'] if r else None,reason=reason,
                    predecessor=r['qualification']['predecessor_attempt'] if r else None))
            case_rows.append(dict(group=group,seed=seed,condition=condition,highest_passed_stage=highest,
                controller_build=(frozen or {}).get('build') or (candidates[-1]['controller_build'] if candidates else None),
                controller_frozen=bool(frozen),failure_attempt=next((r['attempt'] for r in reversed(candidates) if not r['success']),None),
                failure_reason=next((r['reason'] for r in reversed(candidates) if not r['success']),None)))
        index['development_qualification_history']=[dict(
            attempt=r['attempt'],controller_build=r['controller_build'],case_key=r['case_key'],
            level=r['evaluation_stage']['level'],observed_task_outcome=r['stage_status'],
            qualification_usable=r['qualification_usable'],reason=r['reason'],
            parameter_mismatch=r['declared_parameter_mismatch'])
            for r in records if r['purpose']=='development']
        index['accepted_core_controller']=frozen['build'] if frozen else None
        index['case_coverage']=case_rows
        index['stage_coverage']={level:{status:sum(x['level']==level and x['status']==status for x in stage_rows)
            for status in sorted(set(x['status'] for x in stage_rows))} for level in LEVELS}
        index['evaluation_manifest']='evaluation-manifest.csv'
        for name,rows in [('case-highest-stage.csv',case_rows),('evaluation-manifest.csv',stage_rows)]:
            with (SMALL/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if records:
            cols=['attempt','purpose','condition','controller_build','case_key','qualification_usable','stage_status','success','reason','simulation_s','wall_s','physical_samples','first_completion_s','completed_round_trips','source_commit']
            with (SMALL/'episodes.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k) for k in cols} for r in records)
        print(json.dumps(index['actual_summary']),flush=True)
if __name__=='__main__':main()
