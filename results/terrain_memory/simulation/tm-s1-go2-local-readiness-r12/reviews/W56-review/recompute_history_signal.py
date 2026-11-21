#!/usr/bin/env python3
"""Read only published W56 tables; no simulator or material solver is imported."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56'
    histories = ['same_lap_same_foot', 'same_lap_cross_foot',
                 'earlier_lap_cross_foot', 'earlier_lap_same_foot']
    result = dict(
        schema='w56_review_published_exposure_signal_v1',
        source_commit='3590646bc15a59903c9561381368d985963dfe9e',
        scope='All rows of the published r55 A/B E300 actor-query tables with positive exact_query_next_touch_overlap_m2, grouped by the execution-side same_world_cell_history classification. These are query-cell rows, not independent experiments or all physical load events.',
        read_only=True, robot_or_material_solves=0, cases={}, comparisons=[])
    for case in ['A', 'B']:
        source = base/'exposure'/f'{case}-E300-query-cells.csv.gz'
        with gzip.open(source, 'rt') as file:
            rows = list(csv.DictReader(file))
        summaries = {}
        for history in histories:
            selected = [r for r in rows if r['same_world_cell_history'] == history
                        and float(r['exact_query_next_touch_overlap_m2']) > 0]
            summary = dict(query_cell_rows=len(selected),
                           distinct_next_real_load_ids=len({r['next_real_load_id'] for r in selected
                                                           if r['next_real_load_id']}),
                           exact_overlap_sum_m2=sum(float(r['exact_query_next_touch_overlap_m2']) for r in selected))
            for field in ['observer_q11', 'projected_free_depth_2ms_m',
                          'next_real_load_preload_q11', 'same_state_cap_delta_m_s']:
                values = np.array([abs(float(r[field])) for r in selected if r.get(field)], dtype=float)
                summary[field] = (dict(n=len(values), absolute_median_p95_max=np.quantile(values,[.5,.95,1]).tolist())
                                  if len(values) else None)
            summaries[history] = summary
        result['cases'][case] = dict(source=str(source.relative_to(args.repo)), all_rows=len(rows), histories=summaries)
    for build in ['r57','r58','r60']:
        for case in ['A','B']:
            for stage in ['E60','E300','E600']:
                source=base/'f'/f'F-{case}-{build}'/f'{stage}-paired.json'
                if not source.exists():
                    continue
                data=json.loads(source.read_text())
                old,new=data['baseline'],data['new']
                completion=lambda x: next((t['time_s'] for t in x['phase_transitions'] if t['phase']=='complete'), None)
                before,after=completion(old),completion(new)
                result['comparisons'].append(dict(case=case,build=build,stage=stage,
                    source=str(source.relative_to(args.repo)), baseline_attempt=old['attempt'], new_attempt=new['attempt'],
                    baseline_roundtrips=old['roundtrips'], new_roundtrips=new['roundtrips'],
                    baseline_slip=old['slip_integral'],new_slip=new['slip_integral'],
                    slip_relative_change_percent=100*(new['slip_integral']/old['slip_integral']-1),
                    baseline_peak_strain=old['peak_material_strain'],new_peak_strain=new['peak_material_strain'],
                    peak_strain_relative_change_percent=100*(new['peak_material_strain']/old['peak_material_strain']-1),
                    baseline_completion_s=before,new_completion_s=after,
                    completion_delta_s=after-before if before is not None and after is not None else None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),cases=list(result['cases']),comparisons=len(result['comparisons']))))


if __name__ == '__main__':
    main()
