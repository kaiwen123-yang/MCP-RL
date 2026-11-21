"""Read-only timing audit of published r55 E300 query-cell evidence."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
relative = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/exposure')
output = {
    'schema': 'w56_published_query_to_next_load_delay_v1',
    'scope': 'Published r55 E300 query-cell rows; positive exact query/next-touch overlap. '
             'These correlated rows are not independent episodes. The measured delay is '
             'next_real_load_s minus query time_s, not source-load-to-reload or previous-event age.',
    'robot_or_material_solves': 0,
    'horizon_comparison_s': 0.35,
    'cases': {},
}
for case in ['A', 'B']:
    source = relative / f'{case}-E300-query-cells.csv.gz'
    with gzip.open(args.repo / source, 'rt') as file:
        rows = list(csv.DictReader(file))
    groups = {}
    for history in ['same_lap_cross_foot', 'same_lap_same_foot',
                    'earlier_lap_cross_foot', 'earlier_lap_same_foot']:
        selected = [row for row in rows if row['same_world_cell_history'] == history
                    and float(row['exact_query_next_touch_overlap_m2']) > 0]
        values = {'query_cell_rows': len(selected)}
        for key in ['observer_q11', 'projected_free_depth_2ms_m',
                    'same_state_cap_delta_m_s', 'previous_event_age_s']:
            sample = np.abs([float(row[key]) for row in selected if row[key] != ''])
            values[key] = {'available_rows': len(sample),
                           'absolute_median_p95_max': np.quantile(sample, [0.5, 0.95, 1]).tolist()}
        delay = np.array([float(row['next_real_load_s']) - float(row['time_s'])
                          for row in selected if row['next_real_load_s'] != ''])
        values['query_to_next_actual_load_delay_s'] = {
            'available_rows': len(delay),
            'min_median_p95_max': np.quantile(delay, [0, 0.5, 0.95, 1]).tolist(),
            'within_350ms_rows': int(np.sum((delay >= 0) & (delay <= 0.35))),
            'within_350ms_fraction': float(np.mean((delay >= 0) & (delay <= 0.35))),
        }
        groups[history] = values
    output['cases'][case] = {'source': str(source), 'all_rows': len(rows), 'groups': groups}
args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({case: data['groups']['same_lap_cross_foot']['query_to_next_actual_load_delay_s']
                  for case, data in output['cases'].items()}, indent=2))
