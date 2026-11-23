"""Freeze forward-pass geometry using baseline plant load IDs and clipped measured sole intersections."""
import argparse
import json
from pathlib import Path

from s1_w57_mechanism import matched_pairs, add_footprint_overlap

ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SPEC = ROOT/'experiments/terrain_memory/s1-w58-revisit-v2.json'
BG = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)


def geometry():
    result = {}
    for split, center in [('development', 1.40), ('validation', 1.68)]:
        spec = dict(target_strip_x_m=[round(center-.06, 2), round(center+.06, 2)],
                    target_strip_y_m=[-.24, -.06], center_park_x_m=center,
                    body_reversal_x_m=[round(center-.40, 2), round(center+.40, 2)],
                    free_target_hold_s=6., short_gap_upper_s=3., cycles=23)
        evidence = {}
        for case in 'AB':
            if split == 'development':
                legacy = ROOT.parent/'MCP-RL-w57'
                name = f'w57-mech-b-{case.lower()}-short-a01'
                raw = legacy/'outputs'/REL/name
                receipt = legacy/'results'/REL/'w57/episodes'/(name+'.json')
                windows = json.loads(receipt.read_text())['mechanism']['traversals']
            else:
                legacy = ROOT.parent/'MCP-RL-w56-r60'
                raw, = (legacy/'outputs'/REL).glob(f'job-*-f-{case.lower()}-r60-*-E30-attempt01')
                name = raw.name
                receipt = legacy/'results'/REL/'w56/f'/f'F-{case}-r60/E30.json'
                # A single forward passage of an existing original qualification.
                # Its reverse return cannot enter this 0..15 physical-second interval.
                windows = [dict(cycle=1, direction=1, start_s=0., end_s=15.)]
            forward = [row for row in windows if row['direction'] == 1]
            matches = matched_pairs(raw/'material-events.jsonl', forward, spec, 'short')
            add_footprint_overlap(raw, matches)
            evidence[case] = dict(baseline_attempt=name, baseline_path=str(raw),
                baseline_receipt=str(receipt), forward_crossings=len(forward),
                witnessed=[dict(traversal=x['cycle'], key=x['key'], source_id=x['source_id'],
                                release_id=x['release_id'], target_id=x['target_id'],
                                free_s=x['gap_s'], overlap_m2=x['exact_overlap_m2'])
                           for x in matches if x['source_leg']==1 and x['target_leg']==3
                           and x['condition_met'] and x['exact_overlap_m2']>0.])
        targets = [dict(target_id=f'F{i:02d}', traversal=2*i-1, direction=1, source_leg=1, target_leg=3,
                        region_xy_m=[spec['target_strip_x_m'], spec['target_strip_y_m']])
                   for i in range(1,13)]
        result[split] = dict(**spec, target_events=targets, geometry_basis=evidence,
                             observation_s={'short':168., 'recovered':288.})
    return dict(schema='W58_forward_cross_foot_v2', splits=result,
        history_conditions={'short':'Source release to immediately subsequent target load in (0,3) s.',
                            'recovered':'Park once per forward pass with all target cells free for 6 s. Require actual target gap >=6 s.'},
        burden='12 forward traversals, 11 backward transfers, final turn and home hold. The baseline took 4.6–5.3 s per interior traversal, plus approach and return; 168 s for direct and an additional 10 s per prepared pass. The windows are physical simulation durations.',
        independent_geometry='Development uses the W57 M0/B forward footprints. The independent evaluation uses a new route translated 0.28 m, disjoint target cell keys, and new virgin episodes. Prior original E30 geometry was used only for feasibility. Two nominal cases.',
        geometry_score='Immediate preceding source load, true release with positive unloaded duration, target load with a different declared leg, all within the same forward traversal. Positive area intersects both measured square soles and the same world cell. Report all twelve targets per episode, even if unreached.',
        **BG)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--output', type=Path, default=SPEC)
    dest=args.parse_args().output
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(geometry(), indent=2, ensure_ascii=False)+'\n')

