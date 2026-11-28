"""Generate independently seeded virgin specimens using TM02's six-component implicit reference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tm02.reference import Reference

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'experiments/terrain_memory/w61/material/family-v1.json'
COMPACT = ROOT / 'results/terrain_memory/simulation/w61/material'
WORK = ROOT / 'work/w61/material'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions(seed, config):
    original = json.loads((ROOT / config['reference_config']).read_text())
    p = dict(original['reference_material']['parameters'])
    rng = np.random.default_rng(seed)
    k, g, clock, flow = np.exp(rng.uniform(-.65, .65, size=4))
    p.update(K_eq_MPa=p['K_eq_MPa'] * k, G_eq_MPa=p['G_eq_MPa'] * g,
             K_ov_MPa=p['K_ov_MPa'] * k**.7, G_ov_MPa=p['G_ov_MPa'] * g**.7,
             eta_K_table_value=p['eta_K_table_value'] * k**.7 * clock,
             eta_D_table_value=p['eta_D_table_value'] * g**.7 * clock,
             a_MPa=p['a_MPa'] * flow)
    return p, original['numerics']


def protocol(rng, index, length, holdout=False):
    phase = np.linspace(0, 1, length)
    amplitude = rng.uniform(.006, .014)
    if index < 4:
        cycle = [.5, .68, .42, .55][index]
        load = np.interp(phase, [0, .21, cycle, .81, 1], [0, 1, 1, .1, 0])
        dt = [.032, .064, .024, .045][index]
    elif index == 4:
        load = np.interp(phase, [0,.12,.55,.68,1], [0,1,1,0,0])
        dt = .010 if holdout else .017
    elif index == 5:
        load = np.sin(2*np.pi*phase) * (.75+.25*phase)
        dt = .105 if holdout else .080
    elif index == 6:
        load = np.interp(phase,[0,.14,.27,.34,.47,.64,.78,.90,1], [0,1,1,0,0,.72,.72,0,0])
        dt = .028 if holdout else .035
    else:
        load = np.interp(phase,[0,.15,.33,.55,.70,.81,1], [0,-.45,-.45,1,1,0,0])
        dt = .072 if holdout else .052
    tensor = rng.normal(size=(3,3))
    tensor = (tensor + tensor.T)/2
    tensor /= np.linalg.norm(tensor, ord=2)
    modes = [-np.diag([1.,0,0]), np.eye(3)*.65,
             np.array([[0,1.,0],[1.,0,0],[0,0,0]]), tensor]
    main = modes[index % 4]
    # Small non-proportional loading path; principal strain stays below .02.
    extra = .13*np.sin(2*np.pi*phase)*np.abs(load)
    e = amplitude * (load[:,None,None] * main + extra[:,None,None]*tensor)
    values = e[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]
    step = np.full(length, dt * rng.uniform(.90, 1.10))
    step[0] = 0.
    return values, step


def create(config):
    records = {}
    index = {}
    for split, (lo, hi) in config['material_ids'].items():
        rows = []
        for sid in range(lo, hi+1):
            teacher, settings = conditions(sid, config)
            law = Reference(teacher)
            paths = []
            for pidx in range(8):
                seed = config['history_seed_offset'] + 100*sid + pidx
                rng = np.random.default_rng(seed)
                strain, dt = protocol(rng, pidx, config['length'], split == 'holdout')
                stress = law.path(dt.cumsum(), strain, settings)['stress']
                observed = stress + rng.normal(0,config['support_measurement_sd_MPa'],stress.shape)
                observed[0] = 0
                paths.append((strain, dt, observed, stress))
            rows.append(tuple(np.stack(parts) for parts in zip(*paths)))
        for name, value in zip(('strain','dt','observed','reference'), zip(*rows)):
            records[f'{split}_{name}'] = np.stack(value)
        index[split] = dict(specimen_ids=list(range(lo, hi+1)), independent_histories=len(rows)*8,
            support_programs=[0,1,2,3], query_programs=dict(enumerate(config['query_paths'],4)),
            seed_rule='history_seed_offset + 100*specimen_id + program', steps_per_history=config['length']-1)
    dataset = WORK / 'coupons.npz'
    np.savez_compressed(dataset, **records)
    index.update(dataset_sha256=digest(dataset), bytes=dataset.stat().st_size, generator='scripts/w61/material/coupons.py',
        command='python scripts/w61/material/train.py', initial_condition='Independently virgin per physical trajectory',
        stress='Six nominal tensor components in MPa, physical shear. Truth withheld from support except noisy measurements.',
        geometry='homogeneous strain-controlled coupon; no force/indentation conversion',
        independent_teacher='TM02 analytical nonlinear overstress-dependent viscosity and implicit Newton solver; unmodified.')
    (COMPACT/'samples.json').write_text(json.dumps(index,indent=2)+'\n')


def unlabeled_probes(config):
    # Seeds, input histories and durations are independent of every specimen.
    return list(zip(*(protocol(np.random.default_rng(6130000+i), i, config['length']) for i in [1,2,4,6])))
