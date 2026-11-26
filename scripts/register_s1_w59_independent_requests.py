"""Retain charged A development and lock fresh unperturbed sources for W59 admission/actor checks."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from register_s1_w59_interventions import STRATA, SEEDS, observed, spaced, rows
from run_s1_w59_interventions import ROOT, REL, SMALL, LARGE, immutable, sha
from s1_w59_model_fields import STATE


def original_source(raw):
    spec=json.loads((LARGE/'jobs'/raw.name/'execution-contract.json').read_text())
    attempt=json.loads((SMALL/'attempts'/(raw.name+'.json')).read_text())
    settings=spec['w59'];task=spec['w57']['task_mode'];case=attempt['case']
    if task not in ('original','short','recovered') or not attempt.get('ended_unix_s'):
        raise ValueError('Only complete unperturbed source episodes can be registered')
    if task=='original' and attempt['stage_status']!='passed':
        raise ValueError('An unsuccessful original qualification cannot supply this new controller')
    actual={v['decision_id']:v for v in rows(raw/'w55-f/handover-requests.jsonl') if v['stage']=='request'}
    items=observed(raw,case)
    policy=('real P3 at each subsequent legal request' if settings['arm'].startswith('P3') else
            [0.,.060,.05] if settings['arm']=='Fixed-action' else [0.,.050,0.])
    for r in items:
        rid=r['request_id'];v=actual[rid]
        r.update(source_arm=settings['arm'],source_material=settings['material'],source_task=task,source_settings=settings,
            source_command_after_first=policy,source_request=v['macro_action'],source_attempt=raw.name,
            source_contract_sha256=attempt['execution_contract_sha256'],
            prior_non_nominal=any(max(abs(np.array(e['macro_action'])-[0,.05,0.]))>1e-7
                                  for i,e in actual.items() if i<rid))
        if spec['w57'].get('mechanism'):r['mechanism']=spec['w57']['mechanism']
        argv=attempt['launcher_command']
        if '--wm-recipe' in argv:
            r['source_recipe_path']=argv[argv.index('--wm-recipe')+1]
    with np.load(raw/'trajectory.npz') as z:
        states=np.column_stack((z['qpos'][::10],z['qvel'][::10])).astype(np.float32)
    return items,states


def exclude_exposed(candidates,new_states,bank,extra):
    # Apply numerical closeness on synchronous 20ms frames across every complete trajectory
    # actually exposed in the old model bank. Non-synchronous exact state reuse is checked
    # with rounded body/joint position and velocity signatures as well.
    used=np.zeros(len(new_states),bool)
    decimals=np.r_[np.full(19,1.e6),np.full(18,1.e5)].astype(np.float32)
    def keys(a):
        q=np.rint(a*decimals).astype('<i4')
        return q.view(np.dtype((np.void,q.shape[1]*4))).ravel()
    needle=keys(new_states)
    opened=[]
    def compare(other):
        n=min(len(new_states),len(other))
        used[:n]|=((abs(new_states[:n,:19]-other[:n,:19]).max(1)<=1e-6)&
                   (abs(new_states[:n,19:]-other[:n,19:]).max(1)<=1e-5))
        used[:]|=np.isin(needle,keys(other))
    for f in sorted(bank.glob('*.npz')):
        with np.load(f) as z:
            x=z['state'];compare(np.column_stack((x[:,STATE['qpos']],x[:,STATE['qvel']])))
        opened.append(f.stem)
    if not opened:raise ValueError('The observed-model bank must be available to screen source reuse')
    for other in extra:
        compare(other)
    valid=[]
    for c in candidates:
        tick=round(c['request_s']/.020)
        if tick<25 or tick>=len(used) or used[tick-25:tick+1].any():continue
        if not c['prior_non_nominal'] and c['source_task']=='original':continue
        valid.append(c)
    return valid,dict(complete_bank_episodes=len(opened),extra_trajectories=len(extra),
        excluded_registered_times=len(candidates)-len(valid),kept=len(valid),
        screened_bank_episodes_sha256=hashlib.sha256('\n'.join(opened).encode()).hexdigest())


def register(dev,paths,output):
    old=json.loads(dev.read_text())
    prior=old['states']['development']
    if len(prior)!=12 or len({v['snapshot_id'] for v in prior})!=12:
        raise ValueError('The already charged 12 development states must be retained verbatim')
    bank=ROOT/'work/w59/model/aligned'
    prior_refs=[]
    for p in sorted(LARGE.glob('w59-action-development-*')):
        source=p/'trajectory.npz'
        if source.exists():
            with np.load(source) as z:
                # The trajectory starts at a legal event with an arbitrary offset on the 20ms lattice.
                for offset in range(10):
                    prior_refs.append(np.column_stack((z['qpos'][offset::10],z['qvel'][offset::10])).astype(np.float32))
    pools={};receipts={};previous=[]
    for raw in paths:
        rows_,states=original_source(raw)
        available,receipt=exclude_exposed(rows_,states,bank,prior_refs+previous)
        pools[raw.name]=available
        receipts[raw.name]=dict(**receipt,source=str(raw.resolve()),labels=dict(Counter(row['stratum'] or 'unlabelled' for row in available)))
        previous.append(states)
    occupied={};sealed=[];order=list(pools)
    # Prefer one B episode for the admission comparison when all strata exist there.
    # If no such episode exists, use the registered source ordering within each stratum.
    preferred=next((name for name in order if pools[name] and pools[name][0]['case']=='B' and
        all(sum(v['stratum']==k for v in pools[name])>=4 for k in STRATA)),None)
    anc=pools[preferred] if preferred else [row for n in order for row in pools[n]]
    for k in STRATA:sealed.extend(spaced([v for v in anc if v['stratum']==k],4,occupied,'W59-v3/held/'+k))
    # Two-second first-update continuations have disjoint physical windows from every sealed state
    # on their shared source episode; all 24 IDs are distinct, with balanced cases when possible.
    others=[v for n in order for v in pools[n] if all(
        q['source_attempt']!=v['source_attempt'] or abs(q['request_s']-v['request_s'])>=2.+1e-7 for q in sealed)]
    first={}
    for seed in SEEDS:
        chosen=[]
        for c in ('A','B'):chosen.extend(spaced([v for v in others if v['case']==c],4,occupied,str(seed)+c))
        if len(chosen)<8:chosen.extend(spaced(others,8-len(chosen),occupied,str(seed)+'/remainder'))
        first[seed]=chosen
    ok=len(sealed)==12 and all(len(first[x])==8 for x in SEEDS)
    rec=dict(schema='W59_locked_requests_v3_disjoint_source',status='locked' if ok else 'blocked_incomplete_independent_sources',
        development_registry=str(dev.resolve()),development_registry_sha256=sha(dev),
        revoked='v2 B/actor duplicated the W57 M1/B model development reference',
        original_window_rule='The 0.5 s prefix before each legal request must differ from historical model/source states; original-route control has a prior nonnominal request. A new mechanism geometry can independently produce a divergent prefix.',
        source_screen=receipts,preferred_admission_source=preferred,states=dict(development=prior,sealed_admission=sealed),
        first_update=first,signed=old['signed'],duration_s=old['duration_s'],
        action_witness='For a new source the first request is the same absolute nominal or signed request as development. Both repeats share the full physical/observer/RNG state. The source controller resumes naturally at its next legal request; an originally different selected request can produce a different subsequent reference.',
        sealed_count=len(sealed),first_update_counts={k:len(v) for k,v in first.items()},
        unique_legal_states=len(occupied)+len(prior))
    if ok:immutable(output,json.loads(json.dumps(rec)))
    return rec


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--source',action='append',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=register(a.development,a.source,a.output)
    print(json.dumps({k:r[k] for k in ('status','source_screen','sealed_count','first_update_counts')}))
