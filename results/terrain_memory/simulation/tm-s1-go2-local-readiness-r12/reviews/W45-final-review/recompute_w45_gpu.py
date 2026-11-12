#!/usr/bin/env python3
"""W45 published-original audit using a local repository and two verified ZIPs.

Requires Python 3.11+ and NumPy. Missing originals are extracted automatically
after archive and member hashes match. No simulation, network or GPU calls.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')

def parse_policy(p):
    a=np.fromfile(p,dtype='<f8');h=a[:16];t,ns,nu,nx,nr=map(int,h[3:8]);o=16;out={'header':h}
    shapes={'reference_states':(t,ns),'reference_actions':(t-1,nu),'reference_times':(t,),
            'reference_residual':(t,nr),'K':(t,nu,nx),'H':(t,nu,nu),'Qu':(t,nu),
            'Qxu':(t,nx,nu),'active':(t,nu)}
    for k,s in shapes.items():
        n=int(np.prod(s));out[k]=a[o:o+n].reshape(s);o+=n
    assert o==len(a)
    return out

def costs(res, definitions):
    out={}
    for term in definitions:
        q=res[:,term['start']:term['end']];p=term['norm_parameter']
        if term['norm']==0:v=.5*np.sum(q*q,axis=1)
        elif term['norm']==2:v=np.sqrt(np.sum(q*q,axis=1)+p*p)-p
        elif term['norm']==6:v=np.sum(np.sqrt(q*q+p*p)-p,axis=1)
        else:raise ValueError(term)
        out[term['name']]=float(np.mean(term['weight']*v))
    return out

def components(a):
    return {k:float(np.max(np.abs(a[...,s]))) for k,s in {
        'base_xyz_m':slice(0,3),'quaternion_component':slice(3,7),
        'joint_position_rad':slice(7,19),'base_linear_velocity_m_s':slice(19,22),
        'base_angular_velocity_rad_s':slice(22,25),'joint_velocity_rad_s':slice(25,37)}.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd(),
                   help='MCP-RL checkout containing W45 metadata; default: current working directory.')
    p.add_argument('--audit','--audit-dir',dest='audit',type=Path,default=Path(__file__).resolve().parent,
                   help='Directory containing the two W45 GPU ZIPs; default: this script directory.')
    p.add_argument('--output',type=Path,
                   help='Output JSON; default: AUDIT_DIR/recomputed.json.')
    a=p.parse_args()
    if not __debug__:
        p.error('Run without -O: verification uses assertions.')
    a.repo=a.repo.resolve();a.audit=a.audit.resolve()
    root=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    if not (root/'w45/phase.json').is_file():
        p.error('W45 metadata not found; pass --repo /path/to/MCP-RL.')
    w=root/'w45';orig=a.audit/'originals';phase=read(w/'phase.json');gitgpu=read(w/'gpu.json')
    sources=read(w/'delivery/gpu/sources.json');verified=[]
    for asset in phase['delivery']['gpu']['verified_assets']:
        path=a.audit/asset['name']
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        assert digest==asset['sha256'] and path.stat().st_size==asset['size']
        verified.append({'asset':asset['name'],'size':path.stat().st_size,'sha256':digest})
    zs={r['asset']:zipfile.ZipFile(a.audit/r['asset']) for r in verified}
    for src in sources:
        b=zs[src['archive']].read(src['member'])
        assert len(b)==src['size'] and hashlib.sha256(b).hexdigest()==src['sha256']
        rel=PurePosixPath(src['source_path'].split('/gpu/w45/',1)[1])
        assert not rel.is_absolute() and '..' not in rel.parts
        target=orig.joinpath(*rel.parts)
        assert target.resolve().is_relative_to(orig.resolve())
        if target.exists():
            assert target.read_bytes()==b, f'Existing extracted original differs: {rel}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b)
    for archive in zs.values():
        archive.close()
    fixtures={x['label']:x for x in read(w/'mechanism-evidence.json')['fixtures']}
    report={'source_commit':'3e3f69c440d6a33c14131563e5f485d88da3a803',
            'verification':{'assets':verified,'source_count':len(sources),'source_bytes':sum(s['size'] for s in sources)},
            'scope':'published-original numerical audit only; no new physics, rollout, training or GPU execution',
            'problems':[]}
    for gr in gitgpu['results']:
        label=gr['label'];d=orig/label;rr=read(d/'result.json');rank=read(d/'gpu-ranking.json')
        nr=read(d/'native-results.json');cands=read(d/'candidates.json');rows={r['slot']:r for r in nr['rows']}
        assert len(rows)==32
        for key in ['native_top4','top4_recall','native_best_recalled','regret_v','regret_J',
                    'chosen_slot','native_best_slot','end_to_end_top4_confirmation_s','native32_total_s']:
            assert rr[key]==gr[key]
        with np.load(d/'gpu-rollout.npz') as gp:
            states=gp['states'];acts=gp['actions'];v=gp['trunk_violation']
            reference=parse_policy(d/'policy01.bin')['reference_states']
            scales=np.r_[[3,3,.3],[1]*4,[1]*12,[.25]*3,[1]*3,[10]*12]
            proxy=np.mean(np.sum(((states-reference[:,None,:])/scales)**2,axis=2),axis=0)
            assert np.array_equal(proxy,np.asarray(rank['proxy_score']))
            assert np.array_equal(v,np.asarray(rank['trunk_violation']))
            gpu_order=sorted(range(32),key=lambda i:(float(v[i]),float(proxy[i]),i))
            assert gpu_order==rank['ranking']
            newrows=[];costerrors=[];bindingerrors=[];stateerror=[];costmap={}
            for i in range(32):
                cp=d/cands[i]['file'];sha=hashlib.file_digest(cp.open('rb'),'sha256').hexdigest()
                assert sha==cands[i]['sha256']==rows[i]['full_policy_sha256']
                pol=parse_policy(cp)
                assert np.array_equal(acts[i],pol['reference_actions'].astype(np.float32))
                path=d/'native'/f'slot{i:02d}.npz'
                if not path.exists():
                    assert rows[i]['cache_hit']
                    equivalent=[k for k in rows if rows[k]['full_policy_sha256']==sha and not rows[k]['cache_hit']]
                    assert len(equivalent)==1;path=d/'native'/f'slot{equivalent[0]:02d}.npz'
                with np.load(path) as npz:
                    ct=costs(npz['residual'],fixtures[label]['cost_definitions']);costmap[i]=ct
                    recomputed=sum(ct.values());costerrors.append(abs(recomputed-rows[i]['native_J']))
                    binding={k:float(np.max(np.abs(npz[k]-pol[k]))) for k in pol if k!='header'}
                    assert max(binding.values())==0
                    bindingerrors.append(binding)
                    stateerror.append(states[:,i].astype(float)-npz['states'])
                    newrows.append({'slot':i,'identity':cands[i]['identity'],'sha256':sha,
                                    'native_J':rows[i]['native_J'],'native_J_recomputed':recomputed,
                                    'gpu_proxy':float(proxy[i]),'v':rows[i]['trunk_violation'],
                                    'metric_valid':rows[i]['metric_valid'],'failure':rows[i]['failure'],
                                    'cache_hit':rows[i]['cache_hit'],'native_rpc_wall_s':rows[i]['wall_s']})
            key=lambda i:(rows[i]['failure'],rows[i]['trunk_violation'],rows[i]['native_J'],i)
            native=sorted(range(32),key=key);top=gpu_order[:4];chosen=min(top,key=key)
            with_baselines=min(set(top)|{0,1},key=key);best=native[0];baseline=min([0,1],key=key)
            unique=[];seen=set()
            for i in native:
                if rows[i]['full_policy_sha256'] not in seen:
                    unique.append(i);seen.add(rows[i]['full_policy_sha256'])
            gpu_ids={rows[i]['full_policy_sha256'] for i in top}
            unique_recalled=sum(rows[i]['full_policy_sha256'] in gpu_ids for i in unique[:4])/4
            cost_gain=rows[baseline]['native_J']-rows[chosen]['native_J']
            best_gain=rows[baseline]['native_J']-rows[best]['native_J']
            false=[i for i in range(32) if v[i]==0 and (rows[i]['failure'] or rows[i]['trunk_violation']>0)]
            with np.load(d/'native/slot00.npz') as n0:
                other=d/'native/slot01.npz'
                if other.exists():
                    with np.load(other) as n1:
                        baseline_state_difference=components(n0['states']-n1['states'])
                        baseline_action_difference=float(np.max(np.abs(n0['actions']-n1['actions'])))
                else:
                    baseline_state_difference=components(np.zeros((176,37)))
                    baseline_action_difference=0.
            dupinfo=[]
            for i in range(32):
                for j in range(i+1,32):
                    if np.array_equal(acts[i],acts[j]):
                        dif=states[:,i]-states[:,j]
                        dupinfo.append({'slots':[i,j],'full_policy_same':rows[i]['full_policy_sha256']==rows[j]['full_policy_sha256'],
                                        'states_max_component':components(dif),
                                        'terminal_component':components(dif[-1:]),
                                        'first_nonzero_node':int(np.flatnonzero(np.max(np.abs(dif),axis=1)>0)[0]) if np.any(dif) else None,
                                        'proxy_delta':float(proxy[j]-proxy[i])})
            stat={'label':label,'context_origin_s':fixtures[label]['origin_s'],
                'source_call':fixtures[label]['source_call'],'source_plan':fixtures[label]['source_plan'],
                'baseline_source':fixtures[label]['gpu_baseline_source'],
                'gpu_top4':top,'native_top4':native[:4],'slot_top4_recall':len(set(top)&set(native[:4]))/4,
                'identity_deduplicated_top4_recall':unique_recalled,'unique_full_policy_count':len(seen),
                'native_best_recalled':best in top,'chosen_slot':chosen,'chosen_with_confirmed_baselines':with_baselines,
                'native_best_slot':best,'best_baseline_slot':baseline,
                'chosen_native_J':rows[chosen]['native_J'],'best_native_J':rows[best]['native_J'],
                'best_baseline_native_J':rows[baseline]['native_J'],
                'regret_v':rows[chosen]['trunk_violation']-rows[best]['trunk_violation'],
                'regret_J':rows[chosen]['native_J']-rows[best]['native_J'],
                'gain_over_best_baseline':cost_gain,'full_native_gain_over_best_baseline':best_gain,
                'available_cost_improvement_retained':cost_gain/best_gain if best_gain>0 else None,
                'false_feasible_slots':false,'native_positive_v_or_failure_slots':[i for i in range(32) if rows[i]['failure'] or rows[i]['trunk_violation']>0],
                'all_metric_valid':all(r['metric_valid'] for r in rows.values()),
                'new_native_rollouts':nr['new_native_rollouts'],'cache_hits':nr['cache_hits'],
                'maximum_cost_recompute_error':max(costerrors),'maximum_full_policy_binding_error':max(max(x.values()) for x in bindingerrors),
                'gpu_native_state_component_max':components(np.stack(stateerror)),
                'identical_gpu_input_actions':dupinfo,
                'two_baselines_comparison':{'native_J_difference':rows[1]['native_J']-rows[0]['native_J'],
                    'native_state_component_max_difference':baseline_state_difference,
                    'native_action_max_difference_rad':baseline_action_difference,
                    'gpu_proxy_scores':[float(proxy[0]),float(proxy[1])],
                    'gpu_one_based_ranks':[gpu_order.index(0)+1,gpu_order.index(1)+1]},
                'gpu_held_raw_PD_excess_peak_Nm':float(np.max(gp['hold_torque_excess'])),
                'gpu_held_raw_PD_excess_positive_slots':int(np.count_nonzero(gp['hold_torque_excess']>0)),
                'torque_metric_caveat':'post-integration state raw request for still-held action; not actual applied torque or full-hold certificate',
                'gpu_timing':rank['timing'],'native_rpc_wall_sum_s':sum(r['wall_s'] for r in rows.values()),
                'native_timing':{k:v for k,v in nr.items() if k!='rows'},
                'end_to_end_top4_confirmation_s':rr['end_to_end_top4_confirmation_s'],'native32_total_s':rr['native32_total_s'],
                'end_to_end_ratio_gpu_confirmation_to_native32':rr['end_to_end_top4_confirmation_s']/rr['native32_total_s'],
                'all_research_wall_s':rr['all_research_wall_s'],
                'cost_terms_baseline':costmap[baseline],'cost_terms_chosen':costmap[chosen],'cost_terms_best':costmap[best],
                'rows':newrows}
            assert stat['slot_top4_recall']==gr['top4_recall'] and stat['regret_J']==gr['regret_J']
            report['problems'].append(stat)
    ps=report['problems']
    report['summary']={'slots':128,'new_native_rollouts':sum(x['new_native_rollouts'] for x in ps),
        'cache_hits':sum(x['cache_hits'] for x in ps),'slot_recall_mean':np.mean([x['slot_top4_recall'] for x in ps]).item(),
        'deduplicated_recall_mean':np.mean([x['identity_deduplicated_top4_recall'] for x in ps]).item(),
        'native_best_recalled_problems':sum(x['native_best_recalled'] for x in ps),
        'positive_violation_or_failure_slots':sum(len(x['native_positive_v_or_failure_slots']) for x in ps),
        'false_feasible_slots':sum(len(x['false_feasible_slots']) for x in ps),
        'native_J_recompute_max_error':max(x['maximum_cost_recompute_error'] for x in ps),
        'full_policy_binding_max_error':max(x['maximum_full_policy_binding_error'] for x in ps),
        'gpu_device_sum_s':sum(x['gpu_timing']['device_ms']/1000 for x in ps),
        'gpu_preconfirmation_sum_s':sum(x['gpu_timing']['preconfirmation_s'] for x in ps),
        'top4_confirmation_end_to_end_sum_s':sum(x['end_to_end_top4_confirmation_s'] for x in ps),
        'native32_total_sum_s':sum(x['native32_total_s'] for x in ps),
        'all_research_wall_sum_s':sum(x['all_research_wall_s'] for x in ps),
        'job_elapsed_s':gitgpu['ended_unix_s']-gitgpu['started_unix_s'],
        'timed_world_steps':4*32*175*10,'warmup_world_steps':4*32*10,
        'faster_confirmation_problems':sum(x['end_to_end_top4_confirmation_s']<x['native32_total_s'] for x in ps)}
    output=a.output or a.audit/'recomputed.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    write(output,report)
    print('Output:',output)
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
