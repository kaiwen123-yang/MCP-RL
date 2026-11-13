#!/usr/bin/env python3
"""W47 A: server-local, zero-integration inspection of preserved W46 arrays."""
import ctypes, hashlib, json, os, time
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SOURCE=ROOT/REL/'w46-mechanisms'
OUT=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/task-a'
FIELDS=('reference_states','reference_actions','reference_times','K','H','Qu','Qxu','active')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def ma(v):return float(np.max(np.abs(v)))
def first(v,threshold=0):
    ids=np.flatnonzero(v>threshold)
    return int(ids[0]) if len(ids) else None

def arrays(raw,i,T,ns,nu,nx,nr):
    d={}
    for prefix in ('','reference_'):
        for k,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr))]:
            n=int(np.prod(shape));d[prefix+k]=raw[i:i+n].reshape(shape);i+=n
    for k,shape in [('K',(T,nu,nx)),('H',(T,nu,nu)),('Qu',(T,nu)),('Qxu',(T,nx,nu)),('active',(T,nu))]:
        n=int(np.prod(shape));d[k]=raw[i:i+n].reshape(shape);i+=n
    return d,i

def candidates(p):
    raw=np.fromfile(p,dtype='<f8');i=0;out=[]
    while i<len(raw):
        h=raw[i:i+19];i+=19
        T,ns,nu,nx,nr=map(int,h[5:10]);d,i=arrays(raw,i,T,ns,nu,nx,nr)
        for k in ('prefix','feet'):
            n=int(raw[i]);i+=1;d[k]=raw[i:i+n];i+=n
        m=dict(record=len(out),call=int(h[1]),kind=int(h[2]),slot=int(h[3]),alpha=float(h[4]),scale=float(h[10]),
            J=float(h[11]),violation=float(h[12]),failed=bool(h[13]),metric_valid=bool(h[14]),origin=float(h[15]),
            mu=float(h[16]),representation=int(h[17]),input_policy_id=int(h[18]))
        out.append((m,d))
    return out

def ident(m,d):
    h=hashlib.sha256()
    for k in FIELDS:h.update(np.asarray(d[k],dtype='<f8').tobytes())
    h.update(np.asarray([m['scale'],m['representation']],dtype='<f8').tobytes())
    return h.hexdigest()

def terms(r,definitions):
    out={}
    for t in definitions:
        q=r[:,t['start']:t['end']];p=t['norm_parameter']
        if t['norm']==0:v=.5*np.sum(q*q,axis=1)
        elif t['norm']==2:v=np.sqrt(np.sum(q*q,axis=1)+p*p)-p
        elif t['norm']==6:v=np.sum(np.sqrt(q*q+p*p)-p,axis=1)
        else:raise ValueError(t['norm'])
        out[t['name']]=float(np.mean(t['weight']*v))
    return out

def prefix(raw,nr):
    out=[];i=0
    while i<len(raw):
        h=raw[i:i+5];i+=5;nc=int(h[4]);d=dict(node=int(h[0]),sub=int(h[1]),before=float(h[2]),after=float(h[3]))
        for k,n in [('pre',37),('post',37),('raw_pd',12),('actuator',12),('passive',18),('residual',nr)]:
            d[k]=raw[i:i+n];i+=n
        d['contacts']=raw[i:i+nc*8].reshape(nc,8);i+=nc*8;out.append(d)
    return out

class NativeProjection:
    """State difference only. No integration or external projection bridge."""
    def create(self,p):return mujoco.MjModel.from_xml_path(str(p))
    def diff(self,m,ref,x):
        d=np.empty(36);mujoco.mj_differentiatePos(m,d[:18],1.,ref[:19],x[:19]);d[18:]=x[19:]-ref[19:];return d


def bound_info(m,x,u):
    q=m.jnt_qposadr[m.actuator_trnid[:,0]];v=m.jnt_dofadr[m.actuator_trnid[:,0]]
    lower=np.maximum(m.actuator_ctrlrange[:,0],x[q]+(m.actuator_forcerange[:,0]+5*x[m.nq+v])/60)
    upper=np.minimum(m.actuator_ctrlrange[:,1],x[q]+(m.actuator_forcerange[:,1]+5*x[m.nq+v])/60)
    return dict(lower=lower.tolist(),upper=upper.tolist(),center_inside=bool(np.all((u>=lower)&(u<=upper))),
        PD_Nm=(60*(u-x[q])-5*x[m.nq+v]).tolist())

def window_node(k,m,nom,c,bridge,w):
    x=c['states'][k];ref=c['reference_states'][k]
    dx=bridge.diff(w,ref,x);selfdx=bridge.diff(w,ref,ref)
    # Scalar accumulation mirrors mju_mulMatVec; NumPy BLAS would reassociate.
    fb=np.array([sum(float(c['K'][k,i,j])*float(dx[j]) for j in range(36)) for i in range(12)])
    raw=c['reference_actions'][k]+fb;clipped=np.clip(raw,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    bounds=bound_info(m,x,clipped)
    projected=clipped.copy() if bounds['center_inside'] else None
    ff=c['feet'].reshape(-1,29)[k,1:].reshape(4,7);fn=nom['feet'].reshape(-1,29)[k,1:].reshape(4,7)
    return dict(node=k,decision_time_s=float(c['times'][k]),hold_end_s=float(c['times'][k+1]),
        actual_x_equals_reference=bool(np.array_equal(x,ref)),state_tangent=dx.tolist(),quaternion_self_tangent=selfdx[3:6].tolist(),
        quaternion_norm=float(np.linalg.norm(x[3:7])),reference_quaternion_norm=float(np.linalg.norm(ref[3:7])),
        u_ff=c['reference_actions'][k].tolist(),raw_feedback=fb.tolist(),raw=raw.tolist(),clipped=clipped.tolist(),
        projected=projected.tolist() if projected is not None else None,saved_native_held=c['actions'][k].tolist(),nominal_native_held=nom['actions'][k].tolist(),
        projection_reconstruction_abs_max=ma(projected-c['actions'][k]) if projected is not None else None,projection_delta_abs_max=0. if projected is not None else None,
        projection_scope='Analytic identity for interior centers; no historical KKT recomputation claimed for exterior centers',
        bounds=bound_info(m,x,clipped),K_abs_max=ma(c['K'][k]),H_abs_max=ma(c['H'][k]),
        native_fz_N=ff[:,4].tolist(),nominal_native_fz_N=fn[:,4].tolist(),material_fz_N=ff[:,5].tolist(),
        native_contact_counts=ff[:,6].astype(int).tolist(),nominal_contact_counts=fn[:,6].astype(int).tolist())

def main():
    started=time.time();bridge=NativeProjection()
    catalog=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/delivery/mechanism/sources.json').read_text())
    published={r['source_path']:r for r in catalog};verified=[];summaries=[]
    def record(p):
        key=str(p.relative_to(ROOT));actual=dict(path=key,size=p.stat().st_size,sha256=sha(p));old=published.get(key)
        actual['published_manifest_match']=None if old is None else actual['size']==old['size'] and actual['sha256']==old['sha256']
        if old:actual.update(archive=old['archive'],member=old['member'])
        verified.append(actual);return actual
    for label in ('A148','A153','B155','B158'):
        f=json.loads((SOURCE/label/'input.json').read_text());m=mujoco.MjModel.from_xml_path(str(SOURCE/label/'model.xml'));w=bridge.create(SOURCE/label/'model.xml')
        fixture=dict(label=label,source_round='W45',historical_replay=False,origin_s=f['origin_s'],branches={},missing=[])
        for name in ('input.json','input-state.npz','input-policy.bin','original-policy.bin','context.bin','model.xml'):record(SOURCE/label/name)
        for branch in ('none','constant','load'):
            p=SOURCE/label/branch;source=record(p/'r29-current-plan.bin.r33-candidates');ref=record(p/'r29-current-plan.bin.r35-reference-1.bin');record(p/'planner-calls.jsonl')
            cs=candidates(p/'r29-current-plan.bin.r33-candidates');call=json.loads((p/'planner-calls.jsonl').read_text())
            nomm,nom=min((z for z in cs if z[0]['kind']==0),key=lambda z:(z[0]['violation'],z[0]['J']))
            nt=terms(nom['residual'],f['cost_definitions']);rows=[];details=[]
            for attempt in call['search_attempts']:
                selected=[z for z in cs if z[0]['kind']==1 and int(z[0]['slot']/10)==attempt['attempt']-1]
                best=min((z for z in selected if z[0]['alpha']>0),key=lambda z:z[0]['J'])
                for cm,ca in selected:
                    ct=terms(ca['residual'],f['cost_definitions']);a=cm['alpha'];dV=attempt['dV_unscaled']
                    row=dict(**cm,policy_sha256=ident(cm,ca),actual_delta_J=cm['J']-nomm['J'],predicted_decrease=-a*(dV[0]+a*dV[1]),
                        J0=sum(v for k,v in ct.items() if k!='InputReference'),Jref=ct.get('InputReference',0),
                        parts_delta={k:ct[k]-nt[k] for k in ct},reference_time_error=ma(ca['reference_times']-nom['times']),
                        reference_state_error=ma(ca['reference_states']-nom['states']))
                    rows.append(row)
                    if branch=='constant' and (a==0 or cm['record']==best[0]['record']):
                        du=np.max(abs(ca['actions']-nom['actions']),axis=1);ds=np.max(abs(ca['states']-nom['states']),axis=1)
                        u0=first(du);x0=first(ds);large=first(du,1e-6)
                        times=ca['times'];nodes=sorted(set(k for origin in (u0,x0,large) if origin is not None for k in (origin-1,origin,origin+1) if 0<=k<175))
                        details.append(dict(**row,role='alpha0' if a==0 else 'best_positive_of_attempt',
                            first_nonzero_action_node=u0,first_nonzero_state_node=x0,first_action_over_1e_6_rad_node=large,
                            max_action_delta_rad=float(du.max()),max_state_coordinate_delta=float(ds.max()),
                            alpha0_center_action_error=ma(ca['reference_actions']-nom['actions']) if a==0 else None,
                            exact_node_time_match=bool(np.array_equal(times,nom['times'])),
                            nodes=[window_node(k,m,nom,ca,bridge,w) for k in nodes]))
            b=dict(archive=source,reference=ref,nominal={**nomm,'policy_sha256':ident(nomm,nom),'parts':nt},
                candidates=rows,search_attempts=[{k:v for k,v in a.items() if k!='native_candidates'} for a in call['search_attempts']],
                all_positive_true_J_increase=all(r['actual_delta_J']>0 for r in rows if r['alpha']>0))
            if branch=='constant':
                diag=np.load(p/'nominal/selected.npz');record(p/'nominal/selected.npz')
                b['same_complete_policy_repeat_envelope']={k:ma(diag[k]-nom[k]) for k in ('states','actions','times','residual')}
                b['first_divergence_details']=details
                b['prefix_substeps_saved']=len(prefix(nom['prefix'],nom['residual'].shape[1]))
                b['full_horizon_nodes_saved']=len(nom['states'])
            fixture['branches'][branch]=b
        fixture['limitations']=['Original historical pre-call optimizer/private state is not claimed reconstructed.',
            'Native substep contacts/actuator arrays cover20ms only; feet/accepted x/u/residual cover350ms.',
            'Raw/clipped values are offline algebra; interior H-QP is an exact identity. Exterior historical KKT/active sets are not reconstructed.',
            'Existing archives establish first-divergence and actual direction failure, not a unique full-trajectory causal attribution.']
        write(OUT/(label+'.json'),fixture)
        b=fixture['branches']['constant'];z=next(d for d in b['first_divergence_details'] if d['role']=='alpha0');firstnode=next(n for n in z['nodes'] if n['node']==z['first_nonzero_action_node'])
        summaries.append(dict(label=label,nominal_J=b['nominal']['J'],noise_envelope=b['same_complete_policy_repeat_envelope'],
            first_alpha0_action_node=z['first_nonzero_action_node'],first_alpha0_state_node=z['first_nonzero_state_node'],
            first_alpha0_action_delta_rad=ma(np.array(firstnode['saved_native_held'])-firstnode['nominal_native_held']),
            first_state_is_exact_center=firstnode['actual_x_equals_reference'],quaternion_self_tangent=firstnode['quaternion_self_tangent'],
            first_projection_delta=firstnode['projection_delta_abs_max'],first_reconstruction_error=firstnode['projection_reconstruction_abs_max'],
            alpha0_delta_J_range=[min(r['actual_delta_J'] for r in b['candidates'] if r['alpha']==0),max(r['actual_delta_J'] for r in b['candidates'] if r['alpha']==0)],
            best_positive_delta_J=min(r['actual_delta_J'] for r in b['candidates'] if r['alpha']>0)))
    record(ROOT/'work/r11-frozen/S5-r35/agent_server');record(ROOT/'work/r11-frozen/S5-r35/mjpc/utilities.cc')
    result=dict(schema='w47_task_a_saved_array_inspection_v1',status='completed_existing_array_analysis',started_unix_s=started,ended_unix_s=time.time(),
        new_rollouts=0,new_physics_substeps=0,new_solves=0,new_PPO_samples=0,new_PPO_updates=0,
        runtime_mujoco=mujoco.__version__,hypotheses=['Wrong node/time or feedback center','Quaternion self-difference and floating-point feedback amplification','H-QP projection or contact-mode change','Different initial dynamics state'],
        conclusion='Exact same-policy repeats have zero observed discrepancy. All alpha0 centers and time grids equal their actual nominal. First action differences occur while state equals center and are reconstructed by nonzero quaternion self-difference times new K; H-QP does not change those first differing actions. This identifies the first perturbation source; later nonlinear amplification and positive-step failure remain separate.',
        fixtures=summaries,sources=verified,positive_candidates=540,
        unavailable_fields=['Full350ms .2ms prefix contacts and actuator forces are not saved for search candidates; only first20ms.',
            'No new corrected rollout has been run by this script.'],publication='Existing originals retained in place; only compact JSON added.')
    write(OUT/'saved-array-summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','unavailable_fields')},indent=2),flush=True)
if __name__=='__main__':main()
