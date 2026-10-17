"""One functional task-driven proposal, finite evaluation and response pairing.

The executable backend here is the already declared planar analytic family.
It is NOT a neural actor/PPO trainer, robot model, or empirical safety proof.
Policies are immutable values. Outcomes keep missing full costs as None.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math

import numpy as np
from scipy.optimize import minimize

TOL=1e-12
UNKNOWN={'MODEL_UNSUPPORTED','SEARCH_UNRESOLVED','EVALUATION_TRUNCATED'}


@dataclass(frozen=True)
class Feedback:
    r: float
    kind: str='analytic_constant_force_allocation'

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    sign: int
    evidence: str='SUPPORTED'
    weight: float=.5


@dataclass
class Budget:
    case_evaluations: int=0
    completed_path_evaluations: int=0
    exact_phase_maps: int=0
    response_evaluations: int=0
    demand_pullbacks: int=0
    constraint_rows_before_merge: int=0
    constraint_rows_after_merge: int=0
    qp_calls: int=0
    qp_objective_evaluations: int=0
    qp_gradient_evaluations: int=0
    qp_constraint_evaluations: int=0
    qp_constraint_jacobian_evaluations: int=0
    pairing_model_predictions: int=0
    candidate_vectors: int=0
    next_plan_evaluations: int=0


def summarize(outcomes):
    n=len(outcomes);mass=sum(o['weight'] for o in outcomes)
    counts=Counter(o['status'] for o in outcomes)
    complete=[o for o in outcomes if o['status']=='COMPLETED']
    known=sum(o['weight'] for o in complete)/mass
    unknown=sum(o['weight'] for o in outcomes if o['status'] in UNKNOWN)/mass
    return {'total_cases':n,'status_counts':dict(counts),
            'completed_ids':[o['case_id'] for o in complete],
            'known_completion_fraction':known,'unknown_fraction':unknown,
            'case_completion_interval':[known,known+unknown],
            'interval_kind':'finite-case logical interval, not a probability confidence interval',
            'full_mean_task_time':sum(o['weight']*o['task_time'] for o in complete)/mass if len(complete)==n else None,
            'known_completed_conditional_mean':sum(o['weight']*o['task_time'] for o in complete)/sum(o['weight'] for o in complete) if complete else None}


def compare_cases(reference,candidate):
    """Fixed-denominator physical comparison; PPO surrogate is not a quality gate."""
    keys=lambda rows:[(o['case_id'],o['weight'],o['evidence_class'],o.get('task_version'),o.get('sign')) for o in rows]
    if keys(reference)!=keys(candidate):
        return {'passed':False,'reason':'CASE_IDENTITY_MISMATCH'}
    old,new=summarize(reference),summarize(candidate)
    old_mask={o['case_id'] for o in reference if o['status']=='COMPLETED'}
    new_mask={o['case_id'] for o in candidate if o['status']=='COMPLETED'}
    common=old_mask&new_mask
    pairs=[(a,b) for a,b in zip(reference,candidate) if a['case_id'] in common]
    result={'reference':old,'candidate':new,'rescued':sorted(new_mask-old_mask),
            'lost':sorted(old_mask-new_mask),'common_completed_count':len(common),
            'common_completed_fraction':len(common)/len(reference),
            'common_completed_cost_delta':sum(a['weight']*(b['task_time']-a['task_time']) for a,b in pairs)/sum(a['weight'] for a,b in pairs) if pairs else None}
    if any(o['status']=='HARD_CONSTRAINT_FAIL' for o in candidate):
        reason='REJECT_HARD_CONTRACT'
    elif old['unknown_fraction']>0 or new['unknown_fraction']>0:
        reason='DEFER_UNKNOWN_FULL_SCOPE'
    elif new['known_completion_fraction']>old['known_completion_fraction']:
        reason='COMPLETION_PRIORITY_IMPROVEMENT'
    elif new['known_completion_fraction']<old['known_completion_fraction']:
        reason='REJECT_FEWER_COMPLETIONS'
    elif old_mask!=new_mask:
        reason='TRADEOFF_UNRESOLVED'
    elif common and result['common_completed_cost_delta'] < -TOL:
        reason='AVERAGE_TASK_IMPROVEMENT'
    else:
        reason='NO_CLEAR_TASK_GAIN'
    result.update(passed=reason in {'COMPLETION_PRIORITY_IMPROVEMENT','AVERAGE_TASK_IMPROVEMENT'},reason=reason)
    return result


def full_parameter_credit(requests,weights):
    """Keep all shared-parameter contributions, including both direct terms."""
    contributions=[]
    for req in requests:
        h=(np.asarray(req['response_jacobian']).T@np.asarray(req['task_covector'])
           +np.asarray(req['prefix_direct'])+np.asarray(req['suffix_direct']))
        contributions.append(h)
    return np.sum(np.asarray(weights)[:,None]*np.asarray(contributions),axis=0),contributions


def merge_identical_rows(rows):
    """Only exact local duplicates within the same physical facet are merged."""
    unique=[];index={}
    for row in rows:
        key=(row['facet'],float(row['value']).hex(),tuple(float(v).hex() for v in row['gradient']))
        if key in index:
            unique[index[key]]['origins'].extend(row['origins'])
        else:
            index[key]=len(unique)
            unique.append({**row,'gradient':list(row['gradient']),'origins':list(row['origins'])})
    return unique


class AnalyticBackend:
    """Existing a=v0=b=1,L=4 task; no x<=L global task constraint is added."""
    task_version='normalized-stop-band-v1'

    def __init__(self):
        self.budget=Budget()

    def response(self,policy):
        self.budget.response_evaluations+=1
        r=policy.r
        if not 0<=r<1:return None
        s=math.sqrt(1-r*r)
        return np.array([1/s,r/(2*(1-r*r)),r/s,1/(2*s)])

    def suffix_plan(self,response):
        self.budget.next_plan_evaluations+=1
        tau,x,v,y=[float(z) for z in response]
        remaining=4-x
        # This is a limitation of the represented no-overshoot suffix, not of the task.
        if remaining < v*v/2-TOL:
            return {'status':'SEARCH_UNRESOLVED','reason':'represented_no_overshoot_suffix_not_available'}
        peak=math.sqrt(remaining+v*v/2)
        acc=max(0.,peak-v);brake=peak
        return {'status':'PLAN_AVAILABLE','accelerate_duration':acc,'brake_duration':brake,
                'brake_position':x+v*acc+.5*acc*acc,
                'brake_start_time':tau+acc,'completion_time':tau+acc+brake,
                'lateral_endpoint_magnitude':y}

    def evaluate(self,policy,case):
        self.budget.case_evaluations+=1
        out={'case_id':case.case_id,'weight':case.weight,'sign':case.sign,
             'evidence_class':'ANALYTIC_MODEL','policy_identity':policy.identity,
             'task_version':self.task_version,'task_time':None,'response':None,'constraints':{},'details':{}}
        if case.evidence!='SUPPORTED':
            if case.evidence not in UNKNOWN:raise ValueError('Unknown evidence status')
            return {**out,'status':case.evidence,'reason':'declared_evidence_availability_intervention'}
        y=self.response(policy)
        if y is None:return {**out,'status':'MODEL_UNSUPPORTED','reason':'outside_declared_feedback_coordinate_domain'}
        tau,x,v,dy=y;out['response']=y.tolist()
        out['constraints']={'clearance':float(dy-1),'input_norm':0.}
        plan=self.suffix_plan(y)
        out['details']['suffix_plan']=plan
        if plan['status']!='PLAN_AVAILABLE':
            if dy>1+TOL:
                return {**out,'status':'HARD_CONSTRAINT_FAIL','reason':'known_lateral_clearance_violation_in_recovery'}
            return {**out,'status':'SEARCH_UNRESOLVED','reason':plan['reason']}
        phases=[(tau,np.array([policy.r,-case.sign*math.sqrt(1-policy.r**2)])),
                (plan['accelerate_duration'],np.array([1.,0.])),
                (plan['brake_duration'],np.array([-1.,0.]))]
        p=np.zeros(2);vel=np.array([0.,float(case.sign)]);t=0.;trace=[];maximum_y=0.;maximum_u=0.
        for duration,u in phases:
            oldp=p.copy();oldv=vel.copy()
            points=[0.,float(duration)]
            if u[1]!=0 and 0 < -vel[1]/u[1] < duration:points.append(float(-vel[1]/u[1]))
            maximum_y=max(maximum_y,*[abs(p[1]+vel[1]*z+.5*u[1]*z*z) for z in points])
            maximum_u=max(maximum_u,float(np.linalg.norm(u)))
            p=p+vel*duration+.5*u*duration**2;vel=vel+u*duration
            trace.append({'t_start':float(t),'duration':float(duration),'position_start':oldp.tolist(),
                          'velocity_start':oldv.tolist(),'input':u.tolist(),
                          'position_end':p.tolist(),'velocity_end':vel.tolist()})
            t+=duration;self.budget.exact_phase_maps+=1
        self.budget.completed_path_evaluations+=1
        out['constraints']={'clearance':float(maximum_y-1),'input_norm':float(maximum_u-1)}
        out['details'].update(phases=trace,terminal_position=p.tolist(),terminal_velocity=vel.tolist(),
                              diagnostic_path_end_time=float(t),gate_crossed=bool(p[0]>=2-TOL),
                              no_overshoot_is_family_property_not_task_requirement=True)
        if any(g>TOL for g in out['constraints'].values()):
            return {**out,'status':'HARD_CONSTRAINT_FAIL','reason':'lateral_clearance' if maximum_y>1+TOL else 'input_norm'}
        if abs(p[0]-4)>TOL or np.linalg.norm(vel)>TOL:
            return {**out,'status':'MODEL_UNSUPPORTED','reason':'terminal_arithmetic_identity_failed'}
        return {**out,'status':'COMPLETED','task_time':float(t),'reason':'full_declared_task_completed'}

    def demand(self,policy,outcome):
        if outcome['status']!='COMPLETED':return None
        self.budget.demand_pullbacks+=1
        r=policy.r;s=math.sqrt(1-r*r);tau,x,v,dy=outcome['response']
        q=4-x+.5*v*v
        jac=np.array([[r/s**3],[(1+r*r)/(2*(1-r*r)**2)],[1/s**3],[r/(2*s**3)]])
        covector=np.array([1.,-1/math.sqrt(q),v/math.sqrt(q)-1,0.])
        return {'case_id':outcome['case_id'],'weight':outcome['weight'],
                'response_jacobian':jac.tolist(),'task_covector':covector.tolist(),
                'prefix_direct':[0.],'suffix_direct':[0.],
                'direct_term_reason':'Recovery time is already in Y; this analytic suffix has no separate r dependence. Not assumed for a shared neural actor.',
                'constraints':[{'facet':'clearance','value':outcome['constraints']['clearance'],
                                'gradient':jac[3].tolist(),'origins':[outcome['case_id']]},
                               {'facet':'input_norm','value':0.,'gradient':[0.],'origins':[outcome['case_id']]}]}


def propose_step(backend,anchor,requests):
    if not requests:return {'status':'NO_DEFINED_TASK_DIRECTION','candidate':None}
    mass=sum(r['weight'] for r in requests)
    weights=[r['weight']/mass for r in requests]
    gradient,contributions=full_parameter_credit(requests,weights)
    rows=[row for req in requests for row in req['constraints']]
    merged=merge_identical_rows(rows)
    backend.budget.constraint_rows_before_merge+=len(rows)
    backend.budget.constraint_rows_after_merge+=len(merged)
    values=np.array([r['value'] for r in merged]);matrix=np.array([r['gradient'] for r in merged])
    span=np.column_stack((gradient,matrix.T))
    rank=int(np.linalg.matrix_rank(span))
    result={'gradient':gradient.tolist(),'individual_parameter_credits':[c.tolist() for c in contributions],
            'proposal_scope':[r['case_id'] for r in requests],'scope_weight_in_full_cases':mass,
            'rows_before_merge':rows,'rows_after_merge':merged,'local_gradient_span_rank':rank,
            'metric':'identity on analytic coordinate, not empirical PPO Fisher',
            'trust_radius':.1,'trust_is_not_PPO_KL':True}
    if np.linalg.norm(gradient)<=TOL:
        return {**result,'status':'NO_SHARED_TASK_DIRECTION','candidate':None}
    backend.budget.qp_calls+=1
    def objective(d):
        backend.budget.qp_objective_evaluations+=1
        return float(gradient@d+.5*(d@d))
    def jac(d):
        backend.budget.qp_gradient_evaluations+=1
        return gradient+d
    def constraints(d):
        backend.budget.qp_constraint_evaluations+=1
        return -(values+matrix@d)
    def constraint_jacobian(d):
        backend.budget.qp_constraint_jacobian_evaluations+=1
        return -matrix
    solved=minimize(objective,np.zeros(1),jac=jac,method='SLSQP',
                    bounds=[(max(-.1,-anchor.r),min(.1,1-1e-12-anchor.r))],
                    constraints={'type':'ineq','fun':constraints,'jac':constraint_jacobian},
                    options={'maxiter':100,'ftol':1e-12})
    d=solved.x
    ok=bool(solved.success and np.isfinite(d).all() and np.max(values+matrix@d)<=TOL and np.linalg.norm(d)<=.1+TOL)
    result.update(solver={'success':bool(solved.success),'message':str(solved.message),
                          'nit':int(solved.nit),'nfev':int(solved.nfev)},step=d.tolist())
    if ok and np.linalg.norm(d)<=TOL:
        return {**result,'status':'CONTRACT_BLOCKED_DIRECTION','candidate':None}
    return {**result,'status':'PROPOSAL_AVAILABLE' if ok else 'LOCAL_SOLVER_UNRESOLVED',
            'candidate':Feedback(float(anchor.r+d[0])) if ok else None}


def pair_response(backend,policy,outcomes,model_override=None):
    """Build the current-policy response; a mismatching model cannot be published."""
    if any(o['status'] in UNKNOWN for o in outcomes):
        return {'status':'PAIR_DEFERRED_UNKNOWN','policy_identity':policy.identity}
    rows=[];maximum_error=0.
    for outcome in outcomes:
        if outcome['policy_identity']!=policy.identity or outcome['task_version']!=backend.task_version:
            return {'status':'REJECT_PAIR_IDENTITY','policy_identity':policy.identity}
        # Preserve known failure flags too. A paired model is not obliged to
        # predict success for every case; UNKNOWN still blocks full pairing.
        backend.budget.pairing_model_predictions+=1
        prediction=backend.evaluate(policy,Case(outcome['case_id'],outcome['sign'],weight=outcome['weight']))
        if prediction['status']!=outcome['status']:
            return {'status':'REJECT_MODEL_OUTCOME_MISMATCH','policy_identity':policy.identity}
        predicted=prediction['response'] if model_override is None else model_override
        if predicted is None or outcome['response'] is None:
            if predicted is not None or outcome['response'] is not None or outcome['status']=='COMPLETED':
                return {'status':'REJECT_MISSING_RESPONSE','policy_identity':policy.identity}
            rows.append({'case_id':outcome['case_id'],'status':outcome['status'],'response':None,
                         'plan':{'status':'NO_VERIFIED_PLAN','reason':outcome['status']}})
            continue
        predicted=np.asarray(predicted,dtype=float)
        actual=np.asarray(outcome['response'],dtype=float)
        if predicted.shape!=(4,) or actual.shape!=(4,) or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
            return {'status':'REJECT_NONFINITE_OR_INVALID_RESPONSE','policy_identity':policy.identity}
        plan=backend.suffix_plan(predicted)
        if outcome['status']=='COMPLETED' and (plan['status']!='PLAN_AVAILABLE' or not math.isfinite(plan['completion_time'])):
            return {'status':'REJECT_INVALID_RESPONSE_PLAN','policy_identity':policy.identity}
        error=float(np.max(np.abs(predicted-actual)))
        if outcome['status']=='COMPLETED':
            error=max(error,abs(plan['completion_time']-outcome['task_time']))
        maximum_error=max(maximum_error,error)
        rows.append({'case_id':outcome['case_id'],'status':outcome['status'],'response':predicted.tolist(),
                     'plan':plan if outcome['status']=='COMPLETED' else {'status':'NO_VERIFIED_PLAN','reason':outcome['status']}})
    return {'status':'PAIRED' if maximum_error<=TOL else 'REJECT_MODEL_POLICY_MISMATCH',
            'policy_identity':policy.identity,'response_max_error':maximum_error,
            'label_origin':'exact analytic recomputation from this virtual feedback, not learned G or robot data',
            'scope':[o['case_id'] for o in outcomes],'predictions':rows}


def run_round(deployed,anchor,cases,external_candidate=None,model_override=None):
    """Pure complete round. Nothing mutates the input policies or persistent weights."""
    backend=AnalyticBackend();cache={}
    def evaluate(policy):
        if policy.identity not in cache:cache[policy.identity]=[backend.evaluate(policy,c) for c in cases]
        return cache[policy.identity]
    old=evaluate(deployed);base=evaluate(anchor)
    requests=[r for o in base if (r:=backend.demand(anchor,o)) is not None]
    if external_candidate is None:
        proposal=propose_step(backend,anchor,requests)
        candidate=proposal['candidate']
        ratios=(1.,.5,.25,.125) if candidate is not None else ()
    else:
        candidate=external_candidate;ratios=(1.,)
        proposal={'status':'EXTERNAL_FINITE_CANDIDATE','candidate':candidate,
                  'source':'predeclared negative control, not generated by local QP'}
    attempts=[];accepted=None;pair=None
    for alpha in ratios:
        trial=Feedback(anchor.r+alpha*(candidate.r-anchor.r))
        backend.budget.candidate_vectors+=1
        outcomes=evaluate(trial)
        anchor_test=compare_cases(base,outcomes);deployment_test=compare_cases(old,outcomes)
        support_ids={r['case_id'] for r in requests}
        partial=compare_cases([o for o in base if o['case_id'] in support_ids],
                              [o for o in outcomes if o['case_id'] in support_ids]) if support_ids else None
        candidate_pair=pair_response(backend,trial,outcomes,model_override)
        pass_all=anchor_test['passed'] and deployment_test['passed'] and candidate_pair['status']=='PAIRED'
        attempts.append({'alpha':alpha,'candidate':asdict(trial),'candidate_identity':trial.identity,
                         'outcomes':outcomes,'against_anchor':anchor_test,'against_deployed':deployment_test,
                         'supported_subset_only':partial,'pair_check':candidate_pair,'accepted':bool(pass_all)})
        if pass_all:accepted=trial;pair=candidate_pair;break
        if anchor_test['reason']=='DEFER_UNKNOWN_FULL_SCOPE' or deployment_test['reason']=='DEFER_UNKNOWN_FULL_SCOPE':
            # Missing evidence cannot be repaired by shrinking a parameter step.
            break
    active=accepted or deployed
    if pair is None:
        pair=pair_response(backend,deployed,old)
    # Even on an unresolved full scope, a known case can retain its old plan;
    # this is not a new partial-scope controller or a dispatch gate.
    next_plans=[]
    for outcome in evaluate(active):
        next_plans.append({'case_id':outcome['case_id'],'active_policy_identity':active.identity,
                           'plan':backend.suffix_plan(outcome['response']) if outcome['status']=='COMPLETED'
                           else {'status':'NO_VERIFIED_PLAN','reason':outcome['status']}})
    return {'status':'ACCEPTED_ANALYTIC_CASESET' if accepted else 'NO_FULL_SCOPE_UPDATE',
            'deployed_input':asdict(deployed),'anchor_input':asdict(anchor),
            'proposal':{k:(asdict(v) if isinstance(v,Feedback) else v) for k,v in proposal.items()},
            'anchor_outcomes':base,'old_deployment_outcomes':old,'requests':requests,'attempts':attempts,
            'accepted_feedback':asdict(accepted) if accepted else None,'active_feedback':asdict(active),
            'active_response_pair':pair,'next_planning_decisions':next_plans,
            'ppo_surrogate':'No genuine PPO batch or actor distribution exists for this analytic coordinate; no KL/advantage is fabricated.',
            'model_fitting':'None. Pairing is exact functional recomputation, not trained system identification.',
            'budget':asdict(backend.budget),
            'scope':'Only the fixed finite analytic cases. Not robot deployment, PPO learning or a global guarantee.'}
