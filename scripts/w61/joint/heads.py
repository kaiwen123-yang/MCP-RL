"""Support-only material identification and a state-conditioned finite deployment-budget proposal."""
import numpy as np
from w61.mpc.proposal import finite_proposal

# Three initial material coordinates, a rank-one learned preconditioner in the encoder
# coordinates and two proposal rows, each conditioned on the same three causal inputs.
NA = 6
NC = 6


class MaterialHead:
    def __init__(self, checkpoint, fit_seed):
        import torch
        from tm02.w61_material import Family
        fitted=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=fitted['config']
        net=Family(cfg,'lowrank_response_gn')
        # The exact method name is supplied by the original saved checkpoint.
        key=next(k for k in fitted['state_dicts'] if k.endswith('_'+str(fit_seed)) and
                 'lowrank' in k and 'isotropic' not in k and 'anil' not in k)
        net.load_state_dict(fitted['state_dicts'][key])
        self.name=key
        self.base=net.base.detach().numpy().astype(float)
        self.gate=net.gate.weight.detach().numpy().astype(float)
        self.bias=net.gate.bias.detach().numpy().astype(float)
        self.factors=np.einsum('ur,rv->ruv',net.left.detach().numpy(),net.right.detach().numpy()).reshape(len(self.bias),16)
        self.scale=np.array(cfg['stiffness_scale_MPa'])
        self.tau=np.array(cfg['tau_s'])
        self.strain_scale=cfg['strain_scale']

    def coefficients(self, code):
        logits=self.base + np.tanh(self.gate@code+self.bias)@self.factors
        return np.logaddexp(0,logits)*self.scale

    def stiffness(self,code):
        w=self.coefficients(code)
        k=w[0]+w[4::2].sum()
        g=w[1]+w[5::2].sum()
        return float(9*k*g/(3*k+g)*1e6*.025*.001/.05)

    def predict(self, code, support):
        # Reconstruct the measured footprint in world coordinates. Each cell sees
        # one sole displacement; unloaded cells recover between foot visits.
        from tm02.w61_material import MaterialResponse
        from tm02.w61_material_columns import FamilyColumns
        columns=FamilyColumns(MaterialResponse(self.coefficients(code),self.tau,self.strain_scale))
        frames=[];index={};half=np.sqrt(.001)/2
        for row in support:
            legs=[];keys=[];areas=[]
            for leg in np.flatnonzero(row['soft']):
                lo,hi=row['xy'][leg]-half,row['xy'][leg]+half
                for x in range(int(np.floor(lo[0]/.02)),int(np.floor(hi[0]/.02))+1):
                    for y in range(int(np.floor(lo[1]/.02)),int(np.floor(hi[1]/.02))+1):
                        corner=np.array([x,y])*.02
                        area=np.maximum(0.,np.minimum(hi,corner+.02)-np.maximum(lo,corner)).prod()
                        if area>0:
                            keys.append(index.setdefault((x,y),len(index)));legs.append(leg);areas.append(area)
            frames.append((np.array(keys,int),np.array(legs,int),areas))
        history=columns.virgin(len(index));force=[]
        for row,(keys,legs,areas) in zip(support,frames):
            # Distinct feet do not share a cell in this observed support.
            if len(np.unique(keys))!=len(keys):raise ValueError('Overlapping world-cell support in the F protocol')
            disp=np.zeros(len(index));a=np.zeros(len(index))
            disp[keys]=row['bottom'][legs];a[keys]=areas
            history,f,_=columns.contact(history,disp,a,row['dt'])
            force.append(np.bincount(legs,weights=f[keys],minlength=4))
        return np.asarray(force)

    def adapt(self, theta, support, steps):
        c=.6*np.tanh(theta[:3]).copy()
        u=np.array([.55,-.25,.35])+np.tanh(theta[3:6])
        pre=np.eye(3)+np.outer(u,u)
        true=np.asarray([s['fz'] for s in support])
        mask=np.asarray([s['soft'] for s in support],dtype=bool)
        # Withhold the latest complete prefix block for calibration, even in joint training.
        split=max(1,len(support)*2//3)
        use=mask.copy();use[split:]=False
        for _ in range(steps):
            y=self.predict(c,support[:split])
            cols=[(self.predict(c+.008*e,support[:split])-self.predict(c-.008*e,support[:split]))/.016 for e in np.eye(3)]
            j=np.stack(cols,-1)[use[:split]]/40
            r=(y-true[:split])[use[:split]]/40
            delta=-np.linalg.solve(j.T@j+len(r)*.12*np.linalg.inv(pre),j.T@r)
            c+=delta*min(1.,.45/max(1e-12,np.linalg.norm(delta)))
        pred=self.predict(c,support)
        validation=mask.copy();validation[:split]=False
        rmse=float(np.sqrt(np.mean((pred[validation]-true[validation])**2)))
        return c,self.stiffness(c),rmse


def proposal(theta, vel, speed, loads, stiffness, noise, fit=True):
    weight=loads.sum()
    imbalance=(loads[:2].sum()-loads[2:].sum())/max(weight,1.)
    cond=np.array([1.,np.clip((speed-vel)/.20,-2.,2.),np.clip(1.e6/stiffness-1.,-2.,2.)])
    learned=np.tanh(theta.reshape(2,3)@cond) if fit else np.zeros(2)
    warm=np.tile(.55*learned,(2,1))
    # Modes are a standard handover and a longer load-transfer handover. Both use
    # the same diagonal-stance library, bounds and low-level constrained force QP.
    def residual(z):
        bounded=np.tanh(z); dx=.014*bounded[...,0]; height=.002*bounded[...,1]
        ramp=np.array([.050,.072])[:,None]
        nominal_dent=weight*.25/stiffness
        pred_speed=vel+.55*(speed-vel)+.65*dx/(.20+ramp)-.025*(ramp-.05)/.022
        pred_pitch=imbalance*.055-.7*dx+2.8*height
        compression=(weight*.25/stiffness)*(.05/ramp)**.35 - height*.15
        return np.stack([(pred_speed-speed)/.10, pred_pitch/.12, (compression-nominal_dent)/.0015,
                         dx/.065, height/.009],axis=-1)
    result=finite_proposal(warm,residual,noise)
    z=np.tanh(result['z'])
    return dict(mode=int(result['mode']), dx=float(.014*z[0]),height=float(.002*z[1]),
                ramp=(.050,.072)[result['mode']],candidates=int(result['evaluated_candidates']),
                warm=warm.tolist(), per_mode_cost=result['per_mode_cost'].tolist())
