"""Contact-chart partial task decoupling for the r55 articulated WBC.

Only loaded contacts define the local kinematic chart. A compliant normal is
excluded. The final QP operates in the original ambient coordinates and keeps
all six floating rows and the 10-sample torque/actuator constraints.
"""
import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation


def inverse_psd(matrix):
    return np.linalg.pinv((matrix+matrix.T)*.5, rcond=1e-9, hermitian=True)


def contact_chart(mass,jac,drift,selector,task,task_drift,bias):
    n = len(mass)
    if len(jac):
        left, sigma, vt = np.linalg.svd(jac,full_matrices=True)
        rank = int(np.sum(sigma > 1.e-9*sigma[0]))
        tangent = vt[rank:].T
        affine_acc = -vt[:rank].T @ ((left[:,:rank].T @ drift)/sigma[:rank])
    else:
        tangent,affine_acc,rank = np.eye(n),np.zeros(n),0
    reduced = tangent.T @ mass @ tangent
    inverse_tangent = tangent @ np.linalg.solve(reduced,tangent.T)
    wrench_map = task @ inverse_tangent
    inertia = inverse_psd(wrench_map @ task.T)
    # This is the ambient representation of N_dot nu. The connection
    # contributes M*a_aff to reduced drift, as well as J*a_aff to task drift.
    free = task_drift + task @ (affine_acc-inverse_tangent @ (bias+mass@affine_acc))
    mobility = wrench_map @ task.T
    reachable = wrench_map @ selector.T
    projector = reachable @ np.linalg.pinv(reachable,rcond=1.e-8)
    return inertia, mobility, projector, free, rank, affine_acc


class NaturalWBCMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.d_role = self.cfg['w61']['role']
        if self.d_role == 'hybrid':
            self.d_role = 'decoupled'
        if self.d_role not in ('coupled','decoupled','hierarchy'):
            raise ValueError('Choose coupled, decoupled, or hierarchy')
        self.d_chart = None
        self.d_values = []
        self.d_switched = 0
        self.d_tick = None

    def wbc(self,t,data,mq,mission,scheduled,observed,grf,soft,material_info):
        self.d_tick=(mq,scheduled & observed & self.confirmed & self.load_bearing,soft,grf)
        return super().wbc(t,data,mq,mission,scheduled,observed,grf,soft,material_info)

    def _solve(self,P,q,A,lo,hi,kind,t,warm=None):
        if kind!='wbc':
            return super()._solve(P,q,A,lo,hi,kind,t,warm)
        mq,active,soft,measured=self.d_tick
        cfg=self.cfg['wbc']
        rows,drifts=[],[]
        labels=[]
        for leg in np.flatnonzero(active):
            width=2 if soft[leg] else 3
            rows.extend(mq['Jroll'][leg,:width])
            drifts.extend(mq['Jrolldotv'][leg,:width])
            labels.append((int(leg),width))
        nv=self.model.nv
        jc=np.asarray(rows).reshape(-1,nv)
        djv=np.asarray(drifts)
        scale=np.r_[np.ones(3),np.full(3,.25)]
        J=np.vstack([mq['Jcom'],mq['Jr']])
        dJv=np.r_[mq['Jcomdotv'],mq['Jrdotv']]
        T=scale[:,None]*J
        selector=np.eye(nv)[self.va]
        inertia,W,proj,free,rank,affine=contact_chart(
            mq['M'],jc,djv,selector,T,scale*dJv,mq['h'])
        key=(tuple(labels),rank)
        if key!=self.d_chart:
            self.d_chart=key
            self.d_enter=t
            self.d_switched+=1
        phase=np.clip((t-self.d_enter)/.018,0.,1.)
        transition=phase*phase*(3.-2.*phase)
        # All optimization variables remain [qdd, four GRFs]. The tangent
        # chart is used for the response operator; material normal constraints
        # and contact inequalities stay in the original WBC problem.
        if len(jc):
            A=np.vstack([A,np.pad(jc,((0,0),(0,12)))])
            lo=np.r_[lo,-djv];hi=np.r_[hi,-djv]
        dt=self.cfg['mpc']['dt_s']
        plan=self.last_plan
        x=plan['state']
        a=np.clip(t-plan['time'],0.,dt)/dt
        xd=(1.-a)*x[0]+a*x[1]
        ff=np.r_[(x[1,9:12]-x[0,9:12])/dt,(x[1,6:9]-x[0,6:9])/dt]
        ori=Rotation.from_matrix(Rotation.from_euler('xyz',xd[:3]).as_matrix()@mq['R'].T).as_rotvec()
        fb=np.r_[cfg['body_position_kp']*(xd[3:6]-mq['com'])+cfg['body_velocity_kd']*(xd[9:12]-mq['com_velocity']),
                 cfg['orientation_kp']*ori+cfg['angular_velocity_kd']*(xd[6:9]-mq['omega'])]
        weights=np.r_[np.full(3,cfg['body_weight']),np.full(3,cfg['orientation_weight'])]
        D=block_diag(inertia[:3,:3],inertia[3:,3:])
        Wd=inverse_psd(D)
        # Prior solved force and current measured force are causal. This
        # estimates a net task wrench; it never adds a force to the plant.
        residual=measured-self.last_force
        disturbance=np.r_[residual.sum(0),np.cross(mq['contact_force_point']-mq['com'],residual).sum(0)/.25]
        wrench=D@(scale*fb)
        passive=free+W@(inertia@(scale*ff-free)+wrench)
        delta=proj@(Wd-W)@(wrench+disturbance)
        desired=(passive+(transition*delta if self.d_role=='decoupled' else 0.))/scale
        Z=np.pad(J,((0,0),(0,12)))
        if self.d_role=='hierarchy':
            primary=np.eye(len(q))*1e-8+2.*(Z.T*weights)@Z
            gradient=-2.*Z.T@(weights*(ff+fb-dJv))
            first,_=super()._solve(primary,gradient,A,lo,hi,'wbc_priority',t)
            # Preserve the attainable primary task, then optimize the existing
            # posture, foot and force objectives within that solution set.
            A=np.vstack([A,Z]);achieved=Z@first
            lo=np.r_[lo,achieved-1e-6];hi=np.r_[hi,achieved+1e-6]
        else:
            # Retain original foot/posture/force objectives. Replace only the
            # body feedback target; the original J' W J Hessian is identical.
            q=q-2.*Z.T@(weights*(desired-(ff+fb)))
        z,status=super()._solve(P,q,A,lo,hi,kind,t,warm)
        connection_error=float(np.linalg.norm(jc@affine+djv)) if len(jc) else 0.
        kin_error=float(np.linalg.norm(jc@z[:nv]+djv)) if len(jc) else 0.
        floating_error=float(np.linalg.norm(A[:6]@z-hi[:6]))
        self.d_values.append((t,rank,np.linalg.matrix_rank(proj,tol=1e-7),transition,
            np.linalg.norm(W[:3,3:]),np.linalg.norm(delta),connection_error,kin_error,
            floating_error,np.linalg.norm(desired-(ff+fb))))
        return z,status

    def w61_d_stats(self):
        a=np.asarray(self.d_values)
        if not len(a):
            return {'solved':0,'charts':self.d_switched}
        return dict(solved=len(a),charts=self.d_switched,rank_counts={
            str(int(r)):int(n) for r,n in zip(*np.unique(a[:,1],return_counts=True))},
            reachable_task_ranks=np.unique(a[:,2]).astype(int).tolist(),
            mean_mobility_cross_block=float(a[:,4].mean()),
            mean_decoupling_accel_correction=float(a[:,5].mean()),
            connection_residual_max=float(a[:,6].max()),kinematic_constraint_residual_max=float(a[:,7].max()),
            floating_dynamics_residual_max=float(a[:,8].max()),
            shaped_vs_nominal_accel_mean=float(a[:,9].mean()))
