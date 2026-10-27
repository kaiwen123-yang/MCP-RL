"""Static Go2 support references using the unchanged native forward model."""
import numpy as np
import mujoco
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation

def solve_stance(model,q_seed,body_xy,heading,height,anchor_xy):
    """Solve zero-velocity forward balance with fixed foot xy and base height.

    The 27 unknowns are base z/roll/pitch, 12 joint angles and 12 PD targets.
    The 27 equations are native qacc=0, eight world foot xy and base height.
    No plant trajectory, controller reset or added execution torque is used.
    """
    m=model;d=mujoco.MjData(m);feet=[m.geom(n).id for n in ['FL','FR','RL','RR']]
    base=m.body('trunk').id;joints=m.actuator_trnid[:,0];qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
    q_seed=q_seed.copy()
    d.qpos[:]=q_seed;d.qvel[:]=0;mujoco.mj_forward(m,d)
    floor=float(d.geom_xpos[m.geom('floor').id,2])
    # Solve reachable landing geometry first, retaining established contact xy.
    # All solves use this isolated mjData; the physical episode is untouched.
    landing=np.c_[anchor_xy, floor+m.geom_size[feet,0]-4e-6]
    d.qpos[:2]=body_xy;d.qpos[2]=height
    quat=Rotation.from_euler('xyz',[0.,0.,heading]).as_quat()
    d.qpos[3:7]=quat[[3,0,1,2]]
    def ik(q):
        d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
        return (d.geom_xpos[feet]-landing).ravel()
    pose=least_squares(ik,q_seed[qa],bounds=(m.jnt_range[joints,0],m.jnt_range[joints,1]),
        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=200)
    ik(pose.x);q_seed=d.qpos.copy()
    d.ctrl[:]=q_seed[qa];mujoco.mj_forward(m,d)
    com=d.subtree_com[base];r=d.geom_xpos[feet]-com
    balance=np.stack([np.ones(4),r[:,1],-r[:,0]])
    normal=lsq_linear(balance,np.array([m.body_subtreemass[base]*9.81,0,0]),bounds=(0,np.inf),tol=1e-12).x
    generalized=np.zeros(m.nv)
    for j,foot in enumerate(feet):
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacGeom(m,d,jp,jr,foot)
        generalized+=jp[2]*normal[j]
    tau=(d.qfrc_bias-d.qfrc_passive-generalized)[va]
    u=np.clip(q_seed[qa]+tau/60,m.actuator_ctrlrange[:,0]+1e-8,m.actuator_ctrlrange[:,1]-1e-8)
    rpy=Rotation.from_quat(q_seed[[4,5,6,3]]).as_euler('xyz')
    floor=float(d.geom_xpos[m.geom('floor').id,2])
    gap=np.max(d.geom_xpos[feet,2]-m.geom_size[feet,0]-floor)
    # Start the reference solve with all four known native foot geometries
    # loaded; home itself is 1.84mm above contact and has a flat no-contact Jacobian.
    x0=np.r_[q_seed[2]-gap-3e-6,rpy[:2],q_seed[qa],u]
    lower=np.r_[height-.03,-.15,-.15,m.jnt_range[joints,0],m.actuator_ctrlrange[:,0]]
    upper=np.r_[height+.03,.15,.15,m.jnt_range[joints,1],m.actuator_ctrlrange[:,1]]
    def evaluate(x):
        d.qpos[:]=q_seed;d.qpos[:2]=body_xy;d.qpos[2]=x[0]
        quat=Rotation.from_euler('xyz',[x[1],x[2],heading]).as_quat();d.qpos[3:7]=quat[[3,0,1,2]]
        d.qpos[qa]=x[3:15];d.qvel[:]=0;d.ctrl[:]=x[15:27];d.qacc_warmstart[:]=0
        mujoco.mj_forward(m,d)
        return np.r_[d.qacc/10,(d.geom_xpos[feet,:2]-anchor_xy).ravel()/.01,(d.qpos[2]-height)/.04]
    solved=least_squares(evaluate,x0,bounds=(lower,upper),xtol=1e-11,ftol=1e-11,gtol=1e-11,
        max_nfev=250,x_scale=np.r_[.001,.05,.05,np.full(12,.1),np.full(12,.1)],diff_step=1e-7)
    residual=evaluate(solved.x);force=np.zeros((4,3))
    for k in range(d.ncon):
        contact=d.contact[k];g1,g2=int(contact.geom1),int(contact.geom2)
        for j,foot in enumerate(feet):
            if (g1==foot and m.geom_bodyid[g2]==0) or (g2==foot and m.geom_bodyid[g1]==0):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f)
                force[j]+=contact.frame.reshape(3,3).T@f[:3]*(1 if g2==foot else -1)
    return dict(qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),joint_pose=d.qpos[qa].copy(),
        actuator_torque=d.qfrc_actuator[va].copy(),unclipped_pd=60*(d.ctrl-d.qpos[qa]),
        qacc=d.qacc.copy(),feet=d.geom_xpos[feet].copy(),foot_force=force,
        head_position=d.site_xpos[m.site('head').id].copy(),body_height_m=float(d.xpos[base,2]),relative_height_m=float(d.xipos[base,2]-d.geom_xpos[feet,2].mean()),
        anchor_xy=anchor_xy,body_xy=body_xy,heading=heading,requested_height_m=height,
        max_equation_residual=float(abs(residual).max()),solver_nfev=solved.nfev,solver_message=solved.message,
        qacc_norm=float(np.linalg.norm(d.qacc)),max_foot_xy_error_m=float(abs(d.geom_xpos[feet,:2]-anchor_xy).max()),
        pd_reference_identity_max_error_rad=float(abs(d.ctrl-d.qpos[qa]-d.qfrc_actuator[va]/60).max()),
        friction_ratio=[float(np.linalg.norm(f[:2])/f[2]) if f[2]>0 else None for f in force])


def stand_reference(model,data,heading,height):
    """Freeze loaded contact xy; airborne feet receive native feasible landings."""
    feet=[model.geom(n).id for n in ['FL','FR','RL','RR']]
    contact=np.zeros(4,dtype=bool)
    for k in range(data.ncon):
        c=data.contact[k];f=np.zeros(6);mujoco.mj_contactForce(model,data,k,f)
        for j,g in enumerate(feet):
            if ((c.geom1==g and model.geom_bodyid[c.geom2]==0) or
                (c.geom2==g and model.geom_bodyid[c.geom1]==0)) and f[0]>0:
                contact[j]=True
    ref=solve_stance(model,data.qpos,data.qpos[:2].copy(),heading,height,data.geom_xpos[feet,:2].copy())
    ref['loaded_at_entry']=contact
    ref['reference_kind']=['loaded_contact_anchor' if x else 'reachable_floor_landing' for x in contact]
    ref['entry_time_s']=float(data.time)
    # Feasibility is part of reference construction, never a fallback controller.
    ref['feasible']=bool(ref['qacc_norm']<.01 and ref['max_foot_xy_error_m']<1e-5 and
        np.all(ref['foot_force'][:,2]>1.) and max(ref['friction_ratio'])<.5 and
        ref['pd_reference_identity_max_error_rad']<1e-6)
    return ref


def reference_userdata(ref):
    return np.r_[1.,ref['relative_height_m'],ref['body_height_m'],ref['qpos'][3:7],
        ref['head_position'],ref['feet'].ravel(),ref['joint_pose'],ref['ctrl'],ref['actuator_torque'],ref['foot_force'][:,2]]
