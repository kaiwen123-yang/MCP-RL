"""Apply the two r54 counterexamples to the frozen N kernel."""
import sys
from pathlib import Path
import w55_n_prepare as old
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'work/w55-n-r56';SRC=OUT/'source'
EXP=ROOT/'experiments/terrain_memory'
def main():
    old.OUT=OUT;old.SRC=SRC;old.main()
    header=(EXP/'w55_n/support.h').read_text().replace('W55Data=W55Surface+W55Nx*W55Ny','W55Motion=W55Surface+W55Nx*W55Ny, W55Data=W55Motion+3')
    header+='\nextern "C" int w56_progress(const mjModel*,mjData*,const double*,double*);\n'
    (OUT/'support.h').write_text(header);(SRC/'mjpc/w55_support.h').write_text(header)
    c=(EXP/'w55_n/support.cc').read_text();a='out[j]=d->qpos[m->jnt_qposadr[joint]]+(braking[j]+5*d->qvel[m->jnt_dofadr[joint]])/60;'
    b='''double q=d->qpos[m->jnt_qposadr[joint]],v=d->qvel[m->jnt_dofadr[joint]];
    // Convert back to the exactly representable original position interval.
    // Reconstruct actual PD torque and use it in the very same response.
    out[j]=std::clamp(q+(braking[j]+5*v)/60,m->actuator_ctrlrange[2*j],m->actuator_ctrlrange[2*j+1]);
    braking[j]=60*(out[j]-q)-5*v;'''
    assert c.count(a)==1;c=c.replace(a,b)
    a='  for(int i=0;i<F;++i){double* e=log+i*W55Diag;e[6]='
    assert c.count(a)==1;c=c.replace(a,'  for(const auto& row:qp.rows)if(dot(row.a,braking)<row.b-2.e-5)return fail(3);\n'+a)
    (OUT/'support.cc').write_text(c+'\n'+(EXP/'w55_n_r56/progress.inc').read_text())
    p=SRC/'mjpc/planners/ilqg/planner.cc';s=p.read_text()
    s=s.replace('num_rollouts_gui_, kMaxTrajectory-1','num_rollouts_gui_, kMaxTrajectory-2')
    a='  if(const char* dir=std::getenv("MJPC_W54_N_RECORD_DIR")){'
    candidate='''  const int progress=recovery+1;
  auto& advancing=candidate_policy[progress];advancing.CopyFrom(policy,horizon);
  advancing.representation=0;advancing.feedback_scaling=0.;
  std::fill(advancing.feedback_gain.begin(),advancing.feedback_gain.end(),0.);
  std::fill(advancing.action_improvement.begin(),advancing.action_improvement.end(),0.);
  bool generator_ok=true;double rejected_time=-1;
  auto advance_action=[&](double* u,const double* x,double t){
    double shifted[12];policy.Action(shifted,x,t);
    int reason=w56_progress(model,data_[0].get(),shifted,u);
    if(reason&&generator_ok){generator_ok=false;rejected_time=t;}
  };
  trajectory[progress].Rollout(advance_action,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
  ++w53_rollouts;
  if(!generator_ok){auto& tr=trajectory[progress];tr.failure=true;tr.total_return=1.e6;
    if(!tr.w55_reason||rejected_time<tr.w55_first_time){tr.w55_reason=9;tr.w55_first_time=rejected_time;}}
  advancing.trajectory=trajectory[progress];
  if(!trajectory[progress].failure && (best_nominal<0 ||
       trajectory[progress].total_return<trajectory[best_nominal].total_return))best_nominal=progress;
'''
    assert s.count(a)==1;s=s.replace(a,candidate+a)
    s=s.replace('i<=recovery;++i','i<=progress;++i').replace('i<=num_trajectory_;++i','i<=num_trajectory_+1;++i')
    s=s.replace('i==recovery?"support_recovery":"shifted_nominal"','i==progress?"supported_progress":i==recovery?"support_recovery":"shifted_nominal"')
    s=s.replace('i==num_trajectory_?"support_recovery":"action_search"','i==num_trajectory_+1?"supported_progress":i==num_trajectory_?"support_recovery":"action_search"')
    s=s.replace('w54_nominal_slot==num_trajectory_?"support_recovery":"current_reassessed_nominal"','w54_nominal_slot==num_trajectory_+1?"supported_progress":w54_nominal_slot==num_trajectory_?"support_recovery":"current_reassessed_nominal"')
    s=s.replace('best_nominal==num_trajectory_?0.', 'best_nominal>=num_trajectory_?0.')
    p.write_text(s)
if __name__=='__main__':main()
