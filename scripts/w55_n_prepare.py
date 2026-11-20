"""Reconstruct frozen r48 then apply W55 projection and recovery policy selection."""
import io, shutil, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/w55-n-r54'
SRC=OUT/'source'
EXP=ROOT/'experiments/terrain_memory'
def change(file,old,new,count=1):
    p=SRC/'mjpc'/file;s=p.read_text()
    if s.count(old)!=count:raise ValueError((file,old[:90],s.count(old),count))
    p.write_text(s.replace(old,new))
def main():
    up=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
    SRC.mkdir(parents=True,exist_ok=False)
    payload=subprocess.check_output(['git','archive','e00c47a5adb9856af2e0f24231bb3a60d5be23c4','mjpc'],cwd=up)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:archive.extractall(SRC,filter='data')
    for name in ['startup','equilibrium','support']:
        patch=subprocess.check_output(['git','show',f'fb7f924f2d519582a127d15ff2e1e2cb2cd5bc9e:experiments/terrain_memory/s1-go2-wb-{name}-upstream.patch'],cwd=ROOT)
        subprocess.run(['patch','-p1','--quiet','--batch','--forward'],input=patch,cwd=SRC,check=True)
    for rev in (42,48):
        subprocess.run(['patch','-p1','--quiet','--batch','--forward'],input=(EXP/f's1-go2-r11-S5-r{rev}.patch').read_bytes(),cwd=SRC,check=True)
    pj='planners/ilqg/planner.cc';tr='trajectory.cc'
    file=SRC/'mjpc/w54_e_native.h';old=file.read_text()
    a=old.index('inline void W54NormalJdotV');b=old.index('inline void W54Transition(',a)
    s=old[:a]+'''inline void W54ApplyE(const mjModel* m,mjData* d,double* saved=nullptr){
  if(!W54Enabled(m))return;
  double raw[12];mju_copy(raw,d->ctrl,12);w55_project(m,d,raw,d->ctrl,saved);
}
'''+old[b:]
    s=s.replace('#include "mjpc/w54_e_core.h"','#include "mjpc/w55_support.h"')
    s=s.replace('if(terminal)mj_forward(m,d);else mj_step(m,d);','w55_native_step(m,d,!terminal);')
    s=s.replace('if(!terminal)mj_step(m,d);','if(!terminal)w55_native_step(m,d,true);')
    file.write_text(s)
    shutil.copyfile(EXP/'w55_n/support.h',SRC/'mjpc/w55_support.h')
    change('trajectory.h','bool w54_domain_feasible=true;', 'int w55_reason=0; double w55_first_time=-1.,w55_margin=1.e9; bool w54_domain_feasible=true;')
    change(tr,'w54_domain_feasible=true;w54_domain_min_gap=1.e9;', 'w54_domain_feasible=true;w54_domain_min_gap=1.e9;w55_reason=0;w55_first_time=-1.;w55_margin=1.e9;',3)
    f=SRC/'mjpc'/tr;s=f.read_text().replace('40*T','(4*W55Diag)*T').replace('t*40','t*(4*W55Diag)').replace('(horizon-1)*40','(horizon-1)*(4*W55Diag)').replace('(horizon-2)*40','(horizon-2)*(4*W55Diag)').replace('),40);','),4*W55Diag);')
    s=s.replace('mj_step(model, data);','w55_native_step(model,data,true);').replace('mj_forward(model, data);','w55_native_step(model,data,false);')
    marker='w54_domain_feasible &= W54PredictedDomainOkay(model,data,&w54_domain_min_gap);'
    assert s.count(marker)==4
    s=s.replace(marker,'''{int reason=w55_checks(model,data,&w54_domain_min_gap,&w55_margin);
        if(w54_execution_diagnostics[t*(4*W55Diag)+23]!=0)reason=8;
        if(reason&&!w55_reason){w55_reason=reason;w55_first_time=data->time;}
        w54_domain_feasible &= reason==0;}''')
    f.write_text(s)
    change(pj,'#include "mjpc/planners/ilqg/planner.h"','#include "mjpc/planners/ilqg/planner.h"\n#include "mjpc/w55_support.h"')
    change(pj,'  w53_installed=false;w53_rollouts=0;', '  w55_native_contact_filter();\n  w53_installed=false;w53_rollouts=0;')
    start='  int best_nominal = this->BestRollout();'
    change(pj,start,start+'''
  const int recovery=num_trajectory_;
  auto& ref=candidate_policy[recovery];
  ref.CopyFrom(policy,horizon);
  ref.representation=0;ref.feedback_scaling=0.;
  std::fill(ref.feedback_gain.begin(),ref.feedback_gain.end(),0.);
  std::fill(ref.action_improvement.begin(),ref.action_improvement.end(),0.);
  auto recovery_action=[&](double* u,const double*,double){w55_recovery(model,data_[0].get(),u);};
  trajectory[recovery].Rollout(recovery_action,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
  ++w53_rollouts;
  ref.trajectory=trajectory[recovery];
  if(!trajectory[recovery].failure && (best_nominal<0 ||
       trajectory[recovery].total_return<trajectory[best_nominal].total_return))best_nominal=recovery;
''')
    change(pj,'  w54_nominal_slot=best_nominal;',r'''  if(const char* dir=std::getenv("MJPC_W54_N_RECORD_DIR")){
    std::ofstream record(std::string(dir)+"/nominal-candidates.jsonl",std::ios::app);
    record<<std::setprecision(17);
    for(int i=0;i<=recovery;++i){const auto& tr=trajectory[i];
      record<<"{\"call_id\":"<<(w53_call+1)<<",\"time_s\":"<<time<<",\"slot\":"<<i
            <<",\"generator\":\""<<(i==recovery?"support_recovery":"shifted_nominal")<<"\",\"J\":"<<tr.total_return
            <<",\"feasible\":"<<(tr.failure?"false":"true")<<",\"first_constraint_code\":"<<tr.w55_reason
            <<",\"first_constraint_time_s\":"<<tr.w55_first_time<<",\"min_arrest_margin_m\":"<<tr.w55_margin<<"}\n";
    }
  }
  w54_nominal_slot=best_nominal;''')
    change(pj,'num_trajectory_ = mju_min(num_rollouts_gui_, kMaxTrajectory);',
               'num_trajectory_ = mju_min(num_rollouts_gui_, kMaxTrajectory-1);')
    change(pj,'    feedback_scaling = linesearch_steps[best_nominal];','    feedback_scaling = best_nominal==num_trajectory_?0.:linesearch_steps[best_nominal];')
    f=SRC/'mjpc'/pj;s=f.read_text()
    s=s.replace('int64_t hdr[10]={54,3,','int64_t hdr[10]={55,1,').replace('horizon*40','horizon*(4*W55Diag)').replace('k*40+foot*10','k*(4*W55Diag)+foot*W55Diag')
    s=s.replace('if(full)for(int i=0;i<num_trajectory_;++i)', 'if(full)for(int i=0;i<=num_trajectory_;++i)')
    s=s.replace('for(int i=0;i<num_trajectory_;++i){if(i)out', 'for(int i=0;i<=num_trajectory_;++i){if(i)out')
    s=s.replace('out<<"}";',r'''out<<",\"generator\":\""<<(i==num_trajectory_?"support_recovery":"action_search")<<"\",\"first_constraint_code\":"<<tr.w55_reason<<",\"first_constraint_time_s\":"<<tr.w55_first_time<<",\"minimum_arrest_margin_m\":"<<tr.w55_margin<<"}";''')
    s=s.replace('w54_nominal_selected?"current_reassessed_nominal":"action_rollout"', 'w54_nominal_selected?(w54_nominal_slot==num_trajectory_?"support_recovery":"current_reassessed_nominal"):"action_rollout"')
    f.write_text(s)
    print(SRC,flush=True)
if __name__=='__main__':main()
