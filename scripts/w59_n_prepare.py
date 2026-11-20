"""Build the W56/r59 support/selector sources on the pinned N/r56 patch chain."""
from pathlib import Path
import w56_n_prepare as parent

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/w56-n-r59'
EXT=ROOT/'experiments/terrain_memory/w56_n_r59'

def replace(s,old,new):
    if s.count(old)!=1:raise ValueError(f'Unexpected source at {old[:70]}')
    return s.replace(old,new)

def main():
    parent.OUT=OUT;parent.SRC=OUT/'source'
    parent.main()
    cc=OUT/'support.cc'
    s=cc.read_text().split('// One additional candidate.')[0]
    a=s.index('  // Dual coordinate minimization');b=s.index('\n};',a)
    s=s[:a]+(EXT/'projection.inc').read_text()+s[b:]
    # Preserve row scales for the original-unit check on the mapped PD command.
    s=replace(s,'struct Row{Vec a{};double b=0.;};','struct Row{Vec a{};double b=0.,scale=1.;};')
    s=replace(s,'rows.push_back({a,b});','rows.push_back({a,b,n>1.e-13?n:1.});')
    mapping="""bool map_and_check(const mjModel* m,const mjData* d,const QP& qp,Vec& tau,double* out){
  for(int j=0;j<U;++j){
    int id=m->actuator_trnid[2*j];double q=d->qpos[m->jnt_qposadr[id]],v=d->qvel[m->jnt_dofadr[id]];
    out[j]=std::clamp(q+(tau[j]+5*v)/60,m->actuator_ctrlrange[2*j],m->actuator_ctrlrange[2*j+1]);
    tau[j]=60*(out[j]-q)-5*v;
  }
  for(const auto& r:qp.rows){
    double residual=r.b-dot(r.a,tau);
    if(residual>2.e-5 || residual*r.scale>1.e-9*(1.+std::abs(r.b*r.scale)))return false;
  }
  return true;
}
"""
    s=replace(s,'Vec row_of(',mapping+'Vec row_of(')
    a=s.index('  for(int j=0;j<U;++j){int joint=m->actuator_trnid[2*j];double q=')
    b=s.index('  for(int i=0;i<F;++i){double* e=log+i*W55Diag;e[6]',a)
    s=s[:a]+'  if(!map_and_check(m,d,qp,braking,out))return fail(3);\n'+s[b:]
    s=replace(s,'if(!qp.project(raw,x))return 7;double available=', 'double mapped[U];if(!qp.project(raw,x)||!map_and_check(m,d,qp,x,mapped))return 7;double available=')
    cc.write_text('#include <string>\n'+s+(EXT/'handoff.inc').read_text())
    proto='extern "C" int w59_handoff(const mjModel*,mjData*,const double*,double,double,double,double,double*);'
    for p in [OUT/'support.h',OUT/'source/mjpc/w55_support.h']:
        p.write_text(replace(p.read_text(),'extern "C" int w56_progress(const mjModel*,mjData*,const double*,double*);',proto))
    # Sample the same phase/cadence/duty/amplitude as the objective. The
    # residual instance is an immutable per-planning-call snapshot.
    p=OUT/'source/mjpc/tasks/quadruped/quadruped.h';s=p.read_text()
    s=replace(s,'    ResidualFn(const ResidualFn&) = default;','''    ResidualFn(const ResidualFn&) = default;
    void HandoffGait(double t,double* g) const {
      g[0]=GetPhase(t);g[1]=phase_velocity_/(2*mjPI);
      g[2]=parameters_[duty_param_id_];g[3]=GetGait()==kGaitTrot?parameters_[amplitude_param_id_]:0.;
    }''')
    p.write_text(s)
    p=OUT/'source/mjpc/planners/ilqg/planner.cc';s=p.read_text()
    s=replace(s,'#include "mjpc/task.h"','#include "mjpc/task.h"\n#include "mjpc/tasks/quadruped/quadruped.h"')
    s=replace(s,'  auto advance_action=', '''  auto snapshot=task->Residual();
  const auto& gait=dynamic_cast<const QuadrupedFlat::ResidualFn&>(*snapshot);
  auto advance_action=''')
    s=replace(s,'int reason=w56_progress(model,data_[0].get(),shifted,u);','double g[4];gait.HandoffGait(t,g);\n    int reason=w59_handoff(model,data_[0].get(),shifted,g[0],g[1],g[2],g[3],u);')
    s=s.replace('"supported_progress"','"trot_support_handoff"')
    p.write_text(s)

if __name__=='__main__':main()
