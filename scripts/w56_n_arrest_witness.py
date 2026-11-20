"""Recover native first rejected substeps and independently solve their exact linear constraints."""
import json,os,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from analyze_s1_w55_n import records,read_policy
from run_s1_w56_n import ROOT,SMALL,LARGE,write
from s1_go2_w56_n import SIZE,SURFACE,MOTION

def main():
    src=json.loads((SMALL/'N-A-r56/E30.json').read_text());root=LARGE/src['attempt'];native=root/'w54-n-native'
    c=next(x for x in records(native/'calls.jsonl') if x['full_candidates'] and x['time_s']>12.5)
    call=c['call_id'];ctx=next(x for x in records(root/'w54-n-context.jsonl') if x['call_id']==call)
    work=ROOT/'work/w55-n-r56';dest=work/'arrest-witness';dest.mkdir(exist_ok=True)
    native_src=(work/'support.cc').read_text()
    old='if(!qp.project(raw,x))return 7;double available='
    new='''if(!qp.project(raw,x)){
        std::ofstream o(std::getenv("W56_ARREST_ROWS"));o<<std::setprecision(17);
        o<<required<<" "<<drift<<" "<<closing<<" "<<clearance<<"\\n";
        for(const auto& r:qp.rows){for(double a:r.a)o<<a<<" ";o<<r.b<<"\\n";}
        return 7;
      } double available='''
    assert native_src.count(old)==1
    (dest/'support.cc').write_text('#include <fstream>\n#include <iomanip>\n#include <cstdlib>\n'+native_src.replace(old,new))
    (dest/'support.h').write_bytes((work/'support.h').read_bytes())
    harness='''#include "support.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  char msg[2048];mjModel*m=mj_loadXML(argv[1],nullptr,msg,sizeof(msg));if(!m){std::cerr<<msg;return 2;}
  mjData*d=mj_makeData(m);m->opt.timestep=.001;m->opt.disableflags|=mjDSBL_WARMSTART;
  for(int i=0;i<m->njnt;++i)m->jnt_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->ngeom;++i)m->geom_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->npair;++i)m->pair_solimp[mjNIMP*i]=0;
  std::ifstream in(argv[2],std::ios::binary);auto get=[&](double*x,int n){in.read((char*)x,n*sizeof(double));};
  get(&d->time,1);get(d->qpos,m->nq);get(d->qvel,m->nv);get(d->ctrl,m->nu);get(d->mocap_pos,3*m->nmocap);get(d->mocap_quat,4*m->nmocap);get(d->userdata,m->nuserdata);
  w55_native_contact_filter();double gap=1e9,margin=1e9;int reason=0;
  for(int i=0;i<std::atoi(argv[3]);++i){w55_native_step(m,d,true);reason=w55_checks(m,d,&gap,&margin);}
  std::cout<<reason<<" "<<d->time; mj_deleteData(d);mj_deleteModel(m);
}'''
    (dest/'probe.cc').write_text(harness)
    deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps';lib=ROOT/'work/r11-frozen/S5-r56/lib'
    subprocess.run(['g++','-std=c++17','-O3','-I'+str(deps/'mujoco-src/include'),str(dest/'support.cc'),str(dest/'probe.cc'),
        '-L'+str(lib),'-Wl,-rpath,'+str(lib),'-l:libmujoco.so.3.2.3','-o',str(dest/'probe')],check=True)
    u=np.zeros(SIZE);u[:SURFACE]=ctx['userdata_prefix'];grid=u[SURFACE:MOTION].reshape(70,60);u[MOTION:]=ctx['epoch_direction_command']
    for ix,iy,h in ctx['world_surface_m']:grid[ix-40,iy+30]=h
    rows=[]
    for a in c['candidates']:
        if a['dx_m']<=0 or a['first_constraint_code']!=7:continue
        slot=a['slot'];p=read_policy(native/f'call-{call:06d}-candidate-{slot:02d}.bin')
        end=a['first_constraint_time_s'];k=int(np.searchsorted(p['accepted_times'],end-1.e-8)-1);begin=p['accepted_times'][k];n=int(round((end-begin)/.001))
        data=dest/f'slot-{slot}.bin';table=dest/f'slot-{slot}-rows.txt'
        np.concatenate([[begin],p['accepted_states'][k],p['applied_actions'][k],np.ravel(ctx['mocap_pos']),np.ravel(ctx['mocap_quat']),u]).tofile(data)
        log=subprocess.check_output([str(dest/'probe'),str(root/'model/G1-project-go2.xml'),str(data),str(n)],text=True,env=dict(os.environ,W56_ARREST_ROWS=str(table)))
        reason=int(log.split()[0]);r=dict(slot=slot,original_code=a['first_constraint_code'],reconstructed_code=reason,
            first_constraint_time_s=end,k=k,native_substeps=n,predicted_dx_m=a['dx_m'])
        if reason==7 and table.exists():
            detail=np.fromstring(table.read_text().splitlines()[0],sep=' ');mat=np.loadtxt(table,skiprows=1);A,b=mat[:,:12],mat[:,12]
            lp=linprog(np.zeros(12),A_ub=-A,b_ub=-b,bounds=[(None,None)]*12,method='highs')
            support=linprog(np.zeros(12),A_ub=-A[:-1],b_ub=-b[:-1],bounds=[(None,None)]*12,method='highs')
            r.update(rows=len(b),required_point_acceleration_m_s2=detail[0],drift=detail[1],closing_speed_m_s=detail[2],
                clearance_m=detail[3],highs_full_status=int(lp.status),highs_support_without_arrest_status=int(support.status))
            if lp.success:r.update(feasibility_witness_start_torque_Nm=lp.x.tolist(),max_normalized_violation=max(0.,float(np.max(b-A@lp.x))))
        rows.append(r)
    report=dict(source_attempt=src['attempt'],source_call=call,source_time_s=c['time_s'],rows=rows,robot_physics_steps=0,
        note='Reconstructed only the first rejected saved native substep from its exact accepted policy and causal grid. HiGHS tests identical normalized inequalities rather than relaxing a bound. A numerical feasible witness does not qualify an entire candidate or an original task.')
    # Follow the highest positive-dx candidate from its first demonstrated
    # numerical false rejection to the first independently infeasible node.
    positive=next(r for r in rows if r.get('highs_full_status')==0)
    slot=positive['slot'];saved=read_policy(native/f'call-{call:06d}-candidate-{slot:02d}.bin')
    tail=[];later=None
    original_input=np.fromfile(dest/f'slot-{slot}.bin')
    packet=original_input[1+37+12:]
    for k in range(positive['k'],175):
        data=dest/'tail-input.bin';np.concatenate([[saved['accepted_times'][k]],saved['accepted_states'][k],
            saved['applied_actions'][k],packet]).tofile(data)
        for sub in (1,2):
            t=saved['accepted_times'][k]+.001*sub
            if t+1e-9<positive['first_constraint_time_s']:continue
            table=dest/f'tail-{k}-{sub}-rows.txt'
            out=subprocess.check_output([str(dest/'probe'),str(root/'model/G1-project-go2.xml'),str(data),str(sub)],
                 env=dict(os.environ,W56_ARREST_ROWS=str(table)),text=True)
            code=int(out.split()[0]);row=dict(time_s=float(t),code=code)
            if code==7 and table.exists():
                mat=np.loadtxt(table,skiprows=1)
                lp=linprog(np.zeros(12),A_ub=-mat[:,:12],b_ub=-mat[:,12],bounds=[(None,None)]*12,method='highs')
                row['highs_status']=int(lp.status)
            tail.append(row)
            if code and not (code==7 and row.get('highs_status')==0):
                later=row;break
        if later:break
    report['slot8_after_proven_false_first_rejection']=dict(start_s=positive['first_constraint_time_s'],
        original_horizon_end_s=float(saved['accepted_times'][-1]),checked_saved_substeps=len(tail),
        numeric_false_rejections=sum(x.get('highs_status')==0 for x in tail),
        next_independent_or_other_hard_constraint=later,full_tail_checked=later is None,rows=tail)
    write(SMALL/'r56-native-arrest-witness.json',report);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
