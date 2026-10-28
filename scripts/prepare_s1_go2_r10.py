#!/usr/bin/env python3
"""Compile the declared R10 formulation and preserve its own source/binary."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--form', choices=['G','GN','GNC'], required=True)
    p.add_argument('--build',default=None)
    p.add_argument('--jobs',type=int,default=12)
    a = p.parse_args()
    source = ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
    relative = Path('mjpc/tasks/quadruped/quadruped.cc')
    prior = ROOT/'work/r10-source-before'/relative
    prior.parent.mkdir(parents=True, exist_ok=True)
    if not prior.exists():
        shutil.copy2(source/relative, prior)
    old = prior.read_text()
    new = old.replace('ground_height + kFootRadius + step[foot]',
        'ground_height + model->geom_size[3*foot_geom_id_[foot]] + step[foot]')
    new = new.replace('ResidualFn::kFootRadius + ground[foot]',
        'model->geom_size[3*residual_.foot_geom_id_[foot]] + ground[foot]')
    new = new.replace('ResidualFn::kFootRadius + step[foot] + ground[foot]',
        'model->geom_size[3*residual_.foot_geom_id_[foot]] + step[foot] + ground[foot]')
    header = source/'mjpc/tasks/quadruped/quadruped.h'
    header_prior = prior.with_suffix('.h')
    if not header_prior.exists():
        shutil.copy2(header, header_prior)
    header_new = header_prior.read_text()
    if a.form in ('GN','GNC'):
        header_new = header_new.replace('int foot_geom_id_[kNumFoot];',
            'int foot_geom_id_[kNumFoot];\n    bool soft_landing_ = false;')
        new = new.replace('  // ----------  model identifiers  ----------',
            '  residual_.soft_landing_ = GetNumberOrDefault(0, model, "r10_normal_landing") == 1;\n\n'
            '  // ----------  model identifiers  ----------')
        new = new.replace('    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));',
            '''    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
    if (soft_landing_) {
      double u = (mju_clip(angle, -mjPI/2, mjPI/2) + mjPI/2) / mjPI;
      double s = u < 0.5 ? 2*u : 2*(1-u);
      value = s*s*s*(10 + s*(-15 + 6*s));
    }''')
        landing = (ROOT/'experiments/terrain_memory/s1-go2-r10-normal.inc').read_text()
        new = new.replace('  // sensor dim sanity check', landing+'\n  // sensor dim sanity check')
    header.write_text(header_new)
    (source/relative).write_text(new)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
        fromfile='a/'+str(relative), tofile='b/'+str(relative)))
    patch += ''.join(difflib.unified_diff(header_prior.read_text().splitlines(True),header_new.splitlines(True),
        fromfile='a/mjpc/tasks/quadruped/quadruped.h',tofile='b/mjpc/tasks/quadruped/quadruped.h'))
    if a.form == 'GNC':
        contact=ROOT/'experiments/terrain_memory/s1-go2-r10-contact.h'
        shutil.copy2(contact,source/'mjpc/tasks/quadruped/r10_contact.h')
        for name in ('mjpc/agent.cc','mjpc/trajectory.cc','mjpc/planners/model_derivatives.cc','mjpc/grpc/agent_service.cc'):
            path=source/name;before=ROOT/'work/r10-source-before'/name
            before.parent.mkdir(parents=True,exist_ok=True)
            if not before.exists():shutil.copy2(path,before)
            text=before.read_text()
            if name=='mjpc/agent.cc':
                for model in ('model','model_'):
                    text=text.replace(f'GetNumberOrDefault(1, {model}, "s1_control_hold_substeps") == 2',
                        f'GetNumberOrDefault(1, {model}, "s1_control_hold_substeps") > 1')
                    text=text.replace('horizon_ / (2 * timestep_)',
                        f'horizon_ / (GetNumberOrDefault(1, {model}, "s1_control_hold_substeps") * timestep_)',1)
            elif name=='mjpc/trajectory.cc':
                text=text.replace('const bool held_control = GetNumberOrDefault(1, model, "s1_control_hold_substeps") == 2;',
                    'const int control_substeps = GetNumberOrDefault(1, model, "s1_control_hold_substeps");')
                text=text.replace('(held_control ? 2 : 1)', 'control_substeps')
                text=text.replace('if (held_control) mj_step(model, data);',
                    'for (int substep=1;substep<control_substeps;++substep) mj_step(model,data);')
            elif name=='mjpc/planners/model_derivatives.cc':
                start=text.index('        if (GetNumberOrDefault(1, m, "s1_control_hold_substeps") == 2) {')
                end=text.index('\n        }',start)+len('\n        }')
                text=text[:start]+'''        const int control_substeps = GetNumberOrDefault(1,m,"s1_control_hold_substeps");
        if (control_substeps > 1) {
          const int nx=dim_state_derivative, nu=dim_action;
          double* total_a=DataAt(A,t*nx*nx);
          double* total_b=DataAt(B,t*nx*nu);
          std::vector<double> old_a(nx*nx),old_b(nx*nu),next_a(nx*nx),next_b(nx*nu);
          for (int substep=1;substep<control_substeps;++substep) {
            mju_copy(old_a.data(),total_a,nx*nx);mju_copy(old_b.data(),total_b,nx*nu);
            mj_step(m,d);
            mjd_transitionFD(m,d,tol,mode,next_a.data(),next_b.data(),nullptr,nullptr);
            mju_mulMatMat(total_a,next_a.data(),old_a.data(),nx,nx,nx);
            mju_mulMatMat(total_b,next_a.data(),old_b.data(),nx,nx,nu);
            mju_addTo(total_b,next_b.data(),nx*nu);
          }
        }'''+text[end:]
            else:
                text=text.replace('#include "mjpc/trajectory.h"',
                    '#include "mjpc/trajectory.h"\n#include "mjpc/tasks/quadruped/r10_contact.h"')
                text=text.replace('  agent_.SetTaskList(tasks_);',
                    '  mjcb_contactfilter = mjpc::R10ContactFilter;\n  mjcb_passive = mjpc::R10ContactForce;\n  agent_.SetTaskList(tasks_);',1)
            path.write_text(text)
            patch+=''.join(difflib.unified_diff(before.read_text().splitlines(True),text.splitlines(True),
                fromfile='a/'+name,tofile='b/'+name))
    label=a.build or a.form
    (ROOT/('experiments/terrain_memory/s1-go2-r10-'+label+'.patch')).write_text(patch)
    begin = time.time()
    subprocess.run(['cmake','--build',str(source/'build-autodl'),
        '--target','agent_server','-j',str(a.jobs)], check=True)
    frozen = ROOT/'work/r10-frozen'/label
    frozen.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'mjpc', frozen/'mjpc')
    shutil.copytree(source/'build-autodl/mjpc/tasks', frozen/'runtime-tasks')
    shutil.copy2(source/'SOURCE_COMMIT', frozen/'SOURCE_COMMIT')
    shutil.copy2(source/'build-autodl/bin/agent_server', frozen/'agent_server')
    (frozen/'lib').mkdir()
    shutil.copy2(source/'build-autodl/lib/libmujoco.so.3.2.3', frozen/'lib/libmujoco.so.3.2.3')
    identity = dict(formulation=a.form, build=label, compiled_unix_s=time.time(),build_wall_s=time.time()-begin,
        source_sha256=hashlib.sha256(new.encode()).hexdigest(),
        binary_sha256=hashlib.sha256((frozen/'agent_server').read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        geometry='Per-foot radius from compiled model; physical geometry unchanged')
    identity['source_files_sha256']={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (frozen/'mjpc').rglob('*') if p.is_file() and p.suffix in ('.cc','.h')}
    (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity),flush=True)


if __name__ == '__main__':
    main()
