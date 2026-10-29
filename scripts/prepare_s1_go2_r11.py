#!/usr/bin/env python3
"""Build S5-r1 from frozen R10 GNC-r3, retaining all prior binaries and sources."""
import argparse,difflib,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--build',default='S5-r1');a=p.parse_args()
    source=OLD/'work/terrain-memory/upstreams/mujoco_mpc-go2'
    prior=OLD/'work/r10-frozen/GNC-r3'
    if a.build=='S5-r6':prior=ROOT/'work/r11-frozen/S5-r3'
    if a.build=='S5-r7':prior=ROOT/'work/r11-frozen/S5-r6'
    if a.build=='S5-r8':prior=ROOT/'work/r11-frozen/S5-r7'
    if a.build=='S5-r10':prior=ROOT/'work/r11-frozen/S5-r9'
    if a.build=='S5-r11':prior=ROOT/'work/r11-frozen/S5-r8'
    if a.build=='S5-r13':prior=ROOT/'work/r11-frozen/S5-r12'
    if a.build=='S5-r14':prior=ROOT/'work/r11-frozen/S5-r13'
    if a.build=='S5-r15':prior=ROOT/'work/r11-frozen/S5-r14'
    if a.build=='S5-r16':prior=ROOT/'work/r11-frozen/S5-r15'
    if a.build=='S5-r17':prior=ROOT/'work/r11-frozen/S5-r16'
    if a.build=='S5-r18':prior=ROOT/'work/r11-frozen/S5-r17'
    frozen=ROOT/'work/r11-frozen'/a.build
    if frozen.exists():raise ValueError('This controller build already exists; preserve its identity')
    if a.build in ('S5-r2','S5-r4','S5-r5','S5-r12'):
        specification=json.loads((ROOT/f'experiments/terrain_memory/s1-go2-r11-{a.build}.json').read_text())
        if a.build=='S5-r5':prior=ROOT/'work/r11-frozen/S5-r3'
        if a.build=='S5-r12':prior=ROOT/'work/r11-frozen/S5-r11'
        frozen.mkdir(parents=True)
        for name in ('mjpc','runtime-tasks','lib'):(frozen/name).symlink_to(prior/name)
        for name in ('agent_server','SOURCE_COMMIT'):shutil.copy2(prior/name,frozen/name)
        identity=json.loads((prior/'identity.json').read_text())
        identity.update(controller='S5',build=a.build,based_on=specification['based_on'],
            configuration_overrides=specification['configuration_overrides'],
            reused_frozen_binary=True,physics_unchanged=True)
        (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print(f'Prepared {a.build} controller with unchanged frozen C binary.',flush=True)
        return
    if a.build not in ('S5-r1','S5-r3','S5-r6','S5-r7','S5-r8','S5-r9','S5-r10','S5-r11','S5-r13','S5-r14','S5-r15','S5-r16','S5-r17','S5-r18'):raise ValueError('An undeclared S5 controller build was requested')
    relative=Path('mjpc/tasks/quadruped/quadruped.cc')
    old=(prior/relative).read_text()
    new=old.replace('// 15.206408*9.81 / (2*375000) m: two-foot equilibrium scale.',
        '// S5: same instantaneous C stiffness for force, support indentation and normal-load scale.')
    new=new.replace('double over = -z - 0.00019889981664;',
        '''constexpr double stiffness = .025*(9.*1500.*1000./(3.*1500.+1000.))*1e6*.001/.05;
        const double support_indent = model->body_subtreemass[torso_body_id_]*
                                      mju_abs(model->opt.gravity[2])/(2*stiffness);
        double over = -z - support_indent;''')
    new=new.replace('0.5*(over+mju_sqrt(over*over+1e-12))/0.001;',
        '0.5*(over+mju_sqrt(over*over+1e-12))/support_indent;')
    if a.build=='S5-r3':
        new=old.replace('  // ---------- Upright ----------\n  if (stand_equilibrium) {',
            '''  // S5-r3: first-order roll/pitch error on the original material footprint.
  bool r11_material_support = false;
  if (soft_landing_ && GetCustomNumericData(model, "r10_local_contact")) {
    constexpr double half = 0.015811388300841896;
    for (A1Foot foot : kFootAll) {
      const double* p = foot_pos[foot];
      r11_material_support |= p[0]>=.8+half && p[0]<=2.2-half && mju_abs(p[1])<=.6-half;
    }
  }
  // ---------- Upright ----------
  if (stand_equilibrium || r11_material_support) {''')
    if a.build=='S5-r6':
        domain=(ROOT/'experiments/terrain_memory/s1-go2-r11-domain-margin.inc').read_text()
        new=old.replace('  // sensor dim sanity check',domain+'\n  // sensor dim sanity check')
    if a.build=='S5-r7':
        new=old
        start=new.index('void QuadrupedFlat::ResidualFn::FootStep(')
        end=new.index('\n// height during flip',start)
        new=new[:start]+(ROOT/'experiments/terrain_memory/s1-go2-r11-slow-approach.inc').read_text()+new[end:]
        new=new.replace('  double step[kNumFoot];\n  FootStep(step, GetPhase(data->time), gait);',
            '  double step[kNumFoot], step_velocity[kNumFoot];\n  FootStep(step, GetPhase(data->time), gait, step_velocity);',1)
        new=new.replace('        double desired_vz = -0.05*mju_tanh(mju_max(z,0)/0.001);',
            '        double desired_vz = -0.05*mju_tanh(mju_max(z,0)/0.001);\n        if (slow_approach_) desired_vz=mju_max(mju_min(step_velocity[foot],0.),desired_vz);')
        new=new.replace('  residual_.soft_landing_ = GetNumberOrDefault(0, model, "r10_normal_landing") == 1;',
            '  residual_.soft_landing_ = GetNumberOrDefault(0, model, "r10_normal_landing") == 1;\n  residual_.slow_approach_ = GetNumberOrDefault(0, model, "r11_slow_approach") == 1;')
    if a.build in ('S5-r8','S5-r9','S5-r10','S5-r11','S5-r17'):new=old
    if a.build=='S5-r13':
        new=old.replace('    residual[counter++] = step[foot] ? height_difference : 0;',
            '    const bool moving_stance_height = gait != kGaitStand &&\n'
            '        GetNumberOrDefault(0, model, "r11_moving_stance_height") == 1;\n'
            '    residual[counter++] = (step[foot] || moving_stance_height) ? height_difference : 0;')

    if a.build=='S5-r14':
        new=old.replace('  bool r11_material_support = false;',
            '  const bool r11_moving_body_support = current_mode_ == kModeQuadruped &&\n'
            '      GetGait() != kGaitStand &&\n'
            '      GetNumberOrDefault(0, model, "r11_moving_body_support") == 1;\n'
            '  bool r11_material_support = r11_moving_body_support;')
        new=new.replace('      residual[counter++]=material_support ?',
            '      residual[counter++]=(material_support || r11_moving_body_support) ?')
    if a.build=='S5-r15':
        new=old.replace('      if (material) {\n        double z = p[2]',
            '      const double approach_width = GetNumberOrDefault(0, model, "r11_landing_approach_band");\n'
            '      const bool approaching = p[0] >= .8+half_sole-approach_width &&\n'
            '          p[0] <= 2.2-half_sole+approach_width &&\n'
            '          mju_abs(p[1]) <= .6-half_sole+approach_width;\n'
            '      if (material || approaching) {\n        double z = p[2]')
        new=new.replace('        indentation_residual[foot] = 0.5*(over+mju_sqrt(over*over+1e-12))/0.001;',
            '        indentation_residual[foot] = material ? 0.5*(over+mju_sqrt(over*over+1e-12))/0.001 : 0;')
    if a.build=='S5-r16':
        new=old.replace('GetNumberOrDefault(0, model, "r11_landing_approach_band")',
            'GetNumberOrDefault(0., model, "r11_landing_approach_band")')
    if a.build=='S5-r18':
        target='  if (stand_equilibrium) {\n    mju_scl(residual + counter, data->qvel, 1.0 / 0.05, 3);'
        new=old.replace(target,
            '  const double* moving_twist = GetCustomNumericData(model, "r11_body_twist_reference");\n'+target)
        target='    mju_scl(residual + counter + 3, data->qvel + 3, 1.0 / 0.1, 3);\n  } else {'
        new=new.replace(target,
            '    mju_scl(residual + counter + 3, data->qvel + 3, 1.0 / 0.1, 3);\n'
            '  } else if (moving_twist && current_mode_ == kModeQuadruped && GetGait() != kGaitStand) {\n'
            '    mju_sub(residual + counter, data->qvel, data->userdata + 78, 3);\n'
            '    mju_scl(residual + counter, residual + counter, 1.0/moving_twist[0], 3);\n'
            '    double world_omega[3];\n'
            '    mju_rotVecQuat(world_omega, data->qvel + 3, data->qpos + 3);\n'
            '    mju_sub(residual + counter + 3, world_omega, data->userdata + 81, 3);\n'
            '    mju_scl(residual + counter + 3, residual + counter + 3, 1.0/moving_twist[1], 3);\n'
            '  } else {')
    if new==old and a.build not in ('S5-r8','S5-r9','S5-r10','S5-r11','S5-r17'):raise ValueError('Frozen residual was not found')
    for path in (prior/'mjpc').rglob('*'):
        if path.is_file() and path.suffix in ('.cc','.h'):
            shutil.copy2(path,source/path.relative_to(prior))
    (source/relative).write_text(new)
    header_relative=Path('mjpc/tasks/quadruped/quadruped.h')
    header_old=(prior/header_relative).read_text()
    header_new=header_old
    if a.build=='S5-r7':
        header_new=header_new.replace('bool soft_landing_ = false;',
            'bool soft_landing_ = false;\n    bool slow_approach_ = false;')
        header_new=header_new.replace('void FootStep(double step[kNumFoot], double time, A1Gait gait) const;',
            'void FootStep(double step[kNumFoot], double time, A1Gait gait, double* velocity = nullptr) const;')
        (source/header_relative).write_text(header_new)
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile=prior.name+'/'+str(relative),tofile=a.build+'/'+str(relative)))
    patch+=''.join(difflib.unified_diff(header_old.splitlines(True),header_new.splitlines(True),
        fromfile=prior.name+'/'+str(header_relative),tofile=a.build+'/'+str(header_relative)))
    if a.build in ('S5-r8','S5-r9'):
        derivative_path=Path('mjpc/planners/model_derivatives.cc')
        derivative_old=(prior/derivative_path).read_text()
        target='  evaluate_.push_back(T - 2);'
        if derivative_old.count(target)!=1:raise ValueError('Expected derivative endpoint schedule not found')
        derivative_new=derivative_old.replace(target,
            '  // The last interior node may already be present when skip==0.\n'
            '  // One job owns each output block during held-input A/B composition.\n'
            '  if (evaluate_.back() != T - 2) evaluate_.push_back(T - 2);')
        (source/derivative_path).write_text(derivative_new)
        patch+=''.join(difflib.unified_diff(derivative_old.splitlines(True),derivative_new.splitlines(True),
            fromfile=prior.name+'/'+str(derivative_path),tofile=a.build+'/'+str(derivative_path)))
    if a.build in ('S5-r10','S5-r11'):
        agent_path=Path('mjpc/agent.cc');agent_old=(prior/agent_path).read_text()
        target='  // plan\n  if (!allocate_enabled) {'
        replacement='  // R11: numerical predictor state is independent of worker solver history.\n' + \
            '  if (GetNumberOrDefault(0, model_, "r11_prediction_coldstart") == 1) {\n' + \
            '    model_->opt.disableflags |= mjDSBL_WARMSTART;\n  }\n\n' + target
        if agent_old.count(target)!=1:raise ValueError('Planning entry was not uniquely located')
        agent_new=agent_old.replace(target,replacement)
        (source/agent_path).write_text(agent_new)
        patch+=''.join(difflib.unified_diff(agent_old.splitlines(True),agent_new.splitlines(True),
            fromfile=prior.name+'/'+str(agent_path),tofile=a.build+'/'+str(agent_path)))
    if a.build=='S5-r17':
        agent_path=Path('mjpc/agent.cc');agent_old=(prior/agent_path).read_text()
        target='  // R11: numerical predictor state is independent of worker solver history.'
        addition='  // S5-r17: retain original native contact impedance in C predictions.\n' + \
            '  if (differentiable && GetNumberOrDefault(0, model_, "r11_native_contact_impedance") == 1) {\n' + \
            '    for (int i = 0; i < model_->ngeom; ++i)\n' + \
            '      model_->geom_solimp[mjNIMP*i] = geom_solimp_[i];\n' + \
            '    for (int i = 0; i < model_->npair; ++i)\n' + \
            '      model_->pair_solimp[mjNIMP*i] = pair_solimp_[i];\n  }\n\n'
        agent_new=agent_old.replace(target,addition+target)
        (source/agent_path).write_text(agent_new)
        patch+=''.join(difflib.unified_diff(agent_old.splitlines(True),agent_new.splitlines(True),
            fromfile=prior.name+'/'+str(agent_path),tofile=a.build+'/'+str(agent_path)))
    (ROOT/('experiments/terrain_memory/s1-go2-r11-'+a.build+'.patch')).write_text(patch)
    begin=time.time()
    subprocess.run(['cmake','--build',str(source/'build-autodl'),'--target','agent_server','-j','8'],check=True)
    frozen.mkdir(parents=True)
    shutil.copytree(source/'mjpc',frozen/'mjpc')
    shutil.copytree(source/'build-autodl/mjpc/tasks',frozen/'runtime-tasks')
    shutil.copy2(source/'SOURCE_COMMIT',frozen/'SOURCE_COMMIT')
    shutil.copy2(source/'build-autodl/bin/agent_server',frozen/'agent_server')
    (frozen/'lib').mkdir();shutil.copy2(source/'build-autodl/lib/libmujoco.so.3.2.3',frozen/'lib/libmujoco.so.3.2.3')
    prior_config=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/dev11-GNC-r3-cap0215-resolved.json').read_text())['r10_configuration']
    normal=prior_config['normal_landing']
    stiffness=prior_config['local_contact_model']['instantaneous_stiffness_N_m'];indent=15.206408*9.81/(2*stiffness)
    normal.update(indentation_target_m=indent,indentation_scale_m=indent,
        reference='Same actual C stiffness, compiled Go2 torso-subtree mass and gravity; two-foot mg/2 loading',
        mechanism='Prior optimized RL plans penetrated1.42-1.59mm despite1mm material limit; old residual used375000N/m versus actual1227272.727N/m and a1mm scale. Correct common normal support reference/normalization inside MPC, retain original quadratic weight. No truth or task change.')
    overrides=dict(normal_landing=normal)
    if a.build=='S5-r3':
        overrides=dict(r11_material_upright=dict(
            residual='Body world z-axis minus world up: [R02,R12,R22-1]',
            norm='Existing Upright smooth norm6, parameter.05; applied weight1 unchanged',
            activation='At least one candidate footprint inside original material strip; absent for rigid regression',
            baseline='Original dev11 GNC-r3 body reference and landing objective, no S5-r1 or S5-r2 stacking',
            evidence='S5-r1 ends at pitch.61447rad with trunk contact. Original moving Upright uses R22-1, zeroing two available residual channels, and has vanishing first-order tilt sensitivity near upright. Yaw residual is heading-only. S5-r2 separately loses height, not the same mechanism.'))
    if a.build=='S5-r6':
        overrides=json.loads((ROOT/'work/r11-frozen/S5-r5/identity.json').read_text())['configuration_overrides']
        overrides['r11_domain_margin']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r6.json').read_text())['r11_domain_margin']
    if a.build=='S5-r7':
        overrides=json.loads((ROOT/'work/r11-frozen/S5-r6/identity.json').read_text())['configuration_overrides']
        overrides['r11_slow_approach']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r7.json').read_text())['r11_slow_approach']
        landing=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/dev11-GNC-r3-cap0215-resolved.json').read_text())['r10_configuration']['normal_landing']
        landing['height_profile']='S5-r7 three C2 quintic segments; final5mm over.20s, shared with analytic normal velocity reference'
        landing['velocity_target']='During ascent/stance0; during descent max(min(phase_height_derivative,0),-.05*tanh(max(bottom_z,0)/.001))'
        overrides['normal_landing']=landing
    if a.build=='S5-r8':
        overrides=json.loads((ROOT/'work/r11-frozen/S5-r7/identity.json').read_text())['configuration_overrides']
        overrides['r11_derivative_schedule']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r8.json').read_text())['r11_derivative_schedule']
    if a.build=='S5-r9':
        overrides=dict(r11_derivative_schedule=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r9.json').read_text())['r11_derivative_schedule'])
    if a.build=='S5-r10':
        overrides=json.loads((ROOT/'work/r11-frozen/S5-r9/identity.json').read_text())['configuration_overrides']
        overrides['r11_prediction_coldstart']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r10.json').read_text())['r11_prediction_coldstart']
    if a.build=='S5-r11':
        overrides=json.loads((ROOT/'work/r11-frozen/S5-r8/identity.json').read_text())['configuration_overrides']
        overrides['r11_prediction_coldstart']=json.loads((ROOT/'work/r11-frozen/S5-r10/identity.json').read_text())['configuration_overrides']['r11_prediction_coldstart']
        overrides['r11_combined_repair']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r11.json').read_text())
    if a.build=='S5-r13':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_moving_stance_height']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r13.json').read_text())['r11_moving_stance_height']
    if a.build=='S5-r14':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_moving_body_support']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r14.json').read_text())['r11_moving_body_support']
    if a.build=='S5-r15':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_landing_approach_band']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r15.json').read_text())['r11_landing_approach_band']
    if a.build=='S5-r16':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_landing_band_scalar_fix']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r16.json').read_text())['r11_landing_band_scalar_fix']
    if a.build=='S5-r17':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_native_contact_impedance']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r17.json').read_text())['r11_native_contact_impedance']
    if a.build=='S5-r18':
        overrides=json.loads((prior/'identity.json').read_text())['configuration_overrides']
        overrides['r11_body_twist_reference']=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r18.json').read_text())['r11_body_twist_reference']
    identity=dict(controller='S5',build=a.build,based_on=prior.name,
        prior_binary_sha256=sha(prior/'agent_server'),binary_sha256=sha(frozen/'agent_server'),
        source_files_sha256={str(x.relative_to(frozen)):sha(x) for x in (frozen/'mjpc').rglob('*') if x.is_file() and x.suffix in ('.cc','.h')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration_overrides=overrides,
        build_started_unix_s=begin,build_wall_s=time.time()-begin)
    (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({k:v for k,v in identity.items() if k!='source_files_sha256'}),flush=True)
if __name__=='__main__':main()
