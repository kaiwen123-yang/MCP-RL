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
    frozen=ROOT/'work/r11-frozen'/a.build
    if frozen.exists():raise ValueError('This controller build already exists; preserve its identity')
    if a.build in ('S5-r2','S5-r4','S5-r5'):
        specification=json.loads((ROOT/f'experiments/terrain_memory/s1-go2-r11-{a.build}.json').read_text())
        if a.build=='S5-r5':prior=ROOT/'work/r11-frozen/S5-r3'
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
    if a.build not in ('S5-r1','S5-r3','S5-r6'):raise ValueError('An undeclared S5 controller build was requested')
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
    if new==old:raise ValueError('Frozen residual was not found')
    for path in (prior/'mjpc').rglob('*'):
        if path.is_file() and path.suffix in ('.cc','.h'):
            shutil.copy2(path,source/path.relative_to(prior))
    (source/relative).write_text(new)
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile=prior.name+'/'+str(relative),tofile=a.build+'/'+str(relative)))
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
    identity=dict(controller='S5',build=a.build,based_on='S5-r5' if a.build=='S5-r6' else 'R10-GNC-r3',
        prior_binary_sha256=sha(prior/'agent_server'),binary_sha256=sha(frozen/'agent_server'),
        source_files_sha256={str(x.relative_to(frozen)):sha(x) for x in (frozen/'mjpc').rglob('*') if x.is_file() and x.suffix in ('.cc','.h')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration_overrides=overrides,
        build_started_unix_s=begin,build_wall_s=time.time()-begin)
    (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({k:v for k,v in identity.items() if k!='source_files_sha256'}),flush=True)
if __name__=='__main__':main()
