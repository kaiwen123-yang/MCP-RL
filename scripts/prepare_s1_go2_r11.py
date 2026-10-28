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
    frozen=ROOT/'work/r11-frozen'/a.build
    if frozen.exists():raise ValueError('This controller build already exists; preserve its identity')
    if a.build=='S5-r2':
        specification=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r2.json').read_text())
        frozen.mkdir(parents=True)
        for name in ('mjpc','runtime-tasks','lib'):(frozen/name).symlink_to(prior/name)
        for name in ('agent_server','SOURCE_COMMIT'):shutil.copy2(prior/name,frozen/name)
        identity=json.loads((prior/'identity.json').read_text())
        identity.update(controller='S5',build=a.build,based_on='R10-GNC-r3',
            configuration_overrides=specification['configuration_overrides'],
            reused_frozen_binary=True,physics_unchanged=True)
        (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print('Prepared S5-r2 reference controller with unchanged frozen C binary.',flush=True)
        return
    if a.build!='S5-r1':raise ValueError('An undeclared S5 controller build was requested')
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
    if new==old:raise ValueError('Frozen GNC-r3 landing residual was not found')
    for path in (prior/'mjpc').rglob('*'):
        if path.is_file() and path.suffix in ('.cc','.h'):
            shutil.copy2(path,source/path.relative_to(prior))
    (source/relative).write_text(new)
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='R10-GNC-r3/'+str(relative),tofile=a.build+'/'+str(relative)))
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
    identity=dict(controller='S5',build=a.build,based_on='R10-GNC-r3',
        prior_binary_sha256=sha(prior/'agent_server'),binary_sha256=sha(frozen/'agent_server'),
        source_files_sha256={str(x.relative_to(frozen)):sha(x) for x in (frozen/'mjpc').rglob('*') if x.is_file() and x.suffix in ('.cc','.h')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration_overrides=dict(normal_landing=normal),
        build_started_unix_s=begin,build_wall_s=time.time()-begin)
    (frozen/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({k:v for k,v in identity.items() if k!='source_files_sha256'}),flush=True)
if __name__=='__main__':main()
