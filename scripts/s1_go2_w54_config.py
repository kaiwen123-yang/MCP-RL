"""W54 original-task configuration; independent compact ledger and frozen workers."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
from s1_go2_r11_protocol import ROOT,stage
RUN='tm-s1-go2-local-readiness-r12'
def configuration(build,level,cap=.215,terrain='tm02_anchor',training_duration=None,controller_artifact=None):
    artifact=controller_artifact or build
    identity=json.loads((ROOT/'work/r11-frozen'/artifact/'identity.json').read_text())
    route=identity['route']
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-wb-support.json').read_text())
    c.update(run_id=RUN,stage='S1R12',controller_build=build,controller_artifact=artifact,
        implementation_status='W54_original_evaluation',w54_route=route,
        actual_terrain=terrain,fixed_cap_m_s=cap,r11_stage=stage(level,training_duration))
    c['outputs']['root']='results/terrain_memory/simulation/'+RUN+'/w54'
    c['local_contact_model']={'enabled':False,'causal_force_anchor':False}
    if route=='N':
        c.update(copy.deepcopy(identity['configuration_overrides']))
        names=identity['execution_source_names']
    else:
        config_path=identity['configuration_source']
        c['w54_f']=json.loads((ROOT/config_path).read_text())
        c['w54_backend_source']=identity['backend_source']
        c['task']['move_and_turn_settings'].update(Cadence=2.,**{'Duty ratio':.60,'Amplitude':.03})
        c['controller_formulation']='independent_force_MPC_WBC'
        names=[str(Path(identity['backend_source']).relative_to('scripts'))]
    names=list(names)+['s1_go2_w54_config.py','run_s1_go2_r12.py']
    c['w54_execution_source_names']=names
    c['w54_execution_sources']=['scripts/'+n for n in names]
    c['task']['command_speed_m_s']=cap
    c['task']['maximum_episode_s']=c['r11_stage']['duration_s']
    if c['r11_stage']['task_kind'] in ('continuous_finite','continuing_acquisition'):
        c['long_mission_protocol'].update(duration_s=c['r11_stage']['duration_s'],
            full_round_trips=c['r11_stage']['target_circuits'],
            route_completion_deadline_s=c['r11_stage']['route_deadline_s'])
    return c,json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
def arguments(attempt,build,condition,purpose,kind,workers=10,controller_artifact=None):
    frozen=ROOT/'work/r11-frozen'/(controller_artifact or build)
    identity=json.loads((frozen/'identity.json').read_text())
    source=frozen if identity['route']=='N' else ROOT/identity['physical_model_source']
    assets=ROOT/'work/assets'
    return SimpleNamespace(config='experiments/terrain_memory/s1-go2-local-readiness-r12.json',
        stage='G1',purpose=purpose,condition=condition,arm='S' if identity['route']=='N' else 'F',
        mission='long' if kind=='continuous_finite' else 'single',cohort='W54_original_frozen',
        attempt=attempt,workers=workers,author_source=str(source),
        agent_server=str(frozen/'agent_server') if identity['route']=='N' else None,
        original_robot_dir=str(assets/'go2'),original_scene=str(assets/'scene_flat.xml'),
        original_resolved=str(assets/'rigid-T1R-resolved.json'))

