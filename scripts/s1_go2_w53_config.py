"""W53 controller-specific configuration over the unchanged original task."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
from s1_go2_r11_protocol import ROOT,stage
RUN='tm-s1-go2-local-readiness-r12'
def configuration(build,level,cap=.215,terrain='tm02_anchor',training_duration=None,controller_artifact=None):
    if build not in ('S5-r42','S5-r43'):raise ValueError(build)
    route={'S5-r42':'N','S5-r43':'F'}[build]
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-wb-support.json').read_text())
    artifact=controller_artifact or build
    identity=json.loads((ROOT/'work/r11-frozen'/artifact/'identity.json').read_text())
    c.update(run_id=RUN,stage='S1R12',controller_build=build,controller_artifact=artifact,
        implementation_status='W53_original_evaluation',w53_route=route,
        actual_terrain=terrain,fixed_cap_m_s=cap,r11_stage=stage(level,training_duration))
    c['outputs']['root']='results/terrain_memory/simulation/'+RUN
    c['local_contact_model']={'enabled':False,'causal_force_anchor':False}
    if route=='N':
        c.update(copy.deepcopy(identity['configuration_overrides']))
        names=['s1_go2_w53_n_model.py','s1_go2_w53_n_records.py']
    else:
        c['w53_f']=json.loads((ROOT/'experiments/terrain_memory/s1-w53-f.json').read_text())
        c['w53_backend_source']='scripts/s1_w53_f_backend.py'
        c['task']['move_and_turn_settings'].update(Cadence=2.,**{'Duty ratio':.60,'Amplitude':.03})
        c['controller_formulation']='independent_force_MPC_WBC'
        names=['s1_w53_f_backend.py']
    names+=['s1_go2_w53_config.py','run_s1_go2_r12.py']
    c['w53_execution_source_names']=names
    c['w53_execution_sources']=['scripts/'+n for n in names]
    c['task']['command_speed_m_s']=cap
    c['task']['maximum_episode_s']=c['r11_stage']['duration_s']
    if c['r11_stage']['task_kind'] in ('continuous_finite','continuing_acquisition'):
        c['long_mission_protocol'].update(duration_s=c['r11_stage']['duration_s'],
            full_round_trips=c['r11_stage']['target_circuits'],
            route_completion_deadline_s=c['r11_stage']['route_deadline_s'])
    return c,json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
def arguments(attempt,build,condition,purpose,kind,workers=10,controller_artifact=None):
    frozen=ROOT/'work/r11-frozen'/(controller_artifact or build)
    source=frozen if build=='S5-r42' else ROOT/'work/r11-frozen/S5-r41'
    assets=ROOT/'work/assets'
    return SimpleNamespace(config='experiments/terrain_memory/s1-go2-local-readiness-r12.json',
        stage='G1',purpose=purpose,condition=condition,arm='S' if build=='S5-r42' else 'F',
        mission='long' if kind=='continuous_finite' else 'single',cohort='W53_original_frozen',
        attempt=attempt,workers=workers,author_source=str(source),
        agent_server=str(frozen/'agent_server') if build=='S5-r42' else None,
        original_robot_dir=str(assets/'go2'),original_scene=str(assets/'scene_flat.xml'),
        original_resolved=str(assets/'rigid-T1R-resolved.json'))
