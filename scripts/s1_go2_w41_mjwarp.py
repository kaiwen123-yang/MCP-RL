"""One selected alternative: isolated MJWarp4 FP32 contact predictor."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
def run_selected_alternative(report,publish):
    # A G4 summary is not completion of the G5 invocation now starting.
    report.pop('completion',None)
    invocation=report.get('continuation',{}).get('invocation')
    stage_dir=OUT/'continuations'/invocation if invocation else OUT
    stage_dir.mkdir(parents=True,exist_ok=True)
    report['stages']['G5']=dict(status='selected',backend='mujoco-warp4.0.0 / warp1.19.0',
        rationale='Existing fixed MJX contact mismatch and measured forward/derivative throughput; compare one forward backend.',
        identity='New FP32 MuJoCo4/MJWarp predictor, not equivalent CUDA migration',
        retained='Original CPU Go2/TM02 plant, native r30 task cost/private phase/feedback and physical qualification',
        numerical='Native predictor tolerances/iterations retained explicitly. FP32 FD sensitivity checked at three declared scales before independent closed loop.')
    publish('G5_mjwarp_new_predictor_running')
    contracts=report['stages'].get('G3',{}).get('evaluation_contracts',{})
    source_contract=contracts.get('native') or contracts.get('mjx') or str((OUT/'native-evaluation-contract.json').relative_to(ROOT))
    env=dict(os.environ,PYTHONPATH='',S1_W41_GPU_STAGE_DIR=str(stage_dir),
             S1_W41_GPU_SOURCE_CONTRACT=str(ROOT/source_contract))
    if invocation:env['S1_W41_GPU_INVOCATION']=invocation
    command=[str(ROOT/'work/envs/w41-mjwarp/bin/python'),'-u','scripts/s1_go2_w41_warp_backend.py']
    with (stage_dir/'mjwarp.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=stage_dir/'mjwarp-result.json'
    report['stages']['G5'].update(exit_code=result.returncode,result_record=str(path.relative_to(ROOT)),
        result=json.loads(path.read_text()) if path.exists() else {'status':'technical_interruption_before_result'})
    report['stages']['G5']['status']=report['stages']['G5']['result']['status']
    publish('G5_mjwarp_process_ended')
