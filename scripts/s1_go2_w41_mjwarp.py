"""One selected alternative: isolated MJWarp4 FP32 contact predictor."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
def run_selected_alternative(report,publish):
    report['stages']['G5']=dict(status='selected',backend='mujoco-warp4.0.0 / warp1.19.0',
        rationale='Existing fixed MJX contact mismatch and measured forward/derivative throughput; compare one forward backend.',
        identity='New FP32 MuJoCo4/MJWarp predictor, not equivalent CUDA migration',
        retained='Original CPU Go2/TM02 plant, native r30 task cost/private phase/feedback and physical qualification',
        numerical='Native predictor tolerances/iterations retained explicitly. FP32 FD sensitivity checked at three declared scales before independent closed loop.')
    publish('G5_mjwarp_new_predictor_running')
    env=dict(os.environ,PYTHONPATH='')
    command=[str(ROOT/'work/envs/w41-mjwarp/bin/python'),'-u','scripts/s1_go2_w41_warp_backend.py']
    with (OUT/'mjwarp.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=OUT/'mjwarp-result.json'
    report['stages']['G5'].update(exit_code=result.returncode,result=json.loads(path.read_text()) if path.exists() else {'status':'technical_interruption_before_result'})
    report['stages']['G5']['status']=report['stages']['G5']['result']['status']
    publish('G5_mjwarp_completed')
