#!/usr/bin/env python3
"""Narrow launch/aggregation/real Warp-option checks; never run a GPU or robot task.

Run in an isolated Python with mujoco=4.0.0,mujoco-warp=4.0.0,warp-lang=1.19.0.
The command regression executes the unchanged R12 parser and validation AST,
ending immediately before evaluate(). No mock physics or simulated success.
"""
import argparse,ast,contextlib,copy,io,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'scripts'))
from s1_go2_gpu_evaluation import evaluation_command,summarize_completion
from s1_go2_job_limits import goal_based

def check_parser(contract):
    contract=contract.resolve()
    source=ast.parse((ROOT/'scripts/run_s1_go2_r12.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr)
             and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
             and n.value.func.id=='evaluate')
    fn.body=fn.body[:end]+[ast.Return(value=ast.Name(id='a',ctx=ast.Load()))]
    namespace=dict(argparse=argparse,json=json,Path=Path,ROOT=ROOT,CONFIG=str(contract),goal_based=goal_based)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'scripts/run_s1_go2_r12.py'),'exec'),namespace)
    command=evaluation_command(ROOT,contract,'parser-only-not-executed',ROOT/'unused-checkpoint.pt','E30',1794276000.0)
    old=command[:-2]
    prior=sys.argv
    old_error=io.StringIO()
    try:
        sys.argv=old[2:]
        try:
            with contextlib.redirect_stderr(old_error):namespace['main']()
        except SystemExit as e:old_exit=e.code
        else:raise AssertionError('Original missing start time unexpectedly accepted')
        sys.argv=command[2:]
        parsed=namespace['main']()
    finally:sys.argv=prior
    assert old_exit==2 and 'needs --attempt and --evaluation-start-unix' in old_error.getvalue()
    assert parsed.window_start_unix==1794276000.0 and parsed.deadline_unix is None
    return dict(original_exit_code=old_exit,original_error=old_error.getvalue().splitlines()[-1],
                fixed_parser_accepted=True,start_unix_s=parsed.window_start_unix,deadline_unix_s=None,
                physical_evaluation_invoked=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).with_name('gpu-fix-validation.json'))
    args=ap.parse_args()
    original=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu/result.json').read_text())
    review=copy.deepcopy(original);status=summarize_completion(review)
    assert status=='completed_with_technical_failures'
    assert review['stages']['G3']['status']=='physical_integration_not_observed'
    assert len(review['completion']['technical_failures'])==3
    scoped=copy.deepcopy(original);scoped['continuation']={'resume_from':'G5'}
    summarize_completion(scoped)
    assert len(scoped['completion']['active_technical_failures'])==1
    native_scoped=copy.deepcopy(original);native_scoped['continuation']={'resume_from':'G4'}
    native_scoped['stages']['G3']['requested_backends']=['native'];summarize_completion(native_scoped)
    assert len(native_scoped['completion']['active_technical_failures'])==2
    import mujoco,mujoco_warp as mjw,warp as wp
    assert mujoco.__version__=='4.0.0' and wp.__version__=='1.19.0'
    with wp.ScopedDevice('cpu'):
        native=mujoco.MjModel.from_xml_string('<mujoco><option tolerance="1e-8"/><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
        model=mjw.put_model(native)
        before=model.opt.tolerance.numpy().copy();identity=id(model.opt.tolerance)
        # Exact production assignment, applied to a real put_model result.
        backend_ast=ast.parse((ROOT/'scripts/s1_go2_w41_warp_backend.py').read_text())
        cls=next(n for n in backend_ast.body if isinstance(n,ast.ClassDef) and n.name=='WarpPredictor')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assignment=next(n for n in init.body if isinstance(n,ast.Expr)
                        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                        and n.value.func.attr=='fill_')
        from types import SimpleNamespace
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),
                     'production-tolerance-assignment','exec'),dict(self=SimpleNamespace(wm=model),m=native))
        after=model.opt.tolerance.numpy().copy()
        assert id(model.opt.tolerance)==identity and model.opt.tolerance.dtype==wp.float32
        assert after.shape==(1,) and after[0]==np.float32(native.opt.tolerance)
    result=dict(status='passed_scoped_non_GPU_checks',cli=check_parser(args.contract),
        scoped_failure_classification=dict(source='Original delivered failure outcomes; only invocation selection varied',
            G5_only=dict(total_technical_failures=len(scoped['completion']['technical_failures']),
                         active_technical_failures=len(scoped['completion']['active_technical_failures'])),
            G4_native_and_G5=dict(total_technical_failures=len(native_scoped['completion']['technical_failures']),
                                 active_technical_failures=len(native_scoped['completion']['active_technical_failures']))),
        original_report_reclassified=dict(original_status=original['status'],corrected_status=status,
            G3_status=review['stages']['G3']['status'],technical_failures=len(review['completion']['technical_failures']),
            physical_evaluations_started=review['completion']['g4_physical_evaluations_started']),
        real_Warp_option=dict(mujoco=mujoco.__version__,warp=wp.__version__,mujoco_warp='4.0.0',
            device='cpu',put_model_default=before.tolist(),declared_native_tolerance=float(native.opt.tolerance),
            after_fill=after.tolist(),array_identity_retained=True,dtype=str(model.opt.tolerance.dtype)),
        not_validated=['CUDA kernel launch','full original Go2/MJWarp step','G4/G5 physical evaluation','GPU throughput'],
        physics_experiments_run=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
