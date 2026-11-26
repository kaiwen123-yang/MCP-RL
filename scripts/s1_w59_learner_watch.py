"""Adopt W59 physical/model receipts; trigger only their earned matched or standalone learning series."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
W=ROOT/"work/w59"
REL=Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59")
OUT=ROOT/REL
ACTION=OUT/"action"
LEARN=OUT/"learning"
REG2=ACTION/"registry-v2.json"
REG3=ACTION/"registry-v3.json"
DEV=ACTION/"development-scored.json"
SEALED=ACTION/"sealed-scored.json"
FREEZE=ACTION/"model-recipe-freeze.json"
STAGE=ACTION/"stage.json"
STATES=W/"learner/first-states"
CRITIC=W/"learner/historical-task-critic.pt"
REAL=LEARN/"real-development-task-credit.json"
GATE=LEARN/"final-action-gate.json"
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)

def read(p):
    return json.loads(p.read_text()) if p.is_file() else {}

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");tmp.replace(p)

def mark(state,**details):
    path=LEARN/"dispatch.json"
    existing=read(path)
    if existing.get("state")==state and existing.get("details")==details:return
    stamp=dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    write(path,dict(state=state,details=details,pid=os.getpid(),time_shanghai=stamp,**BG))
    print(stamp,state,json.dumps(details,ensure_ascii=False),flush=True)

def use(rt,script,*args):
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(rt/"src"),str(rt/"scripts"),os.environ.get("PYTHONPATH","")]),
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    subprocess.run([sys.executable,"-u",str(rt/"scripts"/script),*map(str,args)],cwd=rt,env=env,check=True)

def physical(rt):
    if read(REAL).get("fully_paired_directions")==72:return
    if not REG2.exists():return
    reg=read(REG2);rows=reg["states"]["development"]
    names=[f'w59-action-development-{r["case"].lower()}-r{r["request_id"]:05d}-{s}-a01' for r in rows for s in reg["signed"]]
    if len(names)==96 and all((OUT/"episodes"/(k+".json")).is_file() for k in names):
        mark("scoring_real_development",registered=96)
        use(rt,"s1_w59_learner_admission.py","--role","development","--physical-only","--registry",REG2,"--output",REAL)

def horizons(rt):
    ans=ACTION/"r3-development-horizons.json"
    base=read(DEV)
    if len(base.get("checkpoints",[]))!=3 or base.get("summary",{}).get("measured_branches")!=96:return
    hashes=[c["sha256"] for c in base["checkpoints"]]
    if read(ans).get("wm_sha256")==hashes:return
    comp={}
    for n in (1,5,17,32):
        path=LEARN/f"development-r3-{n:02d}-substeps.json"
        if [c for c in read(path).get("wm_sha256",[])]!=hashes:
            mark("scoring_development_horizon",substeps=n,wm_sha256=hashes)
            use(rt,"s1_w59_learner_admission.py","--role","development","--planning",DEV,"--registry",REG2,
                "--critic",CRITIC,"--substeps",n,"--output",path)
        comp[n]=read(path)["summary"]
    chosen=max((k for k,v in comp.items() if v.get("admitted")),default=0)
    from hashlib import sha256
    write(ans,dict(chosen_substeps=chosen,r3_substeps=chosen,wm_sha256=hashes,checks=comp,
         status="development_admitted" if chosen else "development_no_trusted_path",
         critic_sha256=sha256(CRITIC.read_bytes()).hexdigest(),
         learner_rule_sha256=sha256((rt/"experiments/terrain_memory/s1-w59-gradient-admission.json").read_bytes()).hexdigest(),
         online_recipe_sha256=sha256((rt/"experiments/terrain_memory/s1-w59-online-wm.json").read_bytes()).hexdigest(),**BG))

def prepare_gate(rt):
    if GATE.exists():return True
    note=read(STAGE);stage=note.get("state")
    if stage=="sealed_scored" and SEALED.exists() and REG3.exists() and FREEZE.exists():
        mark("scoring_frozen_action_admission",receipt=str(SEALED))
        # Sealed physical receipts plus the exact frozen models/critic are checked inside.
        use(rt,"s1_w59_learner_admission.py","--role","sealed_admission","--registry",REG3,"--planning",SEALED,
            "--material",read(FREEZE).get("material","M1"),"--critic",CRITIC,"--freeze",FREEZE,
            "--first-output",STATES,"--output",GATE)
        return True
    if stage in ("blocked_independent_sources","technical_block","scientifically_blocked","resource_blocked"):
        if read(REAL).get("fully_paired_directions")!=72 or not REG3.exists():
            mark("blocked",reason="Terminal action-stage blocker with incomplete measured action comparison or independent first-update registration",
                 action_stage=stage,evidence=[str(STAGE),str(REAL),str(REG3)])
            return None
        reg=read(REG3)
        if len(reg.get("first_update",{}).get("5901",[]))!=8:
            mark("blocked",reason="The independent real 8-state first-actor update check is unavailable for seed 5901",
                 action_stage=stage,evidence=[str(REG3),str(STAGE)])
            return None
        stop=LEARN/"standalone-ppo-reason.json"
        write(stop,dict(status="blocked",reason=note.get("reason") or stage,evidence=[str(STAGE),str(REG3)]))
        mark("checking_standalone_true_ppo",action_stage=stage)
        use(rt,"s1_w59_learner_fallback.py","--registry",REG3,"--development",REAL,"--blocker",stop,
            "--critic",CRITIC,"--first-dir",STATES,"--output",GATE)
        return True
    mark("waiting_for_frozen_action_evidence",action_stage=stage or "not yet registered",gate=str(GATE))
    return False

def ready_for_series(gate):
    if not gate.get("planning_eligible") and not gate.get("ppo_eligible"):return True
    if not (STATES/"seed-5901.json").is_file():return False
    if gate.get("r3_gradient_eligible"):
        required=(("B3","M1"),("Fixed-action","M1"),("P3-full",gate["material"]),("P3-private",gate["material"]))
    elif gate.get("planning_eligible"):
        required=(("P3-full",gate["material"]),)
    else:
        for c in "ab":
            for level in ("e30","e60"):
                if not any(read(p).get("stage_status")=="passed" for p in (OUT/"episodes").glob(f"w59-b3-{c}-original-{level}-a[0-9][0-9].json")):return False
        return True
    return all((OUT/"mechanism"/f"{arm.lower()}-{mat.lower()}-development.json").is_file() for arm,mat in required)

def main(a):
    W.mkdir(parents=True,exist_ok=True)
    lock=(W/"learner-watch.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    try:
        while True:
            if not a.runtime_pointer.is_file():
                mark("waiting_for_frozen_research_runtime",pointer=str(a.runtime_pointer))
                time.sleep(10);continue
            rt=Path(a.runtime_pointer.read_text().strip()).resolve()
            if rt==ROOT or not (rt/"scripts/s1_w59_learner_series.py").is_file():
                mark("blocked",reason="Selected research runtime lacks the frozen learner series",runtime=str(rt));return
            physical(rt)
            horizons(rt)
            status=prepare_gate(rt)
            if status is None:return
            if not status:
                time.sleep(15);continue
            gate=read(GATE)
            if not ready_for_series(gate):
                mark("waiting_for_own_qualification_or_planning_development",runtime=str(rt),gate=str(GATE))
                time.sleep(20);continue
            mark("dispatching_earned_series",runtime=str(rt),matched=gate.get("r3_gradient_eligible"),model_rank=gate.get("planning_eligible"),true_ppo=gate.get("ppo_eligible"))
            use(rt,"s1_w59_learner_series.py","--runtime",rt,"--admission",GATE,"--first-states",STATES,"--critic",CRITIC)
            result=read(LEARN/"learning-series.json")
            if not result:raise RuntimeError("Series ended without a scientific outcome")
            mark("terminal",reason=result["reason"],outcome=str(LEARN/"learning-series.json"));return
    except Exception as e:
        mark("blocked",reason=repr(e),action_evidence=str(STAGE),recover_from="Retain original jobs and charged event ledgers; fix the verified cause and adopt the same identity.")
        raise
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-pointer",type=Path,default=W/"runtime-research.txt");main(p.parse_args())

