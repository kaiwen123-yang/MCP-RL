"""Load a frozen W61 event actor inside the native R12 qualification/revisit physics."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"w61"/"learning"))
from collect import EventSession
from orbits import FController as HybridOrbits
from s1_w59_learner_critic import atomic_json
from tm02.s1_material import ReferenceColumns


class FController(HybridOrbits):
    def __init__(self,model,data,feet,base,out,config):
        super().__init__(model,data,feet,base,out,config)
        item=config["w59"]["w61_final"]
        role=SimpleNamespace(method=item["method"],seed=item["learning_seed"],case=item["case"],role="evaluate",
            version=item["version"],weights=Path(item["checkpoint"]),variant="",episode=0,offset=0,quota=100000)
        self.session=EventSession(role,Path(out))
        self.final_trial=item
        self.installed=False
        self.completion_s=None

    def update(self,t,data,mission,contact,grf,soft,material_info=None,replan=False):
        if not self.installed:
            joint=self.w59_joint
            joint.w59_learner=self
            scenario=self.final_trial["condition"]
            if scenario is not None:
                joint.ground.law=ReferenceColumns(scenario["parameters"])
                original=joint.before_physics
                def loaded_step(dt,phase):
                    ok=original(dt,phase)
                    now=float(data.time)
                    if ok and scenario["push_time_s"]<=now<scenario["push_time_s"]+scenario["push_duration_s"]:
                        data.xfrc_applied[self.base,1]+=scenario["lateral_force_N"]
                    return ok
                joint.before_physics=loaded_step
            self.installed=True
        return super().update(t,data,mission,contact,grf,soft,material_info,replan)

    def complete(self,t,mission,grf):
        self.completion_s=float(t)
        if self.session.pending:
            obs=self.session.state(self,t,mission,self.current_mq,grf)
            self.session.settle(self.w59_joint.snapshot(t,mission),obs,"success")

    def finish(self,*args):
        # Native R12 finishes the controller recorder first, then settles the material and actor.
        if len(args)==2:
            return super().finish(*args)
        result=args[0]
        joint=self.w59_joint
        if self.session.pending is not None:
            reason=result["reason"]
            t=float(joint.data.time)
            failed=reason.startswith(("material_domain_exit","body_fall","invalid_contact"))
            if result.get("error"):
                self.session.cut+=1
            else:
                grf=joint.measurements()[3]
                obs=self.session.state(self,t,joint.mission,self.current_mq,grf)
                self.session.settle(joint.snapshot(t,joint.mission),obs,"physical_failure" if failed else "truncated")
        rec=dict(actor_sha=self.session.actor_sha,frozen_version=self.final_trial["version"],
            observed=len(self.session.rows),requested=self.session.requests,used=0,excluded_open=self.session.cut,
            first_completion_s=self.completion_s,elapsed_holding_after_completion_s=(float(joint.data.time)-self.completion_s) if self.completion_s is not None else 0.,
            hold_mode_events=sum(r["mode"]==1 for r in self.session.rows),original_task_credit=sum(r["reward"] for r in self.session.rows),
            event_durations_s=[r["dt_s"] for r in self.session.rows])
        result["w61_frozen_actor"]=rec
        atomic_json(Path(self.out)/"w61-frozen-actor.json",rec)
