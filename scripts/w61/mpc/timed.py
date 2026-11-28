"""Anytime topology search, with one measured full-controller budget for every search arm."""
import json
import time
from pathlib import Path
import numpy as np
from .controller import TopologyMPC
from .problem import build_problem
from .solver import search


class TimedTopologyMPC(TopologyMPC):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.round_wall=[]
        self.control_log=[]
        # Prepare each common kernel signature from one past development publication;
        # this is an installation cost, included separately from resident control latency.
        fixture=Path(self.c_cfg['shape_source'])
        with np.load(fixture) as f:
            obs=dict(f)
        obs.update(torque_lower=self.model.actuator_forcerange[:,0],torque_upper=self.model.actuator_forcerange[:,1])
        p=build_problem(obs,.18,np.ones(4),self.c_cfg['belief'],0.)
        t=time.perf_counter()
        search(p,self.c_method,0,np.zeros((2,16)),candidates=self.c_cfg['samples'],iterations=1,mode_ids=(0,1))
        self.install_s=time.perf_counter()-t
        self.rate=self.c_cfg['initial_round_ms']/1000

    def update(self,*args,**kw):
        start=time.perf_counter()
        self.begin_cycle=start
        previous=self.c_count
        answer=super().update(*args,**kw)
        ms=(time.perf_counter()-start)*1000
        self.control_log.append((float(args[0]),ms,self.c_count!=previous))
        return answer

    def search_transfer(self,problem):
        start=time.perf_counter()
        run=0
        state=None
        warm=self.c_warm
        total=0
        costs=[]
        reserve=self.c_cfg['common_qp_and_wbc_reserve_ms']/1000
        budget=self.c_cfg['resident_end_to_end_budget_ms']/1000
        while (run==0 or (time.perf_counter()-self.begin_cycle+reserve+self.rate<budget and run<self.c_cfg['finite_round_ceiling'])):
            before=time.perf_counter()
            result=search(problem,self.c_method,self.c_cfg['seed']+self.c_count*36+run,warm,
                          candidates=self.c_cfg['samples'],iterations=1,mode_ids=(0,1),proposal_state=state)
            state=result.get('proposal_state')
            elapsed=time.perf_counter()-before
            self.round_wall.append(elapsed)
            self.rate=float(np.quantile(self.round_wall[-24:],.90))
            warm=result['per_mode']
            total+=result['evaluated_candidates']
            costs.extend(result['transfer_rollout_refinement_sync_s'])
            run+=1
        result.update(evaluated_candidates=total,transfer_rollout_refinement_sync_s=costs,search_wall_s=time.perf_counter()-start)
        return result

    def at_rollout_end(self,**kw):
        super().at_rollout_end(**kw)
        data=np.array(self.control_log)
        targets=data[:,2].astype(bool)
        budget=self.c_cfg['resident_end_to_end_budget_ms']
        (self.c_dir/'timed-controller.json').write_text(json.dumps(dict(installation_s=self.install_s,
            decision_count=int(targets.sum()),all_controller_ticks=len(data),budget_ms=budget,
            decision_controller_p50_p95_p99_ms=np.percentile(data[targets,1],[50,95,99]).tolist() if targets.any() else [],
            decision_controller_deadline_misses=int((data[targets,1]>budget).sum()),
            all_tick_controller_p50_p95_p99_ms=np.percentile(data[:,1],[50,95,99]).tolist(),
            round_wall_ms=(1000*np.asarray(self.round_wall)).tolist(),round_wall_units='milliseconds',
            scope='includes observation/model, transfers, synchronized GPU predictions, refinement and same-tick force QP/WBC; outside C simulator stepping and transport to hardware are measured by base runner/untested respectively'),indent=2)+'\n')
        np.savez_compressed(self.c_dir/'timed-ticks.npz',time_s=data[:,0],controller_ms=data[:,1],decision=targets)
