import json,sys
from pathlib import Path
import numpy as np
from w61.joint.plant import CONTRACT,rollout
from w61.joint.heads import NA,NC
if __name__=="__main__":
    spec=json.loads(CONTRACT.read_text())
    result=rollout(Path(sys.argv[1]),np.zeros(NA+NC),spec["development"],610981)
    print(json.dumps({k:result[k] for k in ("reason","physical_s","query_steps","wall_s","support_samples","calibration_rmse_N","decisions","progress_m","joint_cost")}),flush=True)
