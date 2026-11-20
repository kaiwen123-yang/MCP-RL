"""W55 N r54 runtime, causal frozen material/support packet, and native E ABI."""
import ctypes
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from s1_go2_w54_n_model_r48 import build_project_model, write_flat_ground_height, build_prediction_model
from s1_go2_w54_n_model_r48 import write_equilibrium_settings as original_settings
from s1_go2_w54_n_records_r48 import W54NRecords as PreviousRecords
ROOT=Path(__file__).resolve().parents[1]
SURFACE=79
SIZE=SURFACE+70*60
FIELDS=['gap_m','vertical_velocity_m_s','achieved_acceleration_m_s2','onset_distance_m','braking_gate',
        'target_acceleration_m_s2','achieved_acceleration_m_s2_check','material_interior','feasible_baseline_acceleration_m_s2',
        'requested_increment_m_s2','bearing','observed_or_estimated_normal_N','achieved_end_velocity_m_s','unattained_acceleration_m_s2',
        'baseline_base_vertical_acceleration','projected_base_vertical_acceleration','baseline_support_N','projected_support_N',
        'max_leg_torque_change_Nm','reference_epoch_s','number_hard_supports','braking_shortfall','normal_tangent_N_m','projection_error']

def write_equilibrium_settings(xml_path, config, arm):
    result=original_settings(xml_path,config,arm)
    tree=ET.parse(xml_path);tree.getroot().find('size').set('nuserdata',str(SIZE))
    tree.write(xml_path,encoding='unicode')
    result['xml_sha256']=hashlib.sha256(Path(xml_path).read_bytes()).hexdigest()
    result['support_reference']='62 original stand values; epoch/observed GRF/world-cell stress tangents in fixed userdata'
    return result

class PrecontactE:
    def __init__(self,model):
        self.model=model
        self.lib=ctypes.CDLL(str(ROOT/'work/r11-frozen/S5-r54/lib/libw55_support_python.so'))
        self.fn=self.lib.w55_project
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,array,array,array]
        self.fn.restype=ctypes.c_int

    def capture(self,data,contact,grf,observer):
        # Same epoch snapshot feeds actual E, all native workers and every
        # FD direction. Each follows its own q/v against this frozen field.
        data.userdata[62]=data.time
        measurements=data.userdata[63:79].reshape(4,4)
        measurements[:,0]=contact & (grf[:,2]>0.)
        measurements[:,1:]=grf
        grid=data.userdata[SURFACE:].reshape(70,60)
        g=observer.ground
        # z_rest=z_current+force/k, with the identical cell area on both
        # terms. Use the independent 2ms PANN estimate, never plant e/q.
        if g.ids:
            keys=np.asarray(list(g.ids),dtype=int)
            stiffness_MPa=1227272.7272727273*g.h/(g.gamma*g.area*1e6)
            grid[keys[:,0]-40,keys[:,1]+30]=g.h*(g.e[:,0]-g.stress[:,0]/stiffness_MPa)

    def apply(self,data,raw):
        raw=np.ascontiguousarray(raw,dtype=np.float64);applied=np.empty(12);diagnostic=np.empty((4,24))
        status=self.fn(self.model._address,data._address,raw,applied,diagnostic)
        if status:raise RuntimeError(f'controller_infeasible: execution support projection status {status} at t={data.time:.9f}')
        return applied,diagnostic

class W54NRecords(PreviousRecords):
    def __init__(self,*args):
        super().__init__(*args);self.controller='S5-r54'
    def execution(self,t,raw,applied,diagnostic):
        self._write(self.execution_log,dict(time_s=t,raw_action=raw,applied_action=applied,
            diagnostic=diagnostic,fields=FIELDS,position_hold_s=.002,reference_epoch_s=float(diagnostic[0,19])))
    def progress(self,t,qpos,mission_v,contact):
        import json
        if qpos[2]<.25:self.event(t,'first_base_below_0.25m',first=True,base_z_m=float(qpos[2]))
        if self.control_samples%250==0:
            (self.out/'progress.json').write_text(json.dumps(dict(time_s=t,call_id=self.call_id,
                policy_call_id=self.policy_call_id,qpos=qpos.tolist(),command_velocity=mission_v.tolist(),
                contact=contact.tolist(),controller=self.controller,deadline=None,wall_deadline_enabled=False))+'\n')
    def before_call(self,t,data,mission,joint):
        grid=data.userdata[SURFACE:].reshape(70,60);ix,iy=np.nonzero(grid)
        self._write(self.context,dict(time_s=t,call_id=self.call_id+1,qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            mocap_pos=data.mocap_pos.copy(),mocap_quat=data.mocap_quat.copy(),userdata_prefix=data.userdata[:SURFACE].copy(),
            world_surface_m=[[int(i+40),int(j-30),float(grid[i,j])] for i,j in zip(ix,iy)],
            mission_phase=mission.phase,command_velocity=mission.v.copy(),command_omega=mission.w.copy(),
            cap=float(joint.cap)))
    def finish(self,t,reason):
        import json
        result=super().finish(t,reason)
        result['controller']=self.controller;result['execution_diagnostic_fields']=FIELDS
        result['projection_failure_codes']=['none','singular constrained dynamics','no feasible PD/hold/friction baseline']
        result['candidate_failure_codes']=['none','attitude or base termination margin','illegal trunk/ground contact','joint limit',
            'actuator constraint','material principal normal domain','nonfoot contact before next reachable support',
            'vertical arrest acceleration infeasible','support projection failed']
        (self.out/'w54-n-recording.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
