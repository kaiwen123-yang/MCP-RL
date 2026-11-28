"""Construct one fixed W61 gait and WBC treatment on the frozen W59 B3 stack."""
from s1_w59_f_backend import FController as B3
from w61_support_bridge import SupportBridge
from w61_wbc_internal_model import HybridWBC


def FController(model,data,feet,base,out,config):
    assignment=config['w61']
    classes=[]
    if assignment.get('style_cpg_seed') is not None:
        from w61_style_orbit import StyleOscillator
        classes.append(StyleOscillator)
    elif assignment['gait_role'] in ('library','cpg'):
        from w61_controlled_orbits import ControlledOrbit
        classes.append(ControlledOrbit)
    if assignment.get('orbit_decoder_seed') is not None:
        from w61_orbit_decoder import LearnedOrbit
        classes.append(LearnedOrbit)
    elif assignment['gait_role']=='orbit':
        classes.append(SupportBridge)
    if assignment['role']=='hybrid':
        classes.append(HybridWBC)
    if assignment['role']=='reduced':
        from w61_equality_wbc import EqualityReducedWBC
        classes.append(EqualityReducedWBC)
    cls=type('W61AssignedB3',tuple(classes+[B3]),{})
    return cls(model,data,feet,base,out,config)
