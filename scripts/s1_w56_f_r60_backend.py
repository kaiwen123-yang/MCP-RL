"""Same frozen r58 WBC/reference. Its observer supplies the original law on CPU."""
from s1_w56_f_r58_backend import FController as R58


class FController(R58):
    def soft_reference(self,t,leg,pos_z,measured,target,material_info):
        obs=material_info['material_observer']
        obs.mark('normal_inverse',t)
        result=super().soft_reference(t,leg,pos_z,measured,target,material_info)
        obs.mark('normal_inverse_returned',t)
        return result
