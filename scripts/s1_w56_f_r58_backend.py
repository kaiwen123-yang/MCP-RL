"""r55 force horizon with history-dependent soft normal execution references."""
import json

from s1_w55_f_r55_backend import FController as R55
from tm02.s1_material_prediction import normal_reference


class FController(R55):
    def __init__(self, model, data, feet, base, out, config):
        super().__init__(model, data, feet, base, out, config)
        self.manifest.update(material=config['material_reference'],
                             observer=config['causal_observation'],
                             normal_execution_records='normal-execution.jsonl')
        self._write_manifest()

    def soft_reference(self, t, leg, pos_z, measured, target, material_info):
        obs = material_info['material_observer']
        if abs(obs.time-t) > 1e-9:
            raise ValueError(f'Material publication {obs.time} is stale at {t}')
        increment, detail = normal_reference(obs.ground, material_info['force_boundary_bottom'],
                                             leg, target, measured, self.cfg['wbc']['dt_s'])
        with (self.record_dir/'normal-execution.jsonl').open('a') as f:
            f.write(json.dumps(dict(leg=leg, consume_s=t, publication_s=obs.time,
                                    input_start_s=obs.packet_start, input_last_s=obs.packet_last,
                                    reference_m=pos_z+increment, **detail))+'\n')
        return pos_z+increment
