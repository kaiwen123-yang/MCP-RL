"""Shared W56 native/plant execution semantics on the appropriate MuJoCo ABI."""
from pathlib import Path
from s1_go2_w56_n import (PrecontactE as PreviousE,W54NRecords as PreviousRecords,
    build_project_model,write_flat_ground_height,build_prediction_model,write_equilibrium_settings)

class PrecontactE(PreviousE):
    library=Path(__file__).resolve().parents[1]/'work/r11-frozen/S5-r59/lib/libw55_support_python.so'

class W54NRecords(PreviousRecords):
    def __init__(self,*args):
        super().__init__(*args)
        self.controller='S5-r59'
