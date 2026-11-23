"""W58 G1 touch control with the same P2 candidates and execution permissions."""
from s1_w58_gait import FController as GaitController
from s1_w58_p2 import P2ControlMixin


class FController(P2ControlMixin, GaitController):
    pass
