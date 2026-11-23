"""W58 common G0 gait and event-conditioned planner."""
from s1_w57_f_backend import FController as BaseController
from s1_w58_p2 import P2ControlMixin


class FController(P2ControlMixin, BaseController):
    pass
