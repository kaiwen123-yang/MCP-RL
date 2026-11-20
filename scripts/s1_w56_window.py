"""Publish only completed 10-substep packets, including at terminal events."""
from s1_go2_r10_observer import MaterialObserver


class WindowObserver(MaterialObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packet = []
        self.windows = 0
        self.packet_start = self.packet_last = 0.

    def completed(self, start, end, bottom, velocity):
        expected = self.packet[-1][1] if self.packet else self.time
        if abs(start-expected)>1e-9 or abs(end-start-.0002)>1e-9:
            raise ValueError(f'Discontinuous foot measurement window {expected}, {start}, {end}')
        self.packet.append((start, end, bottom.copy(), velocity.copy()))

    def observe(self, t, bottom, velocity):
        if len(self.packet) < 10:
            # t=0, or the final terminal observation after a partial window.
            # A consumer of an unpublished estimate sees the last publication.
            return
        if len(self.packet) != 10 or abs(t-self.packet[-1][1]) > 1e-9:
            raise ValueError(f'Missed a 500Hz observer publication at {t}')
        self.packet_start = self.packet[0][0]
        self.packet_last = self.packet[-1][0]
        for _, end, measurement, motion in self.packet:
            super().observe(end, measurement, motion)
        self.packet.clear()
        self.windows += 1

    def timing(self):
        result = super().timing()
        result.update(sampling='Ten measured left boundaries at 0.2ms; publish at window end',
                      complete_windows=self.windows, terminal_unpublished_substeps=len(self.packet))
        return result
