"""Offline hybrid M1: CUDA nominal propagation and the original measured-force correction on CPU."""
import ctypes as ct
from pathlib import Path
import time
import weakref

import numpy as np

from s1_w57_material import W57MaterialObserver
from s1_w60_cpu_observer import ARRAY, BYTE, INDEX

BUILD = Path(__file__).resolve().parents[1]/'work/w59/m1-gpu-study'


class _HybridPacket:
    def __init__(self, model, propagation_library):
        gpu = ct.CDLL(str(propagation_library))
        gpu.w59_gpu_create.argtypes = [ARRAY, ct.POINTER(ct.c_void_p)]
        gpu.w59_gpu_create.restype = ct.c_int
        gpu.w59_gpu_destroy.argtypes = [ct.c_void_p]
        gpu.w59_gpu_destroy.restype = None
        gpu.w59_gpu_error.argtypes = [ct.c_int]
        gpu.w59_gpu_error.restype = ct.c_char_p
        shared = [ct.c_int, ct.c_int, ARRAY, ARRAY, ARRAY, INDEX, ct.c_double,
                  ARRAY, BYTE, BYTE, ARRAY, ARRAY, BYTE]
        gpu.w59_gpu_propagate.argtypes = [ct.c_void_p, *shared, ARRAY, BYTE, ct.POINTER(ct.c_int)]
        gpu.w59_gpu_propagate.restype = ct.c_int
        cpu = ct.CDLL(str(BUILD/'correction.so'))
        cpu.w59_correct_last.argtypes = [ARRAY, *shared, ct.c_int, ARRAY, ARRAY, ARRAY, BYTE]
        cpu.w59_correct_last.restype = None
        self.gpu, self.cpu = gpu, cpu
        self.handle = ct.c_void_p()
        self.check_cuda(gpu.w59_gpu_create(model, ct.byref(self.handle)))
        self.close = weakref.finalize(self, gpu.w59_gpu_destroy, self.handle)
        self.before = np.empty((0, 6), dtype=np.float64)
        self.was_loaded = np.empty(0, dtype=np.uint8)
        self.propagation_wall_s = 0.
        self.correction_wall_s = 0.
        self.calls = 0

    def check_cuda(self, code):
        if code:
            raise RuntimeError(self.gpu.w59_gpu_error(code).decode())

    def w57_packet(self, model, n, ticks, dt, normal, area, owners, gamma, state,
                   settled, loaded, forces, stats, trace, correct, measured, diagnostics):
        self.calls += 1
        start = time.perf_counter()
        if len(self.was_loaded) < n:
            self.before = np.empty((n, 6), dtype=np.float64)
            self.was_loaded = np.empty(n, dtype=np.uint8)
        before = self.before[:n]
        was_loaded = self.was_loaded[:n]
        bad = ct.c_int(-1)
        args = (n, ticks, dt, normal, area, owners, gamma, state, settled, loaded, forces, stats, trace)
        self.check_cuda(self.gpu.w59_gpu_propagate(self.handle, *args, before, was_loaded, ct.byref(bad)))
        returned = time.perf_counter()
        self.propagation_wall_s += returned - start
        if bad.value >= 0:
            return bad.value
        self.cpu.w59_correct_last(model, *args, correct, measured, diagnostics, before, was_loaded)
        self.correction_wall_s += time.perf_counter() - returned
        return -1


class HybridObserver(W57MaterialObserver):
    """Same causal input, geometry, event ledger, CPU queries and publication order as M1."""
    def __init__(self, parameters, model_path, mode='M1', propagation_library=None):
        super().__init__(parameters, model_path, mode=mode)
        library = BUILD/'propagate.so' if propagation_library is None else propagation_library
        self.native = _HybridPacket(self.ground.law.model, library)

    def close(self):
        self.native.close()

    def timing(self):
        return dict(super().timing(), backend='W59 offline FP64 CUDA propagation plus CPU force assimilation',
                    native_packet_calls=self.native.calls,
                    synchronous_propagation_wall_s=self.native.propagation_wall_s,
                    cpu_force_correction_wall_s=self.native.correction_wall_s)
