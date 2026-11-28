"""Execute a frozen W61 collector, replacing only its M1 packet implementation."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCIENCE=ROOT/'scripts/w61/learning'
FIXED=ROOT/'work/w61/learning/cpu-exact-v1'
SOURCE=SCIENCE/'collect.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(runtime):
    sys.path.insert(0,str(SCIENCE))
    sys.path.insert(0,str(ROOT/'scripts'))
    from s1_go2_r11_policy import PolicyMaterial
    original=PolicyMaterial.attach
    candidate=ct.CDLL(runtime['binary'])
    def attach(self,*args):
        result=original(self,*args)
        candidate.w57_packet.argtypes=self.observer.native.w57_packet.argtypes
        candidate.w57_packet.restype=self.observer.native.w57_packet.restype
        self.observer.native=candidate
        (self.out/'cpu-implementation.json').write_text(json.dumps(runtime,indent=2)+'\n')
        return result
    PolicyMaterial.attach=attach

def main():
    path=os.environ.get('W61_EXACT_BUILD',str(FIXED/'build/identity.json'))
    identity=json.loads(Path(path).read_text())
    if sha(identity['binary'])!=identity['sha256']:
        raise ValueError('Selected CPU material binary differs from the registered build')
    install(dict(name='W61 exact CPU packet reuse',binary=identity['binary'],sha256=identity['sha256'],
                 registered_build=str(path),base_collector_sha256=sha(SOURCE),source_wrapper_sha256=sha(__file__)))
    sys.argv[0]=str(SOURCE)
    runpy.run_path(str(SOURCE),run_name='__main__')
if __name__=='__main__':main()
