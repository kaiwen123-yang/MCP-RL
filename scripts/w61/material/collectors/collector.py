"""Run the unchanged W61 E contact policy with the preregistered benchmark publication boundary."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from transport import Client

LEARNING=Path('/root/autodl-tmp/MCP-RL-w61-learning')
SOURCE=LEARNING/'scripts/w61/learning/collect.py'
sys.path.insert(0,str(LEARNING/'scripts'))
sys.path.insert(0,str(LEARNING/'scripts/w61/learning'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('spec',type=Path);parser.add_argument('arm');parser.add_argument('case')
    parser.add_argument('window');parser.add_argument('out',type=Path);parser.add_argument('ipc',type=Path)
    args=parser.parse_args();spec=json.loads(args.spec.read_text());w=spec['windows'][args.window]
    frozen=importlib.util.spec_from_file_location('w61_benchmark_collect',spec['frozen_collector'])
    mod=importlib.util.module_from_spec(frozen);frozen.loader.exec_module(mod)
    wrapper={}
    original=mod.PolicyMaterial
    class BenchMaterial(original):
        def attach(self,*a):
            ret=super().attach(*a)
            obs=self.observer
            c=Client(obs.native,obs.ground.law.model,args.arm,('A','B').index(args.case),args.ipc,args.out)
            c.original_completed=obs.completed
            obs.completed=c.completed;obs.native=c;wrapper['client']=c
            return ret
    mod.PolicyMaterial=BenchMaterial
    a=argparse.Namespace(out=args.out,weights=Path(w['weights']),method='PPO',seed=6001,version=w['version'],
        episode=0,case=args.case,role='train',offset=0,quota=128,duration=60.,prefix_s=w['prefix_s'],variant='')
    try:mod.collect(a)
    finally:
        if 'client' in wrapper:wrapper['client'].finish()


if __name__=='__main__':main()
