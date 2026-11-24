"""Score physical work using the recorded cost at the same simulated clock and case."""
from bisect import bisect_right
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path

ROOT=Path('/root/autodl-tmp/MCP-RL-w57')
RAW=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
LOG=ROOT/'work/w57/expanded-evaluations'
def read(p):return json.loads(p.read_text())
def when(s):return datetime.fromisoformat(s).timestamp()

@lru_cache(maxsize=32)
def curve(name,historical=False):
    base=ROOT.parent/'MCP-RL-w56-r60/outputs/terrain_memory/simulation'/RAW.name if historical else RAW
    p=base/'jobs'/name/'physics.log'
    if not p.is_file():return [],[]
    rows=[]
    for line in p.read_text(errors='replace').splitlines():
        if not line.startswith('{"attempt"') or not line.endswith('}'):continue
        row=json.loads(line)
        if 'time_s' in row and 'wall_s' in row:rows.append((row['time_s'],row['wall_s']))
        elif 'simulation_s' in row and 'wall_s' in row and (not rows or row['simulation_s']>rows[-1][0]):rows.append((row['simulation_s'],row['wall_s']))
    return [t for t,w in rows],[w for t,w in rows]

def cost(name,t,historical=False):
    xs,ys=curve(name,historical)
    if not xs or t<xs[0]-1e-6 or t>xs[-1]+1e-6:return None
    if t>=xs[-1]:return ys[-1]
    i=max(0,bisect_right(xs,t)-1)
    return ys[i]+(ys[i+1]-ys[i])*(t-xs[i])/(xs[i+1]-xs[i])

class PhaseComparison:
    def __init__(self):
        grants=[json.loads(l) for l in (LOG/'events.jsonl').read_text().splitlines() if '"lease_granted"' in l]
        cutoff=when(grants[0]['at'])
        names=['w57-m1-p-'+c+'-recovered-a01' for c in 'ab']
        with (RAW/'w57/resource-trace.jsonl').open('rb') as f:
            # The first grant fixes exactly which observations predate four slots.
            f.seek(max(0,f.seek(0,2)-32*1024*1024));f.readline()
            data=[r for l in f if l.endswith(b'\n') for r in [json.loads(l)]
                  if cutoff-430<=r['unix_s']<cutoff and all(i in r['physics'] for i in names)]
        end=data[-1];start=next(x for x in data if 180<=end['unix_s']-x['unix_s']<=360)
        self.calibration={};self.pair_end={}
        for n in names:
            t0=start['physics'][n]['physical_s'];t1=end['physics'][n]['physical_s']
            b=n.replace('-p-','-b-')
            self.calibration[n]=(end['unix_s']-start['unix_s'])/(cost(b,t1)-cost(b,t0))
            self.pair_end[n]=t1
    def interval(self,n,t0,t1):
        _,_,arm,c,mode,*_=n.split('-')
        if mode=='recovered':
            p=f'w57-m1-p-{c}-recovered-a01'
            if arm=='pr' and t1<=self.pair_end[p]:name=p;factor=1.
            else:
                name=f'w57-m1-b-{c}-recovered-a01'
                factor=self.calibration[p] if arm in ('p','pr') else 1.
        else:
            name=n.replace('-e60-','-e30-')
            factor=1.
            if t1>30.:
                if arm in ('p','pr'):name='w57-m1-p-a-private-original-e60-a01'
                else:
                    root=ROOT.parent/'MCP-RL-w56-r60/outputs/terrain_memory/simulation'/RAW.name
                    names=list(root.glob(f'job-*-f-{c}-r60-*-E60-attempt01'))
                    if len(names)!=1:return None
                    name=names[0].name
                    a,b=cost(name,t0,True),cost(name,t1,True)
                    return None if a is None or b is None else (b-a,name)
        a,b=cost(name,t0),cost(name,t1)
        return None if a is None or b is None else ((b-a)*factor,name)
    def observations(self):
        data=[json.loads(l) for l in (LOG/'resources.jsonl').read_text().splitlines()]
        spans={4:[],6:[]};base=None;last=None
        for r in data:
            active=r['active'];n=len(active)
            if n not in spans or any(v['clock'] is None for v in active.values()):base=last=None;continue
            if not last or active.keys()!=last['active'].keys() or when(r['at'])-when(last['at'])>45:base=r
            last=r
            if base is None:base=r;continue
            dt=when(r['at'])-when(base['at'])
            if dt<115:continue
            units={};raw={};refs={}
            for k,v in active.items():
                t0=base['active'][k]['clock'];t1=v['clock']
                z=self.interval(k,t0,t1)
                if z is None or t1<=t0:break
                units[k]=z[0]/dt;refs[k]=z[1];raw[k]=(t1-t0)/dt
            if len(units)==n:
                spans[n].append(dict(start=base['at'],end=r['at'],seconds=dt,units=units,
                    expected_one_stream_work_per_wall_second=sum(units.values()),raw_physical_s_per_wall_s=raw,
                    raw_total=sum(raw.values()),reference_attempts=refs))
            base=r
        return spans
    def six_decision(self):
        spans=self.observations()
        if not spans[4] or len(spans[6])<2:return None
        trials=spans[6][-2:];four=spans[4][0]
        if trials[0]['units'].keys()!=trials[1]['units'].keys():return None
        common=four['units'].keys() & trials[0]['units'].keys()
        if len(common)<2:return None
        baseline=sum(four['units'][k] for k in common)
        ratios=[sum(t['units'][k] for k in common)/baseline for t in trials]
        per={k:[t['units'][k]/four['units'][k] for t in trials] for k in common}
        delta=[t['expected_one_stream_work_per_wall_second']/four['expected_one_stream_work_per_wall_second'] for t in trials]
        retain=(min(delta)>1 and min(ratios)>=.85 and all(max(v)>=.75 for v in per.values()))
        return dict(keep_six=retain,reference_four=four,two_disjoint_six_windows=trials,
            phase_adjusted_aggregate_ratios=delta,previous_stream_retention=ratios,per_common_stream_ratios=per,
            calibration_P_vs_saved_B=self.calibration,
            interpretation='Historical B recovered cost at the same simulated clock for P0, pre-trial scaled B for incumbent P, frozen P clock for PR, and own original E30 for early E60. Differences between arms and trajectories still matter. These are scheduling measurements.')
