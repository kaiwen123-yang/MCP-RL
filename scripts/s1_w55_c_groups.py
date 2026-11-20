"""Compare complete applied loading shapes before scoring held-out episodes."""
import json
import numpy as np
from s1_w55_c_data import RAW,COMPACT,REVIEW,LEGACY,write,geometry
from train_s1_w55_c import runs,contact_families,synthetic


def descriptor(n):
    depth=np.maximum(-n,0.)
    spans=[(a,b) for a,b in runs(depth>0) if depth[a:b].max()>=1e-5]
    if not spans:
        return None
    output=[]
    for j,(a,b) in enumerate(spans):
        trace=depth[a:b]
        sampled=np.interp(np.linspace(0,len(trace)-1,32),np.arange(len(trace)),trace)
        output.append((b-a,0 if j==0 else a-spans[j-1][1],sampled))
    return output


def similar(a,b):
    if a is None or b is None or len(a)!=len(b):
        return False
    for (width,gap,x),(width2,gap2,y) in zip(a,b):
        if abs(width-width2)>max(3,.1*max(width,width2)) or abs(gap-gap2)>max(3,.1*max(gap,gap2)):
            return False
        amplitude=max(x.max(),y.max())
        if abs(x.max()-y.max())>max(.0001,.1*amplitude):
            return False
        if np.sqrt(np.mean((x-y)**2))>max(.00004,.075*amplitude):
            return False
    return True


def signatures(name):
    with np.load(RAW/f'{name}-histories.npz') as z:
        n=z['normal'];keys=z['keys'];owners=z['owner']
    groups=contact_families(n,owners)
    return [(f'{name}:{key[0]},{key[1]}',groups[j],descriptor(n[:,j])) for j,key in enumerate(keys)]


def old_nine():
    summary=json.loads((REVIEW/'pann-replay-summary.json').read_text())
    records=[]
    for case,c in summary['cases'].items():
        d=LEGACY/c['job'];keys=np.array(c['keys'])
        feet=np.concatenate([np.load(p)['foot'][:,6:18] for p in sorted(d.glob('material-[0-9]*.npz'))]).reshape(-1,4,3)
        n,_,owners=geometry(feet,keys);groups=contact_families(n,owners)
        records += [(f'prior_{case}:{k[0]},{k[1]}',groups[j],descriptor(n[:,j])) for j,k in enumerate(keys)]
    return records


def main():
    seen=signatures('N44')
    sn,_,_=synthetic()
    seen += [(f'synthetic:{j}',(j,),descriptor(sn[:,j])) for j in range(sn.shape[1])]
    out={}
    for name in ('F51','B46','F53'):
        if name=='B46':
            seen+=old_nine()+signatures('N48')+signatures('F52')
        current=signatures(name);matched={}
        for key,g,shape in current:
            for peer,_,saved in seen:
                if similar(shape,saved):
                    matched.setdefault(g,[]).append((key,peer))
                    break
        out[name]={'kept_complete_history_keys':[k.split(':')[-1] for k,g,s in current if g not in matched and s is not None],
                   'kept_families':len({g for k,g,s in current if g not in matched and s is not None}),
                   'excluded_complete_history_keys':[k.split(':')[-1] for k,g,s in current if g in matched or s is None],
                   'excluded_event_family_matches':{','.join(map(str,g)):matches for g,matches in matched.items()},
                   'all_families':{k.split(':')[-1]:list(g) for k,g,s in current}}
        seen+=current
    write(COMPACT/'disjoint-holdouts.json',dict(
        split='Entire robot episodes in single cohorts. Use physically connected foot-loading event identities within each episode. If one member resembles a previously seen complete boundary, exclude its whole event family from primary held-out scores.',
        distance='Entire compression-pulse sequence; same pulse count, compare every width and intervening release duration within 10% or 3 fine ticks; peak within 10% or 0.0001 strain; 32-sample waveform RMS within 7.5% or 0.00004 strain. Ignore absolute onset and terminal free-tail duration for conservative duplicate detection. Pulses below 1e-5 strain (0.5 um) are treated as negligible only in grouping. Full measured input retained by all evaluations.',
        cohorts=out))
    print({k:(len(v['kept_complete_history_keys']),v['kept_families']) for k,v in out.items()},flush=True)


if __name__=='__main__':main()
