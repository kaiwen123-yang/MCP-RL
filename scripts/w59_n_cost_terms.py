"""Recover paired original cost components from each recorded native residual ring."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from analyze_s1_w55_n import read_policy,records
from publish_s1_go2_w54 import ROOT,write

REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')

def recover(episode,resolved,call_ids=None):
    settings=json.loads(resolved.read_text())['applied_settings']
    tree=ET.parse(episode/'model/G1-project-go2.xml')
    sensors=[(r.attrib['name'],int(r.attrib['dim']),list(map(float,r.attrib['user'].split())))
             for r in tree.findall('.//sensor/user')]
    risk_node=tree.find(".//custom/numeric[@name='task_risk']")
    risk=float(risk_node.attrib['data']) if risk_node is not None else 0.
    native=episode/'w54-n-native';data=[]
    def decompose(path,weights):
        pi=read_policy(path);res=pi['accepted_residual'];offset=0;terms={}
        for name,dim,meta in sensors:
            x=res[:,offset:offset+dim];offset+=dim
            if meta[0]==0:y=.5*np.sum(x*x,axis=1)
            elif meta[0]==2:y=np.sqrt(np.sum(x*x,axis=1)+meta[4]**2)-meta[4]
            elif meta[0]==6:y=np.sum(np.sqrt(x*x+meta[4]**2)-meta[4],axis=1)
            else:raise ValueError(f'Unrecognized original norm {meta[0]}')
            terms[name]=y*weights[name]
        total=sum(terms.values());node=np.expm1(risk*total)/risk if abs(risk)>1e-6 else total
        gap=float(np.max(abs(node-pi['accepted_costs'])))
        return dict(weighted_terms={k:float(x.mean()) for k,x in terms.items()},risk=risk,
                    mean_original_node_cost=float(node.mean()),max_reconstructed_node_cost_error=gap)
    for call in records(native/'calls.jsonl'):
        n=call['call_id']
        if call_ids and n not in call_ids:continue
        if not call_ids and not call['full_candidates']:continue
        weight=max((r for r in settings if r['time_s']<=call['time_s']+1e-8),key=lambda r:r['time_s'])['cost_weights']
        rows=[]
        p=native/f'policy-{n:06d}.bin'
        if p.is_file():rows.append(dict(source='installed',**decompose(p,weight)))
        for c in call['candidates']:
            p=native/f"call-{n:06d}-candidate-{c['slot']:02d}.bin"
            if p.is_file():
                rows.append(dict(source=c['generator'],slot=c['slot'],first_constraint=c['first_constraint_code'],
                    dx_m=c['dx_m'],selector_J=c['J'],**decompose(p,weight)))
        data.append(dict(call=n,time_s=call['time_s'],selection=call['selection_source'],winner=call['winner'],rows=rows))
    return dict(method='Native recorded residual at each of 176 nodes, logged time-specific weights, XML norm and risk. Mean weighted terms are before the per-node risk transform. Selector J also applies hard rejection.',events=data)

def analyze(workspace,folder,case,output,call_ids=None):
    path=workspace/'results'/REL/folder
    stage=json.loads((path/case/'E30.json').read_text())
    attempt=stage['attempt'];raw=workspace/'outputs'/REL/attempt
    resolved=next(path.glob(f'*/{attempt}-resolved.json'))
    result=recover(raw,resolved,call_ids)
    result.update(source_attempt=attempt)
    write(output,result)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--original',action='store_true');a=p.parse_args()
    out=ROOT/'results'/REL/'w56/n'
    if a.original:
        analyze(ROOT.parent/'MCP-RL-w55-n-r56','w55/n','N-A-r56',out/'r56-original-objective.json',{809,856})
    else:
        analyze(ROOT,'w56/n','N-A-r59',out/'N-A-r59/E30-native-objective.json')
