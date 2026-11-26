"""Count every predeclared W58 development/validation target without rescoring."""
import argparse,json,pathlib,collections
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();r=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w58';agg=collections.Counter();by=collections.defaultdict(collections.Counter);missing=[];all_targets=0;clear=[];all_rows=[]
for typ in ['development','validation']:
 for ep in json.loads((r/f'{typ}-matrix.json').read_text())['episodes']:
  data=json.loads((r/'mechanics'/f"{ep['attempt']}.json").read_text())
  for t in data['targets']:
   all_targets+=1;ms=t['matches'];hold_ok=ep['history']=='short' or t['continuous_clear_s']>=6-1e-8
   if ep['history']=='recovered':clear.append(t['continuous_clear_s'])
   cat='realized' if t['realized'] else 'no_completed_traversal' if not t['completed_traversal'] else 'clear_hold_unmet' if not hold_ok else 'no_ledger_pair' if not ms else 'all_zero_overlap' if max(x['exact_overlap_m2']for x in ms)<=0 else 'gap_mismatch' if not any(x['condition_met']and x['exact_overlap_m2']>0 for x in ms) else 'other'
   agg[cat]+=1;by[f"{typ}/{ep['history']}"][cat]+=1;row={'attempt':ep['attempt'],'case':ep['case'],'history':ep['history'],'target':t['target_id'],'category':cat,'matches':len(ms),'clear_s':t['continuous_clear_s'],'gap_range_s':[min(x['gap_s']for x in ms),max(x['gap_s']for x in ms)]if ms else None,'max_overlap_m2':max((x['exact_overlap_m2']for x in ms),default=None),'any_condition_met':any(x['condition_met']for x in ms)};all_rows.append(row)
   if not t['realized']:missing.append(row)
a.out.write_text(json.dumps({'source_commit':'ef500f31c7a96d94f83c9afb5c7a880864b447dc','all_targets':all_targets,'counts':dict(agg),'by_split_history':{k:dict(v)for k,v in by.items()},'recovered_continuous_clear_s_range':[min(clear),max(clear)],'missing':missing,'all_targets_compact':all_rows,'scope':'Descriptive counts; multiple arms share trajectories. No old criterion changed. No claim of 384 independent samples.'},indent=2));print(dict(agg));print('clear range',min(clear),max(clear))
