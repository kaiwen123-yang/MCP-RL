"""Classify absent source-release-target pairs in one raw W58 event stream."""
import argparse,json,pathlib,collections
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--attempt',required=True);p.add_argument('--events',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();r=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w58';m=json.loads((r/'attempts'/f'{a.attempt}.json').read_text())['mechanism'];events=[json.loads(s)for s in a.events.open()];loads={x['event_id']:x for x in events if x['type']!='release'};releases={(tuple(x['world_key']),x['previous_load_event_id']):x for x in events if x['type']=='release'};out=[]
for tt in m['all_target_rows']:
 if tt['realized']:continue
 w=next(w for w in m['traversals']if w['cycle']==tt['traversal']);lo,hi=tt['region_xy_m'][0];yl,yh=tt['region_xy_m'][1]
 def region(e):x,y=[(c+.5)*.02 for c in e['world_key']];return lo<=x<hi and yl<=y<yh
 winloads=[e for e in loads.values()if w['start_s']<=e['time_s']<=w['end_s']];fr=[e for e in winloads if e['leg']==1];rrloads=[e for e in winloads if e['leg']==3];stats=collections.Counter();samples=[];outsidepairs=[]
 for e in rrloads:
  s=loads.get(e['previous_load_event_id']);rel=releases.get((tuple(e['world_key']),e['previous_load_event_id']));
  if s and rel and s['leg']==1 and w['start_s']<=s['time_s'] and rel['time_s']<e['time_s']:
   if not region(e):outsidepairs.append({'key':e['world_key'],'source_id':s['event_id'],'release_id':rel['event_id'],'target_id':e['event_id'],'gap_s':e['time_s']-rel['time_s']})
  if not region(e):continue
  if not s:why='no_prior_load'
  elif s['leg']!=1:why='previous_load_other_leg_'+str(s['leg'])
  elif s['time_s']<w['start_s']:why='source_before_traversal'
  elif not rel:why='no_release_for_source'
  elif rel['time_s']>=e['time_s']:why='nonpositive_free_gap'
  else:why='valid_event_pair'
  stats[why]+=1;samples.append({'target_id':e['event_id'],'key':e['world_key'],'time':e['time_s'],'previous_id':e['previous_load_event_id'],'previous_leg':s['leg']if s else None,'previous_time':s['time_s']if s else None,'reason':why})
 frkeys={tuple(e['world_key'])for e in fr if region(e)};rrkeys={tuple(e['world_key'])for e in rrloads if region(e)};row={'target_id':tt['target_id'],'window':w,'FR_loads_in_region':sum(region(e)for e in fr),'RR_loads_in_region':sum(region(e)for e in rrloads),'FR_keys':sorted(frkeys),'RR_keys':sorted(rrkeys),'common_keys':sorted(frkeys&rrkeys),'RR_predecessor_reasons':dict(stats),'RR_region_events':samples,'valid_crossfoot_pairs_outside_region':outsidepairs};out.append(row);print(tt['target_id'],'FR',row['FR_loads_in_region'],'RR',row['RR_loads_in_region'],'common_keys',len(row['common_keys']),'reasons',stats,'valid_pair_outside_ROI',len(outsidepairs))
a.out.write_text(json.dumps({'attempt':a.attempt,'source':'HTTP Range extracted ZIP member; ZIP CRC verified; whole ZIP SHA not rehashed','targets':out},indent=2))
