"""Read verified H: trajectory members in memory and report W59 historical coverage."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

import numpy as np


def find_object(catalogs, pathlist):
    for label, conn in catalogs:
        for source in pathlist:
            row=conn.execute("SELECT f.n,f.h,o.archive,o.member FROM files f JOIN objects o ON f.h=o.h WHERE f.path=? AND o.archive IS NOT NULL",(source,)).fetchone()
            if row:
                return (label,source,*row)
    return None


def member(base, item):
    _,_,size,digest,archive,name=item
    with zipfile.ZipFile(base/archive) as outer:
        data=outer.read(name)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError('Archive member differs from the catalog: '+name)
    return data


def inspect(base,requested):
    catalogs=[]
    for day in ('20261124-process','20261123'):
        db=base/'00_catalog'/('migration-'+day)/'archive-index.sqlite'
        catalogs.append((day,sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)))
    jobs=[]
    for row in requested:
        name=row['name']
        prefixes=list(dict.fromkeys((row['root'], '/root/autodl-tmp/MCP-RL-w57/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/'+name,'/root/autodl-tmp/MCP-RL-w57/work/w57-raw-scratch/'+name,'/autodl-fs/data/MCP-RL/W57/originals/'+name)))
        traj=find_object(catalogs,[p+'/trajectory.npz' for p in prefixes])
        req=find_object(catalogs,[p+'/w55-f/handover-requests.jsonl' for p in prefixes])
        jobs.append((traj[4] if traj else '',name,traj,req,row))
    out=[]
    for _,name,traj,req,row in sorted(jobs,key=lambda x:x[:2]):
        r=dict(episode=name,case=row['case'],source_group=row['group'],archived_trajectory=traj is not None,archived_request_stream=req is not None)
        if traj:
            raw=member(base,traj)
            with np.load(io.BytesIO(raw)) as z:
                t=z['time'];q=z['qpos'];phase=z['phase']
            del raw
            r.update(n500=len(t),start_s=float(t[0]),end_s=float(t[-1]),cadence_median_s=float(np.median(np.diff(t))) if len(t)>1 else None,
              path_m=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).sum()),min_body_height_m=float(q[:,2].min()),
              first_completion_s=float(t[np.flatnonzero(phase==5)[0]]) if (phase==5).any() else None,
              trajectory=dict(catalog=traj[0],original_path=traj[1],size=traj[2],sha256=traj[3],pack=traj[4],member=traj[5]))
        if req:
            samples=[]
            for line in member(base,req).splitlines():
                if line:
                    v=json.loads(line)
                    if v.get('stage')=='request':
                        samples.append(v['requested_due_s']-v.get('original_due_s',v['nominal_due_s']))
            r.update(requests=len(samples),due_offset_min_s=min(samples,default=None),due_offset_max_s=max(samples,default=None),
                     nonzero_due_offsets=sum(abs(s)>1.e-9 for s in samples),request_sha256=req[3])
        out.append(r)
    return dict(source='The two verified H catalogs were opened SQLite read-only, and selected ZIP members were read in memory. Original trajectory/request hashes and sizes checked separately. Supplements the frozen D57 model bank.',
              missing_from_frozen_bank=len(requested),with_verified_trajectory=sum(v['archived_trajectory'] for v in out),
              with_verified_requests=sum(v['archived_request_stream'] for v in out),control_samples=sum(v.get('n500',0) for v in out),
              requests=sum(v.get('requests',0) for v in out),nonzero_due_offsets=sum(v.get('nonzero_due_offsets',0) for v in out),
              episodes=out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',required=True,type=Path)
    parser.add_argument('--bank',required=True,type=Path,help='w59/model/bank.json from the frozen D57 fit')
    args=parser.parse_args()
    bank=json.loads(args.bank.read_text())
    missing=[dict(name=r['episode'],root=r['root'],case=r['case'],group=r['source_group']) for r in bank['episodes'] if not r.get('intervals')]
    print(json.dumps(inspect(args.archive_root,missing),separators=(',',':')))
