"""Re-evaluate only the finite initial cohort as W57 publishes the remaining closed collectors."""
from pathlib import Path
import json,sqlite3,time
from inventory import CTL,now,save
from policy import Policy,EP,W57,PERSONAL,RAW,COMPACT,PROTECTED_EP
def eligible(req):
    db=sqlite3.connect((CTL/'inventory.sqlite').as_uri()+'?mode=ro',uri=True,timeout=60)
    p=Policy(db,fresh=bool(req.get('final')))
    known=set(req.get('seen',[]));start=json.loads((CTL/'roots.json').read_text())['start_unix']
    rows=[];considered=[]
    for name in sorted(p.delivered.keys()-known):
        if name in PROTECTED_EP or not EP.fullmatch(name):continue
        state=json.loads((COMPACT/'attempts'/(name+'.json')).read_text())
        if state['started_unix_s']>start:continue
        original=(RAW/name).resolve()
        if original!=PERSONAL/'W57/originals'/name and not original.is_relative_to(W57):continue
        prefix=str(original)+'/'
        for i,path,root,k,tr,nl,opened,src in db.execute('select id,path,root,kind,tracked,nlink,opened,state from files where path>=? and path<?',(prefix,prefix+'\U0010ffff')):
            why=p.reason(path,root,k,tr,nl,opened,src)
            if why.startswith('remove:'):rows.append([i,why])
        considered.append(name)
    db.close();return dict(considered=considered,eligible=rows,checked=now())
