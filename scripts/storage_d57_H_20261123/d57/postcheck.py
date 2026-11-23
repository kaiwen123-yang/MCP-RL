"""Verify retired paths and retained live inputs, using directory listings and actual frozen identities."""
import collections,fcntl,gzip,hashlib,json,os,re,sqlite3,stat,tempfile,time
from pathlib import Path
from inventory import BASE,CTL,PERSONAL,BG,now,save
from policy import COMPACT,RAW,live_paths
def run():
    db=sqlite3.connect((CTL/'inventory.sqlite').as_uri()+'?mode=ro',uri=True,timeout=60)
    db.execute("attach database ? as done",((CTL/'retired.sqlite').as_uri()+'?mode=ro',))
    gone=collections.defaultdict(dict)
    for i,p,n,a in db.execute("select f.id,f.path,f.size,f.allocated from done.actions d join files f on f.id=d.id where d.state='removed'").fetchall():
        parent,name=p.rsplit('/',1);gone[parent][name]=(i,n,a)
    restored=[];counts=0;logical=0;allocated=0;W57_counts=collections.defaultdict(lambda:[0,0,0])
    for p,entries in gone.items():
        try:present={v.name for v in os.scandir(p)}
        except FileNotFoundError:present=set()
        for f,(i,n,a) in entries.items():
            counts+=1;logical+=n;allocated+=a
            match=re.search(r'/(w57-pilot0[12]-(?:full|blind)-[ab]-e60-[0-9]{4})(?:/|$)',p)
            if match:
                group=W57_counts[match[1]]
                group[0]+=1;group[1]+=n;group[2]+=int(bool(re.fullmatch(r'mpc-[0-9]+\.npz',f)))
            if f in present:restored.append(i)
    c=CTL/'dependency-pins.json';pins=json.loads(c.read_text()) if c.exists() else []
    affected=[p for p in pins if p.rpartition('/')[0] in gone and p.rpartition('/')[2] in gone[p.rpartition('/')[0]]]
    # Recheck concrete W57/W58 consumers. Paths are taken from actual frozen case receipts.
    cases={}
    for case in 'AB':
        f=BASE/'MCP-RL-w57/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f'/f'F-{case}-r60/job.json'
        d=json.loads(f.read_text())
        for prop,dg in [('checkpoint','checkpoint_sha256'),('resume_checkpoint','resume_sha256')]:
            p=BASE/'MCP-RL-w57'/d[prop]
            with p.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
            cases[str(p)]=h==d[dg]
    reads={}
    for root,rev in [('MCP-RL-w55-f','r55'),('MCP-RL-w56-r60','r60')]:
        large=BASE/root/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
        for p in large.glob(f'job-*-{rev}-*-E600-attempt01'):
            reads[p.name]=dict(trajectory=(p/'trajectory.npz').is_file(),physical_chunks=len(list(p.glob('physical-*.npz'))),
                               material_chunks=len(list(p.glob('material-[0-9]*.npz'))))
    stream=Path('/root/autodl-tmp/MCP-RL-w57/work/w57-r/pilot02')
    for head in ('full','blind'):
        d=json.loads((stream/head/'node-08192.json').read_text())
        p=stream/head/'policy-032.pt'
        with p.open('rb') as s:cases[str(p)]=hashlib.file_digest(s,'sha256').hexdigest()==d['checkpoint_sha256']
    q=read(COMPACT/'dispatch/frozen-evaluations.json')
    # Only update already-verified publication metadata, under the publisher's existing lock.
    with (BASE/'MCP-RL-w57/work/w57/publish.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        for name,(num,nb,nplans) in W57_counts.items():
            fp=COMPACT/'delivery'/(name+'.json');d=read(fp)
            if d['status']!='verified':raise ValueError('W57 delivery changed: '+name)
            lr=d.setdefault('local_retention',{})
            before=lr.get('H_20261123',{}).get('prior_mpc_removed',lr.get('removed_mpc_files',0))
            lr['removed_mpc_files']=before+nplans
            lr['H_20261123']=dict(files=num,logical_bytes=nb,prior_mpc_removed=before,
                archive_catalog='H:/MPC+RL/00_catalog/migration-20261123/archive-index.sqlite',complete_release=True)
            d.setdefault('originals_retained_at_verification',d['originals_retained']);d['originals_retained']=False
            tmp=CTL/(name+'-metadata.pending');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
            os.replace(tmp,fp)
    return dict(H_retained_membership_changes=dict(W57_counts),observed_at=now(),removed_files=counts,removed_bytes=logical,removed_allocated_bytes=allocated,
        reappeared_new_versions=restored,protected_named_paths_unlinked=affected,verified_inputs=cases,
        four_E600=reads,W57_dispatch=q['stage'],personal_free_inodes=os.statvfs(PERSONAL).f_favail,
        scratch_free_bytes=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize,**BG)
def read(p):return json.loads(p.read_text())
