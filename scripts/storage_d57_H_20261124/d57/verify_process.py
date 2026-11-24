"""Verify the actual unlink ledger, the protected reference data, and the running or subsequent simulation."""
import collections,fcntl,hashlib,json,os,sqlite3,stat,time
from pathlib import Path
from inventory import BASE,ROOT,CTL,PERSONAL,BG,now,save
from policy import RAW,COMPACT,W57,W58,SOURCE_PIN
from packet import recent_physics
def read(p):return json.loads(p.read_text())
def own_attempt(path):
    for root in (str(W57/'work/w57-raw-scratch'),str(RAW),str(PERSONAL/'W57/originals')):
        if path.startswith(root+'/'):
            first=path[len(root)+1:].split('/',1)[0]
            if (COMPACT/'attempts'/(first+'.json')).is_file():return first
    return None
def hashmatch(path,expected):
    with Path(path).open('rb') as f:val=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),sha256=val,matched=(val==expected))
def postcheck():
    with (CTL/'postcheck.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (CTL/'postcheck.json').is_file():return read(CTL/'postcheck.json')
        db=sqlite3.connect((CTL/'inventory.sqlite').as_uri()+'?mode=ro',uri=True)
        db.execute('attach database ? as retired',((CTL/'retired.sqlite').as_uri()+'?mode=ro',))
        removed=db.execute("select f.id,f.path,f.root,f.size,f.allocated,f.dev,f.ino from retired.actions a join files f using(id) where a.state='removed'").fetchall()
        pinned=set(read(CTL/'dependency-pins.json'));byp=collections.defaultdict(dict);episodes=collections.defaultdict(lambda:[0,0,0]);allocs=collections.Counter()
        for i,path,root,n,al,dev,ino in removed:
            d,name=path.rsplit('/',1);byp[d][name]=(i,dev,ino)
            if any(str(z) in pinned for z in Path(path).parents) or path in pinned:raise ValueError('Removed a named dependency '+path)
            allocs['personal' if root.startswith(str(PERSONAL)) else 'scratch']+=al
            ep=own_attempt(path)
            if ep: 
                x=episodes[ep];x[0]+=1;x[1]+=n;x[2]+=int(name.startswith('mpc-') and name.endswith('.npz'))
        regenerated=[]
        for parent,found in byp.items():
            if not os.path.isdir(parent):continue
            for ent in os.scandir(parent):
                if ent.name not in found:continue
                i,dev,ino=found[ent.name];st=ent.stat(follow_symlinks=False)
                regenerated.append(dict(id=i,path=ent.path,prior_inode=ino,current_inode=st.st_ino))
        verify_dirs=set()
        for name in SOURCE_PIN:
            p=RAW/name
            if p.is_dir():verify_dirs.add(p.resolve())
            else:
                alt=BASE/'MCP-RL-w56-r60/outputs/terrain_memory/simulation'/RAW.name/name
                if alt.is_dir():verify_dirs.add(alt.resolve())
        geo=read(W58/'experiments/terrain_memory/s1-w58-revisit-v2.json')
        def geo_dirs(o):
            if isinstance(o,dict):
                for k,v in o.items():
                    if k=='baseline_path':verify_dirs.add(Path(v).resolve())
                    else:geo_dirs(v)
            elif isinstance(o,list):
                for v in o:geo_dirs(v)
        geo_dirs(geo)
        baselines={}
        for dirname,rev in [('MCP-RL-w55-f','r55'),('MCP-RL-w56-r60','r60')]:
            location=BASE/dirname/'outputs/terrain_memory/simulation'/RAW.name
            for d in location.glob(f'job-*-{rev}-*-E600-attempt01'):
                verify_dirs.add(d.resolve())
                baselines[d.name]=dict(trajectory=(d/'trajectory.npz').is_file(),physical=len(list(d.glob('physical-*.npz'))),
                    material=len(list(d.glob('material-[0-9]*.npz'))),video=(d/'full-mission-material-overlay.mp4').is_file())
        if len(baselines)!=4 or not all(q['trajectory'] and q['physical'] and q['material'] for q in baselines.values()):raise ValueError('Baseline reference unavailable')
        verified={}
        for d in sorted(verify_dirs):
            prefix=str(d)+'/'
            files=db.execute("select path,size,mtime,dev,ino from files where path>=? and path<? and kind='file'",(prefix,prefix+'\U0010ffff')).fetchall()
            if not files:raise RuntimeError('Protected directory missing from process watermark: '+str(d))
            for path,n,mt,dev,ino in files:
                s=os.lstat(path)
                if (s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino)!=(n,mt,dev,ino):raise ValueError('Protected input changed: '+path)
            verified[str(d)]=dict(regular_files=len(files),bytes=sum(v[1] for v in files))
        checks={}
        rel=Path('results/terrain_memory/simulation')/RAW.name
        for c in 'AB':
            j=read(W57/rel/'w56/f'/f'F-{c}-r60/job.json')
            for a,b in [('checkpoint','checkpoint_sha256'),('resume_checkpoint','resume_sha256')]:
                checks[c+'-'+a]=hashmatch(W57/j[a],j[b])
        for head in ['full','blind']:
            d=W57/'work/w57-r/pilot02'/head
            checks[head]=hashmatch(d/'policy-032.pt',read(d/'node-08192.json')['checkpoint_sha256'])
        if not all(v['matched'] for v in checks.values()):raise ValueError('Frozen checkpoint digest changed')
        frozen={}
        for c in 'ab':
            a=read(COMPACT/'attempts'/f'w57-m1-pr-{c}-original-e300-a01.json')
            wt=BASE/'MCP-RL-w57-p1-final';bad=[str(x) for x,y in a['active_sources'].items() if not hashmatch(wt/x,y)['matched']]
            if bad:raise ValueError('Frozen E300 scientific build changed: '+str(bad))
            frozen[c]=dict(source=a['source_commit'],controller=a['controller'],sources_and_builds=len(a['active_sources']),
                           status=a['status'],pid=a.get('pid'),timing_sha256=a.get('timing_checkpoint_sha256'))
        # Record the real retained layout only after checked unlinks, under the same Release lock.
        with (W57/'work/w57/publish.lock').open('a') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            for ep,(num,n,plans) in episodes.items():
                p=COMPACT/'delivery'/(ep+'.json');v=read(p)
                if v['status']!='verified':raise ValueError('Release not verified for '+ep)
                lr=v.setdefault('local_retention',{});key='H_process_20261124'
                before=lr.get(key,{}).get('previous_removed_mpc_files',lr.get('removed_mpc_files',0))
                lr['removed_mpc_files']=before+plans
                lr[key]=dict(files=num,logical_bytes=n,mpc_files=plans,previous_removed_mpc_files=before,
                       archive_catalog='H:/MPC+RL/00_catalog/migration-20261124-process/archive-index.sqlite',
                       complete_release=True)
                v.setdefault('originals_retained_at_verification',v['originals_retained']);v['originals_retained']=False
                tmp=p.with_name(p.name+'.storage-part');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
        sp=os.statvfs(BASE);pe=os.statvfs(PERSONAL)
        result=dict(checked_at=now(),removed_files=len(removed),removed_logical_bytes=sum(x[3] for x in removed),
             reclaimed_allocated_bytes=sum(x[4] for x in removed),allocated_by_volume=dict(allocs),regenerated_paths=regenerated,
             protected_path_unlinks=0,named_reference_trees=verified,four_E600=baselines,checkpoint_checks=checks,
             original_E300_builds=frozen,physics_at_check=recent_physics(),
             W57_dispatch=read(COMPACT/'dispatch/frozen-evaluations.json')['stage'],
             W58_stage=read(W58/'work/w58/stage.json')['stage'],changed_delivery_memberships=dict(episodes),
             scratch_free_bytes=sp.f_frsize*sp.f_bavail,personal_free_bytes=pe.f_frsize*pe.f_bavail,personal_free_nodes=pe.f_favail,**BG)
        save('postcheck.json',result);db.close();return result
