"""Publish closed CPU execution evidence from D57, reusing byte-identical originals."""
import json,sys,time
from pathlib import Path
from build import HERE,ROOT,WORK,digest
from experiment import OUT,STUDY,read,save,now
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/w61'));sys.path.insert(0,str(ROOT/'scripts/w61/learning'))
from publish_learning import setup,publisher,DELIVERY

def identical_store(stage):
    directory=WORK/stage
    mapping={};objects={}
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        h=digest(p);mapping[p.relative_to(directory).as_posix()]=dict(sha256=h,size=p.stat().st_size)
        objects.setdefault(h,p)
    idx=WORK/(stage+'-original-map.json')
    if not idx.exists():save(idx,dict(layout='Every original path references its verified content member objects/<sha256>',files=mapping))
    return [('original-index.json',idx),*[(f'objects/{h}',p) for h,p in objects.items()]]

def native_store():
    from native_compare import RAW,ID,GOLD,START,LEDGER
    gold=START/'final-originals'/GOLD;new=RAW/ID
    canonical=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/learning/delivery'/(GOLD+'.json')
    origin=read(canonical)
    if origin['status']!='verified':raise ValueError('Published baseline originals not verified')
    refs={};objects={}
    for p in sorted(new.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(new);h=digest(p);old=gold/rel
        if old.is_file() and old.stat().st_size==p.stat().st_size and digest(old)==h:
            refs[rel.as_posix()]=dict(sha256=h,size=p.stat().st_size,original_bundle_member='episode/'+rel.as_posix())
        else:
            refs[rel.as_posix()]=dict(sha256=h,size=p.stat().st_size,member='objects/'+h);objects.setdefault(h,p)
    idx=WORK/'native-original-map.json'
    if not idx.exists():save(idx,dict(baseline_receipt=origin,files=refs))
    return [('original-index.json',idx),('result.json',LEDGER/'ppo-s6001'/(ID+'.json')),
            ('resolved.json',LEDGER/'ppo-s6001'/(ID+'-resolved.json')),('physical.json',LEDGER/'ppo-s6001'/(ID+'-physical.json')),
            *[(f'objects/{h}',p) for h,p in objects.items()]]

def compact(paths,title):
    import subprocess
    files=[str(p.relative_to(ROOT)) for p in paths if p.is_file()]
    subprocess.run(['git','add','--sparse','--',*files],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*files],cwd=ROOT).returncode:
        subprocess.run(['git','commit','--only','-qm',title,'--',*files],cwd=ROOT,check=True)

def deliver(label,work):
    path=OUT/('release-'+label+'.json')
    if path.is_file():return
    ident=STUDY+'-'+label
    setup();publisher.publish(ident,work)
    src=DELIVERY/(ident+'.json')
    receipt=read(src)
    if receipt['status']!='verified':raise ValueError('Incomplete D57 release')
    save(path,dict(label=ident,status='verified',assets=receipt['assets'],original_index=str(src)))
    compact([path,src,*OUT.glob('*.json')], 'Publish measured W61 CPU speed and checked '+label+' originals')

def main():
    runtime=[('build/identity.json',WORK/'build/identity.json'),('build/exact-material.so',WORK/'build/exact-material.so'),
             ('protocol.json',ROOT/'experiments/terrain_memory/w61/learning/cpu-exact-v1.json'),
             ('sampled-stacks.json',OUT/'profile.json'),('saved-packet-contrast.json',OUT/'recorded-packet-parity.json')]
    runtime.extend(('source/'+p.name,p) for p in sorted(HERE.glob('*')) if p.is_file() and p.suffix in ('.py','.hh'))
    deliver('runtime',runtime)
    while True:
        for stage in ('entry','full','native-e30'):
            receipt=OUT/(stage+'.json');released=OUT/('release-'+stage+'.json')
            if receipt.is_file() and not released.is_file():
                paths=native_store() if stage=='native-e30' else identical_store(stage)
                deliver(stage,[('comparison.json',receipt),*paths])
        if (OUT/'production.json').exists():
            cert=read(OUT/'production.json');r=Path(cert['runtime']['driver']).parent
            if not (OUT/'release-installed.json').exists():
                deliver('installed',[('registration.json',OUT/'production.json'),*[(f'frozen/{p.name}',p) for p in sorted(r.iterdir()) if p.is_file() and not p.name.endswith('.pyc')]])
            compact([OUT/'production.json',OUT/'status.json'],'Record the qualified CPU execution selections')
        if (OUT/'release-native-e30.json').exists() and (OUT/'release-full.json').exists():
            if (OUT/'production.json').is_file() or not read(OUT/'full.json').get('qualifies_for_stage',False):break
        if (OUT/'entry.json').exists() and not read(OUT/'entry.json').get('qualifies_for_stage',False):break
        time.sleep(12)
if __name__=='__main__':main()
