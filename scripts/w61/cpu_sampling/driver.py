"""Explicit launcher for new, certified W61 physics leases; the controller and update are unchanged."""
import json,os,runpy,sys
from pathlib import Path
from entry import ROOT,SCIENCE,FIXED,install,sha

HERE=Path(__file__).resolve().parent
SCRIPTS=ROOT/'scripts'
for path in (SCRIPTS,SCRIPTS/'w61',SCIENCE):sys.path.insert(0,str(path))

def identity():
    source=Path(os.environ.get('W61_EXACT_BUILD',str(FIXED/'build/identity.json')))
    build=json.loads(source.read_text())
    if sha(build['binary'])!=build['sha256']:raise ValueError('CPU execution library differs from its frozen registration')
    return dict(name='W61 exact CPU packet reuse',binary=build['binary'],sha256=build['sha256'],
        registered_build=str(source),base_collector_sha256=sha(SCIENCE/'collect.py'),source_wrapper_sha256=sha(HERE/'entry.py'))

def bind_native(native,build):
    install(build)
    previous=native.case_identity
    def case_identity(*args,**kwargs):
        data,_=previous(*args,**kwargs)
        data.update(w61_material_execution=dict(implementation=build['name'],binary_sha256=build['sha256']))
        return data,native.stable_digest(data)
    native.case_identity=case_identity

def main():
    build=identity()
    if sys.argv[1]=='collect':
        # Solver recovery spawns this file as its child; the original event/return code
        # records its real successor before any data from the new episode can be used.
        install(build)
        src=SCRIPTS/'w61/resume_learning_batch.py'
        sys.argv[0]=str(src);runpy.run_path(str(src),run_name='__main__');return
    original=Path(sys.argv[1]).resolve();args=sys.argv[2:]
    import campaign
    from s1_w59_learner_critic import file_digest
    validator=campaign.verified_runtime
    def backend_revision(manifest):
        return validator(manifest)+'+CPU-exact-'+build['sha256'][:12]
    campaign.verified_runtime=backend_revision
    campaign.COLLECT=HERE/'entry.py'
    if original==Path(campaign.SELF).resolve() and args[0]=='paired':
        method,seed,version,role=args[1:]
        return campaign.paired(method,int(seed),int(version),role)
    if original==SCRIPTS/'w61/resume_learning_batch.py' and args[0]=='resume':
        import resume_learning_batch as recovery
        # case_continuation uses this module's filename for every newly required child.
        recovery.__file__=str(Path(__file__).resolve())
        return recovery.resume(args[1],int(args[2]),int(args[3]))
    if original==SCRIPTS/'w61/evaluation_physics.py':
        import evaluation_physics as physics
        mode,method,seed,version,*case=args
        if mode=='curve':
            return physics.physical_curve(method,int(seed),int(version))
        if mode=='native':
            bind_native(physics.native,build)
            physics.physical_native(method,int(seed),int(version),*case)
            return 0
    raise ValueError(f'Unregistered W61 CPU execution command: {original} {args[:1]}')
if __name__=='__main__':raise SystemExit(main())
