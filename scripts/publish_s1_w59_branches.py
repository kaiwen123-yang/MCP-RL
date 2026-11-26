"""Publish the eight physical continuations for each registered state once the state is complete."""
import argparse
import hashlib
import json
from pathlib import Path

from publish_s1_w59 import STATE, RAW, WORK, ensure_release, publish
from publish_s1_go2_w54 import write


def publish_source(state,role,actions):
    prefix=f'w59-action-{role.split("_")[0]}-{state["case"].lower()}-r{state["request_id"]:05d}'
    completed=[]
    for axis in actions:
        name=prefix+'-'+axis+'-a01'
        source=STATE/'attempts'/(name+'.json')
        if not source.exists():
            return False
        receipt=json.loads(source.read_text())
        if not receipt.get('ended_unix_s'):
            return False
        completed.append((name,receipt))
    label=prefix+'-branches'
    delivered=STATE/'delivery'/(label+'.json')
    if delivered.exists() and json.loads(delivered.read_text())['status']=='verified':
        return True
    paths=[]
    shared={}
    per_branch={}
    for name,receipt in completed:
        paths.append((name+'/job',RAW/'jobs'/name))
        if (RAW/name).is_dir():
            # MJB and XML/mesh model copies often dominate a two-second continuation.
            # For these only, store identical bytes once per eight-way state, with a full path mapping.
            model_files=sorted((RAW/name/'model').rglob('*'))
            if (RAW/name/'model.mjb').is_file():
                model_files.append(RAW/name/'model.mjb')
            links={}
            for f in model_files:
                if not f.is_file():
                    continue
                with f.open('rb') as stream:
                    digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if digest not in shared:
                    shared[digest]=('shared-model/'+digest,f)
                links[str(f.relative_to(RAW/name))]=dict(member=shared[digest][0],size=f.stat().st_size,sha256=digest)
            per_branch[name]=links
            for f in sorted((RAW/name).iterdir()):
                if f.name not in ('model','model.mjb'):
                    paths.append((name+'/episode/'+f.name,f))
        for f in (STATE/'episodes'/(name+'.json'),STATE/'episodes'/(name+'-resolved.json'),STATE/'attempts'/(name+'.json')):
            if f.exists():
                paths.append((name+'/receipt/'+f.parent.name+'/'+f.name,f))
    mapping=WORK/'release-manifests'/(label+'.json')
    write(mapping,dict(state=state['snapshot_id'],branches=per_branch,registered=len(actions),
        scope='Each branch keeps its original path mapping. Shared-model objects are byte-identical by local SHA-256.'))
    paths+=list(shared.values())+[('model-paths.json',mapping)]
    publish(label,paths)
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--role',choices=('development','sealed_admission'),required=True)
    args=parser.parse_args()
    reg=json.loads(args.registry.read_text())
    ensure_release()
    total=0
    for source in reg['states'][args.role]:
        total+=int(publish_source(source,args.role,reg['signed']))
    print('verified states',total,'of',len(reg['states'][args.role]),flush=True)
