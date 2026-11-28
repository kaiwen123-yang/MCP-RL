"""Deliver new original-M1 maneuver evidence after each terminal on-policy trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from bd_route_evidence import collect

REPO='kaiwen123-yang/MCP-RL'
TAG='tm-s1-go2-local-readiness-r12-w61'


def asset_info(name):
    response=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{TAG}']))
    return next((a for a in response['assets'] if a['name']==name),None)


def publish(args,name):
    receipt=args.raw/'episodes'/(name+'.json')
    while not receipt.is_file():
        time.sleep(15)
    out=args.output/name
    if (out/'release.json').exists():return
    if not (out/'result.json').exists():
        collect(SimpleNamespace(name=name,root=args.raw,out=args.output,protocol=args.protocol,renderer=args.renderer))
    staging=args.raw/'release'/name;staging.mkdir(parents=True,exist_ok=True)
    # A 16-second window, including raw WBC matrices, stays below one GHA release asset.
    # The long directed QP follow-up uses streaming 700 MiB members as well.
    marker=staging/'parts.json'
    if not marker.exists():
        names=[f'raw/{name}',f'jobs/{name}',f'episodes/{name}.json',f'episodes/{name}-resolved.json',f'streams/{name}']
        tar=subprocess.Popen(['tar','-C',str(args.raw),'-cf','-',*names],stdout=subprocess.PIPE)
        gzip=subprocess.Popen(['gzip','-1'],stdin=tar.stdout,stdout=subprocess.PIPE);tar.stdout.close()
        split=subprocess.run(['split','-b','700M','-d','-a','2','-',str(staging/(name+'-originals.tar.gz.part-'))],stdin=gzip.stdout)
        gzip.stdout.close()
        if any([split.returncode,gzip.wait(),tar.wait()]):
            raise RuntimeError('Originals packing failed')
        marker.write_text(json.dumps(sorted(p.name for p in staging.glob('*part-*')))+'\n')
    receipts=[]
    for part in json.loads(marker.read_text()):
        p=staging/part
        with p.open('rb') as f:expected='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
        remote=asset_info(part)
        if remote is None:
            subprocess.run(['gh','release','upload',TAG,'--repo',REPO,str(p)],check=True)
            remote=asset_info(part)
        assert (remote['state'],remote['size'],remote['digest'])==('uploaded',p.stat().st_size,expected)
        receipts.append({k:remote[k] for k in ('id','name','digest','size','browser_download_url','state')})
    (out/'release.json').write_text(json.dumps(dict(parts=receipts,restore='Concatenate parts by suffix and unpack the tar.gz.'),indent=2)+'\n')
    subprocess.run(['git','add','--sparse',str(out)],check=True)
    subprocess.run(['git','commit','-m',f'Publish {name} evidence and verified native originals'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','output','protocol','renderer'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('names',nargs='+');a=p.parse_args()
    for n in a.names:publish(a,n)
