"""Deliver only complete real F queries, their videos, physical logs and optimizer records."""
import argparse,fcntl,hashlib,json,os,subprocess,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from w61.mpc.video import render,digest
from w61.joint.plant import ROOT,CONTRACT,write

RUN='w61-f-ac-joint-real-v2'
WORK=ROOT/'work/w61/joint'
DATA=WORK/RUN
COMPACT=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/joint'/RUN
TAG='tm-s1-go2-local-readiness-r12-w61'
REPO='kaiwen123-yang/MCP-RL'

def movie(folder):
    video=folder/'full-trajectory.mp4'
    if (folder/'video.json').exists():return
    obj=json.loads((folder/'result.json').read_text())
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp:
        proxy=Path(temp)
        with np.load(folder/'trajectory.npz') as data:
            a=data['state']
            if len(a)==0:return
            np.savez(proxy/'trajectory.npz',time=a[:,0],qpos=a[:,1:20],qvel=a[:,20:38])
        write(proxy/'result.json',dict(role=folder.parent.parent.name,case=str(obj['case']['specimen']),physical_s=obj['physical_s']))
        meta=render(proxy,video)
    write(folder/'video.json',meta)


def assets():
    args=['gh','api',f'repos/{REPO}/releases/tags/{TAG}','--jq','.id']
    rid=subprocess.check_output(args,text=True).strip()
    rows=subprocess.check_output(['gh','api','--paginate',f'repos/{REPO}/releases/{rid}/assets?per_page=100',
              '--jq','.[] | [.name,.state,.size,.digest,.browser_download_url]|@tsv'],text=True)
    return {v[0]:v[1:] for v in (l.split('\t') for l in rows.splitlines())}


def send(label,folders,records,remote):
    name='w61-f-'+label+'.zip'
    receipt=COMPACT/'originals'/(label+'.json')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():return
    package=DATA/'release'/name
    package.parent.mkdir(parents=True,exist_ok=True)
    for folder in folders:movie(folder)
    if not package.exists():
        tmp=package.with_suffix('.pending')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for folder in folders:
                for p in sorted(folder.rglob('*')):
                    if p.is_file():z.write(p,folder.name+'/'+p.relative_to(folder).as_posix())
            for node in records:z.write(node,'training/'+node.name)
            z.writestr(zipfile.ZipInfo('trajectory-columns.json',date_time=(2000,1,1,0,0,0)),json.dumps(dict(state=['time_s','qpos[19]','qvel[18]','measured_GRF_world_N[4,3]','contact[4]','position_command[12]','planned_force_world_N[4,3]'],cadence_Hz=500,material_boundary_rate_Hz=5000,trajectory_order='time-major',energy='absolute mechanical joint work')))
        tmp.replace(package)
    n=package.stat().st_size;h=digest(package)
    if name not in remote:
        subprocess.run(['gh','release','upload',TAG,str(package),'-R',REPO],check=True)
        remote.update(assets())
    state,size,sha,url=remote[name]
    if (state,int(size),sha)!=('uploaded',n,'sha256:'+h):raise ValueError('F original mismatch '+name)
    write(receipt,dict(name=name,bytes=n,sha256=h,url=url,source=str(package),
        episodes=[dict(path=str(p),result=json.loads((p/'result.json').read_text())['reason'],
                       video=json.loads((p/'video.json').read_text())) for p in folders]))


def main(once):
    spec=json.loads(CONTRACT.read_text())['study']
    with (WORK/'delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        remote=assets()
        while True:
            dev=WORK/'f-forefoot-cost-v1'
            if (dev/'result.json').exists():send('cost-development-v1',[dev],[],remote)
            for s in spec['training_seeds']:
                for arm in spec['arms']:
                    root=DATA/arm/str(s)
                    for n in range(spec['rounds']):
                        node=root/f'round-{n:02d}.json'
                        if node.exists():send(f'ac2-{arm}-{s}-r{n:02d}',[root/f'r{n:02d}-plus',root/f'r{n:02d}-minus'],[node],remote)
                    ev=root/'evaluation.json'
                    if ev.exists():send(f'ac2-{arm}-{s}-unseen',[root/'eval-0',root/'eval-1'],[ev],remote)
            for arm in spec['fixed_pair_controls']:
                roots=[DATA/'factorial'/str(k)/arm for k in range(2)]
                if all((r/'result.json').exists() for r in roots):send('ac2-factorial-'+arm,roots,[],remote)
            if once or all((COMPACT/'originals'/('ac2-factorial-'+a+'.json')).exists() for a in spec['fixed_pair_controls']):return
            time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true')
    main(p.parse_args().once)
