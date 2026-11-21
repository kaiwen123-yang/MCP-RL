"""Deliver interrupted W56 r58 E300 recordings without starting physics."""
from pathlib import Path
import bisect
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw

TREE=Path(__file__).resolve().parents[1]
F=TREE.parent/'MCP-RL-w56-f'
M=TREE.parent/'MCP-RL-w56-material'
RUN='tm-s1-go2-local-readiness-r12'
REL=Path('results/terrain_memory/simulation')/RUN
RAW=Path('outputs/terrain_memory/simulation')/RUN
TAG=RUN+'-w56'
STAGING=F/RAW/'w56-delivery'
sys.path.insert(0,str(F/'scripts'))
from publish_s1_go2_w55 import prepare
import publish_s1_go2_r12 as transport

def save(path,value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def render(ep,label,out):
    records=[json.loads(l) for l in (ep/'w55-f/mpc-solves.jsonl').open()]
    clocks=np.array([x['time_s'] for x in records]);interval=np.diff(clocks);maximum=float(max(interval))
    if clocks[0]!=0 or np.any(interval<=0) or maximum>.0200001:raise ValueError('Saved pose series is discontinuous')
    if [r['call'] for r in records]!=list(range(1,len(records)+1)):raise ValueError('Missing MPC calls')
    period=.04;times=list(np.arange(0,clocks[-1]-1.e-8,period))+[clocks[-1]]
    meta=dict(source='w55-f/mpc-solves.jsonl time_s and corresponding MPC input qpos',pose_samples=len(clocks),
              first_mpc_s=float(clocks[0]),last_mpc_s=float(clocks[-1]),median_pose_interval_s=float(np.median(interval)),
              min_pose_interval_s=float(min(interval)),max_pose_interval_s=maximum,video_fps=25,frames=len(times),
              display_sampling='closest recorded MPC pose to each 40 ms presentation time plus final recorded pose',
              task_length_s=300.,original_task_result=None,partial_recording=True)
    if out.exists():return meta
    m=mujoco.MjModel.from_binary_path(str(ep/'model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=848;m.vis.global_.offheight=480
    # Goal mocap was not flushed. Hide its appearance in this qpos-only replay.
    for geom in range(m.ngeom):
        body=m.geom_bodyid[geom]
        if m.body_mocapid[body]>=0:m.geom_rgba[geom,3]=0
    cam=mujoco.MjvCamera();cam.azimuth=125;cam.elevation=-23;cam.distance=3.7
    tmp=out.with_name(out.stem+'.partial.mp4')
    with mujoco.Renderer(m,height=480,width=848) as rr,imageio.get_writer(tmp,fps=25,codec='libx264',quality=7,
                       ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
        for t in times:
            j=min(bisect.bisect_left(clocks,t),len(clocks)-1)
            if j and abs(clocks[j-1]-t)<abs(clocks[j]-t):j-=1
            p=ep/'w55-f'/records[j]['file']
            with np.load(p) as z:d.qpos[:]=z['qpos']
            d.time=clocks[j];mujoco.mj_forward(m,d);cam.lookat[:]=[d.qpos[0],d.qpos[1],.24]
            rr.update_scene(d,camera=cam)
            g=rr.scene.geoms[rr.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.7,.6,.0002]),np.array([1.5,0,.0002]),np.eye(3).ravel(),np.array([.15,.45,.8,.22],np.float32));rr.scene.ngeom+=1
            im=Image.fromarray(rr.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,848,55),fill=(15,22,29))
            draw.text((10,5),f'{label}  original E300  |  user-stopped run, saved pose prefix',fill='white')
            draw.text((10,21),f'Physical time {d.time:.3f} s   |   saved 0 - {clocks[-1]:.3f} s   |   25 fps from measured MPC-input pose (~50 Hz)',fill='white')
            draw.text((10,37),f'Base x {d.qpos[0]:+.3f} m   y {d.qpos[1]:+.3f} m   z {d.qpos[2]:.3f} m   |   blue = TM02 footprint',fill='white')
            writer.append_data(np.asarray(im))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                   'stream=nb_frames,width,height,avg_frame_rate,duration','-of','json',str(tmp)]))['streams'][0]
    if int(probe['nb_frames'])!=len(times) or probe['avg_frame_rate']!='25/1':raise RuntimeError('Rendered frame clock differs')
    tmp.replace(out);return meta

def already_delivered(case):
    known={}
    for level in ('E30','E60'):
        p=F/REL/'w56/delivery'/f'f-{case.lower()}-r58-{level.lower()}'
        receipt=read(p/'delivery.json')
        uploaded={a['name'] for a in receipt['assets'] if a.get('remote_state')=='uploaded'}
        for r in read(p/'sources.json'):
            if r['asset'] in uploaded:known.setdefault(r['sha256'],(r['asset'],r.get('member')))
    return known

def coverage(ep,name,pattern,field):
    paths=sorted(ep.glob(pattern))
    values=[]
    for p in (paths[0],paths[-1]):
        with np.load(p) as z:
            a=z[field];values.append((float(a[0] if a.ndim==1 else a[0,0]),float(a[-1] if a.ndim==1 else a[-1,1])))
    return dict(measurement=name,first_s=values[0][0],last_s=values[-1][1],chunks=len(paths))

def prepare_case(case):
    label=f'F-{case}-r58';group=label.lower()+'-e300';stage=read(M/REL/'w56/f'/label/'E300.json')
    if stage['exit_code']!=-15:raise ValueError('Interruption differs from recorded stop')
    dest=TREE/REL/'w56/delivery'/group;raw=M/RAW;ep=raw/stage['attempt'];job=raw/'jobs'/stage['job']
    sg=STAGING/group;sg.mkdir(exist_ok=True,parents=True);dest.mkdir(exist_ok=True,parents=True)
    if (dest/'manifest.json').is_file():return group
    video=sg/f'w56-{group}-recorded-pose-prefix.mp4'
    video_meta=render(ep,label,video)
    video_meta.update(physical_substeps=coverage(ep,'2 ms controls have ten 0.2 ms physics substeps','physical-*.npz','physical'),
                      material=coverage(ep,'0.2 ms material foot windows','material-*.npz','foot'),
                      wbc=coverage(ep,'500 Hz WBC control samples','w55-f/control-*.npz','time'),
                      stop_signal=15,stage_reason=stage['reason'],attempt=stage['attempt'],
                      unresolved_tail='Pose playback ends at the last flushed MPC state; the final unflushed high-rate arrays and a completed E300 result are unavailable.')
    save(dest/'coverage.json',video_meta)
    snapshot=sg/'job-snapshot';snapshot.mkdir(exist_ok=True)
    for n in ('job.json','execution-contract.json',stage['attempt']+'.log','E300-runner.log','E300-runtime.json'):
        p=job/n
        if p.is_file() and not (snapshot/n).exists():(snapshot/n).symlink_to(p)
    with (job/'resources.jsonl').open() as src,(snapshot/'resources.jsonl').open('w') as dst:
        for line in src:
            if json.loads(line)['root_pid']==stage['pid']:dst.write(line)
    save(snapshot/'stage.json',stage)
    # Preserve logical episode paths for reconstruction while avoiding second
    # payload copies. Existing released E30/E60 members remain addressable.
    known=already_delivered(case);sources={};reuse=[];same={}
    payload=sg/'payload';payload.mkdir(exist_ok=True)
    for source,top in [(ep,'episode'),(snapshot,'job'),(raw/'evaluation-policy'/stage['attempt'],'policy')]:
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=Path(top)/p.relative_to(source);h=sha(p);key=str(rel)
            item=dict(file=key,sha256=h,size=p.stat().st_size)
            if h in known:
                asset,member=known[h];reuse.append(dict(item,asset=asset,member=member))
            elif h in same:reuse.append(dict(item,same_content_as=same[h]))
            else:
                same[h]=key;link=payload/rel;link.parent.mkdir(parents=True,exist_ok=True)
                if not link.exists():link.symlink_to(p.resolve())
                sources[key]=item
    with (STAGING/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prepare(group,[payload],sg,prefix='w56',staging_root=STAGING,tag=TAG)
        rows=read(sg/'sources.json');manifest=read(sg/'manifest.json')
    # The detailed per-file map is a Release item, while the Git index is compact.
    for r in rows:
        k=str(Path(r['member']).relative_to('payload'))
        sources[k].update(asset=r['asset'],member=r['member'])
    catalog=sg/f'w56-{group}-original-member-map.json.gz'
    with gzip.open(catalog,'wt',encoding='utf-8') as stream:json.dump(dict(source_attempt=stage['attempt'],new=list(sources.values()),reused=reuse),stream,separators=(',',':'))
    for p,kind in ((video,'recorded_pose_prefix_video'),(catalog,'original_member_map')):
        manifest.append(dict(path=str(p.relative_to(F)),name=p.name,category=kind,size=p.stat().st_size,sha256=sha(p),release_tag=TAG))
    save(dest/'manifest.json',manifest)
    save(dest/'sources.json',dict(episode=str(ep),job=str(job),policy=str(raw/'evaluation-policy'/stage['attempt']),
                new_original_files=len(sources),referenced_original_files=len(reuse),catalog_asset=catalog.name,
                resource_rows=sum(1 for _ in (snapshot/'resources.jsonl').open())))
    return group

def publish(group):
    sg=STAGING/group;dest=TREE/REL/'w56/delivery'/group
    with (STAGING/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (sg/'index.json').exists():save(sg/'index.json',{})
        transport.ROOT=F;transport.SMALL=sg;transport.RUN=TAG
        for action in ('upload','finish'):
            sys.argv=[__file__,action,'--manifest',str(dest/'manifest.json'),'--server-auth','--upload-workers','1','--no-wall-deadline']
            transport.main()
        remote={a['name']:a for a in read(sg/'index.json')['publication']['assets']}
        assets=[]
        for m in read(dest/'manifest.json'):
            a=remote[m['name']]
            if (a['remote_state'],a['status'],a['size'],a['remote_digest'])!=('uploaded','uploaded',m['size'],'sha256:'+m['sha256']):
                raise ValueError(f'Incomplete Release receipt for {m["name"]}')
            assets.append({k:a[k] for k in ('name','size','url','remote_state','remote_asset_id','remote_digest','remote_verified_unix_s')})
        c=read(dest/'coverage.json');originals=read(dest/'sources.json')
        save(dest/'delivery.json',dict(status='published',partial_recording=True,level='E300',attempt=c['attempt'],last_mpc_s=c['last_mpc_s'],
             original_result=None,stage_status='user_stopped',formal_pass=None,original_files=originals['new_original_files'],
             referenced_original_files=originals['referenced_original_files'],assets=assets,verified_unix_s=time.time(),
             deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
    args=['git','-C',str(TREE)]
    subprocess.run(args+['add','--',str(dest)],check=True)
    if subprocess.run(args+['diff','--cached','--quiet','--',str(dest)]).returncode:
        subprocess.run(args+['commit','--only','-m','Verify interrupted '+group+' original recordings and continuous pose prefix','--',str(dest)],check=True)
    print(group,'verified',len(assets),'assets',flush=True)

if __name__=='__main__':
    groups=[prepare_case(case) for case in ('A','B')]
    for group in groups:publish(group)
