"""Render recorded E trajectories off the physical pool and verify each displayed frame clock."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
from fractions import Fraction
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import evaluate_learning as native
from campaign import RUN, SMALL
from evaluation_physics import INTENTS, POOL, QUEUE, all_trained_curves_closed
from hybrid import ROOT
from s1_w59_learner_critic import atomic_json, file_digest

DONE=QUEUE/'rendered'
GPU_HEADROOM_MIB=2048
IO_MAX_SOME_AVG10=12.
DRAW_DUTY=.32


def system_load():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    free,utilization=map(int,gpu.split(','))
    fields=Path('/sys/fs/cgroup/io.pressure').read_text().splitlines()[0].split()
    io=next(float(s.split('=')[1]) for s in fields if s.startswith('avg10='))
    return dict(free_gpu_mib=free,gpu_utilization_percent=utilization,io_some_avg10=io,free_scratch_gib=shutil.disk_usage(QUEUE).free/2**30)


class Pace:
    def __init__(self):
        self.next_measure=0.;self.draw_s=0.;self.resource_wait_s=0.;self.frames=0;self.samples=[]

    def resources(self):
        now=time.monotonic()
        if now<self.next_measure:return
        while True:
            r=system_load();self.samples.append(r)
            if r['free_gpu_mib']>=GPU_HEADROOM_MIB and r['io_some_avg10']<=IO_MAX_SOME_AVG10 and r['free_scratch_gib']>=1.:break
            before=time.monotonic();time.sleep(5);self.resource_wait_s+=time.monotonic()-before
        self.next_measure=time.monotonic()+5

    def before_draw(self):
        self.resources();self.begin=time.monotonic()

    def after_draw(self):
        dt=time.monotonic()-self.begin;self.frames+=1;self.draw_s+=dt
        time.sleep(dt*(1/DRAW_DUTY-1))

    def facts(self):
        return dict(frames=self.frames,draw_s=self.draw_s,resource_wait_s=self.resource_wait_s,
            draw_duty_limit=DRAW_DUTY,minimum_gpu_headroom_mib=GPU_HEADROOM_MIB,
            maximum_io_some_avg10=IO_MAX_SOME_AVG10,resource_samples=len(self.samples),
            free_gpu_mib_min=min(s['free_gpu_mib'] for s in self.samples),
            io_some_avg10_max=max(s['io_some_avg10'] for s in self.samples))


def metadata(movie,expected,fps):
    args=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
       'stream=nb_frames,r_frame_rate,avg_frame_rate,duration,time_base:frame=best_effort_timestamp','-of','json',str(movie)]
    probed=json.loads(subprocess.check_output(args));stream=probed['streams'][0]
    pts=[int(f['best_effort_timestamp']) for f in probed['frames']]
    tb=Fraction(stream['time_base'])
    if len(pts)!=expected or int(stream['nb_frames'])!=expected or Fraction(stream['avg_frame_rate'])!=fps:
        raise ValueError(f'Encoded frame count or nominal rate differs from recorded grid: {movie}')
    error=max((abs(Fraction(p-pts[0])*tb-Fraction(i,fps)) for i,p in enumerate(pts)),default=Fraction(0))
    if not pts or pts[0]!=0 or not all(y>x for x,y in zip(pts,pts[1:])) or error>tb:
        raise ValueError(f'Presentation clock differs from the complete physical frame grid: {movie}')
    return stream,dict(first=pts[0],last=pts[-1],time_base=stream['time_base'],frames=len(pts),
                 max_grid_error_s=float(error),all_strictly_increasing=True,
                 sha256=hashlib.sha256(np.asarray(pts,dtype='<i8').tobytes()).hexdigest())


@contextmanager
def paced_curve(pace):
    base=mujoco.Renderer
    class PacedRenderer(base):
        def render(self,*args,**kwargs):
            pace.before_draw();image=super().render(*args,**kwargs);pace.after_draw();return image
    mujoco.Renderer=PacedRenderer
    try:yield
    finally:mujoco.Renderer=base


def encode(item,destination,pace):
    raw=Path(item['raw']);result=native.read(Path(item['physical_result']))
    source=raw/'trajectory.npz'
    pace.resources()
    if item['kind']=='native':
        import render_s1_go2_r12 as renderer
        renderer.SMALL=Path(item['physical_result']).parent;renderer.LARGE=raw.parent
        renderer.render(item['id'],output_dir=destination,pace=pace)
        with np.load(source) as z:t=z['time']
        expected=len(np.unique(np.r_[np.searchsorted(t,np.arange(0,t[-1],1/30)),len(t)-1]))
        movie=destination/'full-mission-material-overlay.mp4';fps=30
        marker='video-asset.json'
    else:
        import render as renderer
        for field in ('identity.json','result.json','trajectory.npz'):
            (destination/field).symlink_to((raw/field).resolve())
        with paced_curve(pace):renderer.render(destination)
        expected=len(np.arange(0,result['physical_s']-1.e-8,1./25))
        movie=destination/'complete.mp4';fps=25;marker='video.json'
    probe,pts=metadata(movie,expected,fps)
    return movie,marker,probe,pts


def lease_end(item):
    p=POOL/(item['lease']+'.json')
    if not p.is_file():return None
    r=native.read(p)
    return r if r.get('state')=='ended' else None


def render_one(item,proof=False):
    lease=lease_end(item)
    if not lease:return None
    if os.environ.get('W61_SLOTS'):raise RuntimeError('Off-pool rendering must run without a physical lease')
    raw=Path(item['raw']);receipt=Path(item['physical_receipt'])
    started=time.time();pace=Pace()
    with tempfile.TemporaryDirectory(prefix='w61-media-',dir=raw.parent) as tmp:
        d=Path(tmp);movie,marker,probe,pts=encode(item,d,pace);sha=file_digest(movie)
        original=raw/movie.name
        rec=dict(id=item['id'],kind=item['kind'],lease=item['lease'],lease_state=lease['state'],lease_ended=lease['ended'],
             lease_exit_code=lease['exit_code'],physical_receipt=str(receipt),source=str(raw/'trajectory.npz'),
             source_sha256=file_digest(raw/'trajectory.npz'),actor_sha256=item.get('actor_sha256'),
             render_pid=os.getpid(),physical_slots=[],render_started_unix=started,render_closed_unix=time.time(),
             render_wall_s=time.time()-started,video=str(original),bytes=movie.stat().st_size,video_sha256=sha,
             true_fps_pts=probe,pts=pts,pace=pace.facts())
        if started<datetime.fromisoformat(lease['ended']).timestamp():raise ValueError('Draw preceded the released physical lease')
        if proof:
            rec['retained_original']=str(original)
            rec['retained_original_sha256']=file_digest(original)
            rec['original_stream'],rec['original_pts']=metadata(original,pts['frames'],int(Fraction(probe['avg_frame_rate'])))
            rec['same_frame_count_and_pts']=rec['original_pts']['sha256']==pts['sha256']
            rec['byte_identical']=rec['retained_original_sha256']==sha
            # Compare all displayed images in presentation order independently of the encoder/container bytes.
            ff=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-threads','1','-i',str(original),
                 '-threads','1','-i',str(movie),'-filter_complex_threads','1','-lavfi','ssim','-f','null','-'],
                 stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,check=True)
            measure=re.search(r'All:([0-9.]+)',ff.stderr)
            rec['decoded_full_video_mean_SSIM']=float(measure.group(1))
            return rec
        m=native.read(d/marker)
        field='path' if item['kind']=='native' else 'file';m[field]=str(original)
        os.replace(movie,original);atomic_json(raw/marker,m)
        if item['kind']=='native':
            physical=native.read(receipt)
            atomic_json(receipt.parent/(item['id']+'-summary.json'),dict(physical,true_fps_pts=probe,video_sha256=sha,
                media={k:rec[k] for k in ('render_started_unix','render_closed_unix','render_wall_s','lease_ended','pts')}))
        atomic_json(DONE/(item['id']+'.json'),rec)
        return rec


def worker():
    QUEUE.mkdir(exist_ok=True,parents=True)
    with (QUEUE/'renderer-owner.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            pending=sorted((p for p in INTENTS.glob('*.json') if not (DONE/p.name).is_file()),key=lambda p:p.stat().st_mtime)
            for p in pending:
                item=native.read(p)
                failed=QUEUE/'render-errors'/p.name
                stamp=[p.stat().st_mtime_ns,Path(__file__).stat().st_mtime_ns]
                if failed.exists() and native.read(failed).get('input_revision')==stamp:continue
                try:
                    with (QUEUE/'render.lock').open('a') as rendering:
                        fcntl.flock(rendering,fcntl.LOCK_EX)
                        rec=render_one(item)
                    if rec:print(json.dumps({k:rec[k] for k in ('id','lease_ended','render_wall_s','bytes')}),flush=True)
                except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as err:
                    atomic_json(failed,dict(id=item['id'],input_revision=stamp,source=item['source_file'],
                        error=repr(err),raw_retained=True,failed_unix=time.time()))
                    print('render error',item['id'],repr(err),flush=True)
            final_status=RUN/'parallel-final/live.json'
            if (not pending and all_trained_curves_closed('-media') and final_status.is_file()
                    and native.read(final_status).get('completed')):return
            time.sleep(3)


def replay_proof():
    final='w61-e-final-ppo-s6001-v000-a-e30-nominal'
    head='ppo-s6001';root=native.LEDGER/head
    old=native.read(root/(final+'-summary.json'))
    dev=native.read(SMALL.parent/'interface-v1.json')['episodes'][0];raw=Path(dev['raw'])
    items=[dict(id=final,lease=final,kind='native',raw=str(native.DATA/final),physical_receipt=str(root/(final+'.json')),
                physical_result=str(root/(final+'.json')),actor_sha256=old['actor_sha256']),
           dict(id=raw.name,lease='w61-e-dev-v1-'+dev['variant'],kind='curve',raw=str(raw),physical_receipt=str(raw/'result.json'),physical_result=str(raw/'result.json'))]
    evidence=[]
    with (QUEUE/'render.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        for item in items:
            rec=render_one(item,proof=True)
            if rec is None:raise ValueError('Missing closed original physical lease for replay: '+item['lease'])
            evidence.append(rec)
    proof=SMALL/'media-boundary-recorded-replay.json'
    atomic_json(proof,dict(evidence=evidence,physical_runs_started=0,training_used=0,
        frozen_evaluator_sha256=file_digest(native.SELF),registered_evaluator_sha256=native.read(native.LEDGER/'parallel-plan.json')['source_evaluator_sha256']))
    print(proof,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['serve','replay-proof']);a=p.parse_args()
    (worker if a.action=='serve' else replay_proof)()
