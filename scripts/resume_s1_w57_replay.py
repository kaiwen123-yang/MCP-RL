"""Continue the active W57 B movie from its verified, already-encoded prefix."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo
import numpy as np
import render_s1_go2_r12 as renderer
from publish_s1_go2_w54 import write

ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
WORK = ROOT/'work/w57'
COMPACT = ROOT/'results'/REL/'w57'
NAME = 'w57-m1-pr-b-original-e600-a01'
OUT = ROOT/'outputs'/REL/NAME
STAGE = WORK/'e600-b-video-continuation'
PROOF = COMPACT/'dispatch/e600-b-replay-continuation.json'
MOVIE = 'full-mission-material-overlay.mp4'
FRAME_BYTES = 1280*720*3
BG = dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def probe(path):
    cmd=['ffprobe','-v','error','-show_entries',
         'stream=codec_name,width,height,r_frame_rate,time_base,nb_frames,extradata_hash:format=duration',
         '-show_data_hash','sha256','-of','json',str(path)]
    return json.loads(subprocess.check_output(cmd))


def packets(path):
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_data_hash','sha256',
         '-show_entries','packet=pts,data_hash','-of','json',str(path)]
    return json.loads(subprocess.check_output(cmd))['packets']


def process_state(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return None


def reader_bytes(pid):
    values=dict(l.split(':',1) for l in (Path('/proc')/str(pid)/'io').read_text().splitlines())
    return int(values['rchar'])


class DrawBudget:
    """Give the incumbent compute job priority; pace only replay drawing on a busy GPU."""
    def __init__(self):
        self.next_sample=0.;self.next_frame=0.;self.busy=True
        self.spent=0.;self.started=0.;self.frames=0

    def before_draw(self):
        at=time.monotonic()
        if at>=self.next_sample:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
            owner=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).split()
            foreign=any(p.isdigit() and int(p)!=os.getpid() for p in owner)
            self.busy=foreign and (int(lines[0])>=85 or int(lines[1])<6000)
            print(now(),'shared GPU',lines,'foreign compute',foreign,'paced',self.busy,'tail frames',self.frames,flush=True)
            self.next_sample=at+25
        if self.busy and self.next_frame>time.monotonic():
            time.sleep(self.next_frame-time.monotonic())
        self.started=time.monotonic()

    def after_draw(self):
        elapsed=time.monotonic()-self.started
        self.frames+=1;self.spent+=elapsed
        # Synchronous GPU readback includes queueing; it is a conservative demand estimate.
        self.next_frame=self.started+max(1/15.,min(1/1.67,elapsed/.035)) if self.busy else 0.


def close_at_frame_boundary(parent,encoder,proof,prefix):
    old=Path('/proc')/str(parent);child=Path('/proc')/str(encoder)
    args=(old/'cmdline').read_bytes().split(b'\0');ff=(child/'cmdline').read_bytes().split(b'\0')
    movie=OUT/MOVIE
    if (os.readlink(old/'cwd')!=str(ROOT) or b'scripts/publish_s1_w57.py' not in args or
            b'follow-episodes' not in args or str(movie).encode() not in ff or
            int(process_state(encoder)[1])!=parent or os.getpgid(encoder)==os.getpgid(parent) or
            (old/'task'/str(parent)/'children').read_text().split()!=[str(encoder)]):
        raise RuntimeError('The observed replay owners changed')
    if not any(os.path.realpath(fd)==str((WORK/'publish.lock').resolve()) for fd in (old/'fd').iterdir()):
        raise RuntimeError('Legacy sender no longer owns the expected publication descriptor')
    status=dict(l.split(':',1) for l in (old/'status').read_text().splitlines() if ':' in l)
    term_bit=1<<(signal.SIGTERM-1)
    if (int(status['SigCgt'],16)|int(status['SigIgn'],16))&term_bit:
        raise RuntimeError('Producer signal behavior differs; keep the current replay')
    # Learn the actual constant amount of non-frame input from multiple fully drained reads.
    seen=defaultdict(set);previous=reader_bytes(encoder)
    while True:
        time.sleep(.055);value=reader_bytes(encoder)
        if value==previous:
            group=seen[value%FRAME_BYTES];group.add(value)
            if len(group)>=3:break
        previous=value
    remainder=value%FRAME_BYTES
    while True:
        os.kill(parent,signal.SIGSTOP)
        stopped=False
        try:
            while (state:=process_state(parent)) and state[0] not in ('T','t'):
                time.sleep(.01)
            if not state:raise RuntimeError('Producer exited before the frame boundary')
            time.sleep(.20)
            value=reader_bytes(encoder)
            if value%FRAME_BYTES!=remainder:
                continue
            os.link(movie,prefix)
            proof.update(phase='closing_input_on_complete_frame',at=now(),producer_pid=parent,
                         encoder_pid=encoder,source_inode=movie.stat().st_ino,
                         raw_bytes=value,expected_prefix_frames=value//FRAME_BYTES,
                         retained_prefix=str(prefix),**BG)
            write(PROOF,proof)
            # Only the offline producer stops. Its encoder has a separate process group:
            # the OS closes stdin at this exact frame boundary and FFmpeg writes the trailer.
            os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
            stopped=True
            break
        finally:
            if not stopped and process_state(parent):
                os.kill(parent,signal.SIGCONT)
                time.sleep(.25)
    while (state:=process_state(encoder)) and state[0]!='Z':
        time.sleep(.2)
    info=probe(prefix)['streams'][0]
    frames=int(info['nb_frames'])
    if abs(frames-proof['expected_prefix_frames'])>1 or frames<=0:
        raise RuntimeError('Finalized prefix frame count differs from its drained FFmpeg input')
    if movie.stat().st_ino!=prefix.stat().st_ino:raise RuntimeError('Legacy output inode changed')
    # Retain the finished prefix in STAGE and release its old, incomplete canonical name.
    movie.unlink()
    proof.update(phase='prefix_verified',prefix_frames=frames,prefix_bytes=prefix.stat().st_size,
                 prefix_sha256=hashlib.file_digest(prefix.open('rb'),'sha256').hexdigest(),prefix_closed_at=now())
    write(PROOF,proof)
    return frames


def main(a):
    import tempfile
    receipt=json.loads((COMPACT/'attempts'/(NAME+'.json')).read_text())
    if receipt['status']!='passed' or not receipt.get('ended_unix_s'):raise RuntimeError('A completed frozen episode is required')
    STAGE.mkdir(exist_ok=True);lockdir=WORK/'replay-locks';lockdir.mkdir(exist_ok=True)
    prefix=STAGE/'original-prefix.mp4'
    with (lockdir/(NAME+'.lock')).open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        marker=OUT/'video-asset.json'
        if marker.is_file():return
        proof=json.loads(PROOF.read_text()) if PROOF.exists() else dict(attempt=NAME,**BG)
        if proof.get('prefix_frames'):
            frames=proof['prefix_frames']
        else:
            frames=close_at_frame_boundary(a.producer,a.encoder,proof,prefix)
        with np.load(OUT/'trajectory.npz') as data:
            times=data['time'];samples=np.unique(np.r_[np.searchsorted(times,np.arange(0,times[-1],1/30)),len(times)-1])
        if frames>=len(samples):raise RuntimeError('The original encoder reached the complete movie')
        tail=STAGE/'tail';tail.mkdir(exist_ok=True);tailfile=tail/MOVIE
        renderer.SMALL=COMPACT/'episodes'
        if not (tail/'video-segment.json').is_file():
            proof.update(phase='rendering_remaining_frames',first_remaining_index=frames,
                         source_time_first_remaining_s=float(times[samples[frames]]),total_frames=len(samples),
                         tail_started_at=now());write(PROOF,proof)
            renderer.render(NAME,output_dir=tail,first_frame=frames,pace=DrawBudget())
        ai=probe(prefix)['streams'][0];bi=probe(tailfile)['streams'][0]
        for key in ('codec_name','width','height','r_frame_rate','time_base','extradata_hash'):
            if ai[key]!=bi[key]:raise RuntimeError('H264 segment configuration differs at '+key)
        if int(bi['nb_frames'])!=len(samples)-frames:raise RuntimeError('Tail frame count differs')
        inputs=STAGE/'concat.txt';inputs.write_text(f"ffconcat version 1.0\nfile 'original-prefix.mp4'\nduration {frames/30:.12f}\nfile 'tail/{MOVIE}'\n")
        full=STAGE/'complete.mp4'
        subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-f','concat','-safe','0','-auto_convert','0',
                        '-i',str(inputs),'-map','0:v:0','-c:v','copy','-movflags','+faststart',str(full)],check=True)
        output=probe(full);video=output['streams'][0]
        if (int(video['nb_frames']),video['r_frame_rate'],video['width'],video['height'])!=(len(samples),'30/1',1280,720):
            raise RuntimeError('Full replay video format or frame count differs')
        original=packets(prefix);suffix=packets(tailfile);joined=packets(full)
        if [p['data_hash'] for p in joined]!=[p['data_hash'] for p in original+suffix]:
            raise RuntimeError('Packet content changed during stream-copy concatenation')
        from fractions import Fraction
        step=1/Fraction(video['time_base'])/30
        if sorted(int(p['pts']) for p in joined)!=[int(i*step) for i in range(len(samples))]:
            raise RuntimeError('The 30 fps presentation timeline has a gap or duplicate')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','4','-i',str(full),'-map','0:v:0','-f','null','-'],check=True)
        target=OUT/MOVIE
        if target.exists():raise RuntimeError('An independent canonical movie appeared')
        os.rename(full,target)
        write(marker,dict(attempt=NAME,path=str(target),size=target.stat().st_size,
                          scope='Full recorded robot trajectory, visual-only strip overlay; no new integration or exaggerated deformation'))
        proof.update(phase='complete',completed_at=now(),preserved_H264_packets=len(original),tail_packets=len(suffix),
                     verified_packet_hashes=len(joined),decoded_all_frames=True,contiguous_30fps=True,
                     video_bytes=target.stat().st_size,video_sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest())
        write(PROOF,proof);print(now(),json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer',type=int,required=True);parser.add_argument('--encoder',type=int,required=True)
    main(parser.parse_args())
