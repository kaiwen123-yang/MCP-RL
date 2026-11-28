"""Render recorded physics samples, preserve each unfinished trajectory and its time mapping."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import mujoco
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from w61_gait_rollout import DEFAULT_MODEL


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def render(folder,output,label=None):
    result=json.loads((folder/'result.json').read_text())
    with np.load(folder/'trajectory.npz') as d:
        if 'time' in d:
            clock=d['time'].astype(float)
            qpos=d['qpos']
            qvel=d['qvel']
        else:
            # Joint forefoot logger records [time, qpos(19), qvel(18), ...].
            states=d['state']
            clock,qpos,qvel=states[:,0],states[:,1:20],states[:,20:38]
    label=label or (str(result.get('role',result.get('arm','')))+' | '+str(result.get('case','')))
    fps=25
    # Last displayed frame is the last regular presentation instant preceding termination.
    times=np.arange(int(np.ceil((clock[-1]+.002)*fps-1e-7)))/fps
    indices=np.maximum(0,np.searchsorted(clock,times+1e-8,side='right')-1)
    model=mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data=mujoco.MjData(model)
    cam=mujoco.MjvCamera()
    cam.distance=1.42
    cam.azimuth=118
    cam.elevation=-17
    width,height=640,360
    proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
                           '-s',f'{width}x{height}','-pix_fmt','rgb24','-r',str(fps),'-i','-','-an',
                           '-c:v','libx264','-threads','1','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',
                           '-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        for t,i in zip(times,indices):
            data.qpos[:]=qpos[i];data.qvel[:]=qvel[i]
            mujoco.mj_forward(model,data)
            cam.lookat[:]=[qpos[i,0]+.05,qpos[i,1],.16]
            renderer.update_scene(data,camera=cam)
            img=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(img)
            draw.rectangle((0,0,width,24),fill=(22,29,35))
            draw.text((7,5),f"{label} | {t:.2f} s",fill=(255,255,255))
            proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('video encoder failed')
    probe=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=nb_frames,avg_frame_rate,time_base,duration','-of','json',str(output)],text=True)
    return dict(video=output.name,sha256=digest(output),bytes=output.stat().st_size,
        renderer='MuJoCo FK replay from original logged qpos at 500Hz',physics_end_s=result['physical_s'],
        recorded_state_last_s=float(clock[-1]),presentation_fps=fps,frames=len(times),
        maximum_frame_to_source_offset_s=float(np.max(np.abs(times-clock[indices]))),
        mapping='frame k PTS=k/25 seconds; most recent recorded state at or before that time (1e-8 float tolerance)',
        ffprobe=json.loads(probe)['streams'][0])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='w61-c-transfer-closed-v1')
    args=parser.parse_args()
    compact=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/mpc'/args.run
    source=ROOT/'work/w61/mpc'/args.run
    assets=source/'release'
    assets.mkdir(exist_ok=True)
    while True:
        rows=sorted(compact.glob('*.json'))
        for file in rows:
            row=json.loads(file.read_text())
            if 'trajectory_sha256' not in row:
                continue
            folder=Path(row['raw_directory'])
            stem=f'{args.run}-{folder.name}'
            video=assets/(stem+'.mp4')
            metadata=folder/'video-metadata.json'
            if not metadata.is_file():
                metadata.write_text(json.dumps(render(folder,video),indent=2)+'\n')
            pack=assets/(stem+'-simulator-originals.zip')
            receipt=folder/'source-package.json'
            if not receipt.exists():
                pending=pack.with_suffix('.pending')
                with zipfile.ZipFile(pending,'w',allowZip64=True) as z:
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and p!=receipt:
                            z.write(p,p.relative_to(folder),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
                pending.rename(pack)
                receipt.write_text(json.dumps(dict(name=pack.name,bytes=pack.stat().st_size,sha256=digest(pack),
                      contains='complete source identity, state, controller/solver records, mode decisions, termination and video clock'),indent=2)+'\n')
        complete=[r for r in source.glob('*/source-package.json')]
        if len(complete)==15:
            manifest={p.parent.name:dict(originals=json.loads(p.read_text()),video=json.loads((p.parent/'video-metadata.json').read_text())) for p in complete}
            out=compact/'assets.json'
            out.write_text(json.dumps(dict(source_directory=str(assets),trials=manifest),indent=2)+'\n')
            return
        time.sleep(20)


if __name__=='__main__':
    main()
