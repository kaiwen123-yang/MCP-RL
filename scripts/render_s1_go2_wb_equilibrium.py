#!/usr/bin/env python3
"""Render every recorded state interval with EGL and the existing NVENC encoder."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from run_s1_go2_wb_baseline import ROOT,PHASES,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args()
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-wb-equilibrium.json').read_text())
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/args.attempt
    result=json.loads((ROOT/c['outputs']['root']/(args.attempt+'.json')).read_text())
    a=np.load(out/'trajectory.npz');times=a['time'];qpos=a['qpos'];qvel=a['qvel'];mp=a['mocap_pos'];mq=a['mocap_quat'];phases=a['phase']
    m=mujoco.MjModel.from_binary_path(str(out/'model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=1280)
    camera=mujoco.MjvCamera();camera.azimuth=125.;camera.elevation=-23.;camera.distance=4.8
    sample=np.unique(np.r_[np.searchsorted(times,np.arange(0,times[-1],1/30)),len(times)-1])
    command=['/usr/bin/ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r','30','-i','-','-an','-c:v','h264_nvenc','-preset','p4','-cq','22','-b:v','0',
        '-pix_fmt','yuv420p',str(out/'complete-mission.mp4')]
    begin=time.perf_counter()
    with (out/'video-encoder.log').open('w') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        for k,i in enumerate(sample):
            d.time=float(times[i]);d.qpos[:]=qpos[i];d.qvel[:]=qvel[i];d.mocap_pos[:]=mp[i];d.mocap_quat[:]=mq[i]
            mujoco.mj_forward(m,d);camera.lookat[:]=[d.qpos[0],d.qpos[1],.25]
            renderer.update_scene(d,camera=camera)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,76),fill=(15,20,30))
            draw.text((18,10),f"{args.attempt} | {result['stage']} | {result['condition']} | t={d.time:.3f}s | {PHASES[int(phases[i])]}",fill='white')
            laps=sum(v['endpoint']=='home' and v['stable_completion_s']<=d.time for v in result['arrivals'])
            lap_text=f' | completed round trips {laps}/5' if result['mission']=='long' else ''
            draw.text((18,34),f"Body x={d.qpos[0]:.3f} m, y={d.qpos[1]:.3f} m{lap_text} | continuous recorded simulation",fill='white')
            if k==len(sample)-1:
                draw.text((18,55),result['reason'],fill='yellow');frame.save(out/'terminal-frame.png')
            process.stdin.write(np.asarray(frame).tobytes())
        process.stdin.close();code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    renderer.close()
    value=dict(video=str(out/'complete-mission.mp4'),frames=len(sample),fps=30,wall_s=time.perf_counter()-begin,
        renderer='MuJoCo EGL',encoder='h264_nvenc',physics_steps=0,size=(out/'complete-mission.mp4').stat().st_size,sha256=sha(out/'complete-mission.mp4'))
    (out/'video.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
if __name__=='__main__':main()
