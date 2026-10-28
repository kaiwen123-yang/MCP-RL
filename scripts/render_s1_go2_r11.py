#!/usr/bin/env python3
"""Render full recorded R11 trajectories with a visual-only material strip."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw
from s1_go2_r11_protocol import ROOT,CONTRACT
from s1_go2_r11_runtime import PHASES,sha
SMALL=ROOT/CONTRACT['outputs']['compact_root'];LARGE=ROOT/CONTRACT['outputs']['large_root']
def render(name):
    out=LARGE/name
    result=json.loads((SMALL/(name+'.json')).read_text())
    resolved=json.loads((SMALL/(name+'-resolved.json')).read_text())
    terrain=resolved['r11_configuration']['actual_terrain']
    a=np.load(out/'trajectory.npz')
    m=mujoco.MjModel.from_binary_path(str(out/'model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=1280)
    camera=mujoco.MjvCamera();camera.azimuth=125;camera.elevation=-23;camera.distance=4.8
    sample=np.unique(np.r_[np.searchsorted(a['time'],np.arange(0,a['time'][-1],1/30)),len(a['time'])-1])
    target=out/'full-mission-material-overlay.mp4'
    with imageio.get_writer(target,fps=30,codec='libx264',quality=8,ffmpeg_params=['-threads','2']) as writer:
        for i in sample:
            d.time=float(a['time'][i]);d.qpos[:]=a['qpos'][i];d.qvel[:]=a['qvel'][i]
            d.mocap_pos[:]=a['mocap_pos'][i];d.mocap_quat[:]=a['mocap_quat'][i]
            mujoco.mj_forward(m,d)
            camera.lookat[:]=[d.qpos[0],d.qpos[1],.25]
            renderer.update_scene(d,camera=camera)
            if terrain!='rigid':
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.7,.6,.0002]),
                    np.array([1.5,0,.0002]),np.eye(3).ravel(),np.array([.15,.45,.8,.22],np.float32))
                renderer.scene.ngeom+=1
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,76),fill=(15,20,30))
            draw.text((18,9),f"{name} | {result['evaluation_stage']['level']} | t={d.time:.4f}s | {PHASES[int(a['phase'][i])]}",fill='white')
            draw.text((18,31),f"Base x={d.qpos[0]:.3f}m y={d.qpos[1]:.3f}m | full recorded task | {'TM02 strip: x[0.8,2.2], y[-0.6,0.6]m' if terrain!='rigid' else 'Rigid regression'}",fill='white')
            draw.text((18,53),result['reason'] if i==sample[-1] else 'Recorded-state replay; original physics and material limits',fill='yellow' if i==sample[-1] else 'white')
            writer.append_data(np.asarray(frame))
    renderer.close()
    record=dict(attempt=name,path=str(target),size=target.stat().st_size,sha256=sha(target),
        scope='Full recorded robot trajectory, visual-only strip overlay; no new integration or exaggerated deformation')
    (out/'video-asset.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt');a=p.parse_args()
    if a.attempt:render(a.attempt);return
    from campaign_s1_go2_r11 import current
    rows=current()['attempts'];names=[x['attempt'] for x in rows if x.get('result') and x['purpose']!='training'
        and not (LARGE/x['attempt']/'video-asset.json').exists()]
    def one(name):
        with (ROOT/'work'/f'r11-render-{name}-overlay.log').open('w') as log:
            child=subprocess.run([sys.executable,str(Path(__file__)),'--attempt',name],cwd=ROOT,stdout=log,stderr=log)
        return dict(attempt=name,render_exit_code=child.returncode)
    with ThreadPoolExecutor(2) as pool:
        for value in pool.map(one,names):print(json.dumps(value),flush=True)
if __name__=='__main__':main()
