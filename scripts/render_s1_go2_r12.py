#!/usr/bin/env python3
"""Render full recorded R12 trajectories with a visual-only material strip."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/"experiments/terrain_memory/s1-go2-local-readiness-r12.json").read_text())
from s1_go2_r11_runtime import PHASES
SMALL=ROOT/CONTRACT['outputs']['compact_root'];LARGE=ROOT/CONTRACT['outputs']['large_root']
def render(name, output_dir=None):
    out=LARGE/name
    dest=out if output_dir is None else Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    result=json.loads((SMALL/(name+'.json')).read_text())
    resolved=json.loads((SMALL/(name+'-resolved.json')).read_text())
    terrain='tm02_anchor'
    mismatch=False
    fields = ['time', 'qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'phase']
    if '-w51-' in name:
        fields.extend(['actual_grf', 'nonfoot_ground_force'])
    # Each access to an NPZ member decompresses the complete array. Read each used member once.
    with np.load(out/'trajectory.npz') as recorded:
        a = {field: recorded[field] for field in fields}
    m=mujoco.MjModel.from_binary_path(str(out/'model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=1280)
    camera=mujoco.MjvCamera();camera.azimuth=125;camera.elevation=-23;camera.distance=4.8
    sample=np.unique(np.r_[np.searchsorted(a['time'],np.arange(0,a['time'][-1],1/30)),len(a['time'])-1])
    target=dest/('full-mission-qualification-noted.mp4' if mismatch else 'full-mission-material-overlay.mp4')
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
            draw.rectangle((0,0,1280,124 if '-w51-' in name else 98 if mismatch else 76),fill=(15,20,30))
            draw.text((18,9),f"{name} | {result.get('evaluation_stage',{}).get('level','training')} | t={d.time:.4f}s | {PHASES[int(a['phase'][i])]}",fill='white')
            draw.text((18,31),f"Base x={d.qpos[0]:.3f}m y={d.qpos[1]:.3f}m | full recorded task | {'TM02 strip: x[0.8,2.2], y[-0.6,0.6]m' if terrain!='rigid' else 'Rigid regression'}",fill='white')
            draw.text((18,53),result['reason'] if i==sample[-1] else 'Recorded-state replay; original physics and material limits',fill='yellow' if i==sample[-1] else 'white')
            if '-w51-' in name:
                fz=a['actual_grf'][i,:,2];nf=a['nonfoot_ground_force'][i,2]
                draw.text((18,76),f"z={d.qpos[2]:.3f}m vz={d.qvel[2]:+.3f}m/s | foot Fz FL FR RL RR="+','.join(f'{f:.1f}' for f in fz)+f" N | nonfoot Fz={nf:.1f} N",fill='white')
                draw.text((18,98),'W51 recorded actual state | '+resolved['r12_configuration']['controller_build'] if 'r12_configuration' in resolved else 'W51 recorded actual state; no new physics',fill='white')
            if mismatch:
                draw.text((18,76),'DECLARED PARAMETER MISMATCH: retained observation; unusable as declared-version qualification',fill='yellow')
            writer.append_data(np.asarray(frame))
    renderer.close()
    record=dict(attempt=name,path=str(target),size=target.stat().st_size,
        scope='Full recorded robot trajectory, visual-only strip overlay; no new integration or exaggerated deformation')
    (dest/'video-asset.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    render(a.attempt)
if __name__=='__main__':main()
