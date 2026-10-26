#!/usr/bin/env python3
"""Render continuous recorded Go2 missions, including unsuccessful seed zero."""
import argparse
import json
import os
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
PHASES=['initial stand','outbound','turn','return','terminal stand','completed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/terrain_memory/s1-control-repair.json')
    parser.add_argument('--cohort',default='formal-v1')
    parser.add_argument('--rerender',action='store_true',help='Regenerate saved-trajectory videos after a renderer correction.')
    parser.add_argument('--attempts',action='store_true',help='Render every retained development attempt from the run index.')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    small=ROOT/config['outputs']['compact_root']
    out=ROOT/config['outputs']['large_root']
    rows=[json.loads(p.read_text()) for p in (small/args.cohort).glob('*-seed-*.json')]
    if args.attempts:
        idx=json.loads((small/'index.json').read_text())
        rows=[json.loads((ROOT/a['record']).read_text()) for a in idx['attempts'] if a['status']=='finished']
    requested=sorted({(r['terrain'],r['controller']) for r in rows if not r.get('reused_from')})
    selection=[]
    for terrain,method in ([] if args.attempts else requested):
        group=[r for r in rows if r['terrain']==terrain and r['controller']==method]
        chosen=[next(r for r in group if r['seed']==0)]
        if not chosen[0]['success'] and any(r['success'] for r in group):
            chosen.append(min((r for r in group if r['success']),key=lambda r:r['seed']))
        elif chosen[0]['success'] and any(not r['success'] for r in group):
            chosen.append(min((r for r in group if not r['success']),key=lambda r:r['seed']))
        selection.extend(chosen)
    if args.attempts:
        selection=rows
    videos=out/'videos'
    videos.mkdir(exist_ok=True)
    previews=small/'video-previews'
    previews.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    smallfont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    entries=[]
    for row in rows:
        if row.get('reused_from') and row['seed']==0:
            source=row['reused_from']
            asset=next(a for a in source['assets'] if a['filename']==f"{row['terrain']}-C0-seed-0.mp4")
            entries.append(dict(terrain=row['terrain'],controller='B0',seed=0,success=row['success'],
                duration_s=row['duration_s'],url=asset['url'],bytes=asset['bytes'],sha256=asset['sha256'],
                reused_from=source['release'],selection='original C0 seed zero reused without download or rerender'))
    for row in selection:
        name=f"{row['terrain']}-{row['controller']}-seed-{row['seed']}"
        path=videos/(name+'.mp4')
        selection_label=('seed zero' if row['seed']==0 else
            'lowest successful seed, additional to failed seed zero' if row['success'] else
            'lowest failed seed, additional to successful seed zero')
        if args.attempts:
            selection_label='Every development attempt, original full mission until recorded termination'
        if path.exists() and not args.rerender:
            entries.append(dict(terrain=row['terrain'],controller=row['controller'],seed=row['seed'],
                success=row['success'],duration_s=row['duration_s'],video=str(path.relative_to(ROOT)),fps=30,
                frames=int(np.ceil(row['duration_s']*30))+1,
                selection=selection_label))
            continue
        archive=np.load(ROOT/row['trajectory'])
        a={k:archive[k] for k in ['time','qpos','qvel','phase','actual_grf','planned_grf','indentation_m','frequency','velocity','final_qpos','final_qvel']}
        if args.attempts:
            a['commanded_grf']=archive['commanded_grf']
        model=mujoco.MjModel.from_binary_path(str(row.get('model_path',Path(row['trajectory']).parent/'go2.mjb')))
        model.vis.global_.offwidth=1120
        model.vis.global_.offheight=720
        data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=720,width=1120)
        camera=mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance=1.9
        camera.azimuth=125
        camera.elevation=-22
        fps=30
        frames=int(np.ceil(row['duration_s']*fps))
        writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=7,pixelformat='yuv420p',
            macro_block_size=None,ffmpeg_log_level='error')
        samples=[]
        for frame in range(frames+1):
            t=min(frame/fps,row['duration_s'])
            k=min(max(np.searchsorted(a['time'],t,side='right')-1,0),len(a['time'])-1)
            data.qpos[:]=a['final_qpos'] if frame==frames else a['qpos'][k]
            data.qvel[:]=a['final_qvel'] if frame==frames else a['qvel'][k]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=data.qpos[:3]+np.array([0,0,-.05])
            renderer.update_scene(data,camera)
            if row['terrain']!='rigid':
                zone=renderer.scene.geoms[renderer.scene.ngeom]
                # Render-only decal offset avoids z-fighting with the floor;
                # recorded geometry and all simulated coordinates stay intact.
                mujoco.mjv_initGeom(zone,mujoco.mjtGeom.mjGEOM_BOX,np.array([.7,.6,.0001]),
                    np.array([1.5,0,.0002]),np.eye(3).ravel(),np.array([.78,.57,.22,.45]))
                renderer.scene.ngeom+=1
            for j,leg in enumerate(['FL','FR','RL','RR']):
                force=a['actual_grf'][k,j]
                if np.linalg.norm(force)>1:
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    point=data.geom_xpos[model.geom(leg).id].copy()
                    point[2]-=model.geom_size[model.geom(leg).id,0]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),
                        np.eye(3).ravel(),np.array([.0,.45,.7,.85]))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.008,point,point+.001*force)
                    renderer.scene.ngeom+=1
            im=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,1120,109),fill=(246,248,250))
            draw.text((16,8),f"S1 Go2 | {row['terrain']} | {row['controller']} | seed {row['seed']} | t = {t:.3f} s",font=font,fill=(20,30,45))
            status=(('COMPLETED' if row['success'] else 'FAILED: '+row['termination']) if frame==frames else PHASES[a['phase'][k]])
            draw.text((16,35),status+f' | body x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m',font=font,fill=(155,35,25) if frame==frames and not row['success'] else (25,65,80))
            draw.text((16,62),'Continuous recorded trajectory | 1x physical time | robot geometry 1x | force arrows: 100 N = 0.1 m',font=smallfont,fill=(30,45,55))
            region='Native rigid ground everywhere' if row['terrain']=='rigid' else 'Shaded strip: virgin reference surface, x=0.8-2.2 m'
            draw.text((16,84),f"speed = {np.linalg.norm(a['velocity'][k,:2]):.3f} m/s   gait = {a['frequency'][k]:.2f} Hz   {region}",font=smallfont,fill=(30,45,55))
            draw.rectangle((0,583,1120,720),fill=(246,248,250))
            draw.text((16,590),('Actual / held plan / command normal GRF (N)' if args.attempts else 'Actual / planned normal GRF (N) and actual indentation (mm)'),font=font,fill=(25,35,45))
            for j,leg in enumerate(['FL','FR','RL','RR']):
                x=16+j*275
                if args.attempts:
                    draw.text((x,620),f"{leg} actual: {a['actual_grf'][k,j,2]:.1f} N",font=smallfont,fill=(0,85,130))
                    draw.text((x,645),f"held/cmd: {a['planned_grf'][k,j,2]:.1f}/{a['commanded_grf'][k,j,2]:.1f} N",font=smallfont,fill=(30,45,55))
                    continue
                draw.text((x,620),f"{leg}: {a['actual_grf'][k,j,2]:7.1f} / {a['planned_grf'][k,j,2]:7.1f} N",font=smallfont,fill=(0,85,130))
                draw.text((x,645),f"indentation {a['indentation_m'][k,j]*1000:.5f} mm",font=smallfont,fill=(30,45,55))
            draw.text((16,683),'Synthetic simulation; no hardware run. KV is not Go2 calibration. TM02 retains gamma=0.025.',font=smallfont,fill=(65,65,65))
            writer.append_data(np.asarray(im))
            if frame in {min(frames,60),frames//2,frames}:
                samples.append(im.resize((560,360)))
        writer.close()
        renderer.close()
        panel=Image.new('RGB',(560*len(samples),360),'white')
        for j,im in enumerate(samples):panel.paste(im,(j*560,0))
        panel.save(previews/(name+'.png'))
        entry=dict(terrain=row['terrain'],controller=row['controller'],seed=row['seed'],success=row['success'],
            duration_s=row['duration_s'],video=str(path.relative_to(ROOT)),fps=fps,frames=frames+1,
            selection=selection_label)
        entries.append(entry)
        print(json.dumps(entry),flush=True)
    indexpath=small/'index.json'
    index=json.loads(indexpath.read_text())
    index['videos']=entries
    indexpath.write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()
