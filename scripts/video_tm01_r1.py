#!/usr/bin/env python3
"""Render continuous videos by replaying recorded MuJoCo trajectories, including falls."""
import argparse,csv,json,os
from pathlib import Path
os.environ["MUJOCO_GL"]="egl"
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from gym_quadruped.quadruped_env import QuadrupedEnv
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--run",default="r1-v4")
p.add_argument("--config",default="experiments/tm01/r1.json")
p.add_argument("--methods",nargs="+",default=["acados","naive","adaptive"])
p.add_argument("--experiment",choices=["fig3","fig4","table1"])
a=p.parse_args()
R=ROOT/"results/terrain_memory/reproduction/pympc"/a.run
O=ROOT/"outputs/terrain_memory/reproduction/pympc"/a.run
config=json.loads((ROOT/a.config).read_text())
V=config["video"]
camera=mujoco.MjvCamera()
camera.type=mujoco.mjtCamera.mjCAMERA_FREE
camera.distance=2.6
camera.azimuth=135
camera.elevation=-22
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
selection=[]
for experiment,methods in [("fig3",["acados","naive","mppi"]),("fig4",["acados","adaptive"]),("table1",a.methods)]:
    if a.experiment and experiment != a.experiment:continue
    for method in methods:
        with (R/experiment/method/"episodes.csv").open() as f:rr=list(csv.DictReader(f))
        chosen={0}
        if experiment=="table1":
            for outcome in ("0","1"):
                matches=[int(x["seed"]) for x in rr if x["success"]==outcome]
                if matches:chosen.add(min(matches))
        for seed in sorted(chosen):
            row=next(x for x in rr if int(x["seed"])==seed)
            trajectory=ROOT/row["trajectory"]
            d=np.load(trajectory)
            model=mujoco.MjModel.from_binary_path(str(trajectory.parent/"model.mjb"))
            model.vis.global_.offwidth=V["width"]
            model.vis.global_.offheight=V["height"]
            data=mujoco.MjData(model)
            renderer=mujoco.Renderer(model,height=V["height"],width=V["width"])
            dest=O/"videos";dest.mkdir(exist_ok=True)
            name=f"{experiment}-{method}-seed-{seed:03d}"
            video=dest/(name+".mp4")
            status="COMPLETED" if row["success"]=="1" else "FAILED: "+row["termination"]
            frames=int(np.ceil(float(row["duration_s"])*V["fps"]))
            writer=imageio.get_writer(video,fps=V["fps"],codec="libx264",quality=7,
                pixelformat="yuv420p",macro_block_size=None,ffmpeg_log_level="error")
            for frame in range(frames+1):
                time=frame/V["fps"]
                k=min(np.searchsorted(d["time"],time,side="right")-1,len(d["time"])-1)
                data.qpos[:]=d["final_qpos"] if frame==frames else d["qpos"][k]
                data.qvel[:]=d["final_qvel"] if frame==frames else d["qvel"][k]
                mujoco.mj_forward(model,data)
                camera.lookat[:]=data.subtree_com[1]
                renderer.update_scene(data,camera)
                force=d["wrench"][k,:3]
                if np.linalg.norm(force)>0:
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),
                        np.zeros(3),np.eye(3).ravel(),np.array([.85,.2,.1,1]))
                    start=data.subtree_com[1].copy()+np.array([0,0,.12])
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.018,start,start+.012*force)
                    renderer.scene.ngeom+=1
                im=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,V["width"],146),fill=(242,244,247))
                line1=f"TM01 {experiment} | {method} | seed {seed} | t={min(time,float(row['duration_s'])):.2f}s"
                line2=f"f={d['frequency'][k]:.1f} Hz | external F=({force[0]:.1f}, {force[1]:.1f}, {force[2]:.1f}) N"
                moment=d["wrench"][k,3:]
                line3=f"external M=({moment[0]:.1f}, {moment[1]:.1f}, {moment[2]:.1f}) N m"
                line4=status if frame==frames else "Recorded trajectory replay | 1x simulation time"
                draw.text((18,10),line1,font=font,fill=(20,30,45))
                draw.text((18,44),line2,font=font,fill=(20,30,45))
                draw.text((18,78),line3,font=font,fill=(20,30,45))
                draw.text((18,112),line4,font=font,fill=(150,30,20) if "FAILED" in line4 else (20,70,50))
                writer.append_data(np.asarray(im))
                if frame==min(frames,round(3*V["fps"])):im.save(dest/(name+"-preview.png"))
            writer.close()
            renderer.close()
            selection.append(dict(experiment=experiment,method=method,seed=seed,success=int(row["success"]),
                duration_s=float(row["duration_s"]),video=str(video.relative_to(ROOT)),frames=frames+1,
                fps=V["fps"],trajectory=row["trajectory"],selection_rule="All Figure 3 episodes; "+V["selection"]))
            print(json.dumps(selection[-1]),flush=True)
(R/"videos.json").write_text(json.dumps(selection,indent=2)+"\n")
