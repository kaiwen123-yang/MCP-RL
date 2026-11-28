"""Render all of an E frozen evaluation from saved poses and verify real media timestamps."""
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from collect import BASE,SOURCE_REL
from s1_w59_learner_critic import atomic_json,file_digest


def render(out):
    identity=json.loads((out/"identity.json").read_text())
    result=json.loads((out/"result.json").read_text())
    xml=BASE/SOURCE_REL/identity["original"]/"model/G1-project-go2.xml"
    with np.load(out/"trajectory.npz") as archive: track=archive["control"]
    if not len(track):
        return None
    physical=track[:,1]
    fps=25
    frame_times=np.arange(0,result["physical_s"]-1.e-8,1./fps)
    sample=np.searchsorted(physical,frame_times).clip(0,len(track)-1)
    model=mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth=640
    model.vis.global_.offheight=360
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,360,640)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=1.35
    camera.elevation=-17
    camera.azimuth=106
    video=out/"complete.mp4"
    command=["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
             "-s","640x360","-r",str(fps),"-i","-","-an","-threads","1","-c:v","libx264","-crf","22","-pix_fmt","yuv420p",str(video)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for index in sample:
            data.qpos[:]=track[index,2:21]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=data.qpos[:3]+[.1,0,0]
            renderer.update_scene(data,camera)
            encoder.stdin.write(renderer.render().tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed to encode the entire frozen trajectory")
    renderer.close()
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=nb_frames,r_frame_rate,avg_frame_rate,duration,time_base",
                                             "-of","json",str(video)],text=True))["streams"][0]
    rec=dict(file=str(video),bytes=video.stat().st_size,sha256=file_digest(video),source=file_digest(out/"trajectory.npz"),
        rendered_frames=len(frame_times),physical_clock_s=[0.,result["physical_s"]],sample_clock="post-step t; nearest following measured pose, clamp at end",
        largest_time_error_s=float(np.abs(physical[sample]-frame_times).max()),probe=probe)
    atomic_json(out/"video.json",rec)
    return rec


if __name__=="__main__":
    import sys
    print(json.dumps(render(Path(sys.argv[1]))))
