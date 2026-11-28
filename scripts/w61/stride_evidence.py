"""Summarize full observed maneuvers and render every recorded outcome on its physics clock."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import mujoco
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def render(run,receipt,arrays):
    t=arrays['time'];qpos=arrays['qpos']
    model=mujoco.MjModel.from_xml_path(receipt['plant_xml'])
    data=mujoco.MjData(model);scene=mujoco.Renderer(model,360,640)
    camera=mujoco.MjvCamera();camera.azimuth=132;camera.elevation=-17;camera.distance=1.28
    fps=30
    count=max(1,int(np.ceil(receipt['physical_s']*fps-1.e-7)))
    movie=run/'full-30fps.mp4'
    ff=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','640x360','-framerate',str(fps),'-i','-','-c:v','libx264','-threads','1','-preset','veryfast',
        '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
    try:
        for i in range(count):
            target=i/fps
            idx=np.clip(np.searchsorted(t,target),0,len(t)-1)
            if idx and abs(t[idx-1]-target)<abs(t[idx]-target):
                idx-=1
            data.qpos[:]=qpos[idx];mujoco.mj_forward(model,data)
            camera.lookat[:]=[qpos[idx,0]+.08,qpos[idx,1],.19]
            scene.update_scene(data,camera=camera)
            ff.stdin.write(scene.render().tobytes())
    finally:
        ff.stdin.close();rc=ff.wait();scene.close()
    if rc:
        raise RuntimeError(f'ffmpeg exited {rc}')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'stream=r_frame_rate,avg_frame_rate,nb_frames,start_time,duration,time_base', '-of','json',str(movie)]))['streams'][0]
    return dict(file=str(movie),sha256=sha(movie),bytes=movie.stat().st_size,stream=probe,
        clock='Frame k at physical k/30 s, nearest recorded 2 ms qpos. Includes the full available terminal trajectory.',
        recorded_first_last_s=[float(t[0]),float(t[-1])])


def summarize(run):
    result=json.loads((run/'result.json').read_text())
    arrays=np.load(run/'trajectory.npz')
    record={k:result.get(k) for k in ('run_id','case','role','completed','end_reason','physical_s','planned_s','x_m',
        'requested_tracking_ISE','mechanical_abs_work_J','slip_integral_m','maximum_vertical_foot_force_N',
        'torque_violation_Nm','section_count','wall_s','cpu_h','controller_ms_p50_p95_p99','feedback_and_plant_ms_p50_p95_p99',
        'feedback_and_plant_overdue_fraction','effective_physics_steps_s','effective_control_solves_s','mpc_solves','wbc',
        'physics_dt_s','feedback_dt_s','runner','mixins','baseline','baseline_config_sha256','fit_sha256',
        'plant_xml_sha256','contract_sha256','cpu_affinity')}
    f=arrays['force'][:,:,2];contact=arrays['contact'].astype(bool)
    # Each new stance contributes one maximum within the first 20 ms. Exclude incomplete post-event windows.
    initial_peaks=[]
    for i in range(4):
        for k in np.flatnonzero(contact[1:,i]&~contact[:-1,i])+1:
            if k+10<=len(f):initial_peaks.append(float(np.max(f[k:k+10,i])))
    record.update(impact_initial20ms_500Hz=dict(count=len(initial_peaks),
        p95_p99_N=np.percentile(initial_peaks,[95,99]).tolist() if initial_peaks else None),
        max_abs_rp_deg=np.rad2deg(np.abs(arrays['body_rpy'][:,:2]).max(axis=0)).tolist(),
        command_rate_rms=float(np.mean(np.square(np.diff(arrays['ctrl'],axis=0)/result['feedback_dt_s']))**.5))
    for name in ('controlled-orbit.json','support-bridges.json'):
        if (run/name).is_file():
            item=json.loads((run/name).read_text());item.pop('events',None)
            record[name]=item
    files=('result.json','runtime.json','trajectory.npz','sections.json','w55-f/manifest.json',
           'w55-f/mpc-solves.jsonl','w55-f/contact-events.jsonl')
    record['sources']={n:{'sha256':sha(run/n),'bytes':(run/n).stat().st_size} for n in files}
    record['video']=render(run,result,arrays)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--lane',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('runs',nargs='+',type=Path);args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    rows=[]
    start=time.monotonic()
    for path in args.runs:
        row=summarize(path.resolve());rows.append(row);print(row['run_id'],row['completed'],flush=True)
    (args.output/'result.json').write_text(json.dumps(dict(lane=args.lane,host=socket.gethostname(),
        rendering_wall_s=time.monotonic()-start,case_count=len(rows),completed=sum(r['completed'] for r in rows),
        runs=rows),indent=2,allow_nan=False)+'\n')
