"""Cut fixed two-second gait excerpts from an already rendered complete W59 original."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import tempfile

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

from score_s1_w59_episode import ROOT, REL, WORK
from publish_s1_go2_w54 import write


def excerpts(name):
    raw=ROOT/'outputs'/REL/name
    full=raw/'full-mission-material-overlay.mp4'
    with np.load(raw/'trajectory.npz') as z:
        t,phase,x=z['time'],z['phase'],z['qpos'][:,0]
    def first(mask):
        idx=np.flatnonzero(mask)
        return float(t[idx[0]]) if len(idx) else None
    with (raw/'material-events.jsonl').open() as f:
        loads=(json.loads(line) for line in f)
        load=next((v['time_s'] for v in loads if v['type']!='release' and v['normal_N']>1.),None)
    with gzip.open(WORK/'new-routes'/(name+'.jsonl.gz'),'rt') as f:
        cross=[v for line in f if (v:=json.loads(line))['outcome']=='positive_cross_foot_overlap']
    returned=first(phase==3)
    crossed=next((v for v in cross if returned is not None and v['target_s']>=returned),cross[0] if cross else None)
    clocks=[('start',first(phase==1)),('material_entry',load),('straight',first((phase==1)&(x>=1.5))),
            ('turn',first(phase==2)),('cross_foot_reload',crossed['target_s'] if crossed else None),
            ('completed_home_hold',first(phase==5))]
    target=WORK/'videos'/name
    target.mkdir(parents=True,exist_ok=True)
    output=target/(name+'-gait-excerpts.mp4')
    spans=[]
    ff=get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        root=Path(tmp);files=[]
        for label,clock in clocks:
            if clock is None:
                spans.append(dict(phase=label,observed=False))
                continue
            start=max(float(t[0]),min(float(t[-1])-2.,clock-.5))
            lo=int(np.ceil(30*(start-t[0])-1e-7))
            hi=int(np.ceil(30*(min(start+2.,float(t[-1]))-t[0])-1e-7))
            clip=root/f'{len(files):02d}.mp4'
            subprocess.run([ff,'-nostdin','-loglevel','error','-i',str(full),'-vf',
                   f'trim=start_frame={lo}:end_frame={hi},setpts=PTS-STARTPTS','-an','-c:v','libx264',
                   '-crf','18','-preset','veryfast','-threads','1',str(clip)],check=True)
            files.append(clip)
            spans.append(dict(phase=label,observed=True,event_s=float(clock),first_frame=lo,end_frame_exclusive=hi))
        manifest=root/'list.txt'
        manifest.write_text(''.join("file '"+str(f.name)+"'\n" for f in files))
        dest=root/output.name
        subprocess.run([ff,'-nostdin','-loglevel','error','-f','concat','-safe','1','-i',str(manifest),
                        '-c','copy',str(dest)],check=True)
        dest.replace(output)
    write(target/(name+'-gait-excerpts.json'),dict(attempt=name,fps=30,phases=spans,
          material_reload=crossed,video=output.name,full_video=full.name))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode')
    print(excerpts(p.parse_args().episode))
