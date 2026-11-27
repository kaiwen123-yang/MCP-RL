"""Compare two encoders on the full W59 E60, with identical frames in the existing renderer."""
import json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import render_s1_go2_r12 as renderer
import imageio_ffmpeg
r=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59')
renderer.SMALL=ROOT/r/'episodes'
renderer.LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
name='w59-ppo-full-fallback-5901-a-original-e60-a01'
original=renderer.imageio.get_writer
def run(folder,hw):
    folder.mkdir()
    def writer(uri,**kw):
        if hw:
            kw.update(codec='h264_nvenc',quality=None,ffmpeg_params=['-filter_threads','1','-threads','1','-preset','p4','-rc','vbr','-cq','15','-b:v','0'])
        return original(uri,**kw)
    renderer.imageio.get_writer=writer
    # Each fresh writer resolves its own executable, with the original setting restored for x264.
    if hw:os.environ['IMAGEIO_FFMPEG_EXE']='/usr/bin/ffmpeg'
    else:os.environ.pop('IMAGEIO_FFMPEG_EXE',None)
    started=time.perf_counter()
    renderer.render(name,output_dir=folder)
    dt=time.perf_counter()-started
    movie=folder/'full-mission-material-overlay.mp4'
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration','-of','json',str(movie)]))['streams'][0]
    return dict(wall_s=dt,bytes=movie.stat().st_size,video=meta,ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='w59-codec-timing-') as tmp:
    p=Path(tmp)
    cpu=run(p/'cpu',False);gpu=run(p/'gpu',True)
    base='full-mission-material-overlay.mp4'
    source=renderer.LARGE/name/base
    # Use the already published clip to verify fidelity, plus the original renderer control.
    quality={}
    for dest in ('cpu','gpu'):
        out=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-threads','1','-i',str(source),'-threads','1',
             '-i',str(p/dest/base),'-filter_complex_threads','1','-filter_complex','[0:v][1:v]ssim','-f','null','-'],
             capture_output=True,text=True,check=True)
        import re
        quality[dest]=float(re.search(r'All:([\d.]+)',out.stderr).group(1))
    result=dict(attempt=name,full_rerender=True,new_physical_steps=0,processors=sorted(os.sched_getaffinity(0)),
            cpu=cpu,gpu=gpu,similarity_to_existing_published_video=quality,
            wall_speedup=cpu['wall_s']/gpu['wall_s'])
    dest=ROOT/'work/w59/gpu-speed-20261127/codec.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
