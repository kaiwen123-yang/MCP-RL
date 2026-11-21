"""Make traceable sampled video sheets; no generated imagery, no simulation."""
import argparse,concurrent.futures,io,json,pathlib,subprocess,threading
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
assets=json.loads(a.manifest.read_text());records=[];lock=threading.Lock()
def sheet(asset,tag,ts):
 name=asset['name'].replace('-full-mission-material-overlay.mp4','').replace('-recorded-pose-prefix.mp4','-pose-prefix')
 dst=a.output/(name+'-'+tag+'.jpg')
 images=[]
 for t in ([] if dst.exists() else ts):
  data=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-ss',str(t),'-i',asset['local_path'],'-frames:v','1','-f','image2pipe','-vcodec','mjpeg','-threads','1','-q:v','2','-'])
  im=Image.open(io.BytesIO(data)).convert('RGB');im.thumbnail((480,270));images.append(im)
 if not dst.exists():
  canvas=Image.new('RGB',(1920,38+300*((len(ts)+3)//4)),(246,247,249));d=ImageDraw.Draw(canvas);d.text((12,10),name+' | '+tag+' | recorded simulation seconds',font=font,fill=(15,30,46))
  for i,(t,im) in enumerate(zip(ts,images)):
   x=(i%4)*480;y=38+(i//4)*300;canvas.paste(im,(x,y));d.text((x+8,y+274),f'{t:.3f} s',font=font,fill=(15,30,46))
  canvas.save(dst,quality=88)
 row={'asset':asset['name'],'asset_sha256':asset['download_sha256'],'sheet':dst.name,'sampled_video_times_s':ts,'fully_decoded':asset['decode_returncode']==0,'visual_review':'pending'}
 with lock:
  records.append(row);(a.output/'new-manifest.json').write_text(json.dumps(records,indent=2)+'\n');print(dst.name,flush=True)
def one(x):
 dur=float(x['probe']['format']['duration']);name=x['name']
 if 'e600' in name:ts=[0,2.2,6.5,9,13,16.5,20,25,30,36,40,50,100,150,200,250,350,440,470,dur-.1]
 elif 'pose-prefix' in name:ts=[round((dur-.1)*i/11,3) for i in range(12)]
 elif 'e300' in name:ts=[0,6.5,13,20,30,40,60,80,100,120,150,180,210,240,270,dur-.1]
 elif 'e60' in name:ts=[0,6.5,13,20,30,36,38,40,42,46,52,dur-.1]
 else:ts=[0,3,6,9,12,15,18,21,24,27,29,dur-.1]
 sheet(x,'overview',ts)
 if 'r60-e600' in name or 'r55-e600' in name:
  sheet(x,'gait-outbound',[8+0.05*i for i in range(16)])
  sheet(x,'gait-return',[31+0.05*i for i in range(16)])
  sheet(x,'gait-last-lap',[435+0.05*i for i in range(16)])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for f in concurrent.futures.as_completed([pool.submit(one,x) for x in assets]):f.result()
