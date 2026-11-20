import json,pathlib,subprocess,io,concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import argparse
_parser=argparse.ArgumentParser(description='Extract sampled frames from verified recorded videos; no simulation.')
_parser.add_argument('--workspace',type=pathlib.Path,required=True)
_args=_parser.parse_args()
root=_args.workspace; assets=json.loads((root/'w56-audit/video-manifest.json').read_text());out=root/'w56-audit/video';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
rows=[]
def one(a):
 dur=float(a['probe']['format']['duration']);name=a['name'].replace('w55-','').replace('-full-mission-material-overlay.mp4','');p=a['local_path']
 if 'e300' in name: ts=[0,6,13,20,30,40,60,80,100,120,150,180,210,240,270,299.9]
 elif 'r53-e60' in name: ts=[0,6,13,20,30,36,38,39,40,41,min(dur-.12,42),dur-.08]
 elif 'e60' in name:ts=[0,6,13,20,30,36,38,40,42,46,52,59.9]
 else:ts=[0,3,6,9,12,15,18,21,24,27,29,29.9]
 frames=[]
 for t in ts:
  b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-ss',str(t),'-i',p,'-frames:v','1','-f','image2pipe','-vcodec','mjpeg','-threads','1','-q:v','2','-'])
  im=Image.open(io.BytesIO(b)).convert('RGB');im.thumbnail((480,270));frames.append(im)
 cols=4;h=300;sheet=Image.new('RGB',(cols*480,((len(ts)+cols-1)//cols)*h+38),(246,247,249));d=ImageDraw.Draw(sheet);d.text((12,10),name+' | recorded simulation seconds',font=font,fill=(15,30,46))
 for i,(t,im) in enumerate(zip(ts,frames)):
  x=(i%cols)*480;y=38+(i//cols)*h;sheet.paste(im,(x,y));d.text((x+8,y+274),f'{t:.2f} s',font=font,fill=(15,30,46))
 dst=out/(name+'-overview.jpg');sheet.save(dst,quality=88)
 row={'asset':a['name'],'asset_sha256':a['download_sha256'],'sheet':dst.name,'sampled_video_times_s':ts,'fully_decoded':True,'visual_review':'pending'}
 print(dst.name,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: rows=list(ex.map(one,assets))
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
