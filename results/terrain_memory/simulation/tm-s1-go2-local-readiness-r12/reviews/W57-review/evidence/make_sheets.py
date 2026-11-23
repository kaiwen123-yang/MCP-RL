import argparse,json,pathlib,subprocess,io
from PIL import Image,ImageDraw,ImageFont
parser=argparse.ArgumentParser();parser.add_argument('--media-dir',type=pathlib.Path,required=True)
p=parser.parse_args().media_dir.resolve();items=json.loads((p/'download-manifest.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
sheets=[]
for a in items:
 if not a['name'].endswith('.mp4'): continue
 dur=float(a['ffprobe']['streams'][0]['duration'])
 times=([0,2,5,7.8,10,13,17,20,24,27,29,29.9] if dur<40 else [0,2,5,8,12,20,28,35,42,48,54,59.9] if dur<80 else [0,2,8,16,24,32,40,50,60,65,80,99])
 ims=[]
 for t in times:
  b=subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(t),'-i',str(p/a['name']),'-frames:v','1','-vf','scale=480:-1','-f','image2pipe','-vcodec','png','-threads','1','-'])
  im=Image.open(io.BytesIO(b)).convert('RGB');canvas=Image.new('RGB',(im.width,im.height+25),'#111827');canvas.paste(im,(0,25));ImageDraw.Draw(canvas).text((8,3),f'{t:07.3f} s',font=font,fill='white');ims.append(canvas)
 w,h=ims[0].size; sheet=Image.new('RGB',(4*w,3*h+30),'#111827');ImageDraw.Draw(sheet).text((8,5),a['name'],font=font,fill='white')
 for i,im in enumerate(ims):sheet.paste(im,((i%4)*w,30+(i//4)*h))
 dest=p/(a['name'].replace('.mp4','')+'-sheet.jpg');sheet.save(dest,quality=90)
 sheets.append({'video':a['name'],'sheet':str(dest),'times':times,'n':len(times)})
(p/'sheets-manifest.json').write_text(json.dumps(sheets,indent=2));print(json.dumps(sheets,indent=2))
