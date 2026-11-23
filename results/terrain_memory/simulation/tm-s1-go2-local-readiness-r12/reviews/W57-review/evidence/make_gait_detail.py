from pathlib import Path
import argparse,io,subprocess
from PIL import Image,ImageDraw,ImageFont

parser=argparse.ArgumentParser();parser.add_argument('--media-dir',type=Path,required=True)
p=parser.parse_args().media_dir.resolve()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
names=['w57-m1-b-a-original-e30-a01-full.mp4','w57-m1-p-a-original-e30-a01-full.mp4',
       'w57-pilot02-full-a-e60-0010-full.mp4']
times=[9.0+i/10 for i in range(7)]
width,height=300,250
def get_frame(name,frame):
    raw=subprocess.check_output(['ffmpeg','-loglevel','error','-i',str(p/name),'-vf',f'select=eq(n\\,{frame})','-vsync','vfr','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
    return Image.open(io.BytesIO(raw)).convert('RGB')
sheet=Image.new('RGB',(7*width,3*(height+50)),(17,24,39)); draw=ImageDraw.Draw(sheet)
for row,name in enumerate(names):
    for col,t in enumerate(times):
        frame=int(round(t*30));im=get_frame(name,frame)
        im=im.crop((535,295,745,470)).resize((width,height))
        x,y=col*width,row*(height+50)
        draw.text((x+6,y+4),name.split('-full.mp4')[0].replace('w57-',''),font=font,fill='white')
        draw.text((x+6,y+25),f'video PTS {frame/30:.3f}s',font=font,fill='white')
        sheet.paste(im,(x,y+50))
sheet.save(p/'B-P-R-material-cycle-comparison.jpg',quality=94)

name=names[1];times=[9.4+i/30 for i in range(16)]
width,height=370,310
sheet=Image.new('RGB',(4*width,4*(height+30)),(17,24,39));draw=ImageDraw.Draw(sheet)
for i,t in enumerate(times):
    frame=int(round(t*30));im=get_frame(name,frame).crop((535,295,745,470)).resize((width,height))
    x,y=(i%4)*width,(i//4)*(height+30)
    draw.text((x+8,y+6),f'P A E30 | video PTS {frame/30:.3f}s',font=font,fill='white')
    sheet.paste(im,(x,y+30))
sheet.save(p/'P-A-E30-9.4-9.9-contact-detail.jpg',quality=94)
