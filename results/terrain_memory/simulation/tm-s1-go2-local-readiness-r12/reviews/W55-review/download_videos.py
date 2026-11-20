import concurrent.futures, hashlib,json,pathlib,subprocess,urllib.request,os,time
import argparse
_parser=argparse.ArgumentParser(description="Download and verify released full videos into an analysis workspace; no simulation.")
_parser.add_argument("--workspace",type=pathlib.Path,required=True)
_args=_parser.parse_args()
root=_args.workspace
assets=json.loads((root/"w56-audit/release-inventory.json").read_text())["assets"]
out=root/"w56-originals/videos";out.mkdir(parents=True,exist_ok=True)
def one(a):
 p=out/a["name"]; h=hashlib.sha256()
 if not p.exists() or p.stat().st_size!=a["size"]:
  req=urllib.request.Request(a["url"],headers={"User-Agent":"MCP-RL-evidence-review"})
  with urllib.request.urlopen(req,timeout=90) as src, p.open("wb") as dst:
   while True:
    b=src.read(1024*1024)
    if not b: break
    dst.write(b)
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 expected=a["sha256"].removeprefix("sha256:")
 assert p.stat().st_size==a["size"] and h.hexdigest()==expected,a["name"]
 probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames,duration","-of","json",str(p)]))
 decoded=subprocess.run(["ffmpeg","-v","error","-i",str(p),"-f","null","-"],capture_output=True,text=True)
 r={**a,"local_path":str(p),"download_sha256":h.hexdigest(),"sha_verified":True,"probe":probe,"decode_returncode":decoded.returncode,"decode_stderr":decoded.stderr}
 print(json.dumps({"name":a["name"],"bytes":a["size"],"verified":True,"decode_exit":decoded.returncode,"duration":probe.get("format",{}).get("duration")}),flush=True)
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 results=list(ex.map(one,[a for a in assets if a["name"].endswith(".mp4")]))
(root/"w56-audit/video-manifest.json").write_text(json.dumps(results,indent=2))
