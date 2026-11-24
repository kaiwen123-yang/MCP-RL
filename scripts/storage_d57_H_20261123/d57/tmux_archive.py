"""Own D57 pack and H-verified retirement workers in tmux; the H receiver owns the destination disk."""
import contextlib,errno,fcntl,hashlib,json,os,re,select,subprocess,sys,time,uuid
from pathlib import Path
from inventory import CTL,BG,now,save

Q=CTL/'queue';Q.mkdir(exist_ok=True)
def row(p):return json.loads(p.read_text())
def atomic(p,v):
    tmp=p.with_name(p.name+f'.{os.getpid()}.part')
    tmp.write_text(json.dumps(v,separators=(',',':')));os.replace(tmp,p)
def proc_start(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[19]
    except (OSError,IndexError):return None
def jobpath(key,kind):
    if not re.fullmatch('[0-9a-f]{32,64}',key):raise ValueError('Unknown archive ticket')
    return Q/(key+'.'+kind)
def stream(req):
    key=uuid.uuid4().hex;pipe=jobpath(key,'fifo');os.mkfifo(pipe,0o600)
    read_fd=os.open(pipe,os.O_RDONLY|os.O_NONBLOCK)
    done=jobpath(key,'done')
    atomic(jobpath(key,'pack'),dict(req=req,reader=os.getpid(),start=proc_start(os.getpid()),issued=now()))
    try:
        events=select.poll();events.register(read_fd,select.POLLIN|select.POLLHUP|select.POLLERR)
        while True:
            events.poll(500)
            try:data=os.read(read_fd,256*1024)
            except BlockingIOError:continue
            if data:
                sys.stdout.buffer.write(data);continue
            if done.exists():
                end=row(done)
                if end['status']!='completed':raise RuntimeError(end['reason'])
                sys.stdout.buffer.flush();return
            time.sleep(.04)
    finally:
        os.close(read_fd);pipe.unlink(missing_ok=True)
def store_retire(req):
    key=hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    p=jobpath(key,'retire');d=jobpath(key,'done')
    if not p.exists() and not d.exists():atomic(p,req)
    while not d.exists():time.sleep(.25)
    v=row(d)
    if v['status']!='completed':raise RuntimeError(v['reason'])
    return v['result']
def serve(key,kind):
    from packet import pack,retire,affinity
    affinity();req=row(jobpath(key,kind));d=jobpath(key,'done')
    try:
        if kind=='pack':
            if proc_start(req['reader'])!=req['start']:raise RuntimeError('SSH receiver ended before pack admission')
            fd=os.open(jobpath(key,'fifo'),os.O_WRONLY|os.O_NONBLOCK);os.set_blocking(fd,True)
            with os.fdopen(fd,'wb') as dest:pack(req['req'],output=dest)
            atomic(d,dict(status='completed',at=now(),objects=len(req['req']['objects'])))
        else:
            result=retire(req);atomic(d,dict(status='completed',result=result,at=now()))
    except Exception as e:
        atomic(d,dict(status='interrupted',reason=repr(e)[-700:],at=now()))
        raise
def main():
    lock=(CTL/'tmux-archive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={};seen=set();peak=0;last=0
    with (CTL/'archive-workers.log').open('a',buffering=1) as log:
        while True:
            for key,(proc,kind) in list(children.items()):
                if proc.poll() is not None:
                    d=jobpath(key,'done')
                    if not d.exists():atomic(d,dict(status='interrupted',reason=f'Worker exit {proc.returncode}',at=now()))
                    del children[key]
            for kind in ('retire','pack'):
                for req in Q.glob('*.'+kind):
                    key=req.stem
                    if key in seen:continue
                    if jobpath(key,'done').exists():
                        seen.add(key);continue
                    if kind=='retire' and any(k=='retire' for p,k in children.values()):continue
                    # Only requests containing an actual attached H receiver cause a data stream to run.
                    p=subprocess.Popen([sys.executable,'-u','tmux_archive.py',kind,key],cwd=CTL,
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children[key]=(p,kind);seen.add(key)
            cnt=sum(k=='pack' for p,k in children.values());peak=max(peak,cnt)
            if time.monotonic()-last>6:
                save('tmux-archive.json',dict(at=now(),pid=os.getpid(),tmux='mcp-rl-H-archive-20261123',
                    pack_pids=[p.pid for p,k in children.values() if k=='pack'],
                    cleanup_pids=[p.pid for p,k in children.values() if k=='retire'],
                    peak_pack_children=peak,**BG));last=time.monotonic()
            if (CTL/'receiver-closed.json').exists() and not children:return
            time.sleep(.16 if children else .5)
if __name__=='__main__':
    if len(sys.argv)>1:serve(sys.argv[2],sys.argv[1])
    else:main()
