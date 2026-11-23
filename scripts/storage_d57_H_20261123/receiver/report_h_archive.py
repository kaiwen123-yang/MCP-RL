"""Keep the finite archive lanes recoverable and relay verified H progress to D57's tmux publisher."""
from archive_common import *
import ctypes,sys,collections
from ctypes import wintypes
from catalog_db import connect
STAGES={'verify_previous':(62336,'previous-proof-complete.json','previous-verify'),
    'ingest':(10844,'scan-complete.json','ingest'),'retire_client':(56068,'retirement-done.json','retire'),
    'transfer_packs':(6480,'transfer-complete.json','transfer'),'receive_git':(48636,'git-history.json','git-history'),
    'late_retention_client':(43168,'retirement-reconciled.json','late-retention'),
    'finish':(49816,'归档结果.json','finish')}
def process_state(pid):
    k=ctypes.windll.kernel32
    k.OpenProcess.restype=wintypes.HANDLE
    handle=k.OpenProcess(0x1000,False,int(pid))
    if not handle:return None
    try:
        exit_code=wintypes.DWORD();created=wintypes.FILETIME();ended=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
        if not k.GetExitCodeProcess(wintypes.HANDLE(handle),ctypes.byref(exit_code)) or exit_code.value!=259:return None
        if not k.GetProcessTimes(wintypes.HANDLE(handle),ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel),ctypes.byref(user)):return None
        return (created.dwHighDateTime<<32)+created.dwLowDateTime
    finally:k.CloseHandle(wintypes.HANDLE(handle))
def is_transient(e):
    e=e.lower()
    return any(v in e for v in ('connection reset','connection closed','connection timed out','broken pipe','connection aborted','name or service not known',
                    'database is locked','network is unreachable','device is not ready','device is not connected','winerror 21','winerror 1167','winerror 1117'))
def loadjson(p):return json.loads(p.read_text(encoding='utf8'))
def start(name,prefix):
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    with (OPS/(prefix+'.log')).open('ab') as out,(OPS/(prefix+'.err')).open('ab') as err:
        return subprocess.Popen([sys.executable,'-u',name+'.py'],cwd=OPS,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                close_fds=True,startupinfo=startup,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS).pid
def main():
    check_volume();registry=OPS/'receiver-services.json'
    states=loadjson(registry) if registry.exists() else {n:dict(pid=v[0],creation=process_state(v[0]),recoveries=0) for n,v in STAGES.items()}
    write(registry,states);had_disconnect=False
    while True:
        try:check_volume()
        except OSError:had_disconnect=True;time.sleep(15);continue
        attention={}
        for name,(_,terminal,prefix) in STAGES.items():
            if (CAT/terminal).exists():continue
            v=states[name];starttime=process_state(v['pid'])
            if starttime and (not v['creation'] or v['creation']==starttime):continue
            err=(OPS/(prefix+'.err')).read_text(errors='replace')[-3200:]
            transient=had_disconnect or is_transient(err)
            if name=='transfer_packs' and 'require diagnosis' in err:
                with connect() as c:
                    rejected=c.execute("select id,err from batches where status='review'").fetchall()
                    if rejected and all(is_transient(e or '') for _,e in rejected):
                        c.executemany("update batches set status='ready' where id=?",[(i,) for i,_ in rejected]);transient=True
            if transient and v['recoveries']<3:
                v['pid']=start(name,prefix);v['creation']=process_state(v['pid']);v['recoveries']+=1
                v['reason']='verified remount or observed transport interruption'
                write(registry,states)
            else:attention[name]=err.splitlines()[-1][:300] if err.strip() else 'Expected completion receipt absent'
        had_disconnect=False
        t=loadjson(CAT/'transfer-progress.json') if (CAT/'transfer-progress.json').exists() else {}
        rm=loadjson(CAT/'retirement-progress.json') if (CAT/'retirement-progress.json').exists() else {}
        groups={r[0]:r[1:] for r in rm.get('groups',[])};removed=groups.get('removed',[0,0,0])
        status=dict(at=now(),state='needs_attention' if attention else 'running',terminal=False,
                volume='H:/MPC+RL',catalog='00_catalog/migration-20261123/archive-index.sqlite',
                verified_packs=t.get('verified_packs',0),new_H_bytes=t.get('new_verified_bytes',0),
                removed_files=removed[0],removed_bytes=removed[1],reclaimed_allocated_bytes=removed[2],
                peak_streams=t.get('peak_streams',0),sampled_streams=t.get('streams',0),MiB_per_s=t.get('MiB_per_s'),
                source_watermark='2026-11-23T23:29:49+08:00',attention=attention,**BG)
        final=CAT/'归档结果.json'
        if final.exists():
            d=loadjson(final);status.update(state=d['status'],terminal=True,verified_packs=d['new_unique_packs'],new_H_bytes=d['new_H_packed_bytes'],
                removed_files=d['removed_files'],removed_bytes=d['removed_logical_bytes'],reclaimed_allocated_bytes=d['removed_allocated_bytes'],
                h_verified_versions=d['h_verified_versions'],source_versions=d['logical_file_versions'],h_verified_logical_bytes=d['h_verified_logical_bytes'],
                unavailable=len(d['unavailable_paths']),results_sha256=sha(final),finished_shanghai=d['finished_shanghai'])
        write(CAT/'current.json',status)
        try:
            request('H-progress',status)
            if status['terminal']:
                request('receiver-closed',dict(status=status['state'],result_sha256=status['results_sha256'],at=now()))
                receipts=request('git-publications')
                write(CAT/'github-receipts.json',receipts)
                if any(k.endswith('/final') for k in receipts):break
        except (RuntimeError,OSError,ValueError) as e:print(now(),'D57 progress mirror pending',repr(e)[:400],flush=True)
        time.sleep(35 if status['terminal'] else 55)
    print('Finished',status['state'],now(),flush=True)
if __name__=='__main__':main()

