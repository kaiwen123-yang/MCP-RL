"""H: volume, archive paths and D57 transport used by the incremental project archive."""
from pathlib import Path
import ctypes, hashlib, json, os, sqlite3, subprocess, time
from ctypes import wintypes
from datetime import datetime, timezone, timedelta

ROOT=Path('H:/MPC+RL')
OPS=ROOT/'99_transfer_logs/migration-20261124-process'
CAT=ROOT/'00_catalog/migration-20261124-process'
INC=ROOT/'90_incoming/migration-20261124-process'
PAY=ROOT/'40_support_evidence/20261124-d57-process'
OLD=ROOT/'00_catalog/migration-20261123/archive-index.sqlite'
REMOTE='/root/autodl-tmp/.mcp-rl-h-process-20261124'
PY='/root/autodl-tmp/MCP-RL/work/envs/d57/bin/python'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','-o','Compression=no','autodl-d57']
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
ZONE=timezone(timedelta(hours=8))
def now():return datetime.now(ZONE).isoformat(timespec='seconds')
def check_volume():
    name=ctypes.create_unicode_buffer(260); serial=wintypes.DWORD()
    ok=ctypes.windll.kernel32.GetVolumeInformationW('H:\\',name,len(name),ctypes.byref(serial),None,None,None,0)
    if not ok or name.value!='AI_DATA_8TB' or serial.value!=0x6063F4DA:
        raise OSError('The specified AI_DATA_8TB archive volume is unavailable')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,data):
    tmp=p.with_name(p.name+f'.{os.getpid()}.pending')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def olddb():return sqlite3.connect(OLD.as_uri()+'?mode=ro',uri=True)
def rem(action,payload=None,out=None):
    # Action names supplied by the program, paths/IDs transmitted only through stdin.
    cmd=f"cd {REMOTE} && exec ionice -c 2 -n 7 nice -n 12 {PY} process_rpc.py {action}"
    return subprocess.run(SSH+[cmd],input=(json.dumps(payload,separators=(',',':')).encode() if payload is not None else b''),stdout=out if out is not None else subprocess.PIPE,stderr=subprocess.PIPE)
def request(action,payload=None):
    x=rem(action,payload)
    if x.returncode:raise RuntimeError(f'{action}: '+x.stderr.decode(errors='replace')[-900:])
    return json.loads(x.stdout)

