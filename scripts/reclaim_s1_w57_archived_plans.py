"""Reclaim only archived per-MPC snapshots from the closed W57 blind training cohort on D57."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
import zipfile
from zoneinfo import ZoneInfo

import requests


ROOT = Path('/root/autodl-tmp/MCP-RL-w57')
PERSONAL = Path('/autodl-fs/data/MCP-RL/W57/originals')
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
R = ROOT/'results'/REL/'w57'
RAW = ROOT/'outputs'/REL
EVENT = '20261123-w57-archived-mpc'
# The volume first reported zero free inodes at 21:29:37. Episode 0014's
# receipt closed at 21:30:07, before the initial per-episode receipt inventory.
CUTOFF = datetime.fromisoformat('2026-11-23T21:30:08+08:00').timestamp()
WORK = ROOT/'work/w57'/EVENT
RESULT = R/'storage'/EVENT
REPO = 'kaiwen123-yang/MCP-RL'
RUN_NAME = re.compile(r'w57-pilot02-blind-[ab]-e60-[0-9]{4}')
FILE_NAME = re.compile(r'mpc-[0-9]{6}\.npz')
PREFIX = 'episode/w55-f/'
BG = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)
LOCAL = ZoneInfo('Asia/Shanghai')


def now():
    return datetime.now(LOCAL).isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=WORK, delete=False) as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    os.replace(f.name, path)


def nodes():
    s = os.statvfs(PERSONAL)
    return dict(total=s.f_files, free=s.f_favail, free_bytes=s.f_bavail*s.f_frsize)


def checked_episode(name):
    if not RUN_NAME.fullmatch(name):
        raise ValueError('Outside the W57 blind training cohort')
    directory = PERSONAL/name
    if directory.resolve(strict=True) != directory or (RAW/name).resolve(strict=True) != directory:
        raise ValueError('Canonical path changed for '+name)
    a = read(R/'attempts'/(name+'.json'))
    if not (a['attempt']==name and a['status']=='passed' and a['task_mode']=='training' and
            a['arm']=='R' and a['settings']['planner']=='P0' and a['settings']['timing_head']=='blind'
            and a['ended_unix_s']<=CUTOFF):
        raise ValueError('No completed frozen-source P0 blind collector receipt for '+name)
    pub = read(R/'delivery'/(name+'.json'))
    if not (pub['label']==name and pub['status']=='verified' and pub['verified_unix_s']<=CUTOFF and
            Path(pub['roots']['episode']).resolve(strict=True)==directory):
        raise ValueError('The complete episode lacked a verified receipt at the inventory time: '+name)
    return directory, pub


def inventory():
    spec = read(ROOT.parent/'MCP-RL-w58/experiments/terrain_memory/w58/material.json')
    protected = {n for group in spec['recorded_episodes'].values() for n in group}
    cohort=[]
    for d in sorted(PERSONAL.iterdir()):
        pub=R/'delivery'/(d.name+'.json')
        if not RUN_NAME.fullmatch(d.name) or d.name in protected or not pub.is_file():
            continue
        receipt=read(pub)
        if receipt.get('status')!='verified' or receipt['verified_unix_s']>CUTOFF:
            continue
        directory, receipt = checked_episode(d.name)
        m=read(directory/'w55-f/manifest.json')
        if not all(FILE_NAME.fullmatch(row['path']) for row in m['full_mpc_files']):
            raise ValueError('Unexpected MPC manifest name for '+d.name)
        cohort.append(dict(attempt=d.name,manifest_entries=len(m['full_mpc_files']),
                           assets=[{k:a[k] for k in ('asset_id','name','size','digest')} for a in receipt['assets']]))
    return dict(id=EVENT,cutoff_shanghai=datetime.fromtimestamp(CUTOFF,LOCAL).isoformat(),
                personal_root=str(PERSONAL),resource_before=nodes(),cohort=cohort,
                retained_inputs=sorted(protected),scope=PREFIX+'mpc-[0-9]{6}.npz',**BG)


def release_rows():
    release=read(R/'release.json')
    api=f"repos/{REPO}/releases/{release['databaseId']}/assets?per_page=100"
    pages=json.loads(subprocess.check_output(['gh','api','-H','Cache-Control: no-cache','--paginate','--slurp',api]))
    return {a['id']:a for page in pages for a in page}


def verify_assets(entry, remote):
    for a in entry['assets']:
        v=remote[a['asset_id']]
        if (v['name'],v['size'],v.get('digest'),v['state']) != (a['name'],a['size'],a['digest'],'uploaded'):
            raise ValueError('Remote Release size/SHA-256/state changed for '+a['name'])


class GitHubZip:
    """Read the actual central directory and only the ranges holding requested members."""
    def __init__(self, api, token):
        self.id=api['id'];self.size=api['size'];self.s=requests.Session()
        hdr={'Authorization':'Bearer '+token,'Accept':'application/octet-stream','Cache-Control':'no-cache'}
        try:
            with self.s.get(api['url'],headers=hdr,params={'storage_read':time.time_ns()},
                            stream=True,allow_redirects=False,timeout=(12,45)) as response:
                self.url=response.headers.get('location','')
                if response.status_code not in (302,307) or urlsplit(self.url).scheme!='https' or not self.url:
                    raise RuntimeError(f'Asset {self.id}: GitHub redirect HTTP {response.status_code}')
        except requests.RequestException as error:
            raise RuntimeError(f'Asset {self.id}: {type(error).__name__} fetching redirect') from None

    def get(self, start, stop):
        expected=f'bytes {start}-{stop-1}/{self.size}'
        try:
            with self.s.get(self.url,headers={'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity'},
                            stream=True,timeout=(12,60)) as res:
                if res.status_code!=206 or res.headers.get('Content-Range')!=expected:
                    raise RuntimeError(f'Asset {self.id}: unexpected range at {start}..{stop}; status={res.status_code}')
                data=res.content
        except requests.RequestException as error:
            raise RuntimeError(f'Asset {self.id}: {type(error).__name__} reading range {start}..{stop}') from None
        if len(data)!=stop-start:raise IOError(f'Asset {self.id}: incomplete range')
        return data

    def hashes(self, wanted):
        start=max(0,self.size-65557);tail=self.get(start,self.size)
        at=tail.rfind(b'PK\x05\x06')
        end=struct.unpack_from('<4s4H2IH',tail,at)
        _,disk,cd_disk,here,count,cd_size,cd_off,comment=end
        if at<0 or at+22+comment!=len(tail) or disk or cd_disk or here!=count or cd_off+cd_size!=start+at:
            raise ValueError(f'Unexpected W57 part footer, asset {self.id}')
        found={}
        with tempfile.TemporaryFile(dir=WORK) as backing:
            backing.seek(start);backing.write(tail)
            if cd_off<start:
                backing.seek(cd_off);backing.write(self.get(cd_off,cd_off+cd_size))
            backing.flush()
            with zipfile.ZipFile(backing) as z:
                infos=sorted(z.infolist(),key=lambda x:x.header_offset)
                spans=[]
                for i,info in enumerate(infos):
                    if info.filename not in wanted:continue
                    after=infos[i+1].header_offset if i+1<len(infos) else cd_off
                    if not spans or spans[-1][1]!=info.header_offset or after-spans[-1][0]>8*1024**2:
                        spans.append([info.header_offset,after,[info]])
                    else:
                        spans[-1][1]=after;spans[-1][2].append(info)
                for first,last,entries in spans:
                    data=self.get(first,last);backing.seek(first);backing.write(data);backing.flush()
                    for info in entries:
                        if info.filename in found:raise ValueError('Duplicate archived member '+info.filename)
                        payload=z.read(info)  # validates the remote ZIP member's CRC before hashing
                        found[info.filename]=[self.id,info.file_size,hashlib.sha256(payload).hexdigest()]
        self.s.close()
        return found


def open_plans():
    opened=set()
    for d in Path('/proc').iterdir():
        if not d.name.isdecimal():continue
        try:
            for f in (d/'fd').iterdir():
                try:s=os.readlink(f)
                except OSError:continue
                if s.startswith(str(PERSONAL)+'/') and '/w55-f/mpc-' in s:
                    opened.add(s)
        except OSError:continue
    return opened


def archived_hashes(entry, remote, token):
    name=entry['attempt'];directory,_=checked_episode(name)
    src=read(directory/'w55-f/manifest.json')['full_mpc_files']
    wanted={PREFIX+r['path']:r['size'] for r in src}
    if len(wanted)!=entry['manifest_entries'] or any(not FILE_NAME.fullmatch(s[len(PREFIX):]) for s in wanted):
        raise ValueError('Changed MPC manifest for '+name)
    proof=WORK/(name+'-archive.json')
    if proof.exists():
        recorded=read(proof)
        if recorded['assets']!=entry['assets']:raise ValueError('Archive identity differs from the stored verification')
        return recorded['members']
    found={}
    for a in entry['assets']:
        if not a['name'].endswith('.zip'):continue
        part=GitHubZip(remote[a['asset_id']],token).hashes(wanted)
        if found.keys() & part.keys():raise ValueError('Member repeated across Release parts: '+name)
        found.update(part)
    if {k:v[1] for k,v in found.items()}!=wanted:raise ValueError('Release membership differs from original manifest: '+name)
    members={k[len(PREFIX):]:v for k,v in found.items()}
    write(proof,dict(assets=entry['assets'],members=members))
    return members


def append(log, item):
    log.write(json.dumps(item,separators=(',',':'))+'\n')
    log.flush();os.fsync(log.fileno())


def fingerprint(st):
    return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]


def reclaim(entry,members,pool):
    name=entry['attempt'];directory,_=checked_episode(name)
    path=directory/'w55-f'
    if path.resolve(strict=True)!=path:raise ValueError('Unexpected plan directory: '+str(path))
    directory_fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    logpath=WORK/(name+'-actions.jsonl')
    checked={};removed=set()
    if logpath.exists():
        for line in logpath.read_text().splitlines():
            record=json.loads(line)
            if record['action']=='matched':checked.update(record['files'])
            else:removed.update(record['files'])
    def local_hash(filename):
        expected=members[filename]
        fd=os.open(filename,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory_fd)
        with os.fdopen(fd,'rb') as f:
            st=os.fstat(fd);before=fingerprint(st)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size!=expected[1]:
                raise ValueError('Local size/type/link mismatch: '+str(path/filename))
            value=hashlib.file_digest(f,'sha256').hexdigest()
            if value!=expected[2] or before!=fingerprint(os.fstat(fd)):
                raise ValueError('Local SHA-256 mismatch or changing file: '+str(path/filename))
            return filename,before
    def unlink_verified(pair):
        filename,witness=pair
        st=os.stat(filename,dir_fd=directory_fd,follow_symlinks=False)
        if fingerprint(st)!=witness or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
            raise RuntimeError('File changed after verification: '+str(path/filename))
        os.unlink(filename,dir_fd=directory_fd)
        return filename
    try:
        with logpath.open('a') as log:
            pending=sorted(set(members)-removed)
            for i in range(0,len(pending),64):
                batch=pending[i:i+64]
                busy=open_plans()
                batch=[f for f in batch if str(path/f) not in busy]
                present=[];closed=[]
                for f in batch:
                    if f in checked and not (path/f).exists():
                        closed.append(f)  # committed verification preceded a prior interrupted unlink
                    else:present.append(f)
                witness=dict(pool.map(local_hash,present))
                append(log,dict(action='matched',files=witness))
                closed.extend(pool.map(unlink_verified,witness.items()))
                append(log,dict(action='removed',files=closed))
                removed.update(closed)
                write(WORK/'progress.json',dict(at=now(),attempt=name,removed_in_episode=len(removed),
                                                  planned_in_episode=len(members),resource=nodes(),**BG))
    finally:os.close(directory_fd)
    if len(removed)!=len(members):raise RuntimeError('A requested member is open; resume after the owning reader exits')
    return dict(attempt=name,files=len(removed),bytes=sum(members[f][1] for f in removed),assets=entry['assets'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','reclaim'])
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    cohort_path=WORK/'cohort.json'
    with (WORK/'reclaim.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='inventory':
            if not cohort_path.exists():
                write(cohort_path,inventory())
            previous=read(cohort_path)
            stamp=datetime.fromtimestamp(CUTOFF,LOCAL).isoformat()
            if previous['cutoff_shanghai']!=stamp:
                updated=inventory()
                first={r['attempt']:r for r in previous['cohort']}
                newer={r['attempt']:r for r in updated['cohort']}
                if not all(newer.get(n)==v for n,v in first.items()) or set(newer)-set(first)!={'w57-pilot02-blind-a-e60-0014'}:
                    raise ValueError('Cohort differs from the fourteen verified episodes in the receipt inventory')
                write(WORK/'mount-check-cohort.json',previous)
                updated.update(resource_before=previous['resource_before'],
                               mount_check_cutoff=previous['cutoff_shanghai'])
                write(cohort_path,updated)
            data=read(cohort_path)
            print(len(data['cohort']),sum(x['manifest_entries'] for x in data['cohort']));return
        frozen=read(cohort_path)
        token=subprocess.check_output(['gh','auth','token'],text=True).strip()
        remote=release_rows()
        report_path=RESULT/'receipt.json'
        report=read(report_path) if report_path.exists() else dict(id=EVENT,started_at=now(),
                cutoff_shanghai=frozen['cutoff_shanghai'],resource_before=frozen['resource_before'],
                proof='GitHub uploaded asset ID, release size/SHA-256 vs D57 publication receipt; exact ZIP member bytes vs SHA-256 of the local regular file before unlink.',
                scope='Only full_mpc_files from previously delivered P0/seed5701 blind training episodes on the D57 personal volume.',
                local_record_root=str(WORK),planned_files=sum(x['manifest_entries'] for x in frozen['cohort']),
                closed_episodes={},**BG)
        report.update(status='running',cutoff_shanghai=frozen['cutoff_shanghai'],
                      planned_files=sum(x['manifest_entries'] for x in frozen['cohort']))
        write(report_path,report)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for entry in frozen['cohort']:
                    name=entry['attempt'];verify_assets(entry,remote)
                    if name in report['closed_episodes']:continue
                    print(now(),'verifying',name,entry['manifest_entries'],flush=True)
                    members=archived_hashes(entry,remote,token)
                    item=reclaim(entry,members,pool)
                    report['closed_episodes'][name]=item;report['last_event_at']=now()
                    report['resource_after']=nodes();write(report_path,report)
                    print(now(),'removed',name,item['files'],report['resource_after']['free'],flush=True)
            tmp=WORK/'members.jsonl.gz'
            with gzip.open(tmp,'wt') as out:
                for entry in frozen['cohort']:
                    name=entry['attempt']
                    for f,(asset,size,sha) in sorted(read(WORK/(name+'-archive.json'))['members'].items()):
                        out.write(json.dumps([name,f,asset,size,sha],separators=(',',':'))+'\n')
            os.replace(tmp,RESULT/'members.jsonl.gz')
            report.update(status='completed',ended_at=now(),resource_after=nodes(),
                    removed_files=sum(x['files'] for x in report['closed_episodes'].values()),
                    removed_bytes=sum(x['bytes'] for x in report['closed_episodes'].values()),
                    member_columns=['attempt','mpc_file','release_asset_id','bytes','SHA256'])
            write(report_path,report)
            print('COMPLETED',report['removed_files'],report['removed_bytes'],report['resource_after'],flush=True)
        except Exception as error:
            report.update(status='interrupted',last_event_at=now(),resource_after=nodes(),reason=str(error))
            write(report_path,report)
            raise


if __name__=='__main__':main()
