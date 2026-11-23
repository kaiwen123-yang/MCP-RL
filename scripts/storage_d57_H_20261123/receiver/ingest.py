from archive_common import *
from catalog_db import connect,initialize
import gzip,io,zipfile

def main():
    c=connect();initialize(c);old={}
    with olddb() as prev:
        for p,n,mt,h,kind in prev.execute('select path,bytes,mtime_ns,sha256,kind from archives'):
            c.execute('insert or ignore into archives values(?,?,?,?,?,0,NULL)',(p,n,mt,h,'previous:'+kind))
        for n,h,p,member in prev.execute('select c.bytes,c.sha256,a.path,c.member from contents c join archives a on a.id=c.archive_id'):
            old[h]=(n,p,member)
    c.commit()
    print(now(),'prior content entries',len(old),flush=True)
    field='id,path,root,kind,n,mt,ct,dev,ino,nl,alloc,mode,tr,link,op,h,state,capture'
    def ingest(num,rows):
        init={}
        for row in rows:
            if row[3]!='file' or not row[15]:continue
            h=row[15];n=row[4]
            if h in init:continue
            v=old.get(h)
            if v and v[0]!=n:raise ValueError('same digest differs in size')
            init[h]=(h,n,row[0],v[1] if v else None,v[2] if v else None)
        c.executemany('insert or ignore into files('+field+') values('+','.join('?'*18)+')',rows)
        c.executemany('insert or ignore into objects(h,n,rep,archive,member) values(?,?,?,?,?)',init.values())
        c.execute('insert or ignore into waves values(?,?)',(num,now()));c.commit()
    first=c.execute('select coalesce(max(id),0)+1 from waves').fetchone()[0]
    while True:
        check_volume();status=request('status')
        if status['scan'] and not (CAT/'scope.json').exists():write(CAT/'scope.json',status['scan'])
        if (not (CAT/'policy.jsonl.gz').is_file()) and status.get('policy_ready'):
            p=INC/'policy.jsonl.gz.part'
            with p.open('wb') as out:q=rem('policy',out=out)
            if q.returncode:raise RuntimeError(q.stderr.decode(errors='replace'))
            os.replace(p,CAT/'policy.jsonl.gz')
        if first<=status['waves']:
            q=rem('waves',{'first':first,'last':min(first+15,status['waves'])})
            if q.returncode:raise IOError(q.stderr.decode(errors='replace')[-800:])
            with zipfile.ZipFile(io.BytesIO(q.stdout)) as z:
                for zi in z.infolist():
                    num=int(zi.filename.split('.')[0])
                    if num!=first:raise ValueError('Out of sequence wave')
                    ingest(num,json.loads(gzip.decompress(z.read(zi))))
                    first+=1
            print(now(),'manifest through',first-1,'of',status['waves'],flush=True)
            continue
        if status['complete'] and (CAT/'policy.jsonl.gz').exists():break
        time.sleep(5)
    c.execute("insert or replace into meta values('all_waves','yes')");c.commit()
    write(CAT/'scan-complete.json',dict(at=now(),waves=first-1,source=status,**BG))
    c.close()
if __name__=='__main__':main()

