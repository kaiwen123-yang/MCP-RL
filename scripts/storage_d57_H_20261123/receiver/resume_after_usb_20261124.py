"""Finish the already authorized Nov 23 snapshot after the H receiver's USB interruption."""
from archive_common import *
from catalog_db import connect
from retire_client import proofs, send_batch
from finish import main as final_audit
import hashlib,time,traceback,msvcrt

RECOVERY=CAT/'usb-return-20261124'
RECOVERY.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    check_volume()
    lock=(RECOVERY/'verify.lock').open('a+b');lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    c=connect()
    if not (CAT/'transfer-complete.json').is_file():raise RuntimeError('Transfer has not naturally completed')
    new=c.execute("select path,n,mt,h from archives where kind='new_D57_pack' and ok=1 order by path").fetchall()
    peak=max(json.loads(l)['peak_streams'] for l in (OPS/'actual-transfer.jsonl').open() if l.strip())
    checked={};ledger=RECOVERY/'sha256-readback.jsonl';last=0
    if ledger.exists():
        for l in ledger.read_text().splitlines():
            q=json.loads(l);checked[q['path']]=q
    def status(phase,terminal=False,extra=None):
        nonlocal last
        rm=c.execute("select count(*),coalesce(sum(n),0),coalesce(sum(alloc),0) from files where server_status='removed'").fetchone()
        v=dict(at=now(),state='completed' if terminal else 'running',phase=phase,terminal=terminal,
               volume='H:/MPC+RL',catalog='00_catalog/migration-20261123/archive-index.sqlite',
               verified_packs=len(new),new_H_bytes=sum(s[1] for s in new),remaining_new_transfer_bytes=0,
               removed_files=rm[0],removed_bytes=rm[1],reclaimed_allocated_bytes=rm[2],peak_streams=peak,
               post_boot_sha_archives=len(checked),post_boot_sha_bytes=sum(a['bytes'] for a in checked.values()),
               attention={},**BG)
        if extra:v.update(extra)
        write(CAT/'current.json',v);write(RECOVERY/'progress.json',v)
        try:request('H-progress',v)
        except (RuntimeError,OSError) as exc:print(now(),'Progress receipt pending',repr(exc)[:300],flush=True)
        last=time.monotonic();return v
    def verify(rows):
        with ledger.open('a',buffering=1) as out:
            for path,n,mt,h in rows:
                check_volume();f=ROOT/path;s=f.stat()
                if (s.st_size,s.st_mtime_ns)!=(n,mt):raise ValueError('Archive metadata changed: '+path)
                prior=checked.get(path)
                if prior and (prior['bytes'],prior['mtime_ns'],prior['sha256'])==(n,mt,h):continue
                actual=sha(f);end=f.stat()
                if actual!=h or (end.st_size,end.st_mtime_ns)!=(n,mt):raise ValueError('Post-reboot media readback failed: '+path)
                rec=dict(path=path,bytes=n,mtime_ns=mt,sha256=actual,at=now())
                out.write(json.dumps(rec,separators=(',',':'))+'\n');out.flush();os.fsync(out.fileno());checked[path]=rec
                if time.monotonic()-last>45:status('post_boot_H_readback')
    status('post_boot_H_readback')
    verify(new)
    # Re-evaluate only the initially authorized input inventory, with the same dependency protections.
    proofs(c)
    ep=CAT/'publication-eligibility.json';seen=set(load(ep).get('seen',[])) if ep.exists() else set()
    scan=request('late-policy',dict(seen=sorted(seen),final=True))
    c.executemany('insert or replace into classification values(?,?)',scan['eligible'])
    c.executemany('insert or ignore into retire_queue(id) values(?)',[(i,) for i,_ in scan['eligible']]);c.commit()
    seen.update(scan['considered']);write(ep,dict(at=now(),seen=sorted(seen),latest_eligible_count=len(scan['eligible']),**BG))
    remaining=c.execute("""SELECT DISTINCT a.path,a.n,a.mt,a.h FROM retire_queue q JOIN files f ON f.id=q.id
        JOIN objects o ON f.h=o.h AND f.n=o.n JOIN archives a ON a.path=o.archive
        WHERE q.state='pending' AND a.ok=1 AND f.state NOT LIKE 'captured%'""").fetchall()
    verify(remaining);status('H_rechecked_before_any_further_source_retirement')
    while send_batch(c):
        if time.monotonic()-last>30:status('verified_retirement')
    rm=c.execute("select count(*),coalesce(sum(n),0) from files where server_status='removed'").fetchone()
    write(CAT/'retirement-done.json',dict(at=now(),**BG))
    request('retirement-closed',dict(volume_serial='6063F4DA',removed_files=rm[0],bytes=rm[1],at=now()))
    write(CAT/'retirement-reconciled.json',dict(at=now(),removed_files=rm[0],bytes=rm[1],**BG))
    status('final_archive_and_live_dependency_audit');final_audit()
    final=CAT/'归档结果.json'
    d=load(final);record=status('final',True,dict(state=d['status'],h_verified_versions=d['h_verified_versions'],source_versions=d['logical_file_versions'],
            h_verified_logical_bytes=d['h_verified_logical_bytes'],unavailable=len(d['unavailable_paths']),
            results_sha256=sha(final),finished_shanghai=d['finished_shanghai']))
    request('receiver-closed',dict(status=record['state'],result_sha256=record['results_sha256'],at=now()))
    while True:
        request('H-progress',record)
        refs=request('git-publications');write(CAT/'github-receipts.json',refs)
        if any(k.endswith('/final') for k in refs):break
        time.sleep(35)
    print('Verified completion and publication',now(),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(now(),repr(exc),flush=True)
        if CAT.is_dir():
            try:
                v=load(CAT/'current.json');v.update(at=now(),state='needs_attention',terminal=False,attention={'post_boot_recovery':repr(exc)[:300]})
                write(CAT/'current.json',v);write(RECOVERY/'progress.json',v);request('H-progress',v)
            except (OSError,RuntimeError):pass
        raise

