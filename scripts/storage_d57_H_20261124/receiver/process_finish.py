"""Match H's per-file lookup to the final D57 unlink ledger and verified experiment inputs."""
from archive_common import *
from retire_client import send_batch
from pathlib import Path
import json,time,sqlite3

def finish(c):
    # Revisit only the finite cohort already copied in full, if an actual W57 Release was still pending.
    while True:
        todo=[v[0] for v in c.execute("select id from retire_queue where state='archive:pending_release'")]
        if not todo:break
        admitted=[]
        for i in range(0,len(todo),1000):
            states=request('source-reasons',dict(ids=todo[i:i+1000]))
            for ident,why in states:
                if why.startswith('remove:'):admitted.append((ident,))
                elif why!='archive:pending_release':
                    c.execute('update retire_queue set state=? where id=?',(why,ident))
                    c.execute('update files set server_status=? where id=?',(why,ident))
        c.executemany("update retire_queue set state='pending' where id=?",admitted);c.commit()
        if admitted:
            while send_batch(c):pass
        else:
            if c.execute("select 1 from retire_queue where state='archive:pending_release' limit 1").fetchone():
                write(CAT/'current.json',dict(at=now(),phase='awaiting_original_publication_before_unlink',finite_pending=len(todo),terminal=False,**BG))
                time.sleep(45)
    check_volume()
    write(CAT/'current.json',dict(at=now(),phase='verifying_D57_dependencies_and_removed_paths',terminal=False,**BG))
    post=request('postcheck');write(CAT/'D57-after.json',post)
    aliases=request('aliases')
    c.executemany('insert or replace into aliases values(?,?)',aliases.items());c.commit()
    error=c.execute('pragma quick_check').fetchone()[0]
    if error!='ok':raise IOError('H catalog: '+error)
    totals=c.execute("select count(*),sum(f.n) from files f join objects o on f.h=o.h and f.n=o.n join archives a on o.archive=a.path where a.ok=1").fetchone()
    removed=c.execute("select count(*),coalesce(sum(n),0),coalesce(sum(alloc),0) from files where server_status='removed'").fetchone()
    if removed!=(post['removed_files'],post['removed_logical_bytes'],post['reclaimed_allocated_bytes']):raise ValueError('H/D57 retirement totals differ')
    scope=json.loads((CAT/'manifest-ready.json').read_text(encoding='utf8'))
    proof=json.loads((CAT/'archive-complete-proof.json').read_text(encoding='utf8'))
    if totals!=(scope['files'],scope['logical_bytes']):raise ValueError('H versions differ from source watermark')
    group=c.execute("select a.kind,count(*),sum(f.n) from files f join objects o on f.h=o.h join archives a on o.archive=a.path group by a.kind").fetchall()
    by_reason=c.execute('select q.state,count(*),sum(f.n) from retire_queue q join files f using(id) group by q.state').fetchall()
    c.execute('pragma wal_checkpoint(TRUNCATE)');c.close()
    index=CAT/'archive-index.sqlite'
    result=dict(status='completed',finished_shanghai=now(),watermark=json.loads((CAT/'scope.json').read_text(encoding='utf8'))['roots.json']['start_time'],
       archived_files=totals[0],archive_logical_bytes=totals[1],verified_groups=group,verified_new_H_bytes=proof['new_H_bytes'],
       new_packs=proof['new_packs'],reused_H_archives=proof['reused_H_archives'],removed_files=removed[0],removed_logical_bytes=removed[1],
       released_allocated_bytes=removed[2],released_per_mount=post['allocated_by_volume'],server_outcomes=by_reason,
       protected_path_unlinks=post['protected_path_unlinks'],regenerated_paths=post['regenerated_paths'],checked_reference_trees=len(post['named_reference_trees']),
       unverified_versions=0,remaining_new_transfer_bytes=0,scratch_free_bytes=post['scratch_free_bytes'],
       personal_free_bytes=post['personal_free_bytes'],personal_free_nodes=post['personal_free_nodes'],
       catalog=str(index),catalog_sha256=sha(index),restore='读取原件.py',**BG)
    write(CAT/'归档结果.json',result)
    request('final-H',result)
    request('close',dict(catalog_sha256=result['catalog_sha256'],removed_files=removed[0],at=now()))
    write(CAT/'current.json',dict(at=now(),phase='archive_and_retirement_complete; publishing_compact_receipt',terminal=False,**result))
    while True:
        refs=request('git-receipts')
        if refs.get('final'):
            write(CAT/'github-receipts.json',refs);break
        time.sleep(35)
    write(CAT/'current.json',dict(at=now(),phase='completed',terminal=True,github_commit=refs['final']['commit'],**result))
    print(now(),'completed',totals,removed,refs['final']['commit'],flush=True)

