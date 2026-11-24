"""Reconcile H coverage, live-dependent retention, D57 unlink acknowledgments and restorable Git history."""
from archive_common import *
from catalog_db import connect
from collections import defaultdict
import gzip,statistics

def main():
    required=[CAT/'transfer-complete.json',CAT/'retirement-done.json',CAT/'git-history.json',CAT/'scan-complete.json']
    while not all(p.is_file() for p in required):time.sleep(15)
    check_volume();meta=request('project-meta')
    for name,action in [('tail-aliases.json.gz','tail-map'),('scan-issues.json.gz','scan-issues')]:
        f=CAT/name
        if not f.exists():
            tmp=INC/(name+'.part')
            with tmp.open('wb') as out:
                res=rem(action,out=out);out.flush();os.fsync(out.fileno())
            if res.returncode:raise RuntimeError(res.stderr.decode(errors='replace')[-500:])
            with gzip.open(tmp,'rt') as z:json.load(z)
            os.replace(tmp,f)
    c=connect()
    c.execute('create table if not exists tail_alias(id integer primary key,path text,previous_id integer)')
    with gzip.open(CAT/'tail-aliases.json.gz','rt') as f:aliases=json.load(f)
    c.executemany('insert or replace into tail_alias values(?,?,?)',aliases)
    c.execute("update files set retention=coalesce((select reason from classification where classification.id=files.id),'retain:initially_active_episode_final_version')")
    c.execute("update files set server_status=CASE WHEN kind!='file' THEN 'topology_only' WHEN h IS NULL THEN 'unavailable_at_snapshot' WHEN capture IS NOT NULL THEN 'retained:live_version' ELSE 'retained_verified' END WHERE server_status IS NULL")
    c.execute("update files set server_status='earlier_version_preserved' WHERE id in (select previous_id from tail_alias where previous_id is not null) and server_status!='removed'")
    c.execute('drop view if exists archive_lookup')
    c.execute("""CREATE VIEW archive_lookup AS SELECT f.id,coalesce(t.path,f.path) AS server_path,f.n AS bytes,f.h AS sha256,o.archive AS archive_relative_to_H_MPC_RL,o.member AS zip_member,f.kind,f.link,f.mt AS mtime_ns,f.mode,f.server_status,f.retention AS reason,f.state AS source_state,a.ok AS archive_verified FROM files f LEFT JOIN tail_alias t ON t.id=f.id LEFT JOIN objects o ON o.h=f.h AND o.n=f.n LEFT JOIN archives a ON o.archive=a.path""")
    c.commit()
    nfiles,total=c.execute("select count(*),sum(n) from files where kind='file'").fetchone()
    covered,covered_b=c.execute("select count(*),sum(bytes) from archive_lookup where kind='file' and archive_verified=1").fetchone()
    gaps=c.execute("select id,server_path,source_state from archive_lookup where kind='file' and coalesce(archive_verified,0)!=1").fetchall()
    allarch=c.execute("select distinct a.path,a.n,a.mt,a.h,a.kind from archives a join objects o on o.archive=a.path join files f on f.h=o.h and f.n=o.n where a.ok=1").fetchall()
    for p,n,mt,h,k in allarch:
        s=(ROOT/p).stat()
        if (s.st_size,s.st_mtime_ns)!=(n,mt):raise ValueError('H archive changed before final check: '+p)
    post=request('postcheck')
    write(CAT/'D57-after.json',post)
    if post['protected_named_paths_unlinked'] or not all(post['verified_inputs'].values()) or len(post['four_E600'])!=4 or any(not x['trajectory'] or not x['physical_chunks'] or not x['material_chunks'] for x in post['four_E600'].values()):
        raise RuntimeError('A protected D57 reference must be inspected before closure')
    removed=c.execute("select count(*),coalesce(sum(n),0),coalesce(sum(alloc),0) from files where server_status='removed'").fetchone()
    if removed!=(post['removed_files'],post['removed_bytes'],post['removed_allocated_bytes']):raise ValueError(('H/D57 retirement accounting',removed,post))
    state=c.execute("select server_status,retention,count(*),sum(n) from files where kind='file' group by server_status,retention order by sum(n) desc").fetchall()
    reuse=c.execute("select a.kind,count(*),sum(f.n) from files f join objects o on f.h=o.h and f.n=o.n join archives a on a.path=o.archive where f.kind='file' group by a.kind").fetchall()
    progress=[json.loads(l) for l in (OPS/'actual-transfer.jsonl').read_text().splitlines()]
    # The byte counter includes currently receiving partials. Report observed overlap and actual rates separately.
    per={}
    for limit in sorted({s['streams'] for s in progress}):
        actual=[s['MiB_per_s'] for s in progress if s['streams']==limit and s['MiB_per_s']>0]
        if actual:per[str(limit)]=dict(observations=len(actual),median_MiB_s=statistics.median(actual),maximum_MiB_s=max(actual))
    q=meta['catalog.done.json']
    summary=dict(status='completed' if not gaps and not post['reappeared_new_versions'] else 'covered_with_listed_exceptions',
        start_shanghai=meta['roots.json']['start_time'],finished_shanghai=now(),source_roots=meta['roots.json']['roots'],
        registered_worktrees=len(meta['roots.json']['registered']),initial_file_types=meta['inventory.done.json']['kinds'],
        initial_regular_bytes=sum(v[2] for v in meta['inventory.done.json']['logical_by_root']),
        finalized_initially_active=q['finalized_initially_active'],tail_records=q['tail_records'],logical_file_versions=nfiles,
        versioned_bytes=total,h_verified_versions=covered,h_verified_logical_bytes=covered_b,
        unavailable_paths=gaps,source_snapshot_issues=meta['inventory.done.json']['issues'],used_H_containers=len(allarch),
        previously_indexed_source_signature_matches=meta['prior-matched.json']['matches'],by_archive_kind=reuse,
        new_unique_packs=sum(k=='new_D57_pack' for p,n,mt,h,k in allarch),
        new_H_packed_bytes=sum(n for p,n,mt,h,k in allarch if k=='new_D57_pack'),
        removed_files=removed[0],removed_logical_bytes=removed[1],removed_allocated_bytes=removed[2],
        retained_or_removed_reasons=state,reappeared_after_acknowledged_cleanup=post['reappeared_new_versions'],
        scratch_free_bytes=post['scratch_free_bytes'],personal_free_inodes=post['personal_free_inodes'],
        max_receiver_jobs=max(s['peak_streams'] for s in progress),observed_throughput_by_concurrency=per,
        physics_during_postcheck=post['W57_dispatch'],archive_index=str(CAT/'archive-index.sqlite'),
        restore_helper=str(CAT/'读取原件.py'),git_reference_manifest=str(CAT/'git-history.json'),
        source_boundary='All project-owned regular paths in the recorded watermark, plus complete natural tails for the physical jobs active then. New experiments created later have their own publication cycle.',
        retained_boundary='Operating repositories, environments, shared/link targets, frozen checkpoints/builds, baselines and analysis/publication inputs retain their server paths.',**BG)
    if c.execute('pragma quick_check').fetchone()[0]!='ok':raise ValueError('H archive catalog database integrity')
    c.execute("insert or replace into meta values('completed_at',?)",(summary['finished_shanghai'],));c.commit()
    c.execute('pragma wal_checkpoint(truncate)');c.close()
    write(CAT/'归档结果.json',summary);print(now(),summary['status'],covered,removed,flush=True)
if __name__=='__main__':main()

