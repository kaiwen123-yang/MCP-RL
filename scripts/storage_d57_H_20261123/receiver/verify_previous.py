"""Check the exact H archives backing the still-present Nov 22 content; read-only on D57."""
from archive_common import *
def main():
    check_volume()
    p=CAT/'previous-proofs.jsonl';done={}
    if p.exists():
        for l in p.read_text(encoding='utf8').splitlines():
            v=json.loads(l);done[v['path']]=v
    with olddb() as db:
        rows=db.execute("""SELECT a.path,a.bytes,a.mtime_ns,a.sha256,
            sum(CASE WHEN f.reason='saved_input_evaluation_or_regression_episode' THEN c.bytes ELSE 0 END) AS priority
            FROM files f JOIN contents c ON f.content_id=c.id JOIN archives a ON c.archive_id=a.id
            WHERE f.server_status LIKE 'retained%' GROUP BY a.id ORDER BY (priority>0) DESC,
            CASE WHEN priority>0 THEN 1.0*priority/a.bytes ELSE 0 END DESC,a.path""").fetchall()
    print(now(),'H archives queued',len(rows),sum(r[1] for r in rows),flush=True)
    with p.open('a',encoding='utf8',buffering=1) as out:
        for rel,size,mt,h,_ in rows:
            check_volume();path=ROOT/rel
            if rel in done and done[rel]['ok'] and path.is_file() and path.stat().st_size==size and path.stat().st_mtime_ns==done[rel]['mtime_ns']:continue
            st=path.stat() if path.exists() else None
            ok=bool(st and st.st_size==size and sha(path)==h)
            if ok and (path.stat().st_size,path.stat().st_mtime_ns)!=(st.st_size,st.st_mtime_ns):ok=False
            q=dict(path=rel,bytes=size,mtime_ns=st.st_mtime_ns if st else None,sha256=h,ok=ok,verified_at=now())
            out.write(json.dumps(q,separators=(',',':'))+'\n');out.flush();os.fsync(out.fileno())
    write(CAT/'previous-proof-complete.json',dict(at=now(),archives=len(rows),**BG))
if __name__=='__main__':main()

