"""Expose H media receipts and real transfer counts to the finite D57 archive session."""
from archive_common import *
def latest(p):
    return json.loads(p.read_text(encoding='utf8')) if p.is_file() else {}
def main():
    while True:
        check_volume()
        state=latest(CAT/'current.json');manifest=latest(CAT/'manifest-ready.json')
        counts={}
        if manifest:
            with sqlite3.connect((CAT/'archive-index.sqlite').as_uri()+'?mode=ro',uri=True,timeout=40) as c:
                counts['pack_states']=list(c.execute('select status,count(*),sum(n) from batches group by status'))
                counts['new_verified_packs'],counts['new_H_packed_bytes']=c.execute("select count(*),coalesce(sum(n),0) from archives where kind='new_D57_pack' and ok=1").fetchone()
                counts['reused_containers_verified']=c.execute("select count(*) from archives where kind='reused_H_archive' and ok=1").fetchone()[0]
                counts['new_verified_source_bytes']=c.execute("select coalesce(sum(n),0) from batches where status='verified'").fetchone()[0]
                counts['remaining_new_source_bytes']=max(0,manifest['new_source_bytes']-counts['new_verified_source_bytes'])
        v=dict(at=now(),archive_root=str(ROOT),current_phase=state.get('phase'),manifest=manifest,counts=counts,
               transfer=latest(CAT/'transfer-progress.json'),retirement=latest(CAT/'retirement-done.json'),
               terminal=bool(state.get('terminal')),**BG)
        if state.get('removed_files') is not None:v.update(removed_files=state['removed_files'],removed_allocated_bytes=state.get('removed_allocated',state.get('released_allocated_bytes',0)))
        write(CAT/'live.json',v)
        try:request('H-progress',v)
        except (OSError,RuntimeError) as exc:print(now(),str(exc)[:240],flush=True)
        if v['terminal']:return
        time.sleep(35)
if __name__=='__main__':main()

