from archive_common import *
from catalog_db import connect
from retire_client import send_batch,proofs
def main():
    c=connect()
    while not c.execute("select 1 from meta where k='policy_ready'").fetchone():time.sleep(5)
    seen=set();p=CAT/'publication-eligibility.json'
    if p.exists():seen.update(json.loads(p.read_text()).get('seen',[]))
    while True:
        final=(CAT/'retirement-done.json').is_file()
        v=request('late-policy',dict(seen=sorted(seen),final=final))
        if v['considered']:
            c.executemany('insert or replace into classification values(?,?)',v['eligible'])
            c.executemany("insert or ignore into retire_queue(id) values(?)",[(i,) for i,reason in v['eligible']])
            c.commit();seen.update(v['considered'])
            write(p,dict(at=now(),seen=sorted(seen),latest_eligible_count=len(v['eligible']),**BG))
            print(now(),'eligible membership refreshed',len(seen),len(v['eligible']),flush=True)
        if final:break
        time.sleep(55)
    while True:
        check_volume();proofs(c)
        if not send_batch(c):break
    stat=c.execute("select count(*),coalesce(sum(n),0) from files where server_status='removed'").fetchone()
    request('retirement-closed',dict(volume_serial='6063F4DA',removed_files=stat[0],bytes=stat[1],at=now()))
    write(CAT/'retirement-reconciled.json',dict(at=now(),removed_files=stat[0],bytes=stat[1],**BG));c.close()
if __name__=='__main__':main()

