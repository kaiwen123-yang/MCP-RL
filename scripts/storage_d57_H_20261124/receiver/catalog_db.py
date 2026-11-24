from archive_common import *
def connect():
    check_volume()
    c=sqlite3.connect(CAT/'archive-index.sqlite',timeout=90)
    c.execute('pragma busy_timeout=90000')
    c.execute('pragma journal_mode=wal')
    c.execute('pragma synchronous=FULL')
    c.execute('pragma cache_size=-240000')
    c.execute('pragma mmap_size=536870912')
    c.execute('pragma temp_store=MEMORY')
    return c

def initialize(c):
    c.executescript("""
    CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT UNIQUE, root TEXT,kind TEXT,n INTEGER,mt INTEGER,ct INTEGER,dev INTEGER,ino INTEGER,nl INTEGER,alloc INTEGER,mode INTEGER,tr INTEGER,link TEXT,op INTEGER,h TEXT,state TEXT,capture TEXT,retention TEXT,server_status TEXT);
    CREATE TABLE IF NOT EXISTS objects(h TEXT PRIMARY KEY,n INTEGER NOT NULL,rep INTEGER,archive TEXT,member TEXT,batch INTEGER);
    CREATE TABLE IF NOT EXISTS archives(path TEXT PRIMARY KEY,n INTEGER,mt INTEGER,h TEXT,kind TEXT,ok INTEGER,checked TEXT);
    CREATE TABLE IF NOT EXISTS waves(id INTEGER PRIMARY KEY,ingested TEXT);
    CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY,path TEXT UNIQUE,status TEXT,n INTEGER,items INTEGER,err TEXT);
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
    CREATE INDEX IF NOT EXISTS need_object ON objects(archive,batch);
    CREATE INDEX IF NOT EXISTS f_h ON files(h);
    CREATE INDEX IF NOT EXISTS f_cleanup ON files(server_status,retention);
    CREATE VIEW IF NOT EXISTS archive_lookup AS SELECT f.id,f.path AS server_path,f.n AS bytes,f.h AS sha256,o.archive AS archive_relative_to_H_MPC_RL,o.member AS zip_member,f.kind,f.link,f.mt AS mtime_ns,f.mode,f.server_status,f.retention AS reason,f.state AS source_state,a.ok AS archive_verified FROM files f LEFT JOIN objects o ON o.h=f.h AND o.n=f.n LEFT JOIN archives a ON o.archive=a.path;
    """)
    c.commit()

