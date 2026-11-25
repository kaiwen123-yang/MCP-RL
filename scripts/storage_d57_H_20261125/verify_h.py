"""Close the H media catalog and read a representative retired member directly from its archived bytes."""
from pathlib import Path
import hashlib
import json
import sqlite3
import zipfile

from receive import BG, CAT, INC, OPS, PAY, ROOT, atomic, media, now

def main():
    media()
    final=json.loads((CAT/'final.json').read_text(encoding='utf-8'))
    if final['status']!='completed':raise ValueError('The receiver has not completed')
    proofs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OPS.glob('*-H.json'))]
    if len(proofs)!=final['packs'] or list(INC.glob('*.part')):raise ValueError('Incomplete H pack set')
    for proof in proofs:
        if not proof['h_readback'] or proof['serial']!='6063F4DA':raise ValueError(proof['pack'])
        b=proof.get('bundle')
        if b and (ROOT/b['path']).stat().st_size!=b['size']:raise ValueError('H final size '+b['path'])
    catalogs=CAT/'archive-index.sqlite'
    with sqlite3.connect(catalogs.as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(':memory:') as db:
        source.backup(db)
        if db.execute('pragma integrity_check').fetchone()[0]!='ok':raise ValueError('H catalog integrity')
        paths,logical=db.execute('select count(*),sum(bytes) from archive_lookup where archive_verified=1').fetchone()
        count,unique=db.execute('select count(*),sum(n) from objects').fetchone()
        sample=db.execute("select server_path,bytes,sha256,archive_relative_to_H_MPC_RL,zip_member from archive_lookup where server_status='H_verified_D57_retirement_batch_receipt' and server_path like '%/w55-f/mpc-%' limit 1").fetchone()
        if not sample:raise ValueError('Missing archived MPC snapshot')
        path,n,h,relative,member=sample
        base=ROOT/relative
        if member is None:
            size=base.stat().st_size
            with base.open('rb') as src:got=hashlib.file_digest(src,'sha256').hexdigest()
        else:
            with zipfile.ZipFile(base) as z:
                size=z.getinfo(member).file_size
                with z.open(member) as src:got=hashlib.file_digest(src,'sha256').hexdigest()
        if n!=size or h!=got:raise ValueError('Restored member digest')
    with catalogs.open('rb') as src:checksum=hashlib.file_digest(src,'sha256').hexdigest()
    zips=list(PAY.glob('*.zip'))
    value=dict(**final,serial='6063F4DA',verified_shanghai=now(),
               catalog_path=str(catalogs),catalog_sha256=checksum,catalog_size=catalogs.stat().st_size,
               new_zip_count=len(zips),new_zip_bytes=sum(p.stat().st_size for p in zips),
               verified_catalog_paths=paths,catalog_logical_bytes=logical,unique_content_objects=count,unique_content_bytes=unique,
               restore_member_verified=True,restore_example=dict(source=path,bytes=size,sha256=got,archive=relative,member=member))
    media()
    atomic(CAT/'verified-final.json',value)
    print(json.dumps(value,ensure_ascii=True))
if __name__=='__main__':main()
