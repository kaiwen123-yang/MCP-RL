"""Reconstitute the exact D57 shallow Git graph and every registered worktree head on H."""
from archive_common import *
import tempfile
def run():
    check_volume()
    while True:
        info=request('project-meta')
        if 'git-snapshot.json' in info:break
        time.sleep(10)
    manifest=info['git-snapshot.json']
    root=ROOT/'50_code_configs/20261123-d57'
    bare=root/'history.git';root.mkdir(parents=True,exist_ok=True)
    p=INC/'project-history.pack.part'
    source=next((f for f in (bare/'objects/pack').glob('*.pack') if f.stat().st_size==manifest['bytes']),None)
    if not source:
        if not p.exists() or p.stat().st_size!=manifest['bytes'] or sha(p)!=manifest['sha256']:
            with p.open('wb') as f:
                a=rem('git-pack',out=f);f.flush();os.fsync(f.fileno())
            if a.returncode:raise RuntimeError(a.stderr.decode(errors='replace')[-500:])
        source=p
    if source.stat().st_size!=manifest['bytes'] or sha(source)!=manifest['sha256']:raise ValueError('Git history pack differs')
    subprocess.run(['git','init','--bare','-q',str(bare)],check=True)
    if manifest['shallow']:(bare/'shallow').write_bytes(('\n'.join(manifest['shallow'])+'\n').encode())
    with source.open('rb') as f:f.seek(-20,2);oid=f.read(20).hex()
    target=bare/'objects/pack'/('pack-'+oid+'.pack')
    if source!=target:os.replace(source,target)
    actual=subprocess.check_output(['git','--git-dir',str(bare),'index-pack',str(target)],text=True).strip()
    if oid!=actual:raise ValueError('Git pack trailer/index disagree')
    refs=manifest['refs'];updates=''.join('update '+r+' '+h+'\n' for r,h in refs.items())
    subprocess.run(['git','--git-dir',str(bare),'update-ref','--stdin'],input=updates.encode(),check=True)
    if 'refs/heads/main' in refs:subprocess.run(['git','--git-dir',str(bare),'symbolic-ref','HEAD','refs/heads/main'],check=True)
    subprocess.run(['git','--git-dir',str(bare),'fsck','--connectivity-only','--no-dangling'],check=True)
    manifest.update(archive_relative_to_H_MPC_RL=target.relative_to(ROOT).as_posix(),pack_object_id=oid,connectivity_verified_at=now())
    write(CAT/'git-history.json',manifest);print('Verified',len(refs),'refs',len(manifest['worktrees']),'worktrees',manifest['bytes'],now(),flush=True)
if __name__=='__main__':run()

