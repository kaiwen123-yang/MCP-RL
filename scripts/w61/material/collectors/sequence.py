"""Each order holds the identical pair of physical slots/CPU groups across its CPU/GPU arms."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from paired import HERE,SPEC,STUDY,RAW,DEST,SITE,LEARN

POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')
def main():
    cfg=json.loads(SPEC.read_text())
    for src,h in {**cfg['original_dependencies'],cfg['original_collector']:cfg['original_collector_sha256'],
                   cfg['frozen_collector']:cfg['frozen_collector_sha256'],cfg['fused_binary']:cfg['fused_sha256']}.items():
        if hashlib.sha256(Path(src).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source differs: '+src)
    if len(sys.argv)>1:
        w=sys.argv[1];config=cfg['windows'][w]
        if hashlib.sha256(Path(config['weights']).read_bytes()).hexdigest()!=config['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
        for arm in config['order']:
            subprocess.run([sys.executable,'-u',str(HERE/'paired.py'),w,arm],check=True)
        return
    for w in cfg['windows']:
        cmd=[sys.executable,str(POOL),'--runtime',str(SITE),'--job','w61-e-packet-joint-'+w+('-a02' if w=='prefix12' else ''),'--slots','2','--cores','2',
             '--ram-gib','3.0','--gpu-mib','600','--scratch-gib','.45','--',sys.executable,'-u',str(HERE/'sequence.py'),w]
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
