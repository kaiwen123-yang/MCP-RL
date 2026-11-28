"""Reconstruct the recorded measurement-only prefix hook without changing any production collector."""
import hashlib,json
from pathlib import Path
from paired import SPEC,RAW
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cfg=json.loads(SPEC.read_text());before=Path(cfg['original_collector']);target=Path(cfg['frozen_collector'])
    if sha(before)!=cfg['original_collector_sha256']:raise ValueError('Frozen E collector identity differs')
    line='            if mission.phase == "complete" or n == round(a.duration/DT):'
    original=before.read_text()
    if original.count(line)!=1:raise ValueError('Missing frozen execution boundary')
    code='            if a.prefix_s is not None and n == round(a.prefix_s/DT):\n                reason="diagnostic_prefix"\n                break\n'
    data=original.replace(line,code+line)
    if hashlib.sha256(data.encode()).hexdigest()!=cfg['frozen_collector_sha256']:raise ValueError('Unexpected experimental source')
    if target.exists():
        if sha(target)!=cfg['frozen_collector_sha256']:raise ValueError('Existing snapshot differs')
    else:target.write_text(data)
    for name,source in ([('collect.py',target)]+[(Path(s).name,Path(s)) for s in cfg['original_dependencies']]):
        path=target.with_name(name)
        if not path.exists():path.symlink_to(source)
        if sha(path)!=sha(source):raise ValueError('Snapshot dependency mismatch '+name)

if __name__=='__main__':main()
