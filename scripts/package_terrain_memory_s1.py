#!/usr/bin/env python3
"""Package only new S1 evidence on its execution host for direct Release upload."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/terrain_memory/s1-control-repair.json')
    parser.add_argument('--cohort',default='formal-v1')
    parser.add_argument('--release-tag',default='tm-s1-control-repair-evidence-20261024')
    a=parser.parse_args()
    c=json.loads((ROOT/a.config).read_text())
    out=ROOT/c['outputs']['large_root']
    small=ROOT/c['outputs']['compact_root']
    indexpath=small/'index.json'
    index=json.loads(indexpath.read_text())
    if not index['phase_A_complete']:
        raise ValueError('Phase A actual result records are not complete.')
    if index['phase_B_readiness']['ready'] and not index['phase_B_complete']:
        raise ValueError('Authorized ready phase B has not completed.')
    records=[(p,json.loads(p.read_text())) for p in sorted((small/a.cohort).glob('*-seed-*.json'))]
    new_records=[(p,r) for p,r in records if not r.get('reused_from')]
    archive_files={}
    owner={}
    cohort_directories={out/a.cohort}

    def member(path):
        if path.is_relative_to(out):
            return str(path.relative_to(out))
        if path.is_relative_to(small):
            return 'compact/'+str(path.relative_to(small))
        return 'repository/'+str(path.relative_to(ROOT))

    def assign(path,archive):
        path=(ROOT/path).resolve()
        if path not in owner:
            owner[path]=archive
            archive_files.setdefault(archive,[]).append(path)
        return path

    def episode_files(record_path,row):
        trace=(ROOT/row['trajectory']).resolve()
        stem=trace.stem
        cohort_directories.add(trace.parent)
        files=[trace,ROOT/row['material_trajectory'],record_path,
            Path(row.get('events_path',trace.parent/(stem+'-events.json')))]
        if row['controller'] in ('B1','B2','B3'):
            files.append(Path(row.get('plans_path',trace.parent/(stem+'-plans.json'))))
        if row['controller'] in ('B2','B3'):
            files.append(trace.parent/(stem+'-footstep-commitments.json'))
        return files

    for record_path,row in new_records:
        archive=c['run_id']+'-'+row['terrain']+'-'+row['controller']+'-trajectories.tar.gz'
        for path in episode_files(record_path,row):
            assign(path,archive)
    development_archive=c['run_id']+'-development-and-model.tar.gz'
    development=[]
    for attempt in index['attempts']:
        if attempt['kind']!='development':
            continue
        record_path=ROOT/attempt['record']
        row=json.loads(record_path.read_text())
        paths=[assign(path,development_archive) for path in episode_files(record_path,row)]
        development.append(dict(attempt_id=attempt['attempt_id'],record=str(record_path),
            files=[dict(execution_host_path=str(path),archive=owner[path],member=member(path)) for path in paths]))
    # Cohort snapshots retain the exact model, upstream patch and source at each
    # actual execution version, including an adopted development trajectory.
    for folder in sorted(cohort_directories):
        for path in [folder/'go2.mjb',folder/'pinned-upstream-executed.patch',*sorted((folder/'source').rglob('*'))]:
            if path.is_file():
                assign(path,development_archive)
        for path in sorted((small/folder.name).glob('*-resolved.json')):
            assign(path,development_archive)
    for path in [ROOT/a.config,ROOT/c['base']['material_config'],ROOT/'scripts/tm01_env.sh',
                 *[ROOT/'scripts'/name for name in ('run_terrain_memory_s1.py','summarize_terrain_memory_s1.py',
                    'video_terrain_memory_s1.py','package_terrain_memory_s1.py')],
                 *sorted((ROOT/'src/tm02').glob('*.py'))]:
        assign(path,development_archive)
    archives=[]
    for name,files in archive_files.items():
        archive=out/name
        if not archive.exists():
            with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
                for path in files:
                    tar.add(path,arcname=member(path),recursive=False)
        archives.append(archive)
    url='https://github.com/kaiwen123-yang/MCP-RL/releases'
    previous={item['filename']:item for item in index.get('publication',{}).get('assets',[])}
    previous_archives={item['filename']:item for item in index.get('publication',{}).get('archives',[])}
    assets,archive_entries=[],[]

    def asset_entry(path,digest=None):
        stat=path.stat()
        old=previous.get(path.name)
        unchanged=old and old['bytes']==stat.st_size and old.get('execution_host_mtime_ns')==stat.st_mtime_ns
        return dict(filename=path.name,execution_host_path=str(path),bytes=stat.st_size,
            sha256=digest or (old['sha256'] if unchanged else sha(path)),execution_host_mtime_ns=stat.st_mtime_ns,
            url=url+'/download/'+a.release_tag+'/'+path.name)

    for archive in archives:
        stat=archive.stat()
        if stat.st_size<2*1024**3:
            entry=asset_entry(archive)
            assets.append(entry)
            archive_entries.append(dict(filename=archive.name,bytes=stat.st_size,sha256=entry['sha256'],
                execution_host_mtime_ns=stat.st_mtime_ns,assets=[archive.name],split=False))
            continue
        old=previous_archives.get(archive.name)
        if old and old['bytes']==stat.st_size and old.get('execution_host_mtime_ns')==stat.st_mtime_ns:
            parts=[asset_entry(out/name) for name in old['assets']]
            whole_sha=old['sha256']
        else:
            # Only an actually oversized completed archive is split. Parts are
            # byte ranges of that one gzip stream, not independently compressed.
            parts=[]
            whole_hash=hashlib.sha256()
            with archive.open('rb') as source:
                remaining=stat.st_size
                part_number=1
                while remaining:
                    part=archive.with_name(archive.name+f'.part{part_number:03d}')
                    count=min(1_900_000_000,remaining)
                    part_hash=hashlib.sha256()
                    with part.open('wb') as target:
                        written=0
                        while written<count:
                            block=source.read(min(8*1024**2,count-written))
                            target.write(block)
                            whole_hash.update(block)
                            part_hash.update(block)
                            written+=len(block)
                    parts.append(asset_entry(part,part_hash.hexdigest()))
                    remaining-=count
                    part_number+=1
            whole_sha=whole_hash.hexdigest()
        assets.extend(parts)
        archive_entries.append(dict(filename=archive.name,bytes=stat.st_size,sha256=whole_sha,
            execution_host_mtime_ns=stat.st_mtime_ns,assets=[p['filename'] for p in parts],split=True,
            reassembly='Concatenate the listed parts in order to recover this exact tar.gz, then extract it.'))
    assets.extend(asset_entry(out/'videos'/Path(video['video']).name) for video in index['videos'] if 'video' in video)
    index['publication']=dict(release_url=url+'/tag/'+a.release_tag,uploaded=False,
        user_computer_secondary_archive=False,weights_repackaged=False,
        assets=assets,archives=archive_entries,development_attempts=development,
        archive_files=[dict(execution_host_path=str(path),archive=archive,member=member(path),bytes=path.stat().st_size)
            for path,archive in owner.items()],
        reused_B0_reference='Original C0/B0 arrays and videos are referenced by reused_records and videos URLs; not copied or repackaged.',
        development_reuse='Each actual source file belongs to one archive. Adopted development trajectories are in their formal terrain/controller archive and referenced here by development attempt ID.',
        asset_size_limit_bytes=2*1024**3,
        verification_method='Execution-host SHA256 and byte size are computed before direct upload; confirm against GitHub returned size/digest without full download.')
    indexpath.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(index['publication'],indent=2),flush=True)


if __name__=='__main__':main()
