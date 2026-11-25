"""Prepare the canonical W57 Release ZIPs from immutable, completed episode files."""
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile


def groups_for(paths, video, part_size):
    members = []
    for prefix, path in paths:
        if path.is_dir():
            members.extend((prefix+'/'+str(f.relative_to(path)),f) for f in sorted(path.rglob('*'))
                           if f.is_file() and f != video)
        elif path.is_file():
            members.append((prefix,path))
        else:
            raise FileNotFoundError(path)
    groups, group, size = [], [], 0
    for member, path in members:
        n = path.stat().st_size
        if group and size+n > part_size:
            groups.append(group)
            group, size = [], 0
        group.append((member,path))
        size += n
    if group:
        groups.append(group)
    return groups, len(members)


def source_signature(group):
    signature = hashlib.sha256()
    for name, path in group:
        s = path.stat()
        signature.update(json.dumps([name,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode],
                                    separators=(',',':')).encode()+b'\n')
    return signature.hexdigest()


def prepare_part(directory, name, group):
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/name
    record = directory/(name+'.source.json')
    with (directory/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        signature = source_signature(group)
        if target.is_file() and record.is_file():
            saved = json.loads(record.read_text())
            if saved['source'] == signature and saved['bytes'] == target.stat().st_size:
                return target
        temporary = directory/(name+'.part-'+str(os.getpid()))
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for member,path in group:
                archive.write(path,member)
        if signature != source_signature(group):
            raise RuntimeError('Original changed during packing: '+name)
        os.replace(temporary,target)
        pending = record.with_suffix('.tmp-'+str(os.getpid()))
        pending.write_text(json.dumps(dict(source=signature,bytes=target.stat().st_size,members=len(group)))+'\n')
        os.replace(pending,record)
    return target


def episode_paths(large, attempt, movie_dir):
    source = large/attempt
    paths = [('episode',source),('job',large/'jobs'/attempt)]
    if movie_dir != source:
        paths.append(('episode/video-asset.json',movie_dir/'video-asset.json'))
    actor = large/'evaluation-policy'/attempt
    if actor.exists():
        paths.append(('frozen-speed-policy',actor))
    return paths


def prepare_episode(root, attempt, workers):
    from datetime import datetime
    rel = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
    large = root/'outputs'/rel
    receipt = json.loads((root/'results'/rel/'w57/attempts'/(attempt+'.json')).read_text())
    if not receipt.get('ended_unix_s'):
        raise RuntimeError('Packing requires a closed episode')
    movie_dir = large/attempt
    if not (movie_dir/'video-asset.json').is_file():
        movie_dir = root/'work/w57/rendered'/attempt
    movie = movie_dir/'full-mission-material-overlay.mp4'
    record = json.loads((movie_dir/'video-asset.json').read_text())
    if record['attempt'] != attempt or movie.stat().st_size != record['size']:
        raise RuntimeError('Recorded movie identity differs')
    groups,_ = groups_for(episode_paths(large,attempt,movie_dir),movie,512*1024**2)
    directory=root/'work/w57/prepared-release-parts'/attempt
    def run(item):
        i,batch=item
        path=prepare_part(directory,f'{attempt}-originals-{i:03d}.zip',batch)
        print(datetime.now().isoformat(timespec='seconds'),path.name,path.stat().st_size,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run,enumerate(groups,1)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--workers',type=int,default=4,choices=range(1,9))
    args=p.parse_args()
    prepare_episode(args.root,args.attempt,args.workers)
