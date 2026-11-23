"""Fetch only the pinned public assets used in this read-only evidence review.

No SSH, simulator, training, source deletion or archive upload is performed.
Large source files should remain outside the git repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir',type=Path,required=True)
    parser.add_argument('--only-raw',action='store_true',help='Fetch the single P/A/E30 ZIP; omit all seven MP4 files.')
    args=parser.parse_args()
    target=args.media_dir.resolve();target.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path(__file__).with_name('media-sources-and-timecodes.json').read_text())
    verified=[]
    for entry in source['assets']:
        if args.only_raw and not entry['name'].endswith('.zip'):
            continue
        name=entry['name']
        if Path(name).name!=name:
            raise ValueError('Asset name must be a basename')
        dest=target/name
        if dest.exists():
            if dest.stat().st_size!=entry['size'] or sha256(dest)!=entry['actual_sha256']:
                raise RuntimeError(f'Existing local file differs from pinned asset: {dest}')
        else:
            part=target/(name+'.partial')
            with urllib.request.urlopen(entry['url']) as remote,part.open('wb') as stream:
                for block in iter(lambda:remote.read(4*1024*1024),b''):
                    stream.write(block)
            if part.stat().st_size!=entry['size'] or sha256(part)!=entry['actual_sha256']:
                raise RuntimeError(f'Download did not match pinned size/SHA256: {part}')
            part.replace(dest)
        record=dict(entry,local_path=str(dest),hash_verified=True)
        if name.endswith('.mp4'):
            record['ffprobe']=json.loads(subprocess.check_output([
                'ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(dest)]))
        verified.append(record)
        print(name,'SHA256 verified',flush=True)
    (target/'download-manifest.json').write_text(json.dumps(verified,indent=2)+'\n')


if __name__=='__main__':
    main()
