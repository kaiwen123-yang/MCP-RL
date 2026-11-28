"""Render required closed W59 full movies while preceding Release assets transfer."""
import fcntl
import json
import time

from publish_s1_w59 import RAW, STATE, WORK, render_video


def main():
    with (WORK/'video-prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            for f in sorted((STATE/'attempts').glob('*.json')):
                r=json.loads(f.read_text())
                if not r.get('ended_unix_s') or r.get('task_mode') not in ('original','short','recovered'):
                    continue
                raw=RAW/f.stem
                if (raw/'video-asset.json').exists():
                    continue
                receipt=STATE/'delivery'/f.name
                if receipt.exists() and json.loads(receipt.read_text()).get('status')=='verified':
                    continue
                if (raw/'trajectory.npz').is_file() and (STATE/'episodes'/f.name).is_file():
                    render_video(f.stem,raw)
            if (STATE/'outcome.json').exists():
                return
            time.sleep(30)


if __name__=='__main__':
    main()
