"""Verify and deliver W59 model checkpoints and the distinct new aligned bank from D57."""
import json
import os
import time
from pathlib import Path
from s1_w59_model_bank import WORK,OUT,write
from s1_w59_model_fields import BG
from publish_s1_w59 import ensure_release,publish,STATE

def send(label,paths):
    receipt=STATE/'delivery'/(label+'.json')
    if receipt.exists() and json.loads(receipt.read_text())['status']=='verified':return
    publish(label,paths)

def main():
    os.environ['S1_RELEASE_STREAMS']='1'
    ensure_release();seen=set()
    while True:
        for path in sorted((WORK/'checkpoints').glob('*/*/node-*.pt')):
            kind,seed=path.parent.parent.name,path.parent.name;label=f'w59-wm-{kind}-{seed}-{path.stem}'
            if label not in seen:send(label,[(path.name,path)]);seen.add(label)
        for path in sorted((WORK/'action-checkpoints').glob('*/*.pt')):
            if not (OUT/f'actions-{path.parent.name}-{path.stem.split("-")[0]}.json').is_file():continue
            label=f'w59-wm-action-{path.parent.name}-{path.stem}'
            if label not in seen:send(label,[(path.name,path)]);seen.add(label)
        for label,done,paths in [
            ('w59-causal-observation-bank',OUT/'aligned-bank.json',[('bank',WORK/'aligned')]),
            ('w59-prespecified-action-data',OUT/'action-bank.json',[('action-development',WORK/'action-development')]),
            ('w59-fp64-material-recovery',OUT/'free-horizons.json',[('material-events.json.gz',WORK/'material-events.json.gz'),('free-horizons.json.gz',WORK/'free-horizons.json.gz')])]:
            if label not in seen and done.exists():send(label,paths);seen.add(label)
        write(OUT/'model-release-progress.json',dict(verified_labels=sorted(seen),**BG))
        if (OUT/'fit-completion.json').is_file() and len(list(OUT.glob('actions-*.json')))==6 and 'w59-fp64-material-recovery' in seen and 'w59-prespecified-action-data' in seen:return
        time.sleep(30)

if __name__=='__main__':main()
