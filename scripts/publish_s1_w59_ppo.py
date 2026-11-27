"""Publish immutable W59 learner batches, first-update forks and scientific recording streams on D57."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

from publish_s1_go2_w54 import write
from publish_s1_w59 import ensure_release, publish, release, RAW, ROOT, STATE, WORK

AREA=WORK/'ppo-release'
AUTH=WORK/'archive-process-20261127/RESUME_AUTHORIZED.json'
INVENTORY=STATE/'delivery/learning-inventory.json'
H_CATALOG='H:/MPC+RL/00_catalog/migration-20261127-process/archive-index.sqlite'
DIGEST_CACHE={}


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def digest(path):
    st=path.stat()
    key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if key not in DIGEST_CACHE:
        with path.open('rb') as f:
            DIGEST_CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return DIGEST_CACHE[key]


def frozen_map(path,value):
    if path.is_file():
        if read(path)!=value:
            raise ValueError('Previously sealed mapping changed: '+str(path))
    else:
        write(path,value)
    return path


def verified(label):
    path=STATE/'delivery'/(label+'.json')
    if not path.is_file():return False
    row=read(path)
    return row.get('status')=='verified' and bool(row.get('assets')) and all(
        a.get('size',0)>0 and a.get('asset_id') and a.get('url') and
        str(a.get('digest','')).startswith('sha256:') for a in row['assets'])


def send(label,paths,video=None):
    if not verified(label):
        if not AUTH.is_file():raise RuntimeError('Storage operation has not released this publisher')
        publish(label,paths,video)
    if not verified(label):raise RuntimeError('Missing verified Release receipt: '+label)
    return label


def map_labels(existing,key,labels):
    existing[key]=sorted(set(existing.get(key,[]))|set(labels))


class Publisher:
    def __init__(self):
        self.out=read(INVENTORY) if INVENTORY.exists() else dict(schema='w59_ppo_delivery_v1',
            status='partial',assets=[],attempt_receipts={},checkpoint_receipts={},node_receipts={},
            head_ledger_receipts={},representative_training_videos={},closed_exposures={},
            diagnostic_archive=H_CATALOG)
        self.out.setdefault('head_ledger_receipts',{})
        self.attempts=self.out['attempt_receipts']
        self.checkpoints=self.out['checkpoint_receipts']

    def save(self):
        old=read(INVENTORY) if INVENTORY.is_file() else {}
        if self.out!=old:
            self.out['updated_at']=now()
            write(INVENTORY,self.out)

    def shared_plant(self,raw):
        paths=sorted(p for p in (raw/'model').rglob('*') if p.is_file())
        if (raw/'model.mjb').is_file():paths.append(raw/'model.mjb')
        if not paths:return None
        mapping={p.relative_to(raw).as_posix():dict(sha256=digest(p),size=p.stat().st_size) for p in paths}
        tree=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        label='w59-ppo-plant-'+tree[:20]
        if not verified(label):
            data=frozen_map(AREA/'manifests'/(label+'.json'),mapping)
            send(label,[('plant-paths.json',data)]+[(p.relative_to(raw).as_posix(),p) for p in paths])
        return label

    def episode_members(self,attempt,short_tag):
        name=attempt['attempt'];raw=RAW/name
        terminal=STATE/'episodes'/(name+'.json')
        result=read(terminal) if terminal.exists() else {}
        paths=[]; missing=[]; damaged=[]
        tech=attempt['status']=='technical_interruption'
        def add(p,member,required=True):
            if p.is_file():paths.append((short_tag+'/'+member,p));return
            if required:missing.append(member)
        for f in [STATE/'attempts'/(name+'.json'),terminal,terminal.with_name(name+'-resolved.json')]:
            add(f,'receipts/'+f.parent.name+'/'+f.name)
        add(RAW/'jobs'/name/'execution-contract.json','job/execution-contract.json')
        for p in sorted((RAW/'evaluation-policy'/name).glob('*.jsonl')):
            add(p,'frozen-speed-policy/'+p.name)
        for n in ('trajectory.npz','w59-actor-events.jsonl','w59-learner-result.json','material-final.npz',
                  'material-events.jsonl','material-failures.jsonl','cap-decisions.json','pann-action-dependence.json'):
            add(raw/n,'episode/'+n)
        sets={}
        for pat in ('physical-[0-9]*.npz','material-[0-9]*.npz','w55-f/control-[0-9]*.npz'):
            files=sorted(raw.glob(pat))
            sets[pat]=len(files)
            if not files:missing.append(pat)
            for p in files:add(p,'episode/'+p.relative_to(raw).as_posix())
        for n in ('manifest.json','mpc-solves.jsonl','contact-events.jsonl','handover-requests.jsonl','normal-execution.jsonl'):
            add(raw/'w55-f'/n,'episode/w55-f/'+n)
        # Match the recorder's own chunk roster, including the last short material chunk.
        expected=[Path(c['path']).name for c in result.get('chunks',[]) if c['stream']=='physical']
        expected+=list((result.get('material') or {}).get('chunks',[]))
        expected+=['w55-f/'+c['path'] for c in (result.get('w54_f_recording') or {}).get('control_chunks',[])]
        missing.extend(n for n in expected if not (raw/n).is_file())
        if attempt['task_mode']=='branch':
            config=attempt['branch'];snapshot=Path(config['snapshot'])
            if snapshot.is_file() and digest(snapshot)!=config['sha256']:raise ValueError('Registered first-update state changed')
            add(snapshot,'registered-snapshot/'+snapshot.name)
        if tech:
            for member,file in paths:
                if file.suffix=='.npz':
                    try:
                        with ZipFile(file) as z:z.infolist()
                    except BadZipFile:
                        damaged.append(member)
        elif missing:
            raise ValueError(f'{name}: natural record missing required originals {missing}')
        model=self.shared_plant(raw)
        if not model and not tech:raise ValueError('No plant for '+name)
        wm_receipts=[]
        settings=attempt['w59']
        for filename,checksum in zip(settings['model_paths'],settings['model_sha256'],strict=True):
            f=Path(filename)
            if digest(f)!=checksum:raise ValueError('Executed WM differs from the recorded checkpoint')
            wm_receipts.append(send('w59-ppo-executed-wm-'+checksum[:20],[('execution-wm.pt',f)]))
        return paths,dict(attempt=name,status=attempt['status'],source_commit=attempt['source_commit'],
            controller=attempt['controller'],physical_rows=result.get('physical_samples'),streams=sets,
            shared_plant_receipt=model,execution_wm_receipts=wm_receipts,missing=sorted(set(missing)),damaged_as_recorded=damaged,
            optional_legacy_diagnostics='Numbered MPC/drift diagnostics and process probes/logs are outside the required training payload. Archive membership is per original path at the archive cutoff, not assumed for later jobs. Query '+H_CATALOG,
            raw_root=str(raw),scope='Continuous plant, WBC, material, request/version and actor evidence; a technical prefix has its actual surviving members.')

    def episodes(self,group,label,video_name=None):
        names=[a['attempt'] for a in group]
        if all(self.attempts.get(n) and all(verified(q) for q in self.attempts[n]) for n in names):
            return
        paths=[];maps={};movie=None
        for a in group:
            name=a['attempt'];members,info=self.episode_members(a,name)
            paths.extend(members);maps[name]=info
        if video_name:
            # Render only the preselected first naturally completed A/E60 for each head.
            from publish_s1_w59 import render_video
            dest=AREA/'videos'/video_name;dest.mkdir(parents=True,exist_ok=True)
            marker=dest/'video-asset.json'
            if not marker.exists():render_video(video_name,dest)
            movie=dest/'full-mission-material-overlay.mp4'
            if read(marker)['size']!=movie.stat().st_size:raise ValueError('Incomplete representative video')
            paths.append((video_name+'/video-asset.json',marker))
        sidecar=frozen_map(AREA/'manifests'/(label+'.json'),maps)
        send(label,paths+[('recording-inventory.json',sidecar)],movie)
        for name,info in maps.items():
            map_labels(self.attempts,name,[q for q in (label,info['shared_plant_receipt'],*info['execution_wm_receipts']) if q])
            if info['missing'] or info['damaged_as_recorded']:
                self.out['closed_exposures'][name]=dict(missing=info['missing'],damaged=info['damaged_as_recorded'])
        if video_name:self.out['representative_training_videos'][video_name]=label
        self.save()

    def head(self,run,final):
        seed,arm=run.parent.name,run.name
        compact=STATE/'learning'/seed/arm
        campaign=run/'campaign.json'
        if not campaign.is_file():return
        cfg=read(campaign)
        prefix='w59-learner-'+seed+'-'+arm.lower()
        def index_weights(files,label):
            for f in files:map_labels(self.checkpoints,digest(f),[label])
        def done(f):
            labels=self.checkpoints.get(digest(f),[])
            return labels and all(verified(l) for l in labels)
        reports=sorted(compact.glob('update-[0-9][0-9][0-9].json'))
        versions=[(int(p.stem.split('-')[-1]),p) for p in reports]
        initial=run/'policy-000.pt'
        if initial.is_file() and not done(initial):
            paths=[('policy-000.pt',initial),('campaign.json',campaign)]
            for f in (Path(cfg['first_states']),Path(cfg['admission_path']),Path(cfg['initial_critic_source']),Path(cfg['planning_recipe'])):
                paths.append(('identity/'+f.name,f))
            if digest(Path(cfg['first_states']))!=cfg['first_states_sha256'] or digest(Path(cfg['initial_critic_source']))!=cfg['common_critic_sha256'] or digest(Path(cfg['admission_path']))!=cfg['admission_sha256']:
                raise ValueError('Frozen head inputs changed: '+prefix)
            for f in ('experiments/terrain_memory/s1-w59-learner.json','experiments/terrain_memory/s1-w59-learner-nodes.json',
                      'experiments/terrain_memory/s1-w59-gradient-admission.json','experiments/terrain_memory/s1-w59-online-wm.json'):
                paths.append(('identity/'+Path(f).name,ROOT/f))
            label=send(prefix+'-initial-'+digest(initial)[:16],paths)
            index_weights([initial],label);self.save()
        pending=[]
        for v,metric in versions:
            installed=run/f'policy-{v:03d}.pt';proposed=run/f'proposal-{v:03d}.pt'
            if not installed.is_file() or not proposed.is_file():raise ValueError('Closed update lacks its optimizer/checkpoint: '+str(metric))
            if not done(installed) or not done(proposed):pending.append((v,metric,installed,proposed))
        for start in range(0,len(pending),8):
            window=pending[start:start+8];paths=[];weights=[]
            for v,metric,installed,proposed in window:
                files=[installed,proposed,metric,run/f'proposed-{v:03d}.json']
                ordered=[]
                for case in 'AB':
                    ledger=run/f'events-{v-1:03d}-{case}.jsonl'
                    ready=run/f'ready-{v-1:03d}-{case}.json'
                    with ledger.open() as stream:ids=[json.loads(line)['event_id'] for line in stream if line.strip()]
                    if len(ids)!=128 or read(ready)['count']!=128:raise ValueError(f'Unsealed collector batch: {ledger}')
                    ordered.extend(ids);files.extend((ledger,ready))
                if len(set(ordered))!=read(metric)['distinct_used']:raise ValueError('Used event denominator changed: '+str(metric))
                used=frozen_map(AREA/'manifests'/prefix/f'used-{v:03d}.json',dict(version=v-1,ids=ordered))
                for f in [*files,used]:paths.append((f.name,f))
                weights.extend((installed,proposed))
            label=send(f'{prefix}-updates-{window[0][0]:03d}-{window[-1][0]:03d}-'+digest(window[-1][2])[:12],paths)
            index_weights(weights,label);self.save()
        for report in sorted([*compact.glob('node-*.json'),*compact.glob('stopped-*.json')]):
            d=read(report);policy=Path(d['checkpoint'])
            if digest(policy)!=d['checkpoint_sha256']:raise ValueError('Node checkpoint digest differs: '+str(report))
            node=str(report.relative_to(STATE/'learning'))
            paths=[('node/'+report.name,report),('node/'+campaign.name,campaign)]
            frozen=run/f"freeze-{d['used_events']:05d}.json"
            if report.name.startswith('node') and not frozen.exists():continue
            if frozen.exists():paths.append(('node/'+frozen.name,frozen))
            labels=self.checkpoints.get(d['checkpoint_sha256'],[])
            if not labels:raise ValueError('Node has no sealed policy/optimizer bundle')
            mapping=dict(checkpoint_sha256=d['checkpoint_sha256'],weights_in=list(labels),used_events=d['used_events'],
                observed_completed=d['observed_completed'],excluded_natural_tail=d['excluded_natural_tail'],wm_sha256=d['wm_sha256'])
            f=frozen_map(AREA/'nodes'/prefix/(report.stem+'-'+digest(report)[:12]+'.json'),mapping)
            label=send(prefix+'-'+report.stem+'-'+digest(report)[:12],paths+[('node/checkpoint-map.json',f)])
            self.out['node_receipts'][node]=sorted(set([label,*labels]));self.save()
        first=compact/'first-update-real.json';req=run/'first-update-request-001.json'
        if first.exists() and req.exists():
            label=prefix+'-first-update-registration-'+digest(first)[:12]
            records=[first,req,*(run.glob('first-[0-9][0-9]-*.json')),*(run.glob('frozen-*-000.json'))]
            send(label,[(f.name,f) for f in sorted(records)]);map_labels(self.out['head_ledger_receipts'],prefix,[label])
        if final:
            # No collector or optimizer owns these append-only ledgers after series closure.
            files=[run/f for f in ('used-event-ids.jsonl','physics-attempts.jsonl','physics-ends.jsonl','published.json','campaign.json')]
            for p in run.glob('*.pt'):
                if not done(p):
                    orphan=send(prefix+'-terminal-'+p.stem+'-'+digest(p)[:12],[(p.name,p)])
                    index_weights([p],orphan)
            files += [p for p in compact.glob('*.json') if p.name.startswith(('frozen-outcome','first-update-real','stopped-','node-'))]
            files=[f for f in files if f.is_file()]
            key=hashlib.sha256(''.join(f.name+digest(f) for f in sorted(files)).encode()).hexdigest()[:16]
            label=send(prefix+'-final-ledger-'+key,[(('ledger/' if f.parent==run else 'evaluation/')+f.name,f) for f in files])
            index_weights([f for f in files if f.suffix=='.pt'],label);map_labels(self.out['head_ledger_receipts'],prefix,[label])
        self.save()

    def tick(self):
        all_rows=[read(p) for p in sorted((STATE/'attempts').glob('w59-r3-*.json'))]
        roster=[a for a in all_rows if a['task_mode']=='training' or (a['task_mode']=='branch' and '-first-' in a['attempt'])]
        final=(STATE/'learning/learning-series.json').is_file() and all(a.get('ended_unix_s') for a in roster)
        runs=sorted((WORK/'learning').glob('seed-*/*/campaign.json'))
        for campaign in runs:self.head(campaign.parent,final)
        closed=[a for a in roster if a.get('ended_unix_s')]
        training=[a for a in closed if a['task_mode']=='training']
        pairs=defaultdict(list)
        for a in closed:
            if a['task_mode']=='branch':pairs[a['attempt'].rsplit('-',1)[0]].append(a)
        start_count=len(self.attempts)
        for pair,group in sorted(pairs.items()):
            if len(self.attempts)-start_count>=4:break
            roles={a['attempt'].rsplit('-',1)[-1] for a in group}
            if roles!={'before','after'} and not final:continue
            # Prefix qualified by which roles actually exist even in a stopped technical tail.
            label=pair+'-'+'-'.join(sorted(roles))+'-recordings'
            self.episodes(sorted(group,key=lambda x:x['attempt']),label)
        movies={}
        for a in sorted(training,key=lambda x:(x.get('started_unix_s',0),x['attempt'])):
            if a['case']=='A' and a['status']=='passed':
                head=a['attempt'].split('-a-e60-',1)[0]
                movies.setdefault(head,a['attempt'])
        for a in sorted(training,key=lambda a:(a['attempt'] not in movies.values(),a.get('ended_unix_s'),a['attempt'])):
            if len(self.attempts)-start_count>=4:break
            name=a['attempt']
            self.episodes([a],name+'-recordings',name if name in movies.values() else None)
        available={a['attempt'] for a in roster}
        missing=sorted(available-self.attempts.keys())
        unverified=[]
        if final:
            for campaign in runs:
                for base in ('policy-*.pt','proposal-*.pt'):
                    for path in campaign.parent.glob(base):
                        if not any(verified(l) for l in self.checkpoints.get(digest(path),[])):unverified.append(str(path))
        self.out.update(status='verified' if final and not missing and not unverified else 'partial',assets=[],
             registered_attempts=len(available),closed_attempts=len(closed),missing_attempts=missing,
             unverified_checkpoint_files=unverified,physics_authority='read closed W59 attempts only',
             video_scope='First naturally completed training A/E60 for each head. Qualification and mechanism complete movies are published separately.')
        self.save()
        return self.out['status']=='verified'


def main(once=False):
    AREA.mkdir(parents=True,exist_ok=True)
    with (AREA/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not AUTH.is_file():
            write(AREA/'status.json',dict(status='waiting_archive_authorization',marker=str(AUTH),pid=os.getpid()))
            if once:return
            time.sleep(30)
        worker=Publisher()
        while True:
            try:
                write(AREA/'status.json',dict(status='publishing',pid=os.getpid(),checked=now()))
                ensure_release()
                previous_coverage=len(worker.attempts)
                finished=worker.tick()
                write(AREA/'status.json',dict(status=worker.out['status'],pid=os.getpid(),covered=len(worker.attempts),checked=now()))
                if finished or once:return
                if len(worker.attempts)>previous_coverage and worker.out['closed_attempts']>len(worker.attempts):
                    continue
                time.sleep(45)
            except Exception as exc:
                why=str(exc)
                network=release.transient_transfer(why) or 'Interrupted GitHub upload' in why
                write(AREA/'status.json',dict(status='waiting_github' if network else 'blocked',
                    pid=os.getpid(),reason=why,checked=now(),covered=len(worker.attempts)))
                if not network or once:raise
                # The original uploader already checked the remote reservation. Resume the
                # immutable label from its per-part receipts once the D57 route is available.
                time.sleep(90)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    main(parser.parse_args().once)
