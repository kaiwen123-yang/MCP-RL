"""Publish the real incremental process-archive receipts without checking out or changing a live branch."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from inventory import CTL,ROOT,BASE,BG,now,save
from policy import COMPACT
REL='results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/storage/d57-H-process-20261124'
CODE='scripts/storage_d57_H_20261124'
NETWORK=BASE/'MCP-RL-w57/scripts/autodl_network.sh'
SERVER=('inventory.py','hash_inventory.py','legacy_policy.py','policy.py','packet.py','tmux_archive.py','observe.py','prepare_process.py','process_rpc.py','verify_process.py','publish_process.py')
RECEIVER=('archive_common.py','catalog_db.py','transfer_packs.py','retire_client.py','process_receive.py','process_finish.py','process_bridge.py','restore_member.py')
def read(p,default=None):return json.loads(p.read_text()) if p.is_file() else default
def git(*args,env=None,inp=None):
    return subprocess.run(['git','-C',str(ROOT),*args],env=env,input=inp,capture_output=True,check=True).stdout
def net(route,*args):
    command=['bash','-c','source "$1" "$2" >/dev/null && shift 2 && exec git "$@"','process-publish',str(NETWORK),route,'-C',str(ROOT),*args]
    q=subprocess.run(command,capture_output=True)
    if q.returncode:raise RuntimeError(q.stderr.decode(errors='replace')[-520:])
    return q.stdout.decode().strip()
def blob(parent,path):
    q=subprocess.run(['git','-C',str(ROOT),'show',parent+':'+path],capture_output=True)
    return q.stdout if q.returncode==0 else None
def encoded(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def publish(record,final,route):
    net(route,'fetch','--no-tags','origin','main')
    parent=git('rev-parse','refs/remotes/origin/main').decode().strip()
    env={**os.environ,'GIT_INDEX_FILE':str(CTL/'publication.index')}
    git('read-tree',parent,env=env)
    start=read(CTL/'roots.json');scan=read(CTL/'catalog.done.json');cohort=read(CTL/'cohort-summary.json');obs=read(CTL/'real-throughput.json',{})
    watermark=start['start_time']
    day=(record['finished_shanghai'] if final else record['at'])[:10]
    phase='final' if final else 'running'
    event=day.replace('-','')+'/storage/process-H-20261124-'+phase
    hp=read(CTL/'H-progress.json',{})
    brief=dict(event_id=event,state='completed' if final else 'running',as_of_shanghai=record.get('finished_shanghai') or record['at'],
        watermark=watermark,selected=scan['selected'],selected_logical_bytes=scan['selected_bytes'],hash_misses=scan['scoped_hash_failures'],
        old_and_new_payload=hp.get('manifest',{}),measured_h=hp.get('counts',{}),H_receiver=hp.get('current_phase'),
        D57_measured_peak_concurrent_payload_writers=obs.get('peak_concurrent_nonempty_payload_writers',0),
        latest_D57_sample=obs.get('last',{}),D57_transfer_tmux='mcp-rl-process-transfer-20261124',
        H_catalog='H:/MPC+RL/00_catalog/migration-20261124-process/archive-index.sqlite',
        H_restore='H:/MPC+RL/00_catalog/migration-20261124-process/读取原件.py',
        classifications=cohort['groups'],scientific_pins=cohort['scientific_pins'],**BG)
    updates={REL+'/latest.json':encoded(brief)}
    if final:
        post=read(CTL/'postcheck.json')
        brief['final']=record;updates[REL+'/latest.json']=encoded(brief);updates[REL+'/final.json']=encoded(record);updates[REL+'/D57-after.json']=encoded(post)
        for name in post['changed_delivery_memberships']:
            rr=COMPACT/'delivery'/(name+'.json');path=str(rr.relative_to(BASE/'MCP-RL-w57'));current=rr.read_bytes();d=json.loads(current);prior=blob(parent,path)
            if not prior:updates[path]=current;continue
            before=json.loads(prior)
            ids=lambda a:[(v['name'],v['size'],v['digest'],v['asset_id']) for v in a['assets']]
            if ids(before)!=ids(d):raise RuntimeError('Waiting for the original W57 asset receipt: '+name)
            for k in ('local_retention','originals_retained_at_verification','originals_retained'):before[k]=d[k]
            updates[path]=current if before==d else encoded(before)
    for group,src,names in [('d57',CTL,SERVER),('receiver',CTL/'receiver-code',RECEIVER)]:
        for name in names:
            f=src/name
            if not f.is_file():raise RuntimeError('Waiting for source snapshot '+str(f))
            updates[f'{CODE}/{group}/{name}']=f.read_bytes()
    # A closed episode whose original members have moved to H is still represented by its verified Release.
    pub='scripts/publish_s1_w57.py';source=blob(parent,pub).decode()
    if 'H_process_20261124' not in source:
        a="    if prior and prior.get('local_retention',{}).get('removed_mpc_files'):"
        b="    retention = (prior or {}).get('local_retention', {})\n    if retention.get('removed_mpc_files') or retention.get('H_process_20261124'):"
        if source.count(a)!=1:raise RuntimeError('Inspect updated W57 retention entry point')
        updates[pub]=source.replace(a,b).encode()
    doc='docs/S1_DAILY_WORKLOG.md';body=blob(parent,doc).decode()
    if event not in body:
        if final:
            desc=(f"本次 H 盘增量已收尾。{record['archived_files']:,} 个过程文件共 {record['archive_logical_bytes']:,} 逻辑字节完成核验。"
                  f"本轮新写入 {record['new_packs']} 个包、{record['verified_new_H_bytes']:,} 字节。D57 复查原文件身份、摘要、占用和依赖后逐文件释放 {record['removed_files']:,} 个文件、{record['released_allocated_bytes']:,} 个分配字节。"
                  f"命名保护路径误删数 {record['protected_path_unlinks']}。四个 E600 基线、冻结 checkpoint、源码和队列后继输入已复核。原路径可通过 H 盘归档索引找回。")
        else:
            counts=record['counts'];m=record['manifest']
            desc=(f"从 D57 的 {watermark} 有限源清单中选出 {scan['selected']:,} 个过程文件，共 {scan['selected_bytes']:,} 逻辑字节。"
                  f"按 H 盘既有内容索引去重，需新传 {m['new_source_bytes']:,} 字节，复用旧容器 {m['old_H_archives']} 个。"
                  f"{record['at']} 已核验新包 {counts['new_verified_packs']} 个、{counts['new_H_packed_bytes']:,} 字节。D57 实测同时写出有效数据的进程峰值 {brief['D57_measured_peak_concurrent_payload_writers']}。"
                  "W57 运行回合及其输入、W58 指定回放、四条 E600 基线、工作树、环境和 checkpoint 保留原位。整份清单完成 H 盘校验之后，才对仍无依赖的普通文件执行释放。")
        text='\n### '+event+'｜D57 过程文件增量归档\n\n'+desc+' [执行回执](../'+REL+'/latest.json)。\n'
        if '\n## '+day in body:
            at=body.index('\n',body.index('\n## '+day)+1)+1;body=body[:at]+text+'\n'+body[at:]
        else:
            at=body.index('## 以后每天如何追加');body=body[:at]+'## '+day+'｜项目实验与归档\n'+text+'\n'+body[at:]
        updates[doc]=body.encode()
    for name,value in updates.items():
        oid=git('hash-object','-w','--stdin',inp=value).decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+oid+','+name,env=env)
    tree=git('write-tree',env=env).decode().strip()
    if tree==git('rev-parse',parent+'^{tree}').decode().strip():return parent
    tip=git('commit-tree',tree,'-p',parent,'-m',('Complete verified cleanup' if final else 'Record measured transfer')+' for the D57 process archive').decode().strip()
    try:net(route,'push','origin',tip+':refs/heads/main')
    except RuntimeError:
        net(route,'fetch','--no-tags','origin','main')
        if subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',tip,'origin/main']).returncode:raise
    remote=net(route,'ls-remote','origin','refs/heads/main').split()[0]
    if remote!=tip:
        net(route,'fetch','--no-tags','origin','main');git('merge-base','--is-ancestor',tip,'origin/main')
    return tip
def main():
    with (CTL/'receipt-publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);attempt=0;routes=('academic','direct','academic','pc')
        while True:
            done=read(CTL/'git-receipts.json',{});final=read(CTL/'H-final.json');live=read(CTL/'H-progress.json',{})
            if done.get('final'):return
            candidate=final or live;isfinal=bool(final)
            key='final' if isfinal else live.get('at','')[:10]+'/running'
            measured=read(CTL/'real-throughput.json',{}).get('peak_concurrent_nonempty_payload_writers',0)>0
            if candidate and key not in done and (isfinal or (measured and live.get('counts',{}).get('new_verified_packs',0)>0)):
                try:
                    tip=publish(candidate,isfinal,routes[attempt%len(routes)])
                    done[key]=dict(commit=tip,verified_shanghai=now(),event_time=candidate.get('finished_shanghai') or candidate.get('at'))
                    save('git-receipts.json',done);print(now(),'Verified main',tip,key,flush=True);attempt=0
                    if isfinal:return
                except (OSError,RuntimeError,subprocess.CalledProcessError) as e:
                    attempt+=1;print(now(),'Publication pending',str(e)[-420:],flush=True)
            time.sleep(30)
if __name__=='__main__':main()
