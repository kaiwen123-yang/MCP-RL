"""Publish compact real H receipts and Shanghai daily events on top of the latest GitHub main."""
import hashlib,json,os,subprocess,time
from datetime import datetime
from pathlib import Path
from inventory import CTL,ROOT,TZ,BG,now,save
from policy import BASE,COMPACT
REL='results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/storage/d57-H-20261123'
CODE='scripts/storage_d57_H_20261123'
def read(p):return json.loads(Path(p).read_text())
def git(*args,env=None,inp=None,check=True):
    return subprocess.run(['git','-C',str(ROOT),*args],input=inp,capture_output=True,env=env,check=check)
def blob(parent,path):
    p=git('show',parent+':'+path,check=False)
    return p.stdout if not p.returncode else b''
def publish(record,final=False):
    parent=git('fetch','--no-tags','origin','main')
    parent=git('rev-parse','refs/remotes/origin/main').stdout.decode().strip()
    env=dict(os.environ,GIT_INDEX_FILE=str(CTL/'publish-index'));git('read-tree',parent,env=env)
    data=read(CTL/'inventory.done.json');at=record.get('at') or now();date=at[:10]
    data0=read(CTL/'roots.json')
    obs=read(CTL/'real-throughput.json') if (CTL/'real-throughput.json').is_file() else {}
    current=read(CTL/'tmux-archive.json') if (CTL/'tmux-archive.json').exists() else {}
    out=dict(event='20261123/storage/project-H-incremental-and-throughput',status=record.get('state','running'),
        snapshot=data0['start_time'],as_of_shanghai=at,regular_initial=next(v for k,v in data['kinds'] if k=='file'),
        initial_logical_bytes=sum(v[2] for v in data['logical_by_root']),registered_worktrees=len(data0['registered']),
        H_archive=record,measured_D57_concurrent_payload_writers=obs.get('peak_concurrent_nonempty_payload_writers'),
        sampled_peak_D57_sender_MiB_s=obs.get('peak_sample_MiB_per_s'),D57_tmux=current.get('tmux'),D57_pid=current.get('pid'),
        preexisting_W57_original_E30=[['A','B'],['B','P','R','PR','blind']],B_short_M0=[6,7],B_short_M1=[8,8],
        B_short_target=12,M1_B_short_both_returned_home=True,**BG)
    updates={REL+'/latest.json':(json.dumps(out,indent=2,ensure_ascii=False)+'\n').encode()}
    for group,folder in [('d57',CTL),('receiver',CTL/'receiver-code')]:
        if not folder.is_dir():continue
        names=('inventory.py','hash_inventory.py','packet.py','policy.py','tmux_archive.py','postcheck.py','late_retention.py','observe.py','tails.py','archive_publish.py') if group=='d57' else ('archive_common.py','catalog_db.py','ingest.py','verify_previous.py','transfer_packs.py','retire_client.py','late_retention_client.py','receive_git.py','finish.py','report_h_archive.py','restore_member.py')
        for n in names:
            if (folder/n).is_file():updates[f'{CODE}/{group}/{n}']=(folder/n).read_bytes()
    f='docs/S1_DAILY_WORKLOG.md';doc=blob(parent,f).decode('utf8')
    eid=date.replace('-','')+'/storage/'+('D57-H-final' if final else 'D57-H-parallel-running')
    if eid not in doc:
        typ='归档最终核对' if final else '跨日归档、训练与实际并发'
        text=f"\n### {eid}｜{typ}\n\nD57 项目归档状态 {out['status']}；H 盘接收观测 {at}，已核验新包 {record.get('verified_packs',0)} 个、{record.get('new_H_bytes',0)} 字节，已凭 H 回执并复核 D57 源后清理 {record.get('removed_files',0)} 个文件、{record.get('removed_bytes',0)} 逻辑字节。接收实测并发任务峰值 {record.get('peak_streams')}，D57 实测同时写出有效数据的进程峰值 {out['measured_D57_concurrent_payload_writers']}。一致 Git 快照核对了 29 个工作树和 64 个引用。W57 冻结双槽依自己的调度推进；首组 B 的 A/B 正式短重访分别为 8/12、8/12。已归档的活动输入、共享别名、四条 E600 基线、最终 checkpoint 与研究队列按引用保留。核验和逐路径索引位于 H:/MPC+RL/00_catalog/migration-20261123。[本次实测](../{REL}/latest.json)。\n"
        if '## '+date in doc:
            pos=doc.index('\n',doc.index('## '+date))+1;doc=doc[:pos]+text+'\n'+doc[pos:]
        else:
            pos=doc.index('## 以后每天如何追加');doc=doc[:pos]+f'## {date}｜D57 实验及归档\n'+text+'\n'+doc[pos:]
    updates[f]=doc.encode('utf8')
    if final:
        checks=read(CTL/'postcheck-verified.json') if (CTL/'postcheck-verified.json').exists() else {}
        for name in checks.get('H_retained_membership_changes',{}):
            q=COMPACT/'delivery'/(name+'.json');rel=str(q.relative_to(BASE/'MCP-RL-w57'))
            prior=blob(parent,rel)
            if prior:
                d=json.loads(prior);src=read(q)
                if d['assets']==src['assets']:
                    for k in ('originals_retained','originals_retained_at_verification','local_retention'):d[k]=src[k]
                    updates[rel]=(json.dumps(d,indent=2,ensure_ascii=False)+'\n').encode()
    for path,content in updates.items():
        oid=git('hash-object','-w','--stdin',inp=content).stdout.decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+oid+','+path,env=env)
    tree=git('write-tree',env=env).stdout.decode().strip()
    if tree==git('rev-parse',parent+'^{tree}').stdout.decode().strip():return parent
    tip=git('commit-tree',tree,'-p',parent,'-m',('Complete' if final else 'Publish')+' D57 storage and parallel execution evidence').stdout.decode().strip()
    p=git('push','origin',tip+':refs/heads/main',check=False)
    if p.returncode:raise RuntimeError('Git push pending: '+p.stderr.decode(errors='replace')[-480:])
    head=git('ls-remote','origin','refs/heads/main').stdout.decode().split()[0]
    if head!=tip:
        git('fetch','--no-tags','origin','main');git('merge-base','--is-ancestor',tip,'origin/main')
    return tip
def main():
    log=CTL/'github-states.json'
    states=read(log) if log.exists() else {}
    while True:
        if (CTL/'H-progress.json').exists():
            d=read(CTL/'H-progress.json');final=d.get('terminal',False);key=d['at'][:10]+('/final' if final else '/running')
            if key not in states:
                try:
                    tip=publish(d,final);states[key]=dict(commit=tip,verified=now());save(log.name,states)
                    print(now(),'verified',tip,key,flush=True)
                except (OSError,subprocess.CalledProcessError,RuntimeError) as e:
                    print(now(),'Publication pending',repr(e),flush=True)
            if final and key in states:return
        time.sleep(24)
if __name__=='__main__':main()
