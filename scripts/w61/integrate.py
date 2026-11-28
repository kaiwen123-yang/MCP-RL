"""Single integration owner for the registered W61 runs and Shanghai daily worklog."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from w61.joint.plant import write
from w61.joint.results import rows as joint_rows
from w61.joint.deliver import RUN as F_STUDY,COMPACT as F_OUT
from w61.lease import resources
BASE=ROOT.parent
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61')
WORK=ROOT/'work/w61/joint'
E_STUDY='e61-hybrid-m1-01'
AUX=(BASE/'MCP-RL-w61-learning'/REL/'learning')
LANES=('material','gait','mpc','wbc','learning')
LOG=ROOT/'docs/S1_DAILY_WORKLOG.md'
HROOT=BASE/'.mcp-rl-w61-cold-20261128'
H2=BASE/'.mcp-rl-w61-legacy-solves-20261128'
ZONE=ZoneInfo('Asia/Shanghai')

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def read(p):return json.loads(p.read_text())
def head():
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    return subprocess.check_output(['gh','api','repos/kaiwen123-yang/MCP-RL/git/ref/heads/main','--jq','.object.sha','-H','Cache-Control: no-cache'],env=env,text=True).strip()
def blobs_and_merge(remote):
    if subprocess.run(['git','-C',str(ROOT),'cat-file','-e',remote+'^{commit}'],stderr=subprocess.DEVNULL).returncode:
        git('fetch','--no-tags','origin','main')
    git('merge','--no-edit','-q',remote)
    for lane in LANES:git('merge','--no-edit','-q','refs/heads/codex/w61-'+lane)

def evidence():
    formal={};ablation={};copies=[]
    for folder in sorted((AUX/E_STUDY).glob('*-s6*')):
        # Recovery keeps both its interrupted attempt and its final update for
        # reproducibility. Only the canonical version receipt contributes scientific steps.
        canonical=sorted(folder.glob('train-[0-9][0-9][0-9].json'))
        receipts=[read(p) for p in canonical]
        updates=[r for r in receipts if r.get('status')=='updated']
        e=dict(used=max((n['used'] for n in updates),default=0),updates=len(updates),
               requested=sum(n['algorithm']['requested'] for n in updates),
               actor_steps=sum(n['algorithm']['accepted_actor_steps'] for n in updates),
               critic_updates=sum(n['algorithm']['critic_updates'] for n in updates),
               imagined=sum(n['algorithm']['imagined'] for n in updates))
        if (folder/'terminal.json').exists():e['terminal']=read(folder/'terminal.json')
        (formal if folder.name.split('-s6')[0] in ('ppo','spo','e-score') else ablation)[folder.name]=e
        copies.extend(p for p,row in zip(canonical,receipts) if row.get('status')=='updated')
        copies.extend(folder.glob('evaluate-[0-9][0-9][0-9].json'));copies.extend(folder.glob('terminal.json'))
    # final_dispatch commits its own frozen receipts. Importing them a second time
    # can publish an intermediate upload into the integration parent before that merge.
    delivery=AUX/'delivery'
    copies.extend(p for p in delivery.glob('*.json') if not p.name.startswith('w61-e-final-'))
    copies.extend((AUX/E_STUDY).glob('lifecycle.json'))
    copied=[]
    for f in copies:
        target=ROOT/REL/'learning'/f.relative_to(AUX)
        source=f.read_bytes()
        receipt=json.loads(source)
        if f.parent==delivery and receipt.get('status')!='verified':continue
        if not target.exists() or target.read_bytes()!=source:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source)
        copied.append(str(target.relative_to(ROOT)))
    return formal,ablation,copied

def step(force=False):
    remote=head()
    blobs_and_merge(remote)
    lanes={l:git('rev-parse','refs/heads/codex/w61-'+l) for l in LANES}
    rows=joint_rows()
    formal,abl,paths=evidence()
    used=sum(r['used'] for r in formal.values())
    final_ledger=AUX/E_STUDY/'final-evaluation'
    frozen_progress=final_ledger/'parallel-progress.json'
    frozen={}
    if frozen_progress.is_file():
        dispatch=read(frozen_progress)
        outcomes={};movies=set()
        for suffix in ('-summary.json','-physical.json'):
            for path in final_ledger.glob('*/*'+suffix):
                if '-v032-' not in path.name or ('-short-' not in path.name and '-recovered-' not in path.name):continue
                name=path.name.removesuffix(suffix);outcomes[name]=read(path)
                if suffix=='-summary.json':movies.add(name)
        frozen=dict(registered=dispatch['registered_long'],ready_8192_heads=dispatch['checkpoint_heads_8192'],
                    physical_closed=len(outcomes),full_video_closed=len(movies),awaiting_full_video=len(outcomes)-len(movies),
                    remote_receipts=dispatch['long'].get('verified',0),reported_outcomes=dict(Counter(r['stage_status'] for r in outcomes.values())),
                    actual_simulated_seconds=sum(r['physical_s'] for r in outcomes.values()),
                    compact_progress='learning/'+E_STUDY+'/final-evaluation/parallel-progress.json',
                    completed=dispatch['completed'])
    current=AUX/E_STUDY/'independent-dispatch-live.json'
    independent={}
    if current.is_file():
        scheduler=read(current)
        independent=dict(logical_heads=scheduler.get('logical_positions',[]),batch_distinct=256,
            receipt='learning/'+E_STUDY+'/independent-dispatch.json',current_admissions=scheduler.get('training_admissions',{}),
            admission_owner=scheduler.get('feeder_pid'),manager_lock_transferred=scheduler.get('manager_lock_transferred'))
    gpu_receipt=BASE/'MCP-RL-w61-material'/REL/'material/w61-e-packet-throughput-v1/index.json'
    packet={}
    if gpu_receipt.is_file():
        exp=read(gpu_receipt)
        packet=dict(stage=exp.get('stage'),production_backend=exp.get('production_backend'),production_eligible=exp.get('production_eligible'),
                    comparisons=exp.get('comparisons',{}),receipt='material/w61-e-packet-throughput-v1/index.json')
    cpu_dir=AUX/'w61-e-cpu-exact-v1'
    cpu={}
    if (cpu_dir/'production.json').is_file():
        selection=read(cpu_dir/'production.json')
        cpu=dict(selected_shanghai=selection['registered_shanghai'],binary_sha256=selection['runtime']['binary_sha256'],
                 training=selection['training_enabled'],curve=selection['curve_enabled'],native=selection['native_enabled'],
                 full_pair_speedup=selection['training_evidence']['full']['speedup'],
                 native_E30_speedup=(selection.get('frozen_evaluation_evidence') or {}).get('speedup'),
                 receipts='learning/w61-e-cpu-exact-v1')
        for name,key in [('first-production','first_formal_lease'),('first-production-update','first_formal_update')]:
            if (cpu_dir/(name+'.json')).is_file():cpu[key]=read(cpu_dir/(name+'.json'))
    jobs=[read(f) for f in (ROOT/'work/w61/pool/jobs').glob('*.json')]
    counts=Counter(v['state'] for v in jobs)
    active={r['job']:r['slots'] for r in jobs if r['state']=='running'}
    now=datetime.now(ZONE)
    storage={}
    for name,folder in [('historical_process',HROOT),('historical_solver',H2)]:
        if (folder/'H-client-final.json').is_file():
            storage[name]=dict(receipt=read(folder/'H-client-final.json'),proof=read(folder/'complete-H-proof.json'))
    if (HROOT/'products-totals.json').is_file():storage['verified_W61_products']=read(HROOT/'products-totals.json')
    nH=storage.get('verified_W61_products',{}).get('count',0)
    marker=dict(day=now.date().isoformat(),lanes=lanes,
                F_pairs=rows['training_pairs']//3, actual_joint=rows['actual_simultaneous_two_head_updates']//3,
                F_eval=len(rows['unseen']),F_factorial=len(rows['factorial']), F_release=rows['originals_verified']//4,
                E={k:(v['used']//2048,bool(v['updates']),bool(v.get('terminal'))) for k,v in formal.items()},
                ablation={k:v['used']//2048 for k,v in abl.items()}, storage=list(storage),H_products=nH//10)
    marker['frozen_final']=frozen
    marker['cpu_sampling']=(cpu.get('binary_sha256'),bool(cpu.get('first_formal_lease')),bool(cpu.get('first_formal_update')))
    signature=hashlib.sha256(json.dumps(marker,sort_keys=True).encode()).hexdigest()[:13]
    stamp=WORK/'integration-state.json'
    event=now.strftime('%Y%m%d')+'/w61/execution-'+signature
    if not force and stamp.is_file() and read(stamp)['signature']==signature and remote==git('rev-parse','HEAD'):
        return bool((AUX/E_STUDY/'lifecycle.json').is_file() and len(rows['factorial'])==8 and not counts.get('running') and not counts.get('waiting') and (not frozen or frozen['completed']))
    snap=resources()
    F_OUT.mkdir(parents=True,exist_ok=True)
    write(F_OUT/'result.json',rows)
    dst=ROOT/REL/'index.json';dst.parent.mkdir(parents=True,exist_ok=True)
    write(dst,dict(observed_shanghai=now.isoformat(timespec='seconds'),contract='experiments/terrain_memory/s1-w61-research-contract.json',
        W59='c8378beae4be371eb4f104782b75e5d295b6f59a',W59_handoff='results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W59-review/handoff.json',
        branches=lanes,F_runtime=rows['runtime'],F_pairs=rows['training_pairs'],F_training_real_queries=rows['training_real_queries'],
        F_joint_updates=rows['actual_simultaneous_two_head_updates'],F_unseen_frozen_heads=len(rows['unseen']),F_factorial_queries=len(rows['factorial']),
        formal_E= formal,E_used=used,E_target=73728,E_extra_geometry_ablation=abl,frozen_final=frozen,independent_head_dispatch=independent,material_collector_batching=packet,exact_cpu_sampler=cpu,physical_pool=counts,leased_slots=active,resources=snap,
        storage=storage,release='https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-w61',
        runtime_policy=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False),
        run_commands={'F':'python -m w61.joint.train schedule under W61_F_RUNTIME_COMMIT with the registered single-slot leases',
            'E':'scripts/w61/learning/campaign.py paired with the original E study manifest, two slots per independent policy',
            'raw_publication':'scripts/w61/joint/deliver.py and lane release receipts','integrator':'python scripts/w61/integrate.py --watch'}))
    paths.extend([str(dst.relative_to(ROOT)),str((F_OUT/'result.json').relative_to(ROOT))])
    paths.extend(str(p.relative_to(ROOT)) for p in (F_OUT/'originals').glob('*.json'))
    if event not in LOG.read_text():
        changed=git('log','--no-merges','-12','--format=%h %s',remote+'..HEAD').splitlines()
        lines=['','### '+event+' · '+now.strftime('%Y-%m-%d %H:%M 上海'),
               '六线仍使用现有 material、gait、mpc、wbc、learning、joint 工作树。运行固化源码与权重。此次核对的各分支提交为 '+', '.join(l+' '+sha[:8] for l,sha in lanes.items())+'，F 正式训练使用 '+rows['runtime'][:12]+'。']
        if changed:lines+=['本次整合的实现及实际证据包括 '+ ' / '.join(changed)+ '。']
        lines+=['E 正式九头训练累计使用 '+str(used)+' / 73,728 个去重真实事件。已结束更新的 requested 为 '+str(sum(r['requested'] for r in formal.values()))+'，累计接受 actor 步 '+str(sum(r['actor_steps'] for r in formal.values()))+'，critic 更新 '+str(sum(r['critic_updates'] for r in formal.values()))+'，想象事件 '+str(sum(r['imagined'] for r in formal.values()))+'。单头 used 为 '+', '.join(k+' '+str(r['used']) for k,r in formal.items())+'。无几何附加臂单计。',
                'F 已完成 '+str(rows['training_pairs'])+' 对相同实测支撑的正负候选，实际独立查询 '+str(rows['training_real_queries'])+' 次、查询物理子步 '+str(rows['training_real_query_substeps'])+'。同一轮 A 与 C 参数均发生非零变化的更新 '+str(rows['actual_simultaneous_two_head_updates'])+' 次。独立未见材料查询完成的冻结头 '+str(len(rows['unseen']))+'/12，2×2 完整结果数 '+str(len(rows['factorial']))+'/8。逐对列出实际回报、物理终止、支持集摘要及足端命令差。累计真实CPU '+format(rows['summed_rollout_cpu_hours'],'.3f')+' 核时。',
                '物理池此刻占用 '+str(sum(map(len,active.values())))+'/6 槽，等待 '+str(counts.get('waiting',0))+' 项。cgroup 实测 '+format(snap['total_cpu_cores'],'.1f')+'/25 核，可用 RAM '+format(snap['available_ram_gib'],'.1f')+' GiB，GPU 利用率 '+str(snap['gpu_util_percent'])+'%，空余显存 '+str(snap['free_gpu_mib'])+' MiB，scratch 空间 '+format(snap['scratch_free_gib'],'.1f')+' GiB。此处为共享设备同时刻读数。']
        if frozen:
            lines+=['96 条冻结材料长回合按各头的 8192 节点并行调度。目前 '+str(frozen['ready_8192_heads'])+
                    ' 个头达到节点，已核验原件 '+str(frozen['remote_receipts'])+'/96。已有原始终止类型 '+
                    json.dumps(frozen['reported_outcomes'],ensure_ascii=False)+'，累计 '+format(frozen['actual_simulated_seconds'],'.2f')+
                    ' 物理秒。零策略 A/B 资格和曲线评估各自单计。']
        if independent:
            lines+=['独立头调度当前保留的接续位置为 '+', '.join(independent['logical_heads'])+'，物理闭合、完整成片与远端核验分别统计。']
        if packet:
            lines+=['真实跨 collector 材料试验阶段 '+str(packet['stage'])+'，正式采样材料后端 '+str(packet['production_backend'])+'。']
        if cpu:
            lines+=['正式采样启用精确 CPU 材料内核 '+cpu['binary_sha256'][:12]+'。每流 128 事件的同槽完整对照吞吐为原版的 '+format(cpu['full_pair_speedup'],'.3f')+' 倍。']
        if rows['matched_unseen']:lines+=['F 未见材料四臂完整评价汇总 '+json.dumps(rows['matched_unseen'],ensure_ascii=False)+'。']
        if storage:lines+=['H 盘从已写完的过程原件和发布中转件逐项回读，GitHub Release 每个原件由执行端核对 size/digest。具体本轮和历史 H/D57 回执列于 w61/index.json，原件退役按内容与运行身份核对。']
        lines+=['后继按既定工作量继续各独立种子、正式冻结评价及已登记的控制对照。具条件的终局、数值中断和负结果都写回原运行身份。','']
        with LOG.open('a') as out:out.write('\n'.join(lines)+'\n')
    paths.append('docs/S1_DAILY_WORKLOG.md')
    git('add','--sparse','--',*paths)
    if subprocess.call(['git','-C',str(ROOT),'diff','--cached','--quiet'])==1:
        git('commit','-qm','Publish measured W61 research and Shanghai worklog '+signature)
    rev=git('rev-parse','HEAD')
    cfg='credential.helper=!gh auth git-credential'
    git('-c',cfg,'-c','http.version=HTTP/1.1','push','origin',rev+':refs/heads/main')
    seen=head()
    if seen!=rev:raise RuntimeError('Remote main advanced during verification; refresh the recorded parent')
    write(stamp,dict(signature=signature,remote=seen,shanghai=now.isoformat(timespec='seconds'),actual_F_joint=rows['actual_simultaneous_two_head_updates'],E_used=used))
    print(now.isoformat(timespec='seconds'),seen,used,rows['training_pairs'],rows['actual_simultaneous_two_head_updates'],flush=True)
    return False

def main(watch,after_current):
    with (WORK/'integrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if after_current else fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done=step()
            except subprocess.CalledProcessError as e:
                print('transport or git error',e,flush=True)
                if not watch or git('ls-files','-u'):raise
                time.sleep(60);continue
            if done or not watch:return
            time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true')
    p.add_argument('--after-current',action='store_true',help='acquire the one existing writer lock after its previous owner leaves')
    args=p.parse_args(); main(args.watch,args.after_current)
