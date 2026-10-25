#!/usr/bin/env python3
"""Publish the bounded transition evidence directly from the execution host."""
import base64
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
REPO='kaiwen123-yang/MCP-RL'
TAG='tm-s1-contact-transition-evidence-20261025'
CONFIG='experiments/terrain_memory/s1-contact-transition.json'
C=json.loads((ROOT/CONFIG).read_text())
SMALL=ROOT/C['outputs']['compact_root']
INDEX=SMALL/'index.json'
GH='/home/fanyuan/projects/kaiwen-learning/terrain-memory-s0/work/tools/gh_2.103.2_linux_amd64/bin/gh'

def main():
    index=json.loads(INDEX.read_text())
    if not index['phase_A_complete'] or (index['phase_B_readiness']['ready'] and not index['phase_B_complete']):
        raise RuntimeError('The authorized matrix has not finished.')
    token=os.environ['GH_TOKEN']
    git_env=dict(os.environ,GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
        GIT_CONFIG_KEY_1='http.sslCAInfo',GIT_CONFIG_VALUE_1='/etc/ssl/certs/ca-certificates.crt')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,env=git_env,text=True).strip()
    def gh(*args):
        return subprocess.check_output([GH,*args],cwd=ROOT,env=os.environ,text=True)
    git('fetch','origin','main')
    git('merge','--no-edit','origin/main')
    table=['| 地面 | 原冻结B1 | 共同修正B1 | T1 |','|---|---:|---:|---:|']
    for terrain,label in [('rigid','刚地'),('kv_moderate','KV8000'),('tm02_anchor','TM02')]:
        g={x['controller']:x for x in index['groups'] if x['terrain']==terrain}
        table.append('| '+label+' | '+('0/5' if terrain=='tm02_anchor' else '5/5')+' | '+
            f"{g['B1']['successes']}/{g['B1']['episodes']} | {g['T1']['successes']}/{g['T1']['episodes']} |")
    table='\n'.join(table)
    failures={g['terrain']+' '+g['controller']:g['failures'] for g in index['groups']}
    phaseB=('准备条件成立，已执行全部5个T2原seed。' if index['phase_B_complete'] else
        'Phase B未执行：T1未达到冻结的完整任务/跨足重访准备条件，不能据此否定材料历史信息本身的价值。')
    text=(f"**第三轮S1固定任务结束：{index['completed_episodes']}条独立正式结果，{index['successes']}成功、"
          f"{index['completed_episodes']-index['successes']}失败；实际新启动{index['new_mission_starts']}次，开发6/6次。新增训练0、硬件0。**\n\n"
          +table+'\n\n'+
          '共同摆腿修正补入世界坐标逆动力学中遗漏的浮基加速度、完整Jacobian导数及基座—腿惯性耦合项。'
          '它影响三种地面全部15条B1，因此旧B1只保留为历史参照，正式配对均重算；不能把共同修复的变化归为T1。'
          'T1在原名义时钟下使用连续法向接近、实际接触支撑可用性及载荷接管。\n\n'+phaseB+'\n\n'+
          f"同起点查询{index['information_chain']['queries']}次，其中初态不同{index['information_chain']['different_initial_states']}次，"
          f"所计算接近动作不同{index['information_chain']['different_selected_actions']}次。查询差异不等于完整任务收益。\n\n"+
          '首批3次开发任务在轨迹序列化时出错，内存时程未保存；源码、模型、resolved配置和日志保留，并完整计入预算。'
          '其余3次开发结果按相同冻结版本复用为正式结果。这一记录缺失不填零、不臆造物理结局。\n\n'+
          '五个seed是配对优化器种子。时间和机械功差值只对双方完成者计算；提前失败不是更快或节能，机械功不是电池耗电。'
          '并行共享资源计时不代表独占算法速度或NX实时性。当前仍缺稳定TM02完整任务及后续可观测辨识、公平强基线和部署证据。\n\n'+
          f"[运行索引]({C['outputs']['compact_root']}/index.json)、[逐回合结果]({C['outputs']['compact_root']}/episodes.csv)、"
          f"[配对差值]({C['outputs']['compact_root']}/paired.csv)、[共同修正对照]({C['outputs']['compact_root']}/shared-correction-vs-frozen-B1.csv)、"
          f"[同起点查询]({C['outputs']['compact_root']}/same-start-queries.csv)、[Release](https://github.com/{REPO}/releases/tag/{TAG})。")
    readme=ROOT/'README.md';s=readme.read_text()
    a=s.index('## 当前任务：');b=s.index('## S1 第二轮已完成',a)
    readme.write_text(s[:a]+'## S1 第三轮：接触过渡完整任务结果\n\n'+text+'\n\n'+s[b:])
    plan=ROOT/'docs/TERRAIN_MEMORY_PLAN.md';s=plan.read_text()
    a=s.index('### 5.4 当前唯一下一任务：');b=s.index('#### 5.4.1',a)
    plan.write_text(s[:a]+'### 5.4 第三轮已执行：接触过渡与接触前信息条件\n\n'+
        text.replace('(results/','(../results/')+'\n\n以下保留冻结的本轮协议及适用边界。\n\n'+s[b:])
    agents=ROOT/'AGENTS.md';s=agents.read_text()
    marker='## 唯一主线与当前任务\n'
    s=s.replace(marker,marker+'\n**本轮固定任务已执行并进入发布收尾。不得再次启动本文件的历史矩阵；后续工作由用户另行指定。实际范围与失败见运行索引。**\n',1)
    agents.write_text(s)
    index['scientific_conclusions']=dict(summary=text,failures=failures,phase_B=phaseB,
        common_correction_scope='All three terrains; 15 B1 recomputed.',
        supported='Complete-mission outcomes and the implemented contact/query mechanism under the frozen synthetic model.',
        not_established='Persistent-memory task benefit, deployable estimator, improved PPO, natural-terrain transfer or NX real time.')
    index['postprocessing_commands']=['python scripts/summarize_s1_transition.py',
        'python scripts/video_terrain_memory_s1.py --config '+CONFIG,
        'python scripts/package_terrain_memory_s1.py --config '+CONFIG+' --release-tag '+TAG,
        'python scripts/publish_s1_transition.py']
    INDEX.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    files=['src/tm02/s1_transition.py','src/tm02/s1_robot.py','scripts/run_terrain_memory_s1.py',
        'scripts/parallel_terrain_memory_s1.py','scripts/summarize_s1_transition.py','scripts/package_terrain_memory_s1.py',
        'scripts/publish_s1_transition.py',CONFIG,'README.md','docs/TERRAIN_MEMORY_PLAN.md','AGENTS.md',str(SMALL.relative_to(ROOT))]
    git('add',*files)
    git('reset','--',str((SMALL/'index.lock').relative_to(ROOT)))
    for p in (SMALL/'figures').glob('*.pdf'):git('add','-f',str(p.relative_to(ROOT)))
    git('commit','-m','Publish bounded S1 contact-transition missions and paired negative evidence')
    evidence=git('rev-parse','HEAD')
    git('push','origin','HEAD:main')
    notes=ROOT/'work/s1-transition/release-notes.md'
    notes.write_text(text.replace('(results/','(https://github.com/'+REPO+'/blob/main/results/')+
        '\n\nExecution/results commit: '+evidence+'\n')
    gh('release','create',TAG,'--repo',REPO,'--target',evidence,'--draft',
        '--title','S1 contact transition — bounded paired complete missions','--notes-file',str(notes))
    assets=index['publication']['assets']
    gh('release','upload',TAG,'--repo',REPO,*[a['execution_host_path'] for a in assets])
    release=next(r for r in json.loads(gh('api',f'repos/{REPO}/releases?per_page=100')) if r['tag_name']==TAG)
    checks=[]
    for expected in assets:
        actual=next(a for a in release['assets'] if a['name']==expected['filename'])
        check=dict(filename=actual['name'],bytes=actual['size'],digest=actual['digest'],url=actual['browser_download_url'],
            size_matches=actual['size']==expected['bytes'],sha256_matches=actual['digest']=='sha256:'+expected['sha256'],state=actual['state'])
        checks.append(check)
    if not all(c['size_matches'] and c['sha256_matches'] and c['state']=='uploaded' for c in checks):
        raise RuntimeError('Uploaded asset size/digest mismatch; draft retained.')
    gh('release','edit',TAG,'--repo',REPO,'--draft=false')
    release=json.loads(gh('api',f'repos/{REPO}/releases/tags/{TAG}'))
    index['status']='completed_and_published'
    index['publication'].update(uploaded=True,evidence_commit=evidence,release_id=release['id'],draft=release['draft'],
        uploaded_asset_verification=checks,confirmed_utc=datetime.now(timezone.utc).isoformat(),
        verification_method='Execution-host file SHA256/size against GitHub asset digest/size; no archive download.')
    INDEX.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    git('add',str(INDEX.relative_to(ROOT)));git('commit','-m','Confirm transition Release sizes and SHA256 digests');git('push','origin','HEAD:main')
    print(json.dumps(dict(event='published',commit=git('rev-parse','HEAD'),release=release['html_url'],assets=len(checks))),flush=True)

if __name__=='__main__':main()
