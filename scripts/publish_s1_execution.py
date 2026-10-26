#!/usr/bin/env python3
"""Publish only this round's actual attempt evidence from the execution host."""
import argparse
import base64
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
REPO='kaiwen123-yang/MCP-RL'
TAG='tm-s1-contact-execution-evidence-20261026'
CONFIG='experiments/terrain_memory/s1-contact-execution.json'
C=json.loads((ROOT/CONFIG).read_text())
SMALL=ROOT/C['outputs']['compact_root']
INDEX=SMALL/'index.json'
GH='/home/fanyuan/projects/kaiwen-learning/terrain-memory-s0/work/tools/gh_2.103.2_linux_amd64/bin/gh'


def prepare():
    index=json.loads(INDEX.read_text())
    rows=[json.loads((ROOT/a['record']).read_text()) for a in index['attempts']]
    table='| 开发顺序 | seed | 原完整任务结果 | 终止时刻 / s |\n|---|---:|---|---:|\n'
    for a,r in zip(index['attempts'],rows):
        term='机身触地' if r['termination'].startswith('invalid_contact') else '机体跌倒'
        table+=f"| {a['attempt_id']} | {r['seed']} | {term} | {r['duration_s']:.4f} |\n"
    e=index['execution_metrics']
    maxf=max(x['peak_force_difference_norm_N'] for x in e)
    maxm=max(x['peak_moment_difference_norm_Nm'] for x in e)
    summary=(f'**本轮在开发预算内结束：3次原完整任务启动，0次完成，0次记录丢失；正式结果0条，刚地正式阶段、KV8000和TM02均未启动。新增PANN/PPO训练0、硬件0。**\n\n'+table+'\n'+
        '三次均执行同一T1R实现 `96748c9d0d51041177892ef7081d9ac3e02a6e36`，没有调参后挑选结果，也没有可冻结的正式版本。'
        'seed4越过旧T1在1.344s的初始站立失败点，在2s进入去程，最终仍于2.769s机身触地；这不是整程成功。\n\n'+
        '已保存记录中，所有名义支撑且实际不支撑的足均有明确接近控制所有者，初始t=0单样本之后没有持续零关节命令空分支。'
        '失接触足命令力为0；同一映射限制法向力与原摩擦锥近似。实际关节力矩分项与请求相符，执行器饱和另记。'
        '持有四足力只在原50Hz求解时更新，500Hz执行没有重新按接触足数分配mg/n。\n\n'+
        f'三次中同一时点命令相对持有计划的最大合力差范数为{maxf:.3f}N，最大力矩差范数为{maxm:.3f}Nm。'
        '二者是命令差，不是实测外力误差，峰值也不一定同一时刻。连续轨迹显示反复离地/再接触、加载受限和支撑不足伴随机体失稳。'
        '这说明两项接口修复不足以建立可用刚地执行；不能把全部剩余失败归结为某一个未经对照隔离的动态项。\n\n'+
        '**结束决定：停止T1R的连续增量修补，后续执行器方向选择复用已有Go2全身执行底座，首选TM20的Go2路线；本轮没有安装、运行或迁移该平台。** '
        '接入成本包括：关节目标/PD到实际限幅力矩的一致映射；将持续材料内状态、本构反力及其扰动副本接入每条MuJoCo预测；保留软区唯一物理反力、恢复/2%域和原碰撞；把原往返/转向/终点站立条件映射成任务目标。'
        '其中材料状态进入候选与导数计算是主要新工作，不能只接一个外力回调或替换控制器类。工时尚未实测。'
        '作者部署接口仍标注开发中，采用外部动捕，不能称开箱即用或已验证NX。第二轮冻结B1仍只作为刚地/KV局部参照。\n\n'+
        '本轮只执行刚地开发，未检验TM02受载、跨足重访或记忆收益；失败短轨迹不作为更快/节能证据。'
        '所有历史T1/第三轮B1/第二轮B1比较均引用原版本，没有新基线仿真。\n\n'+
        f"[运行索引]({C['outputs']['compact_root']}/index.json)、[全部启动结果]({C['outputs']['compact_root']}/episodes.csv)、"
        f"[执行量]({C['outputs']['compact_root']}/contact-execution.csv)、[历史配对]({C['outputs']['compact_root']}/paired-existing-versions.csv)、"
        f"[连续图]({C['outputs']['compact_root']}/figures/continuous-execution.png)及[新数组与全部三次完整失败视频](https://github.com/{REPO}/releases/tag/{TAG})。")
    p=ROOT/'README.md';s=p.read_text();a=s.index('## 当前任务：');b=s.index('## S1 第三轮已冻结：',a)
    p.write_text(s[:a]+'## S1 第四轮已结束：接口修复未形成可用刚地执行\n\n'+summary+'\n\n'+s[b:])
    p=ROOT/'docs/TERRAIN_MEMORY_PLAN.md';s=p.read_text();a=s.index('### 5.5 当前唯一下一轮：');b=s.index('#### 5.5.1',a)
    s=s[:a]+'### 5.5 第四轮已结束：T1R接触执行修正的实际结果\n\n'+summary.replace('(results/','(../results/')+'\n\n以下为本轮已关闭的固定协议，保留作结果解释，不能再次启动。\n\n'+s[b:]
    # Preserve the original numerical protocol, but replace the now-obsolete launch prompt.
    a=s.index('## 7.');s=s[:a]+'''## 7. 本轮执行结束

第四轮开发3/3均失败，正式版本未冻结，三个正式地面阶段未执行。源码、完整失败轨迹、连续图与视频进入本轮Release；发布按资产size/digest确认。停止连续增量修补，不再次调用旧启动词，也不自动迁移平台或启动下一轮。后续执行底座选择与材料接入成本见第5.5节及本轮运行索引。
'''
    p.write_text(s)
    p=ROOT/'AGENTS.md';s=p.read_text();a=s.index('用户于 **2026-10-26**');b=s.index('## 已确认的研究位置',a)
    p.write_text(s[:a]+'''**2026-10-26第四轮已在固定开发预算内结束：3次完整任务启动，0完成，正式结果0；没有可冻结且完成刚地任务的版本。正式刚地、KV8000与TM02均未执行。** 当前只允许收尾本轮源码/真实结果/视频发布和size/digest确认，不再次启动旧矩阵。结果与后续执行底座决定见计划第5.5节和 `results/terrain_memory/simulation/tm-s1-contact-execution-20261026/index.json`。

停止T1R连续增量修补；后续选择复用成熟Go2全身执行底座，首选TM20 Go2路线，但本轮没有迁移、安装或试跑。不得自动开新轮、训练或硬件任务。以下保留本轮执行约定及既有证据，数字预算不得刷新。

'''+s[b:])
    C['implementation_status']='executed_development_closed_no_formal_version'
    C['actual_outcome']={'new_mission_starts':3,'development_missions':3,'development_completed_missions':0,'formal_records':0,
        'formal_frozen_version':None,'executed_implementation_commit':index['implementation_commit'],
        'unexecuted':['rigid formal','KV8000','TM02'],'immutable_per_attempt_configuration':'The resolved JSON and source snapshot next to each actual trajectory retain the exact pre-execution configuration and fingerprint.'}
    (ROOT/CONFIG).write_text(json.dumps(C,indent=2,ensure_ascii=False)+'\n')
    index['publication_summary']=summary
    index['postprocessing_commands']=['python scripts/summarize_s1_execution.py','python scripts/video_terrain_memory_s1.py --config '+CONFIG+' --attempts',
        'python scripts/publish_s1_execution.py --prepare','python scripts/package_terrain_memory_s1.py --config '+CONFIG+' --release-tag '+TAG,
        'python scripts/publish_s1_execution.py']
    index['reproduction']={'base_main_commit':'e53e16c5cd02ebee5a12d0aa49cc63b7f028226a','implementation_commit':index['implementation_commit'],
        'upstream_base':'c46cb9a3960554cffc467a71c6674b65c0f4a2c8','upstream_full_patch':'experiments/terrain_memory/s1-contact-execution-upstream.patch',
        'environment':'Existing work/envs/pympc; source scripts/tm01_env.sh selects the isolated patched upstream.',
        'setup':'Apply the retained full upstream patch to the pinned clean upstream before loading the original environment; all executed snapshots and source hashes are archived with the actual attempts.',
        'do_not_restart':'This round is closed. The recorded commands describe actual execution; repeating them would be a new study.'}
    INDEX.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    notes=ROOT/'work/s1-execution/release-notes.md'
    notes.write_text(summary.replace('(results/','(https://github.com/'+REPO+'/blob/main/results/')+'\n\nAll three development attempts retained; no formal stage was run.\n')
    print('Prepared actual results and closure decision.')


def publish():
    index=json.loads(INDEX.read_text())
    if not index['experiments_finished'] or len(index['attempts'])!=3 or not all(a['status']=='finished' for a in index['attempts']):
        raise RuntimeError('Physical attempt outcomes are incomplete; no publication performed.')
    token=os.environ['GH_TOKEN']
    env=dict(os.environ,GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
        GIT_CONFIG_KEY_1='http.sslCAInfo',GIT_CONFIG_VALUE_1='/etc/ssl/certs/ca-certificates.crt',
        GIT_CONFIG_KEY_2='http.proxy',GIT_CONFIG_VALUE_2=os.environ['HTTPS_PROXY'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,text=True).strip()
    def gh(*args):return subprocess.check_output([GH,*args],cwd=ROOT,text=True)
    git('fetch','origin','main')
    git('merge-base','--is-ancestor','origin/main','HEAD')
    files=['AGENTS.md','README.md','docs/TERRAIN_MEMORY_PLAN.md',CONFIG,
        'scripts/summarize_s1_execution.py','scripts/publish_s1_execution.py','scripts/video_terrain_memory_s1.py','scripts/package_terrain_memory_s1.py',str(SMALL.relative_to(ROOT))]
    git('add',*files);git('reset','--',str((SMALL/'index.lock').relative_to(ROOT)))
    for p in (SMALL/'figures').glob('*.pdf'):git('add','-f',str(p.relative_to(ROOT)))
    git('commit','-m','Close T1R after three failed rigid missions with complete execution evidence')
    evidence=git('rev-parse','HEAD');git('push','origin','HEAD:main')
    gh('release','create',TAG,'--repo',REPO,'--target',evidence,'--draft','--title','S1 contact execution: bounded rigid failure evidence',
        '--notes-file',str(ROOT/'work/s1-execution/release-notes.md'))
    assets=index['publication']['assets']
    gh('release','upload',TAG,'--repo',REPO,*[a['execution_host_path'] for a in assets])
    releases=json.loads(gh('api',f'repos/{REPO}/releases?per_page=100'))
    release=next(r for r in releases if r['tag_name']==TAG)
    checks=[]
    for expected in assets:
        actual=next(a for a in release['assets'] if a['name']==expected['filename'])
        checks.append(dict(filename=actual['name'],bytes=actual['size'],digest=actual['digest'],url=actual['browser_download_url'],
            size_matches=actual['size']==expected['bytes'],sha256_matches=actual['digest']=='sha256:'+expected['sha256'],state=actual['state']))
    if not all(c['size_matches'] and c['sha256_matches'] and c['state']=='uploaded' for c in checks):
        raise RuntimeError('Asset size/digest mismatch; draft retained.')
    gh('release','edit',TAG,'--repo',REPO,'--draft=false')
    release=json.loads(gh('api',f'repos/{REPO}/releases/tags/{TAG}'))
    index['status']='completed_and_published_development_gate_failed'
    index['publication'].update(uploaded=True,evidence_commit=evidence,release_id=release['id'],draft=release['draft'],
        uploaded_asset_verification=checks,confirmed_utc=datetime.now(timezone.utc).isoformat(),
        verification_method='Execution-host SHA256/size against GitHub asset digest/size; no result/archive downloaded to user computer.')
    INDEX.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    git('add',str(INDEX.relative_to(ROOT)));git('commit','-m','Confirm T1R Release asset sizes and digests');git('push','origin','HEAD:main')
    print(json.dumps(dict(event='published',commit=git('rev-parse','HEAD'),release=release['html_url'],assets=len(checks))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    prepare() if args.prepare else publish()
