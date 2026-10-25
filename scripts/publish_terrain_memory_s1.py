#!/usr/bin/env python3
"""Publish the completed repair round directly from its execution host."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPO = 'kaiwen123-yang/MCP-RL'
RUN = 'tm-s1-control-repair-20261024'
TAG = 'tm-s1-control-repair-evidence-20261024'
SMALL = ROOT/'results/terrain_memory/simulation'/RUN
INDEX = SMALL/'index.json'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare_results(index):
    groups = {(g['terrain'], g['controller']): g for g in index['groups']}
    terrain_names = {'rigid': '刚地', 'tm02_anchor': 'TM02 γ=0.025',
                     'kv_soft': 'KV3500', 'kv_moderate': 'KV8000'}
    table = ['| 地面 | B0 原C0 | B1 接触跟踪 | B2 0.24s | B3 2.5s |',
             '|---|---:|---:|---:|---:|']
    for terrain, label in terrain_names.items():
        values = [f"{groups[terrain, method]['successes']}/{groups[terrain, method]['episodes']}"
                  if (terrain, method) in groups else '—' for method in ['B0', 'B1', 'B2', 'B3']]
        table.append('| '+label+' | '+' | '.join(values)+' |')
    table = '\n'.join(table)
    count, success = index['completed_episodes'], index['successes']
    starts, dev = index['new_mission_starts'], index['development_mission_starts']
    budget = (f'**{count}/{count}条正式结果已完成，{success}成功、{count-success}失败；'
              f'20条B0引用原C0，40条为本轮新增正式结果。实际新增任务启动{starts}次，'
              f'开发尝试{dev}/8次，其中1次按同版本计入正式矩阵；总启动预算{starts}/68。**')
    meanings = ('B1在KV8000由4/5变为5/5，刚地仍为5/5；TM02仍实际应变超域，KV3500仍出现小腿触地。'
                'B1反演共同足底位移的面积加权合力，并在实际WB力矩路径执行支撑运动参考；GRF前馈只计一次。'
                '这是已知材料状态下的接触执行对照，不是完整STANCE c3WBC复现。')
    prediction = ('B2/B3继承正在执行的摆腿、接触与MPC状态，仅改变下一组未承诺落足的一次共享偏移，'
                  '共用执行、信息、代价与约束。刚地B2 seed3在6.0s结束：当次三个候选都预测在6.166s首次违反落足IK条件，'
                  '此前可比较前缀与实际一致；6.166s是预测拒绝时刻，不是实际跌倒时刻。该失败保留。'
                  '本轮比较短、长预测，没有把长时域本身称为事件创新，也未建立跨腿持久材料记忆的因果收益。')
    paired = ('B3在刚地和KV8000均为5/5，B2分别为4/5和3/5；B1在这两种地面也是5/5。'
              f"B3的平均完整任务时间分别为{groups['rigid','B3']['successful_completion_time_mean_s']:.4f}s和"
              f"{groups['kv_moderate','B3']['successful_completion_time_mean_s']:.4f}s，"
              '同五个配对seed相对B1分别减少0.2992s和0.4272s。'
              '长预测修复了本轮短预测中的完成率损失，但相对B1没有完成率增益；计算量明显更高。')
    limits = ('新增PANN/PPO训练0，C4暂停，硬件0。全部成功与失败轨迹保留，失败后的值不补零。'
              '五个seed是优化器配对种子，不是五种独立材料；只对双方完成的配对比较完成时间。'
              '关节机械功不是电池耗电，提前失败的短轨迹不代表更快或更节能。'
              '同步功能仿真的物理时间会等待计算，分项耗时及首调用单列，NX实时性未测。'
              '用户随后授权并行完成剩余独立回合；索引和逐回合记录区分串行、切换中及共享资源运行，'
              '这些混合资源条件下的wall time不能直接作为算法独占计算速度对比。')
    links = (f'[运行索引](results/terrain_memory/simulation/{RUN}/index.json)、'
             f'[全部回合](results/terrain_memory/simulation/{RUN}/episodes.csv)、'
             f'[配对差值](results/terrain_memory/simulation/{RUN}/paired.csv)、'
             f'[分项耗时](results/terrain_memory/simulation/{RUN}/timing.csv)、'
             f'[同绝对时间预测误差](results/terrain_memory/simulation/{RUN}/forecast-errors.csv)、'
             f'[实际落足承诺](results/terrain_memory/simulation/{RUN}/footstep-commitments.csv)与'
             f'[全部新轨迹和连续视频](https://github.com/{REPO}/releases/tag/{TAG})提供实际结果。'
             '旧B0数组和视频引用原Release。新资产从服务器直接上传，按size/digest核对，不在用户电脑下载或归档。')
    body = '\n\n'.join([budget, table, meanings, prediction, paired, limits, links])
    readme_path = ROOT/'README.md'
    readme = readme_path.read_text()
    start = readme.index('## 当前下一步：')
    end = readme.index('## 首轮 S1：', start)
    readme = readme[:start]+'## S1第二轮已完成：接触基线与预测执行修正\n\n'+body+'\n\n'+readme[end:]
    old = ('[运行入口](scripts/run_terrain_memory_s1.py)与[冻结实际配置](experiments/terrain_memory/s1-go2-contact.json)可复现本轮；'
           '下面仅是历史复现命令，不是新的启动任务：\n\n```bash\nsource scripts/tm01_env.sh\n'
           'python scripts/run_terrain_memory_s1.py --matrix --cohort formal-v1\n```')
    readme = readme.replace(old, '首轮运行入口与原85组合属于冻结提交 '
        '[1bfcf24](https://github.com/'+REPO+'/commit/1bfcf244bf5787cf0086f3f1846c19c42576f9d7)及'
        '[原配置](experiments/terrain_memory/s1-go2-contact.json)。当前main入口默认第二轮协议；不以当前默认命令重跑首轮。')
    readme_path.write_text(readme)

    agents_path = ROOT/'AGENTS.md'
    agents = agents_path.read_text()
    begin = agents.index('用户于2026-10-24要求')
    end = agents.index('## 读取与执行顺序', begin)
    agents = agents[:begin]+('## 本轮已完成，发布确认后结束\n\n'+budget+'\n\n'+table+'\n\n'
        +meanings+'\n\n'+prediction+'\n\n'
        '本轮实际成果见[索引](results/terrain_memory/simulation/'+RUN+'/index.json)。'
        '用户于2026-10-25授权完成剩余B2/B3、视频、汇总并直接发布；提交及Release资产size/digest确认后结束。'
        'R1、三轮R2、S0、首轮S1及本轮全部原配置和结果冻结；不自动重跑、加样本、训练PPO或启动硬件。'
        '以后只有新的明确授权才启动新任务，下面保留的是本轮执行规则，不是再次启动命令。\n\n')+agents[end:]
    agents = agents.replace('## 读取与执行顺序', '## 本轮保留的读取与执行顺序', 1)
    agents_path.write_text(agents)

    plan_path = ROOT/'docs/TERRAIN_MEMORY_PLAN.md'
    plan = plan_path.read_text()
    first_bold = plan.index('**')
    bold_end = plan.index('**', first_bold+2)+2
    plan = plan[:first_bold]+'**S1第二轮固定预算已完成。正式结果60条，20条旧B0引用、40条新增；新增训练0、硬件0。确认GitHub与Release后结束，不自动重开任何阶段。**'+plan[bold_end:]
    plan = plan.replace('| **S1 下一轮** | **第 5.3 节：先隔离接触执行，再修正受限落足规划** | 40 个基线结果，条件满足再增加 20 个规划结果；有效旧 C0 可复用 |',
        '| **S1 第二轮** | **固定预算已完成，保留全部失败** | 40条基线＋20条规划结果；旧C0复用20条，实际新启动'+str(starts)+'次 |')
    plan = plan.replace('### 5.3 下一轮唯一任务：先接触基线，再预测执行一致的落足规划',
                        '### 5.3 本轮已完成：接触基线与预测执行一致的落足规划')
    p = plan.index('**交付目标：', plan.index('### 5.3'))
    q = plan.index('#### 5.3.1', p)
    plan = plan[:p]+body.replace('(results/', '(../results/')+'\n\n以下保留本轮冻结协议。实际实现使用 `s1_stance.py` 和 `s1_prediction.py`，预测为同一闭环MuJoCo/MPC/WB的完整展开；没有采用简化长尾。各次实际resolved配置、源码摘要与定向修正均在索引中。\n\n'+plan[q:]
    p = plan.index('## 7. 下一轮启动提示词')
    plan = plan[:p]+('## 7. 本轮结束与后续授权\n\n本轮固定矩阵、连续视频、紧凑汇总与Release完成后停止。'
        '不得把第5.3节的历史协议或首轮85回合再次当作待启动任务。新研究任务需要用户另外明确授权；'
        '展示已发布视频不触发新仿真、训练、下载归档或硬件。\n')
    plan_path.write_text(plan)
    index['scientific_conclusions'] = dict(contact_execution=meanings, aligned_planning=prediction, paired_results=paired,
        interpretation=limits, completion_table=table)
    index['postprocessing_commands'] = [
        'python scripts/summarize_terrain_memory_s1.py --config experiments/terrain_memory/s1-control-repair.json --cohort formal-v1 --figures',
        'python scripts/video_terrain_memory_s1.py --config experiments/terrain_memory/s1-control-repair.json --cohort formal-v1',
        'python scripts/package_terrain_memory_s1.py --config experiments/terrain_memory/s1-control-repair.json --cohort formal-v1',
        'python scripts/publish_terrain_memory_s1.py']
    index['publication']['user_requested_video_display_later'] = True
    index['status'] = 'experiments_completed_publication_pending'
    write_json(INDEX, index)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gh', default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-s0/work/tools/gh_2.103.2_linux_amd64/bin/gh')
    parser.add_argument('--proxy', default='http://127.0.0.1:7897', help='Existing execution-host HTTP proxy for GitHub publication.')
    args = parser.parse_args()
    index = json.loads(INDEX.read_text())
    if not (index['phase_A_complete'] and index['phase_B_complete'] and index['attempt_totals']['ledger_active_count']==0):
        raise RuntimeError('The authorized actual matrix has not finished.')
    token = os.environ['GH_TOKEN']
    env = dict(os.environ, HTTP_PROXY=args.proxy, HTTPS_PROXY=args.proxy, ALL_PROXY=args.proxy,
               http_proxy=args.proxy, https_proxy=args.proxy, all_proxy=args.proxy,
               NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1')
    git_env = dict(env, GIT_CONFIG_COUNT='3', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
                   GIT_CONFIG_KEY_1='http.sslCAInfo', GIT_CONFIG_VALUE_1='/etc/ssl/certs/ca-certificates.crt',
                   GIT_CONFIG_KEY_2='http.proxy', GIT_CONFIG_VALUE_2=args.proxy)

    def git(*args, capture=False):
        return subprocess.run(['git', *args], cwd=ROOT, env=git_env, check=True,
                              capture_output=capture, text=True).stdout

    def gh(*args):
        return subprocess.check_output([args_gh, *args], cwd=ROOT, env=env, text=True)

    args_gh = args.gh
    git('fetch', 'origin', 'main')
    git('merge-base', '--is-ancestor', 'origin/main', 'HEAD')
    notes = prepare_results(index)
    notes_path = ROOT/'work/s1-repair-release-notes.md'
    notes_path.write_text(notes.replace('(results/', '(https://github.com/'+REPO+'/blob/main/results/'))
    git('add', 'scripts/publish_terrain_memory_s1.py', 'scripts/parallel_terrain_memory_s1.py',
        'scripts/summarize_terrain_memory_s1.py', 'scripts/package_terrain_memory_s1.py',
        'AGENTS.md', 'README.md', 'docs/TERRAIN_MEMORY_PLAN.md', str(SMALL.relative_to(ROOT)))
    for pdf in sorted((SMALL/'figures').glob('*.pdf')):
        git('add', '-f', str(pdf.relative_to(ROOT)))
    git('commit', '-m', 'Complete S1 repair round with paired missions, failures and server evidence')
    evidence_commit = git('rev-parse', 'HEAD', capture=True).strip()
    git('push', 'origin', 'HEAD:main')
    notes_path.write_text(notes_path.read_text()+'\n\nExecution and compact-result commit: ['+evidence_commit+'](https://github.com/'+REPO+'/commit/'+evidence_commit+').\n')
    print(gh('release', 'create', TAG, '--repo', REPO, '--target', evidence_commit, '--draft',
             '--title', 'S1 contact baseline and aligned planning — complete fixed-budget results', '--notes-file', str(notes_path)), flush=True)
    assets = index['publication']['assets']
    print(json.dumps(dict(stage='direct_upload', assets=len(assets), bytes=sum(a['bytes'] for a in assets))), flush=True)
    print(gh('release', 'upload', TAG, '--repo', REPO, *[a['execution_host_path'] for a in assets]), flush=True)
    release = next(r for r in json.loads(gh('api', f'repos/{REPO}/releases?per_page=100')) if r['tag_name']==TAG)
    checks = []
    for expected in assets:
        actual = next(a for a in release['assets'] if a['name']==expected['filename'])
        checks.append(dict(filename=actual['name'], asset_id=actual['id'], bytes=actual['size'], digest=actual['digest'],
            url=actual['browser_download_url'], state=actual['state'], size_matches=actual['size']==expected['bytes'],
            sha256_matches=actual['digest']=='sha256:'+expected['sha256']))
    if not all(c['size_matches'] and c['sha256_matches'] and c['state']=='uploaded' for c in checks):
        raise RuntimeError('Release size/digest does not match the execution-host assets: '+json.dumps(checks))
    gh('release', 'edit', TAG, '--repo', REPO, '--draft=false')
    release = json.loads(gh('api', f'repos/{REPO}/releases/tags/{TAG}'))
    index = json.loads(INDEX.read_text())
    index['status'] = 'completed_and_published'
    index['phase_B_execution_status'] = '20/20 actual missions completed; all successful and failed outcomes retained'
    index['publication'].update(uploaded=True, uploaded_asset_verification=checks, release_id=release['id'], draft=release['draft'],
        published_at=release['published_at'], evidence_commit=evidence_commit, upload_host='fy-5080',
        confirmed_utc=datetime.now(timezone.utc).isoformat(),
        verification_method='Execution-host pre-upload SHA256 and bytes match GitHub asset digest/size. No artifact download.',
        final_commit_interpretation='The evidence_commit contains code, configuration and compact results. The following main commit adds these actual publication confirmations only.')
    write_json(INDEX, index)
    git('add', str(INDEX.relative_to(ROOT)))
    git('commit', '-m', 'Confirm S1 Release asset sizes and SHA256 digests')
    git('push', 'origin', 'HEAD:main')
    final_commit = git('rev-parse', 'HEAD', capture=True).strip()
    remote_commit = json.loads(gh('api', f'repos/{REPO}/commits/main'))['sha']
    if remote_commit!=final_commit or release['draft']:
        raise RuntimeError('The final public commit or Release publication is not confirmed.')
    print(json.dumps(dict(stage='completed_and_verified', commit=final_commit, release=release['html_url'],
        assets=len(checks), verified_bytes=sum(c['bytes'] for c in checks), new_starts=index['new_mission_starts'],
        formal_results=index['completed_episodes'], successes=index['successes'])), flush=True)


if __name__=='__main__':
    main()
