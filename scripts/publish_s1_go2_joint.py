#!/usr/bin/env python3
"""Direct execution-host publication of the bounded R9 physical negatives."""
import argparse,base64,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_s1_go2_wb_baseline import ROOT,dump,sha
REPO='kaiwen123-yang/MCP-RL'
TAG='tm-s1-go2-joint-r9'
SMALL=ROOT/'results/terrain_memory/simulation'/TAG

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','upload','finalize'])
    a=p.parse_args();token=sys.stdin.read().strip()
    env=dict(os.environ,GH_TOKEN=token,GIT_TERMINAL_PROMPT='0',
        GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
        GIT_CONFIG_KEY_1='http.proxy',GIT_CONFIG_VALUE_1=os.environ['HTTPS_PROXY'])
    ghpath=ROOT.parent/'terrain-memory-s0/work/tools/gh_2.103.2_linux_amd64/bin/gh'
    def gh(*x):return subprocess.check_output([str(ghpath),*x],env=env,cwd=ROOT,text=True).strip()
    def git(*x):return subprocess.check_output(['git',*x],env=env,cwd=ROOT,text=True).strip()
    def remote_main():
        return json.loads(gh('api',f'repos/{REPO}/git/ref/heads/main'))['object']['sha']
    def publish_source():
        # Source and compact evidence are sent from the execution host.
        # The caller supplies the existing verified network transport.
        git('fetch','origin','main')
        git('merge-base','--is-ancestor','origin/main','HEAD')
        git('push','origin','HEAD:main')
        head=git('rev-parse','HEAD')
        if remote_main()!=head:raise ValueError('Remote main does not match local published HEAD')
        return head
    idx=json.loads((SMALL/'index.json').read_text())
    if a.action=='prepare':
        commit=publish_source()
        url=gh('release','create',TAG,'--repo',REPO,'--target',commit,'--draft',
            '--title','S1 R9: actual TM02 material prerequisite failed under frozen S3',
            '--notes','Eight actual complete-mission attempts failed at the original material strain domain. PPO/PANN and formal evaluation were not run. Full original negative evidence is being uploaded; no joint success claim.')
        idx=json.loads((SMALL/'index.json').read_text())
        idx['publication']=dict(status='draft_uploading',release_url=url,evidence_commit=commit)
        dump(SMALL/'index.json',idx);print(url,flush=True);return
    release=next(x for x in json.loads(gh('api',f'repos/{REPO}/releases?per_page=100')) if x['tag_name']==TAG)
    pages=json.loads(gh('api','--paginate','--slurp',f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"))
    actual={v['name']:v for page in pages for v in page}
    if a.action=='upload':
        folder=ROOT/'outputs/terrain_memory/simulation'/TAG/'release-assets'
        folder.mkdir(exist_ok=True);todo=[]
        for asset in idx['assets']:
            if asset['size']>=2*1024**3:raise ValueError('Asset exceeds the specified release limit: '+asset['name'])
            if asset['name'] in actual:
                b=actual[asset['name']]
                if b['size']!=asset['size'] or b['digest']!='sha256:'+asset['sha256']:
                    raise ValueError('Published bytes do not match original: '+asset['name'])
                continue
            link=folder/asset['name']
            if not link.exists():link.symlink_to(ROOT/asset['path'])
            todo.append(link)
        def upload(path):
            gh('release','upload',TAG,'--repo',REPO,str(path))
            print(json.dumps(dict(uploaded=path.name,bytes=path.stat().st_size)),flush=True)
        with ThreadPoolExecutor(3) as pool:list(pool.map(upload,todo))
        return
    for asset in idx['assets']:
        b=actual[asset['name']]
        if b['state']!='uploaded' or b['size']!=asset['size'] or b['digest']!='sha256:'+asset['sha256']:
            raise ValueError('Release original-byte verification failed: '+asset['name'])
    notes=ROOT/'work/r9-release-notes.md'
    lines=['# Actual R9 result: material prerequisite blocked after 8/8 adaptation starts','',
        'The frozen R8 S3 model and binary were reused. Five constant caps and three predeclared 2Hz sequences all failed during outbound travel on the unchanged TM02 strip.','',
        '| Intent | Physical endpoint (s) | Body x (m) | Failed foot | Final indentation (mm) | Maximum principal strain |',
        '|---|---:|---:|---|---:|---:|']
    for r in idx['summary']:
        label=f"constant {r['cap_m_s']:.3f} m/s" if r['cap_profile']=='constant' else r['cap_profile']
        lines.append(f"| {label} | {r['physics_end_s']:.4f} | {r['max_body_x_m']:.6f} | {r['failed_foot']} | {-r['failed_boundary_bottom_mm']:.6f} | {r['max_abs_principal_strain']:.8f} |")
    lines += ['', 'The original material limit remains 0.02 (1mm for H=0.05m). The final out-of-domain constitutive boundary is retained, but its force is not applied or integrated into the robot. All eight complete recorded failure videos are included.',
        '', '8/8 adaptation physical starts, 8/32 nontraining starts, one zero-physics initialization failure, 306472 integrated physical steps / 61.2944 simulated seconds. No constitutive numerical failures. Native pre-attachment model and MPC binary SHA256 exactly match R8 S3. No strain clipping or relaxation of robot, material, controller, or mission limits.',
        '', 'The material adapter was implemented and executed. The PANN observer and joint PPO environment were not implemented/executed after the actual prerequisite failed. All six PPO streams, training transitions, new policies/weights and formal 21+3 evaluations remain unstarted. There are no training curves or joint-success claims.',
        '', 'All eight predeclared intents failed; this is not a mathematical impossibility proof for every possible time-varying policy. The full eight-start adaptation allowance is exhausted; the 48-hour wall-clock cap is not exhausted.',
        '', 'Assets are original execution-host files: 500Hz robot/control states, 5kHz physics and material boundaries, 500Hz material cell states, world-key load/revisit/release events, complete plans/startup records, allowed cap decisions, models, logs and eight full failure videos. Every asset is below2GiB. No dataset was downloaded to the user computer.',
        '', 'See the single repository R9 index and episodes.csv for exact provenance, timings, unexecuted work, figure definitions and final verified asset metadata.']
    notes.write_text('\n'.join(lines)+'\n')
    git('add','AGENTS.md','README.md','docs/TERRAIN_MEMORY_PLAN.md','experiments/terrain_memory/s1-go2-joint.json',
        'scripts/summarize_s1_go2_joint.py','scripts/publish_s1_go2_joint.py',str(SMALL.relative_to(ROOT)))
    # Lock files are excluded from versioned experiment data.
    git('reset','--',str((SMALL/'index.lock').relative_to(ROOT)))
    git('commit','-m','Retain complete R9 material negatives and original evidence manifest')
    commit=publish_source()
    gh('release','edit',TAG,'--repo',REPO,'--target',commit,'--notes-file',str(notes),'--draft=false')
    release=json.loads(gh('api',f'repos/{REPO}/releases/tags/{TAG}'))
    tag_commit=json.loads(gh('api',f'repos/{REPO}/git/ref/tags/{TAG}'))['object']['sha']
    if release['draft'] or tag_commit!=commit:raise ValueError('Published release tag does not identify the evidence commit')
    pages=json.loads(gh('api','--paginate','--slurp',f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"))
    actual={v['name']:v for page in pages for v in page}
    checks=[]
    for asset in idx['assets']:
        b=actual[asset['name']]
        expected=f"https://github.com/{REPO}/releases/download/{TAG}/{asset['name']}"
        checks.append(dict(name=b['name'],url=b['browser_download_url'],size=b['size'],digest=b['digest'],
            verified=b['state']=='uploaded' and b['size']==asset['size'] and b['digest']=='sha256:'+asset['sha256'] and b['browser_download_url']==expected))
    if not all(x['verified'] for x in checks):raise ValueError('Final asset metadata verification failed')
    idx['status']='blocked_material_domain_published_verified'
    idx['publication'].update(status='published_verified',release_url=release['html_url'],release_id=release['id'],
        evidence_commit=commit,published_tag_commit=tag_commit,asset_count=len(checks),
        asset_bytes=sum(x['size'] for x in checks),all_assets_verified=True,assets=checks,
        verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        method='Authenticated execution-host uploads; original size/SHA256 compared with final URL/size/digest; no asset downloads.')
    dump(SMALL/'index.json',idx)
    git('add',str((SMALL/'index.json').relative_to(ROOT)))
    git('commit','-m','Verify final R9 original asset URLs sizes and digests')
    final=publish_source();remote=remote_main()
    if final!=remote:raise ValueError('Remote main does not identify the final verified index')
    print(json.dumps(dict(final_commit=final,remote_main=remote,release_tag_commit=tag_commit,
        release=release['html_url'],assets=len(checks),bytes=sum(x['size'] for x in checks))),flush=True)
if __name__=='__main__':main()

