"""D57 deletion eligibility for the finite Nov 23 archive, with live and queued dependencies."""
import json,os,re,subprocess,time
from pathlib import Path
from inventory import BASE,ROOT,CTL,PERSONAL
R12='tm-s1-go2-local-readiness-r12'
W57=BASE/'MCP-RL-w57'
COMPACT=W57/'results/terrain_memory/simulation'/R12/'w57'
RAW=W57/'outputs/terrain_memory/simulation'/R12
KNOWN=set(json.loads((BASE/'MCP-RL-migration-20261111/protection.json').read_text())['protected_paths'])
SPLITS=json.loads((BASE/'MCP-RL-w58/experiments/terrain_memory/w58/material.json').read_text())['recorded_episodes']
PROTECTED_EP={e for group in SPLITS.values() for e in group}
EP=re.compile(r'w57-pilot0[12]-(?:full|blind)-[ab]-e60-[0-9]{4}')
PLAN=re.compile(r'(mpc|control|drift)-[0-9]+\.npz$')
def within(path,root):
    return path==root or path.startswith(root+'/')
def ancestors(path):
    while path and path!='/':
        yield path
        path=path.rpartition('/')[0]
def live_paths():
    live=set();busy=set()
    for pid in Path('/proc').iterdir():
        if not pid.name.isdecimal():continue
        try:
            cwd=os.readlink(pid/'cwd')
            if cwd==str(CTL):continue
            if cwd.startswith((str(ROOT),str(PERSONAL))):busy.add(cwd)
            for fd in (pid/'fd').iterdir():
                try:s=os.readlink(fd)
                except OSError:continue
                if s.startswith((str(ROOT),str(PERSONAL))):live.add(s)
            for a in (pid/'cmdline').read_bytes().split(b'\0'):
                s=os.fsdecode(a)
                if s.startswith((str(ROOT),str(PERSONAL))):live.add(os.path.realpath(s))
        except OSError:continue
    return live,busy
def pins(db,foreign=False):
    out=set(KNOWN)
    for s,link in db.execute("SELECT path,link FROM files WHERE kind='link'"):
        if not link:continue
        t=os.path.abspath(os.path.join(os.path.dirname(s),link))
        if not t.startswith((str(ROOT),str(PERSONAL))):continue
        t=os.path.realpath(t)
        if os.path.basename(t) in ('work','outputs','results'):continue
        if EP.fullmatch(os.path.basename(t)) and (within(t,str(PERSONAL)) or within(t,str(W57))):continue
        out.add(t)
    if foreign:
        roots=set(json.loads((CTL/'roots.json').read_text())['roots'])
        others=[str(f) for f in BASE.iterdir() if not f.is_symlink() and f.is_dir() and str(f) not in roots and f!=CTL]
        if others:
            proc=subprocess.run(['find',*others,'-xdev','-type','l','-printf','%p\0%l\0'.replace('\0','\\0')],capture_output=True)
            parts=proc.stdout.split(b'\0')
            for i in range(0,len(parts)-1,2):
                if i+1>=len(parts) or not parts[i]:break
                p,t=map(os.fsdecode,parts[i:i+2])
                candidate=os.path.normpath(os.path.join(os.path.dirname(p),t))
                if candidate.startswith((str(BASE),str(PERSONAL))):
                    target=os.path.realpath(candidate)
                    if target.startswith((str(ROOT),str(PERSONAL))):out.add(target)
            # Ignore concurrent removal of unrelated temporary files, not permission errors.
            if proc.returncode and b'Permission denied' in proc.stderr:raise RuntimeError('Foreign link protection scan was incomplete')
    return out

class Policy:
    def __init__(self,db,fresh=False):
        self.roots=json.loads((CTL/'roots.json').read_text())
        self.registered=set(self.roots['registered'])
        f=CTL/'dependency-pins.json'
        if fresh or not f.exists():
            from inventory import save
            self.pins=pins(db,foreign=True)
            save(f.name,sorted(self.pins))
        else:self.pins=set(json.loads(f.read_text()))
        self.live,self.busy=live_paths()
        self.delivered={}
        for p in (COMPACT/'delivery').glob('w57-pilot*-e60-*.json'):
            if not EP.fullmatch(p.stem):continue
            d=json.loads(p.read_text())
            if d.get('status')=='verified' and d.get('original_file_count',0)>0 and len(d.get('assets',[]))>=1:
                a=COMPACT/'attempts'/(p.stem+'.json')
                if a.exists():
                    row=json.loads(a.read_text())
                    if row.get('status')=='passed' and row.get('task_mode')=='training' and row.get('ended_unix_s'):
                        self.delivered[p.stem]=True
    def reason(self,path,root,kind,tracked,nlink,opened,source_state):
        if kind!='file':return 'retain:topology_or_runtime_object'
        if tracked:return 'retain:git_tracked'
        if source_state.startswith('captured') or opened:return 'retain:live_version'
        if nlink!=1:return 'retain:shared_inode'
        if path in self.live or any(parent in self.busy for parent in ancestors(path) if parent!=root):return 'retain:live_path'
        if any(parent in self.pins for parent in ancestors(path)):return 'retain:named_fixture_or_link_target'
        rel=path[len(root)+1:];bits=rel.split('/');name=bits[-1]
        if '.git' in bits or name.endswith(('.lock','.sock')):return 'retain:repository_or_lock'
        if root.startswith(str(BASE/'MCP-RL-migration-')):
            return 'remove:retired_migration_metadata' if name!='protection.json' else 'retain:protection_ledger'
        if path.startswith(str(PERSONAL)+'/W57/originals/'):
            ep=bits[2] if len(bits)>2 else ''
            tail=bits[3:]
        elif root==str(W57) and rel.startswith('outputs/terrain_memory/simulation/'+R12+'/'):
            ep=bits[4] if len(bits)>4 else '';tail=bits[5:]
        elif root==str(W57) and bits[:2]==['work','w57-raw-scratch']:
            ep=bits[2] if len(bits)>2 else '';tail=bits[3:]
        else:ep=None;tail=[]
        if ep is not None:
            if ep not in self.delivered:return 'retain:active_formal_or_pending_GitHub_receipt'
            if ep in PROTECTED_EP:return 'retain:registered_material_case'
            # Preserve the observed event/foot histories, trajectory, video and node/credit inputs.
            # Final timing reads those histories. Planner physics only consumes P or PR, never P0 collectors.
            if len(tail)==2 and tail[0]=='w55-f' and PLAN.fullmatch(name):return 'remove:published_P0_offline_solver_records'
            if len(tail)==1 and re.fullmatch(r'physical-[0-9]+\.npz',name):return 'remove:published_P0_physical_chunks'
            return 'retain:W57_collector_replay_and_publication_inputs'
        if root==str(PERSONAL) or root.startswith(str(BASE/'MCP-RL-w57')) or root.startswith(str(BASE/'MCP-RL-w58')):
            return 'retain:active_worktree_or_dispatch'
        if root not in self.registered:return 'retain:inspect_project_support'
        if bits[0]=='outputs':
            if 'training' in bits or 'evaluation-policy' in bits or 'resume-history' in bits or name.endswith(('.pt','.pth','.safetensors','.onnx')):
                return 'retain:policy_or_training_state'
            if len(bits)>4 and bits[1:3]==['terrain_memory','simulation']:
                head=bits[4]
                if (('-r55-' in head or '-r60-' in head) and '-eval-' in head or
                    '-r58-' in head and '-E300-' in head or head in ('w55-c','w47') or
                    (head=='w46-delivery' and len(bits)>5 and bits[5] in ('w46-a','w46-b'))):
                    return 'retain:baseline_or_named_replay'
                # Cases read by the existing material replay / release-identification tools.
                if '-w54-' in head and any(x in head for x in ('-r44-','-r48-','-r51-','-r52-')):
                    return 'retain:material_development_history'
                if head.startswith('job-') and '-eval-' in head or head.startswith('train-') or head.endswith('-delivery') or head=='raw-bundles':
                    if name.endswith(('.json','.jsonl','.log','.txt','.sh')) and len(tail)==0 and os.path.basename(name) in ('live.json','video-asset.json'):
                        return 'retain:compact_endpoint_metadata'
                    return 'remove:archived_historical_episode'
                if head=='jobs' or head=='w52-analysis' or 'diagnostics-transfer' in bits or 'selected-release-upload-links' in bits:
                    if name.endswith(('.npz','.zip','.mp4','.jpg','.png','.mjb','.npy','.gz','.csv','.core')):
                        return 'remove:finished_job_or_delivery_payload'
            return 'retain:mechanism_or_execution_metadata'
        if bits[0]=='work' and len(bits)>1:
            if any(x.endswith('-video-review') for x in bits) or name.endswith('.core'):return 'remove:historical_debug_or_visualization'
            if (('CMakeFiles' in bits or 'build' in bits) and name.endswith(('.o','.o.d','.obj.tmp'))) and root!=str(ROOT):
                return 'remove:inactive_rebuildable_object'
        return 'retain:operating_tree_runtime_or_source'
