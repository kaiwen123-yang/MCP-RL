"""File-level retention for closed MCP-RL process outputs; the active physics and future inputs stay local."""
import os,re,json,time
from pathlib import Path
from legacy_policy import Policy as Legacy, BASE,ROOT,CTL,PERSONAL,W57,RAW,COMPACT,PROTECTED_EP,within,ancestors,live_paths

BULK=re.compile(r'(?:material(?:-final|-[0-9]+)|physical-[0-9]+|trajectory)\.npz$')
SOLVER=re.compile(r'(?:control|drift|mpc|material-plan)-[0-9]+\.npz$')
W58=BASE/'MCP-RL-w58'
SOURCE_PIN=set(PROTECTED_EP)|{'w57-m1-b-'+c+'-original-'+l+'-a01' for c in 'ab' for l in ('e30','e60')}|{'w57-m1-p-a-original-e30-a01'}
for f in [W58/'experiments/terrain_memory/s1-w58-revisit-v2.json',W58/'experiments/terrain_memory/w58/material.json']:
    SOURCE_PIN.update(re.findall(r'(?:w57-[a-z0-9-]+-a[0-9]{2}|w57-pilot[0-9]+-(?:full|blind)-[ab]-e60-[0-9]{4})',f.read_text()))
# Explicit frozen inputs of the W57 private-P comparison and the independently composed archive.
SOURCE_PIN.update({'w57-m1-p-a-short-a01','w57-m1-p-a-private-short-a01'})
LEGACY_DEBUG=re.compile(r'(?:r[0-9]+-current-plan\.bin\.r[0-9]+-candidates|r[0-9]+-candidates-call[0-9]+\.npz|(?:accepted-reference|recovery-reference)-.*-plan[0-9]+\.npz)$')
PUBLISHED_BULK={'model.mjb','full-mission-material-overlay.mp4'}
EVIDENCE={'material-events.jsonl','material-failures.jsonl','w57-revisit-pairs.jsonl','timing-decisions.jsonl','timing-transitions.jsonl','timing-final-observation.jsonl','timing-summary.json','cap-decisions.json','pann-action-dependence.json'}
class Policy(Legacy):
    def __init__(self,db,fresh=False):
        super().__init__(db,fresh=fresh)
        self.rows={f.stem:json.loads(f.read_text()) for f in (COMPACT/'attempts').glob('*.json')}
        self.finished_at=json.loads((CTL/'roots.json').read_text())['start_unix']
        self.public=set()
        for f in (COMPACT/'delivery').glob('*.json'):
            row=json.loads(f.read_text())
            if row.get('status')=='verified' and row.get('assets') and row.get('original_file_count',0)>0:
                self.public.add(f.stem)
        self.summarized={f.stem for f in (COMPACT/'mechanics').glob('*.json')}
        self.planner_done={f.stem for f in (COMPACT/'p/physical').glob('*.json')}
        self.refresh_live()
    def refresh_live(self):
        self.live,self.busy=live_paths();self.live_inodes=set()
        for p in Path('/proc').glob('[0-9]*'):
            try:
                for f in (p/'fd').iterdir():
                    try:
                        t=os.readlink(f)
                        if t.startswith((str(ROOT),str(PERSONAL))):
                            st=os.stat(f);self.live_inodes.add((st.st_dev,st.st_ino))
                    except OSError:continue
                with (p/'maps').open(errors='replace') as maps:
                    for line in maps:
                        fields=line.rstrip().split(None,5)
                        if len(fields)==6 and fields[5].startswith((str(ROOT),str(PERSONAL))):
                            self.live.add(fields[5])
            except OSError:continue
        self.live_checked=time.monotonic()
    def reason(self,path,root,kind,tracked,nlink,opened,source_state):
        if kind!='file':return 'retain:directory_link_or_special'
        if tracked:return 'retain:tracked_source_or_scientific_receipt'
        if nlink!=1:return 'retain:shared_inode'
        if opened or source_state.startswith(('captured','changed','missing')) or path in self.live:return 'retain:active_or_changed'
        rel=path[len(root)+1:];bits=rel.split('/');name=bits[-1]
        if '.git' in bits or name.endswith(('.sock','.lock')):return 'retain:repository_and_locks'
        if any(a in self.pins or (a!=root and a in self.busy) for a in ancestors(path)):return 'retain:named_dependency_or_live_directory'
        ep=None;tail=None
        if root==str(W57) and rel.startswith('outputs/terrain_memory/simulation/'+RAW.name+'/') and len(bits)>5:
            ep=bits[4];tail=bits[5:]
        elif root==str(W57) and bits[:2]==['work','w57-raw-scratch'] and len(bits)>3:ep=bits[2];tail=bits[3:]
        elif root==str(PERSONAL) and bits[:2]==['W57','originals'] and len(bits)>3:ep=bits[2];tail=bits[3:]
        if ep in self.rows:
            a=self.rows[ep]
            if not a.get('ended_unix_s') or a['ended_unix_s']>self.finished_at:return 'retain:active_at_snapshot'
            if a['status']=='technical_interruption':return 'retain:technical_prefix_for_final_delivery'
            if ep in SOURCE_PIN:return 'retain:registered_W58_or_private_reference'
            if a['level'] in ('E300','E600'):return 'retain:long_qualification_original'
            if a['task_mode']!='training' and (ep not in self.summarized or a['arm'] in ('P','PR') and ep not in self.planner_done):
                return 'retain:pending_analysis_input'
            # Final W57 timing/collector software still reads these tables, contracts and publications.
            simple=(len(tail)==1 and (BULK.fullmatch(name) or name in PUBLISHED_BULK or name=='r60-python-stacks.log'))
            leaf=(len(tail)==2 and tail[0]=='w55-f' and (SOLVER.fullmatch(name) or name in ('normal-execution.jsonl','mpc-solves.jsonl','material-candidates.jsonl','material-execution.jsonl')))
            model=(tail[0]=='model' and name.endswith(('.xml','.obj','.stl','.msh','.png','.mjb')))
            if simple or leaf or model:
                if ep not in self.public:return 'archive:pending_release'
                return 'remove:completed_W57_process'
            return 'retain:runtime_contract_event_history_or_final_analysis_input'
        if not tracked and bits[0] in ('scripts','src') and '__pycache__' in bits and name.endswith('.pyc') and root not in self.busy:
            source=Path(path).parent.parent/(name.split('.cpython-')[0]+'.py')
            if source.is_file():return 'remove:inactive_regenerable_python_cache'
        old=super().reason(path,root,kind,tracked,nlink,opened,source_state)
        if (old=='retain:mechanism_or_execution_metadata' and bits[0]=='outputs' and len(bits)>=5 and
                root in self.registered and root not in self.busy and LEGACY_DEBUG.fullmatch(name)):
            return 'remove:historical_debug_process'
        return old
