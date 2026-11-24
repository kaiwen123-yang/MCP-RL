"""查询 D57 的原路径；指定 --out 才提取到一个尚不存在的目标文件。"""
import argparse,contextlib,hashlib,os,sqlite3,sys,tempfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out',type=Path);args=p.parse_args()
base=Path(__file__).resolve().parent;root=base.parent.parent
with sqlite3.connect((base/'archive-index.sqlite').as_uri()+'?mode=ro',uri=True) as db:
    cols='SELECT server_path,bytes,sha256,archive_relative_to_H_MPC_RL,zip_member,kind,link,server_status,archive_verified FROM archive_lookup'
    rows=db.execute(cols+' WHERE server_path=? ORDER BY id DESC',(args.source,)).fetchall()
    if not rows:
        candidate=args.source
        aliases=list(db.execute('select path,target from aliases order by length(path) desc'))
        for _ in range(3):
            nxt=next((target+candidate[len(alias):] for alias,target in aliases
                      if candidate==alias or candidate.startswith(alias+'/')),None)
            if not nxt or nxt==candidate:break
            candidate=nxt
            rows=db.execute(cols+' WHERE server_path=? ORDER BY id DESC',(candidate,)).fetchall()
            if rows:break
    if not rows and not args.out:
        for row in db.execute(cols+' WHERE instr(server_path,?)>0 LIMIT 20',(args.source,)):print(row)
        sys.exit()
    if not rows:sys.exit('没有匹配的完整 D57 路径。')
    path,n,h,archive,member,kind,link,status,ok=rows[0]
    print(path,kind,n,h,archive,member,status,sep='\n')
    if kind=='link':print('链接目标',link)
    if args.out:
        if not archive or ok!=1 or kind!='file':sys.exit('所选条目没有已核验的普通文件载荷。')
        dst=args.out.resolve()
        if dst.exists():sys.exit('目标已存在，请选择其他提取位置。')
        tmp=None
        with contextlib.ExitStack() as stack:
            src=stack.enter_context(stack.enter_context(zipfile.ZipFile(root/archive)).open(member)) if member is not None else stack.enter_context((root/archive).open('rb'))
            with tempfile.NamedTemporaryFile(prefix='.restore-',dir=dst.parent,delete=False) as out:
                tmp=Path(out.name);dig=hashlib.sha256();size=0
                while block:=src.read(2**20):out.write(block);dig.update(block);size+=len(block)
                out.flush();os.fsync(out.fileno())
        if (size,dig.hexdigest())!=(n,h):raise ValueError('提取内容的大小或 SHA-256 不一致；未写入目标。')
        os.rename(tmp,dst);print('SHA-256 核验后已提取',dst)

