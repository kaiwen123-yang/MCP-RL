"""Deliver the D57 benchmark runtime and complete paired simulated collector streams to W61 Release."""
import hashlib,json,sys,time,zipfile,subprocess
from pathlib import Path
from paired import RAW,DEST,STUDY,SPEC,SITE,write
sys.path.insert(0,str(SITE/'scripts/w61/material'))
import release_trials as release

def main():
    release.remote=release.inventory()
    staging=RAW/'release';staging.mkdir(exist_ok=True)
    config=json.loads(SPEC.read_text());receipts=DEST/'release.json'
    checked=json.loads(receipts.read_text())['assets'] if receipts.exists() else {}
    while True:
        groups={};keys=[]
        if 'runtime' not in checked:
            groups['runtime']=[(RAW/n,n) for n in ('fused-m1.cu','fused-m1.so','frozen_collect.py','build.json')]+[(SPEC,SPEC.name)]
        for w in config['windows']:
            for arm in ('cpu','gpu_batch'):
                name=f'{w}-{arm}';keys.append(name)
                if name in checked:continue
                f=DEST/(name+'-a02.json') if (DEST/(name+'-a02.json')).exists() else DEST/(name+'.json')
                if not f.is_file():continue
                rec=json.loads(f.read_text())
                if rec['status']!='complete':continue
                folder=Path(rec['raw_directory'])
                groups[name]=[(p,str(p.relative_to(folder))) for p in sorted(folder.rglob('*')) if p.is_file()]+[(f,'arm-summary.json')]
        for label,files in groups.items():
            asset=staging/(STUDY+'-'+label+'.zip')
            if not asset.is_file():
                pending=asset.with_suffix('.pending')
                with zipfile.ZipFile(pending,'w',allowZip64=True) as pack:
                    for p,name in files:pack.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix in ('.npz','.gz','.so') else zipfile.ZIP_DEFLATED)
                pending.replace(asset)
            row=release.push(asset);checked[label]=row
            write(receipts,dict(tag=release.TAG,assets=checked))
            release.stage_list(SITE,'material',STUDY,staging,list(checked.values()))
            print('verified',label,row['size'],row['sha256'],flush=True)
        if groups:
            git=['git','-C',str(SITE)]
            subprocess.run([*git,'add','--sparse','--',str(receipts.relative_to(SITE))],check=True)
            if subprocess.run([*git,'diff','--cached','--quiet']).returncode==1:
                subprocess.run([*git,'commit','-qm','Verify D57 W61 multi-collector evidence on GitHub Release'],check=True)
        if all(k in checked for k in ['runtime',*keys]):return
        time.sleep(10)

if __name__=='__main__':main()
