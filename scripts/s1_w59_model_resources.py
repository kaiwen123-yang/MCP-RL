"""Sample actual training resource overlap through the registered optimizer nodes."""
import json,subprocess,time
from pathlib import Path
from s1_w59_model_bank import OUT,WORK

def run():
    path=WORK/'trainer-resources.jsonl'
    while not (OUT/'fit-completion.json').is_file():
        rates={}
        for name in ('fit-live','second-fit-live'):
            f=OUT/(name+'.json')
            if f.exists():
                v=json.loads(f.read_text())
                rates[name]={k:v[k] for k in ('kind','seed','accepted_updates','updates_per_s','GPU_allocated_bytes')}
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
        pids=[]
        for p in Path('/proc').glob('[0-9]*'):
            try:
                cmd=(p/'cmdline').read_bytes().split(b'\0')
                if len(cmd)<3 or not cmd[2].endswith((b's1_w59_model_train.py',b's1_w59_model_parallel.py')):continue
                stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
                pids.append(dict(pid=int(p.name),cpu_ticks=int(stat[11])+int(stat[12]),rss_pages=int(stat[21])))
            except FileNotFoundError:continue
        with path.open('a') as f:f.write(json.dumps(dict(unix_s=time.time(),GPU_board_util_memMiB_powerW=gpu,trainer_processes=pids,rates=rates))+'\n')
        time.sleep(30)

if __name__=='__main__':run()
