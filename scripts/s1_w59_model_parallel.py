"""Acquire a disjoint registered member at a cumulative fitting node."""
import argparse,fcntl,json,os,time
from s1_w59_model_train import OUT,WORK,create_bank,fit_member,torch,write,BG

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',required=True,type=int);p.add_argument('--through',default=10000,type=int);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,ids,stats=create_bank()
    val=ids['development_validation'][0];val=val[torch.linspace(0,len(val)-1,min(len(val),4096),device='cuda').long()]
    recipe=json.loads((OUT/'recipe.json').read_text())
    for seed in a.seeds:
        rec=OUT/f'structured-{seed}.json'
        if rec.is_file() and json.loads(rec.read_text())['nodes'][-1]['updates']>=a.through:continue
        lock=(WORK/f'structured-{seed}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main=json.loads((OUT/'fit-live.json').read_text())
        if main['seed']>=seed:
            raise RuntimeError('Primary has reached this seed; no second owner admitted.')
        write(OUT/'second-trainer.json',dict(member=['structured',seed],pid=os.getpid(),target=a.through,checked_primary=main,started=time.time(),**BG))
        fit_member('structured',seed,data,ids['fit'],stats,val,recipe,a.through,'second-fit-live.json')
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    write(OUT/'second-trainer-complete.json',dict(seeds=a.seeds,target=a.through,finished=time.time(),**BG))

if __name__=='__main__':main()
