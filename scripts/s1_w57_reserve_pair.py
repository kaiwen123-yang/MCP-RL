"""Acquire both existing physics leases before starting the two W57 collectors."""
import fcntl
import os
from pathlib import Path
import sys
import time

from s1_w56_slot import QUEUE, old_task


def both():
    QUEUE.mkdir(parents=True,exist_ok=True)
    while True:
        held=[]
        with (QUEUE/'dispatch.lock').open('a') as dispatch:
            fcntl.flock(dispatch,fcntl.LOCK_EX)
            for slot in (0,1):
                if old_task(slot):break
                fd=(QUEUE/f'slot-{slot}.lock').open('a')
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    fd.close()
                    break
                held.append(fd)
            if len(held)==2:
                for fd in held:os.set_inheritable(fd.fileno(),True)
                return held
            for fd in held:fd.close()
        time.sleep(3)


if __name__=='__main__':
    args=sys.argv[1:]
    if not args:raise SystemExit('Provide the two-collector command')
    leases=both()
    print(f'Acquired physical slots 0 and 1; supervisor pid={os.getpid()}',flush=True)
    os.execvp(args[0],args)
