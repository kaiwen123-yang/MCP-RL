"""Queue foreground original tasks across W56 worktrees and the two live r55 cases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REL = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
QUEUE = ROOT.parent / 'MCP-RL/work/w56-physical-slots'
BASELINE = ROOT.parent / 'MCP-RL-w55-f' / REL / 'w55/f'


def old_task(slot):
    case = ('A', 'B')[slot]
    status = json.loads((BASELINE / f'F-{case}-r55/E600.json').read_text())
    if status['status'] not in ('declared', 'running'):
        return False
    proc = Path('/proc') / str(status['pid']) / 'cmdline'
    try:
        return status['attempt'].encode() in proc.read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def run(priority, label, command):
    QUEUE.mkdir(parents=True, exist_ok=True)
    since = time.time_ns()
    ticket = QUEUE / f'{priority}-{since}-{os.getpid()}.waiting'
    ticket.write_text(label + '\n')
    print(f'Queued {label} pid={os.getpid()} priority={priority}', flush=True)
    try:
        while True:
            with (QUEUE / 'dispatch.lock').open('a') as dispatch:
                fcntl.flock(dispatch, fcntl.LOCK_EX)
                tickets = sorted(p for p in QUEUE.glob('*.waiting')
                                 if (Path('/proc') / p.stem.rsplit('-', 1)[-1]).exists())
                if tickets and tickets[0] == ticket:
                    for slot in (0, 1):
                        if old_task(slot):
                            continue
                        lease = (QUEUE / f'slot-{slot}.lock').open('a')
                        try:
                            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            lease.close()
                            continue
                        # Slot 0 normally matches original F-A's 10..13. A native stream
                        # occupies ten dedicated CPUs; the adjacent F stream retains four.
                        first = 10 if slot == 0 else 0
                        width = 10 if priority == 'N' else 4
                        os.sched_setaffinity(0, range(first, first + width))
                        os.set_inheritable(lease.fileno(), True)
                        os.environ['S1_W56_SLOT'] = str(slot)
                        ticket.unlink()
                        print(f'Dispatched {label} slot={slot} cpus={sorted(os.sched_getaffinity(0))}', flush=True)
                        os.execvp(command[0], command)
            time.sleep(5)
    finally:
        ticket.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priority', choices=('F', 'N'), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run(args.priority, args.label, args.command[1:] if args.command[:1] == ['--'] else args.command)
