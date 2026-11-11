#!/usr/bin/env python3
"""One final W44-B publication handoff after its active evaluation and old upload.

The first raw queue covers only the earlier301-sample stop. This invokes the
existing finisher once when both current operations have exited; it never
starts training, modifies a clock, kills a process, or retries publication.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REL = Path("outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
SMALL = ROOT / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
JOB = ROOT / REL / "job-20261110T143400Z-w44-b"
KEY = "w44_b_final_delivery_handoff"

def record(status, **fields):
    with (SMALL / "index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = SMALL / "index.json"
        index = json.loads(path.read_text())
        row = index.setdefault("publication", {}).setdefault(KEY, {})
        row.update(status=status, updated_unix_s=time.time(), **fields)
        path.write_text(json.dumps(index, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(status=status, **fields), ensure_ascii=False), flush=True)

def wait_for_original_process(pid, expected_start):
    while True:
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return
        fields = text[text.rfind(")")+2:].split()
        if int(fields[19]) != expected_start or fields[0] == "Z":
            return
        time.sleep(30)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-pid", type=int, required=True)
    parser.add_argument("--compute-start", type=int, required=True)
    parser.add_argument("--upload-pid", type=int, required=True)
    parser.add_argument("--upload-start", type=int, required=True)
    a = parser.parse_args()
    record("waiting_for_current_B_workflow", computation_pid=a.compute_pid,
           computation_start_ticks=a.compute_start, old_upload_pid=a.upload_pid,
           old_upload_start_ticks=a.upload_start, deadline_unix_s=None,
           job=str(JOB.relative_to(ROOT)), publication_attempts=0,
           purpose="After final B evaluation and the earlier301-sample upload, run the existing finisher once for the full14737/57 result.",
           source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
           training_restarted=False, existing_processes_signalled=False)
    wait_for_original_process(a.compute_pid, a.compute_start)
    record("waiting_for_previous_raw_upload")
    wait_for_original_process(a.upload_pid, a.upload_start)
    collection = json.loads((ROOT / REL / "training/w44-b/PANN_PPO_S5-seed2/collection.json").read_text())
    evaluation = json.loads((JOB / "evaluation.json").read_text())
    if (collection["collected_transitions"] != 14737 or collection["updates"] != 57
            or evaluation["status"] == "running" or not (JOB / "evaluation-finished-at.txt").is_file()):
        record("final_result_not_available", collected_transitions=collection["collected_transitions"],
               updates=collection["updates"], evaluation_status=evaluation["status"])
        return 1
    code = int((JOB / "training-exit-code.txt").read_text())
    command = [sys.executable, "-u", "scripts/finish_s1_go2_r12_job.py",
               "--job-dir", str(JOB), "--training-exit-code", str(code)]
    record("running_one_final_postprocess", command=command, publication_attempts=1)
    log = JOB / "diagnostics-transfer/final-delivery-handoff-finisher.log"
    with log.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    record("final_postprocess_queued_or_reused" if result.returncode == 0 else "final_postprocess_failed",
           returncode=result.returncode, finisher_log=str(log.relative_to(ROOT)),
           raw_delivery_status="Use this job diagnostics/evidence.json; this handoff is not a claim that all uploads completed.")
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
