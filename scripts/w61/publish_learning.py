"""Deliver closed W61 learning originals and videos directly from D57, with GitHub digest receipts."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"learning"))
from hybrid import CACHE, RESULT, ROOT, SPEC
from campaign import RUN, SMALL, STUDY, head_name
from s1_w59_learner_critic import atomic_json
import publish_s1_w59 as publisher

TAG="tm-s1-go2-local-readiness-r12-w61"
DELIVERY=RESULT/"delivery"


def setup():
    publisher.STATE=RESULT
    publisher.WORK=CACHE
    publisher.TAG=TAG
    publisher.release.SMALL=RESULT
    publisher.release.LARGE=RUN
    publisher.release.ROOT=ROOT
    publisher.release.TAG=TAG
    row=json.loads(publisher.github("api",f"repos/{publisher.release.REPO}/releases/tags/{TAG}"))
    record=dict(databaseId=row["id"],url=row["html_url"],tagName=row["tag_name"],targetCommitish=row["target_commitish"])
    atomic_json(RESULT/"release.json",record)


def verified(label):
    path=DELIVERY/(label+".json")
    return path.is_file() and json.loads(path.read_text())["status"]=="verified"


def development():
    for case in json.loads((RESULT/"interface-v1.json").read_text())["episodes"]:
        raw=Path(case["raw"])
        label="w61-E-interface-v1-"+case["variant"]
        if verified(label):
            continue
        if not (raw/"video.json").is_file():
            continue
        publisher.publish(label,[("episode",raw),("interface-v1.json",RESULT/"interface-v1.json")],raw/"complete.mp4")


def stage(head,version):
    node=version*256
    label=f"w61-E-{STUDY}-{head}-real-{node:05d}"
    compact=SMALL/head
    raw=RUN/head
    if not verified(label) and (compact/f"train-{version-1:03d}.json").is_file():
        last=json.loads((compact/f"train-{version-1:03d}.json").read_text())
        if last.get("status")=="updated":
            first={8:0,16:8,32:16}[version]
            files=[("study/manifest.json",RUN/"manifest.json"),("study/runtime-v2.json",SMALL/"runtime-v2.json"),
                   ("study/contract.json",ROOT/"experiments/terrain_memory/w61/learning/hybrid.json"),
                   ("study/response-update.json",ROOT/"experiments/terrain_memory/w61/learning/response-update.json")]
            if first==0: files.append(("checkpoint/policy-000.pt",raw/"policy-000.pt"))
            for m in range(first,version):
                files.extend([(f"checkpoint/policy-{m+1:03d}.pt",raw/f"policy-{m+1:03d}.pt"),
                    (f"used/used-{m:03d}.json",raw/f"used-{m:03d}.json"),
                    (f"updates/train-{m:03d}.json",compact/f"train-{m:03d}.json")])
                files.extend((f"physics/train-{m:03d}-{c}",raw/f"train-{m:03d}-{c}") for c in ("A","B"))
            publisher.publish(label,files)
    eval_receipt=compact/f"evaluate-{version:03d}.json"
    if eval_receipt.is_file():
        for case in json.loads(eval_receipt.read_text())["cases"]:
            for record in case.get("episodes",[]):
                video=record.get("video")
                if not video: continue
                movie=Path(video["file"])
                label=f"w61-E-{STUDY}-{head}-frozen-{node:05d}-{case['case']}"
                if not verified(label):
                    publisher.publish(label,[("episode",movie.parent),("evaluation.json",eval_receipt)],movie)


def main(watch):
    setup()
    with (CACHE/"release-learning.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            development()
            for method in [*SPEC["methods"],SPEC["geometry_ablation"]]:
                for seed in SPEC["seeds"]:
                    for v in (8,16,32):
                        stage(head_name(method,seed),v)
            if not watch or (SMALL/"lifecycle.json").exists():
                return
            time.sleep(45)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    main(p.parse_args().watch)
