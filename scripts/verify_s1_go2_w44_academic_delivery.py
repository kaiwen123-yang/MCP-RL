#!/usr/bin/env python3
"""Verify the existing W44 manifest after the academic-only recovery upload."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import publish_s1_go2_r12_diagnostics as diagnostics

ROOT=Path(__file__).resolve().parents[1]
REL=Path("terrain_memory/simulation/tm-s1-go2-local-readiness-r12")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream",choices=("a","b"),required=True)
    args=parser.parse_args()
    stamp="143358" if args.stream=="a" else "143400"
    name=f"job-20261110T{stamp}Z-w44-{args.stream}"
    job=ROOT/"outputs"/REL/name
    destination=ROOT/"results"/REL/"jobs"/name/"diagnostics"
    begin=time.time()
    command=[sys.executable,"-u","scripts/publish_s1_go2_r12.py","finish","--server-auth",
             "--manifest",str(destination/"manifest.json"),"--job-name",name+"-diagnostics",
             "--source-commit","d75e660bbc3b51ab3fc5333156e2784e33d080c2","--no-wall-deadline"]
    log=job/"diagnostics-transfer/academic-recovery-20261112/verify-remote.log"
    with log.open("w") as output:
        result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        print(json.dumps(dict(status="remote_verification_failed",returncode=result.returncode,log=str(log))),flush=True)
        return result.returncode
    index=json.loads((ROOT/"results"/REL/"index.json").read_text())
    registered={r["name"]:r for r in index["publication"]["assets"]}
    manifest=json.loads((destination/"manifest.json").read_text())
    entries=[registered[r["name"]] for r in manifest]
    hashes=json.loads((ROOT/"outputs"/REL/"academic-recovery-20261112/pending-source-hashes.json").read_text())
    hashes=[r for r in hashes if r["stream"]==args.stream]
    comparisons=[]
    for source in hashes:
        remote=registered[source["name"]]
        comparisons.append(dict(name=source["name"],local_sha256=source["sha256"],
            remote_digest=remote.get("remote_digest"),
            match=remote.get("remote_digest")=="sha256:"+source["sha256"]))
    passed=sum(r["match"] for r in comparisons)
    metadata_ok=all(r.get("status") in ("uploaded","git_recorded") and
                    (r.get("status")=="git_recorded" or r.get("remote_verified_unix_s",0)>=begin)
                    for r in entries)
    ctx=diagnostics.context(argparse.Namespace(job_dir=job,delivery_window=None,
                                               no_wall_deadline=True,hours=6))
    evidence,extra=diagnostics.finalize(ctx,verified=metadata_ok and passed==len(comparisons))
    evidence["academic_recovery_verification"]=dict(
        checked_unix_s=time.time(),network="academic_only",
        remote_manifest_assets=len(entries),remote_metadata_all_matched=metadata_ok,
        remote_digest_present=sum(bool(r.get("remote_digest")) for r in entries),
        local_sha256_comparisons=len(comparisons),local_sha256_matches=passed,
        digest_comparisons=comparisons,
        scope="All manifest names/states/sizes re-read from GitHub; local SHA256 comparison covers the412 originally pending recovery assets across A/B only.",
        remote_assets_downloaded=0,new_physics_or_semantic_recomputation=False,
        original_failed_starter_metadata="outputs/"+str(REL)+"/academic-recovery-20261112/starter-cleanup.json")
    diagnostics.write(destination/"evidence.json",evidence)
    diagnostics.git_delivery(destination,ctx[-1],extra)
    print(json.dumps(dict(stream=args.stream,status=evidence["status"],
        manifest_assets=len(entries),metadata_all_matched=metadata_ok,
        digest_compared=len(comparisons),digest_matched=passed,
        unavailable_preplant_records=len(evidence.get("unavailable_before_plant_start",[])),
        unresolved_missing=len(evidence.get("blocking_missing_sources",[]))),indent=2),flush=True)
    return 0 if evidence["status"]=="published" else 2

if __name__=="__main__":
    raise SystemExit(main())
