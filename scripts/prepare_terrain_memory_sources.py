#!/usr/bin/env python3
"""Download the selected papers and obtain the pinned reproduction source."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "terrain-memory"
MANIFEST = ROOT / "literature" / "terrain_memory_sources.json"


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def get_papers(papers):
    destination = WORK / "papers"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for paper in papers:
        target = destination / paper["filename"]
        cached = target.exists()
        if not cached:
            request = Request(paper["pdf_url"], headers={"User-Agent": "terrain-memory-reproduction/1.0"})
            with urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append({
            "id": paper["id"], "url": paper["pdf_url"],
            "path": target.relative_to(ROOT).as_posix(), "sha256": digest,
        })
        print(f"{'existing' if cached else 'downloaded'} {paper['id']}: {target.relative_to(ROOT)}")
    return records


def get_upstream(upstream):
    target = ROOT / upstream["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        git("clone", upstream["url"], str(target))
    if git("status", "--porcelain", cwd=target):
        raise RuntimeError(f"Local changes preserved in {target}; use a clean checkout for reproduction.")
    if git("rev-parse", "HEAD", cwd=target) != upstream["commit"]:
        git("fetch", "origin", upstream["commit"], cwd=target)
        git("checkout", "--detach", upstream["commit"], cwd=target)
    head = git("rev-parse", "HEAD", cwd=target)
    print(f"upstream {target.relative_to(ROOT)} @ {head}")
    return {"url": upstream["url"], "path": target.relative_to(ROOT).as_posix(), "commit": head}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--papers-only", action="store_true")
    mode.add_argument("--upstream-only", action="store_true")
    parser.add_argument(
        "--paper-id", action="append", choices=[paper["id"] for paper in manifest["papers"]],
        help="Limit PDF downloads to these IDs; repeat for multiple papers.",
    )
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    lock_path = WORK / "sources.lock.json"
    records = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    if not args.upstream_only:
        selected = [paper for paper in manifest["papers"] if not args.paper_id or paper["id"] in args.paper_id]
        paper_records = {paper["id"]: paper for paper in records.get("papers", [])}
        paper_records.update({paper["id"]: paper for paper in get_papers(selected)})
        records["papers"] = list(paper_records.values())
        lock_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.papers_only:
        records["upstream"] = get_upstream(manifest["upstream"])
        lock_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Source identities: {lock_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
