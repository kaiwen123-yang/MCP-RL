"""Fetch pinned public inference/model assets into the ignored work directory.

This script does not execute upstream deployment code or connect to hardware.
Use an authenticated git/gh environment. Model weights are not committed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCES = {
    "rl_sar": {
        "url": "https://github.com/fan-ziqi/rl_sar.git",
        "commit": "376d42c9b128f963ab08579762d5a216a976ce39",
        "paths": ["policy/go2"],
    },
    "rl_sar_zoo": {
        "url": "https://github.com/fan-ziqi/rl_sar_zoo.git",
        "commit": "7dd30bdc7806898950b354260655d5a7f0ce844e",
        "paths": ["go2_description"],
    },
}


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/go2-public-assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in SOURCES.items():
        dest = args.output / name
        if not dest.exists():
            run("git", "clone", "--filter=blob:none", "--no-checkout", source["url"], str(dest))
            run("git", "sparse-checkout", "set", *source["paths"], cwd=dest)
            run("git", "checkout", "--detach", source["commit"], cwd=dest)
        actual = run("git", "rev-parse", "HEAD", cwd=dest)
        if actual != source["commit"]:
            raise RuntimeError(f"{name}: expected pinned commit {source['commit']}, found {actual}; use a new output directory")
        if run("git", "status", "--porcelain", cwd=dest):
            raise RuntimeError(f"{name}: source checkout has changes; refusing to silently reuse")
        hashes = {}
        for relative in source["paths"]:
            for path in sorted((dest / relative).rglob("*")):
                if path.is_file():
                    hashes[path.relative_to(dest).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name] = {**source, "files_sha256": hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned public assets prepared at {args.output}; no policy execution or training")


if __name__ == "__main__":
    main()
