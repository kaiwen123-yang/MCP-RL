#!/usr/bin/env python3
"""Read explicitly selected W41 release members using the existing Range reader.

Selection is a JSON list of archive member names. The coverage report records
what was actually decompressed and checked, separately from manifest inventory.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--selection', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    selected = set(json.loads(args.selection.read_text()))
    source = Path(__file__).resolve().parent.parent / 'W40-final-review/read_raw_zip_members.py'
    spec = importlib.util.spec_from_file_location('original_range_reader', source)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reader.selected = lambda member: member in selected
    sys.argv = [sys.argv[0], *remaining]
    reader.main()


if __name__ == '__main__':
    main()
