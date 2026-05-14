#!/usr/bin/env python3
"""Convert per-database jsonl files into a urs_taxid<TAB>pattern TSV.

Each input line is a JSON object with `urs_taxid` plus arbitrary other fields.
Every non-`urs_taxid` value is treated as a pattern; comma-separated values are
split into separate patterns. Empty/null patterns are skipped, and (urs_taxid,
pattern) pairs are de-duplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import re
import nltk


EXTRA_IGNORE = {
    "com", "mir-", "arg", "asn", "cys", "gln", "glu", "gly", "ile", "phe",
    "thr", "trp", "tyr", "val", "asx", "glx", "xaa", "mrna", "rrna", "rnai",
    "trna", "snrna", "ncrna", "12s", "16s", "16s rrna", "18s rrna", "28s",
    "28s rrna",
}


def load_ignore_set() -> set[str]:
    try:
        words = set(nltk.corpus.words.words())
    except LookupError:
        nltk.download("words", quiet=True)
        words = set(nltk.corpus.words.words())
    words = {w.lower() for w in words}
    words.update(EXTRA_IGNORE)
    return words


def iter_patterns(obj: dict):
    for key, value in obj.items():
        if key == "urs_taxid" or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        for part in text.split(","):
            part = part.strip()
            if part:
                yield part


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="jsonl files (or directories containing them)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("-"),
        help="output TSV path (default: stdout)",
    )
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.inputs:
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        else:
            files.append(p)

    ignore = load_ignore_set()

    seen: set[tuple[str, str]] = set()
    n_filtered = 0
    out = sys.stdout if str(args.output) == "-" else args.output.open("w")
    try:
        for f in files:
            with f.open() as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(
                            f"{f}:{lineno}: skipping malformed json ({e})",
                            file=sys.stderr,
                        )
                        continue
                    urs = obj.get("urs_taxid")
                    if not urs:
                        continue
                    for pattern in (urs, *iter_patterns(obj)):
                        # urs_taxid itself is never filtered; only secondary patterns.
                        if pattern != urs and (pattern.lower() in ignore or re.match(r"^\d+$", pattern)):
                            n_filtered += 1
                            continue
                        key = (urs, pattern)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.write(f"{urs}\t{pattern}\n")
        print(f"filtered {n_filtered} common-word patterns", file=sys.stderr)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
