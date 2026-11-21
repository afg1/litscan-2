"""`litscan-2 <command> ...` (or `python -m litscan_2`). See docs/cli.md."""

import argparse
import json
import os
import sys
import time
from importlib.metadata import version

from polars.exceptions import PolarsError

from . import build_scanner, extract_paragraphs, get_filter_model, scan_one, scan_shards

COUNT_KEYS = ("articles_total", "articles_scanned", "articles_below_threshold", "mentions", "notices")
_TSV_SAFE = str.maketrans("\t\n\r", "   ")


def _shards_arg(shards):
    # A single directory is expanded by the library; otherwise pass the list.
    if len(shards) == 1 and os.path.isdir(shards[0]):
        return shards[0]
    for s in shards:
        if not os.path.isfile(s):
            raise FileNotFoundError(f"shard not found: {s}")
    return shards


def _build_scanner(terms, quiet):
    t0 = time.monotonic()
    scanner = build_scanner(terms)
    if not quiet:
        print(f"built scanner: {scanner.n_terms} terms, {scanner.n_patterns} patterns "
              f"in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    return scanner


def _write_tsv(header, rows):
    sys.stdout.write("\t".join(header) + "\n")
    for row in rows:
        sys.stdout.write("\t".join(str(v).translate(_TSV_SAFE) for v in row) + "\n")


def cmd_scan(args):
    shards = _shards_arg(args.shards)
    model = get_filter_model(args.model, args.metadata) if args.model else None
    counts = scan_shards(_build_scanner(args.terms, args.quiet), shards, args.output,
                         filter_model=model, min_probability=args.min_probability,
                         n_decompress=args.n_decompress,
                         progress=not (args.quiet or args.no_progress))
    print(json.dumps(dict(zip(COUNT_KEYS, counts))))


def cmd_paragraphs(args):
    n = extract_paragraphs(args.mentions, _shards_arg(args.shards), args.output,
                           progress=not (args.quiet or args.no_progress))
    print(json.dumps({"paragraphs": n}))


def cmd_scan_one(args):
    if args.xml == "-":
        xml = sys.stdin.read()
    else:
        with open(args.xml, encoding="utf-8") as f:
            xml = f.read()
    hits = scan_one(_build_scanner(args.terms, args.quiet), xml)
    _write_tsv(("term_id", "section", "symbol", "sentence_offset", "sentence"),
               ((h.term_id, h.section, h.symbol, h.sentence_offset, h.sentence) for h in hits))


def cmd_classify(args):
    model = get_filter_model(args.model, args.metadata)
    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]

    def row(text):
        r = model.py_classify(text)
        return r.label, f"{r.proba[1]:.6f}", text[:80]

    _write_tsv(("label", "rna_probability", "text"), map(row, texts))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="litscan-2", description="Scan Europe PMC JATS full text for millions of identifiers.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('litscan-2')}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, func, help, description):
        p = sub.add_parser(name, help=help, description=description)
        p.set_defaults(func=func, parser=p)
        return p

    def model_args(p, required=False):
        p.add_argument("--model", required=required, metavar="SAFETENSORS",
                       help="classifier weights (.safetensors)")
        p.add_argument("--metadata", required=required, metavar="JSON",
                       help="classifier metadata (.json)")

    def progress_args(p, quiet_help):
        p.add_argument("--no-progress", action="store_true", help="hide the progress bar")
        p.add_argument("-q", "--quiet", action="store_true", help=quiet_help)

    terms_help = "terms file: term_id<TAB>pattern, no header"
    shards_help = "a directory of shards, or shard files"

    p = command("scan", cmd_scan, "scan shards and write mentions/articles/notices parquet",
                "Scan JATS shards (.xml / .xml.gz). Writes OUTPUT/{mentions,articles,notices}/"
                "<shard>.parquet and OUTPUT/manifest/<shard>.json, and prints total counts as JSON.")
    p.add_argument("terms", help=terms_help)
    p.add_argument("shards", nargs="+", help=shards_help)
    p.add_argument("-o", "--output", required=True, help="output directory")
    model_args(p)
    p.add_argument("--min-probability", type=float, metavar="P",
                   help="skip articles whose abstract scores below P, 0 to 1 (needs --model)")
    p.add_argument("--n-decompress", type=int, metavar="N",
                   help="shards processed at once (default: ceil(CPUs/4)); bounds memory")
    progress_args(p, "no progress or log output")

    p = command("paragraphs", cmd_paragraphs, "extract the paragraph around each mention",
                "Write one row per unique (pmcid, section, sentence) in MENTIONS with its full "
                "paragraph. Pass the shards you scanned.")
    p.add_argument("mentions", help="mentions parquet file, glob, or the scan's mentions/ directory")
    p.add_argument("shards", nargs="+", help=shards_help)
    p.add_argument("-o", "--output", required=True, help="output parquet path")
    progress_args(p, "no progress output")

    p = command("scan-one", cmd_scan_one, "scan a single <article> and print hits as TSV",
                "Scan one JATS <article> in memory and print hits as TSV. "
                "No double-UTF-8 repair or filter model is applied.")
    p.add_argument("terms", help=terms_help)
    p.add_argument("xml", help="article XML file, or - for stdin")
    p.add_argument("-q", "--quiet", action="store_true", help="no log output")

    p = command("classify", cmd_classify, "score text for RNA relevance",
                "Print label and P(RNA) for each TEXT, or for each non-empty stdin line "
                "if no TEXT is given.")
    model_args(p, required=True)
    p.add_argument("text", nargs="*", help="text to score (default: read lines from stdin)")
    return parser


def _check_args(args):
    error = args.parser.error
    if args.command == "scan":
        if bool(args.model) != bool(args.metadata):
            error("--model and --metadata must be given together")
        if args.min_probability is not None and not args.model:
            error("--min-probability requires --model and --metadata")
        if args.min_probability is not None and not 0 <= args.min_probability <= 1:
            error("--min-probability must be between 0 and 1")
        if args.n_decompress is not None and args.n_decompress < 1:
            error("--n-decompress must be at least 1")
    if args.command in ("scan", "paragraphs") and len(args.shards) > 1:
        for s in args.shards:
            if os.path.isdir(s):
                error(f"a shard directory must be the only shard argument: {s}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    _check_args(args)
    try:
        args.func(args)
        sys.stdout.flush()  # so a closed pipe is caught here, not at interpreter exit
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # Point stdout at devnull so the flush at exit doesn't fail again (exit 120).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (OSError, ValueError, RuntimeError, PolarsError) as e:
        print(f"litscan-2: error: {e}", file=sys.stderr)
        return 1
    return 0
