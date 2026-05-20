"""
Utilities to scan a directory/list of JATS archive shards.
"""

import os
import tempfile
from pathlib import Path

import polars as pl

from . import _core


def get_filter_model(weights, metadata):
    return _core.TfidfClassifier(
        weights,
        metadata,
    )


def scan_shards(scanner, shards, output_path, filter_model=None, n_decompress=None):
    """Scan a list of shards (or a directory of them) and concatenate
    per-shard parquet outputs into `output_path`.

    Multiple shards are decompressed and scanned concurrently in Rust. Set
    `n_decompress` to override the number of decompression workers (default
    is ceil(ncpu/4), with the remaining CPUs used by the per-shard parallel
    scan).

    Returns (total_articles, scanned_articles, filtered_articles, total_hits).
    """
    if isinstance(shards, (str, os.PathLike)) and os.path.isdir(shards):
        shard_list = sorted(str(p) for p in Path(shards).iterdir() if p.is_file())
    else:
        shard_list = [str(s) for s in shards]

    result_files = []
    for _ in shard_list:
        fd, tmp = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        result_files.append(tmp)

    try:
        total_articles, total_scanned, total_filtered, total_hits = _core.scan_shards(
            scanner, shard_list, result_files, filter_model, n_decompress
        )
        pl.scan_parquet(result_files).sink_parquet(output_path)
    finally:
        for f in result_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return total_articles, total_scanned, total_filtered, total_hits
