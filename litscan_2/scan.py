"""
Utilities to scan a directory/list of JATS archive shards.
"""

import os
import tempfile
from pathlib import Path

import polars as pl
from tqdm import tqdm

from . import _core


def get_filter_model(weights, metadata):
    return _core.TfidfClassifier(
        weights,
        metadata,
    )


def scan_shards(scanner, shards, output_path, filter_model=None):
    """Scan a list of shards (or a directory of them) and concatenate
    per-shard parquet outputs into `output_path`.

    Returns (total_articles, scanned_articles, filtered_articles, total_hits).
    """
    if isinstance(shards, (str, os.PathLike)) and os.path.isdir(shards):
        shard_list = sorted(str(p) for p in Path(shards).iterdir() if p.is_file())
    else:
        shard_list = [str(s) for s in shards]

    result_files = []
    total_articles = 0
    total_scanned = 0
    total_filtered = 0
    total_hits = 0
    try:
        for shard_path in tqdm(shard_list):
            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            tot_art, scanned_art, filt_art, hits = _core.scan_shard(
                scanner, shard_path, tmp, filter_model
            )
            result_files.append(tmp)
            total_articles += tot_art
            total_scanned += scanned_art
            total_filtered += filt_art
            total_hits += hits

        pl.scan_parquet(result_files).sink_parquet(output_path)
    finally:
        for f in result_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return total_articles, total_scanned, total_filtered, total_hits
