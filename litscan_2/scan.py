"""
Utilities to scan a directory/list of JATS archive shards.
"""

import os
from pathlib import Path

from tqdm import tqdm

from . import _core


def get_filter_model(weights, metadata):
    return _core.TfidfClassifier(
        weights,
        metadata,
    )


def _shard_list(shards):
    if isinstance(shards, (str, os.PathLike)) and os.path.isdir(shards):
        return sorted(str(p) for p in Path(shards).iterdir() if p.is_file())
    return [str(s) for s in shards]


def scan_shards(
    scanner,
    shards,
    output_dir,
    filter_model=None,
    min_probability=None,
    n_decompress=None,
    progress=True,
):
    """Scan a list of shards (or a directory of them) into `output_dir`.

    For each shard `<shard>` this writes `mentions/<shard>.parquet`,
    `articles/<shard>.parquet`, `notices/<shard>.parquet` and, last,
    `manifest/<shard>.json`. A shard
    is complete only once its manifest exists. Re-running overwrites.

    `filter_model` adds `rna_probability` to the articles output. Articles are
    skipped only if `min_probability` is also set.

    Multiple shards are decompressed and scanned concurrently in Rust. Set
    `n_decompress` to override the number of shard workers (default
    ceil(ncpu/4)). Each worker reads, decompresses and splits one shard at a
    time; parsing and scanning run on one pool that uses every CPU. It mainly
    bounds memory: each in-flight shard holds its decompressed text (~1-1.5 GB).

    Returns (articles_total, articles_scanned, articles_below_threshold, mentions, notices)
    summed over all shards.
    """
    shard_list = _shard_list(shards)
    pbar = tqdm(total=len(shard_list)) if progress else None
    callback = (lambda _shard: pbar.update(1)) if pbar is not None else None
    try:
        return _core.scan_shards(
            scanner,
            shard_list,
            str(output_dir),
            filter_model,
            min_probability,
            n_decompress,
            callback,
        )
    finally:
        if pbar is not None:
            pbar.close()
