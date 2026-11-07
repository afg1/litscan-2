"""
Post-processing pass: given scanner mentions parquet(s) and the shards the
scanner ran over, emit one parquet row per unique `(pmcid, section, sentence)`
containing the full paragraph that surrounds the mention.

    mentions = pl.read_parquet("out/mentions/*.parquet")
    paras = pl.read_parquet("paragraphs.parquet")
    joined = mentions.join(paras, on=["pmcid", "section", "sentence"], how="left")
"""

import os
import tempfile
from collections import defaultdict

import polars as pl
from tqdm import tqdm

from . import _core
from .scan import _shard_list


def extract_paragraphs(mentions, shards, output_path):
    """Walk `shards`, look up the containing paragraph for every unique
    `(pmcid, section, sentence)` in `mentions` (a parquet path, glob, or the
    scanner's `mentions/` directory), and write a parquet with columns
    `(pmcid, section, sentence, paragraph)` to `output_path`.

    `shards` is a directory or a list of shard paths — same shape as
    `scan_shards` accepts.

    Returns the total number of paragraph rows written.
    """
    if isinstance(mentions, (str, os.PathLike)) and os.path.isdir(mentions):
        mentions = os.path.join(mentions, "*.parquet")
    requests = (
        pl.scan_parquet(mentions)
        .select("pmcid", "section", "sentence")
        .unique()
        .collect()
    )

    by_pmcid: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pmcid, section, sentence in requests.iter_rows():
        by_pmcid[pmcid].append((section, sentence))

    shard_list = _shard_list(shards)

    result_files: list[str] = []
    total_rows = 0
    try:
        for shard_path in tqdm(shard_list):
            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            rows = _core.extract_paragraphs_for_shard(shard_path, dict(by_pmcid), tmp)
            if rows > 0:
                result_files.append(tmp)
                total_rows += rows
            else:
                os.remove(tmp)

        if result_files:
            pl.scan_parquet(result_files).sink_parquet(output_path)
        else:
            # Produce an empty parquet with the right schema so downstream
            # joins don't blow up.
            pl.DataFrame(
                schema={
                    "pmcid": pl.Categorical,
                    "section": pl.Categorical,
                    "sentence": pl.Utf8,
                    "paragraph": pl.Utf8,
                }
            ).write_parquet(output_path)
    finally:
        for f in result_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return total_rows
