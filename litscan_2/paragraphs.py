"""
Post-processing pass: given a scanner hits parquet and the same shard
directory the scanner ran over, emit one parquet row per unique
`(pmcid, section, offset, sentence)` containing the full paragraph that
surrounds the hit.

Inner-join the result onto the hits parquet to attach paragraph context to
every hit. Use the 4-key join when an article can have multiple sections
sharing a `SectionKind` (which is common — `(pmcid, section, offset)` alone
is not unique in the hits parquet):

    hits = pl.read_parquet("hits.parquet")
    paras = pl.read_parquet("paragraphs.parquet")
    joined = hits.join(paras, on=["pmcid", "section", "offset", "sentence"], how="left")
"""

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import polars as pl

from . import _core


def extract_paragraphs(hits_parquet, shards, output_path):
    """Walk `shards`, look up the containing paragraph for every unique
    `(pmcid, section, offset)` tuple in `hits_parquet`, and write a parquet
    with columns `(pmcid, section, offset, paragraph)` to `output_path`.

    `shards` is a directory or a list of shard paths — same shape as
    `scan_shards` accepts.

    Returns the total number of paragraph rows written.
    """
    requests = (
        pl.scan_parquet(hits_parquet)
        .select("pmcid", "section", "offset", "sentence")
        .unique(subset=["pmcid", "section", "offset", "sentence"])
        .collect()
    )

    by_pmcid: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for pmcid, section, offset, sentence in requests.iter_rows():
        by_pmcid[pmcid].append((section, int(offset), sentence or ""))

    if isinstance(shards, (str, os.PathLike)) and os.path.isdir(shards):
        shard_list = sorted(str(p) for p in Path(shards).iterdir() if p.is_file())
    else:
        shard_list = [str(s) for s in shards]

    result_files: list[str] = []
    total_rows = 0
    try:
        for shard_path in shard_list:
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
                    "offset": pl.UInt32,
                    "sentence": pl.Categorical,
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
