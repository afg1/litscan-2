# litscan-2

Find mentions of millions of identifiers across the Europe PMC open-access
full-text corpus, fast.

litscan-2 compiles every term (e.g. ~24M RNAcentral gene and transcript names)
into a single [Aho-Corasick](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)
automaton and reads each article once. Scan time therefore scales with the
amount of text, not the number of terms. The core is Rust, and it's used from
Python.

**Documentation:** https://afg1.github.io/litscan-2/

## Install

```bash
pip install litscan-2
```

Wheels are built for Linux (x86_64, aarch64) and macOS (arm64), for CPython 3.12
and later.

## Usage

```python
import litscan_2
import polars as pl

scanner = litscan_2.build_scanner("terms.tsv")        # term_id<TAB>pattern, no header
litscan_2.scan_shards(scanner, "shards/", "out/")     # a directory of JATS .xml / .xml.gz shards

mentions = pl.read_parquet("out/mentions/*.parquet")  # term_id, pmcid, section, symbol, sentence, sentence_offset
```

An optional abstract classifier can score each article for RNA relevance, or
skip articles that score low. See the
[docs](https://afg1.github.io/litscan-2/classifier/).

## Development

```bash
pip install maturin
maturin develop --release
cargo test
```
