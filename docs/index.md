# litscan-2

litscan-2 finds mentions of a large, fixed vocabulary of identifiers (RNA gene
and transcript names, for RNAcentral) in the Europe PMC open-access full-text
corpus. You give it a terms file and a set of JATS XML shards. It writes Parquet
files listing every sentence that mentions a term, plus article metadata.

The core is written in Rust and exposed to Python as the `litscan_2` package
(built with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs)).

## The idea in one paragraph

The obvious way to do this is one search per term. LitScan 1 did that, and it
had 23.9 million search jobs.
litscan-2 compiles **every** term into a single
[Aho-Corasick](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)
automaton once, then reads each article **once**. The cost of that pass grows
with the length of the text and the number of matches, not with the number of
terms. Building the automaton for all 23.9M terms takes about 50 s. After that,
scanning two Europe PMC OA shards (each covering a range of 10,000 PMCIDs) takes
5–8 s on a 10-core laptop. See
[How it works](how-it-works.md).

## Quick start

```bash
pip install litscan-2          # wheels for Linux x86_64/aarch64 and macOS arm64, CPython ≥ 3.12
```

```python
import litscan_2
import polars as pl

scanner = litscan_2.build_scanner("terms.tsv")        # term_id<TAB>pattern, no header
litscan_2.scan_shards(scanner, "shards/", "out/")     # a directory of .xml or .xml.gz shards

mentions = pl.read_parquet("out/mentions/*.parquet")
```

See [Python usage](usage.md) for the full API, and [Output files](outputs.md)
for what each column means.

## Building from source

You need a Rust toolchain and Python ≥ 3.12.

```bash
pip install maturin
maturin develop --release     # builds the extension into the current virtualenv
cargo test                    # Rust unit and parity tests
```

The build embeds the git commit into the version string recorded in every
output manifest, e.g. `0.1.0+d34752e` (see `build.rs`). CI sets it through the
`LITSCAN_GIT_SHA` environment variable, because git doesn't work inside the
manylinux build container.

## Releases

CI (`.github/workflows/CI.yml`) runs `cargo test`, builds abi3 wheels and an
sdist, and deploys these docs on every push to `main`. Pushing a tag such as
`v0.1.0` also publishes to PyPI, using trusted publishing. The tag must match
`version` in `Cargo.toml`.
