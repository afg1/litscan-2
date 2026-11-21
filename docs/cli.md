# Command line

Installing the package adds a `litscan-2` command, which you can also run as
`python -m litscan_2`. Each subcommand wraps one function from the
[Python API](usage.md). Run `litscan-2 <command> --help` for every option.

| Command | Wraps | Output |
|---|---|---|
| `scan` | `scan_shards` | Parquet and manifest files in `-o DIR`; total counts as JSON on stdout |
| `paragraphs` | `extract_paragraphs` | One parquet file at `-o PATH`; the row count as JSON on stdout |
| `scan-one` | `scan_one` | Hits as TSV on stdout |
| `classify` | `TfidfClassifier.py_classify` | `label`, `rna_probability` and the first 80 characters of the text as TSV on stdout |

The commands use the rest of the API internally. `scan` and `scan-one` call
`build_scanner`. `--model` and `--metadata` load a `TfidfClassifier` through
`get_filter_model`. `classify` prints fields from the `ClassificationResult`.
For one shard, `scan` and `paragraphs` produce the same output as
`scan_shard` and `extract_paragraphs_for_shard`.

Logs and the progress bar go to stderr. Errors exit with status 1, and
invalid arguments exit with status 2.

## scan

```bash
litscan-2 scan terms.tsv shards/ -o out/
# {"articles_total": 4, "articles_scanned": 4, "articles_below_threshold": 0, "mentions": 4, "notices": 1}
```

`SHARDS` is either a single directory (every file in it, sorted) or one or
more `.xml` / `.xml.gz` files. Options:

| Option | Meaning |
|---|---|
| `--model`, `--metadata` | Classifier files. Adds `rna_probability` to the articles output. |
| `--min-probability P` | Skip articles whose abstract scores below `P` (0 to 1). Needs `--model`. |
| `--n-decompress N` | Shards in flight at once, at least 1 (default ⌈CPUs / 4⌉). Each holds 1–1.5 GB of text. |
| `--no-progress` | Hide the progress bar. |
| `-q`, `--quiet` | Hide the progress bar and log lines. |

```bash
litscan-2 scan terms.tsv shards/ -o out-filtered/ \
    --model model.safetensors --metadata model_metadata.json --min-probability 0.5
# {"articles_total": 4, "articles_scanned": 2, "articles_below_threshold": 2, "mentions": 3, "notices": 1}
```

## paragraphs

```bash
litscan-2 paragraphs out/mentions/ shards/ -o paragraphs.parquet
# {"paragraphs": 4}
```

`MENTIONS` is a parquet file, a glob, or the scan's `mentions/` directory.
Pass the same shards you scanned. `--no-progress` or `-q` hides the progress
bar.

## scan-one

```bash
litscan-2 scan-one terms.tsv article.xml     # or - to read from stdin
```

```text
term_id	section	symbol	sentence_offset	sentence
2512	title	MIR21	0	MIR21 and HOTAIR
9786	title	HOTAIR	10	MIR21 and HOTAIR
```

As with `scan_one`, no double-UTF-8 repair or filter model is applied.

## classify

```bash
litscan-2 classify --model model.safetensors --metadata model_metadata.json \
    "The long non-coding RNA HOTAIR regulates chromatin state."
```

```text
label	rna_probability	text
1	0.996222	The long non-coding RNA HOTAIR regulates chromatin state.
```

If you give no text arguments, every non-empty line on stdin is scored.
