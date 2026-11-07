# Output files

`scan_shards` and `scan_shard` write four files per shard, named after the
shard's filename with any `.gz` and `.xml` removed:

```text
out/
├── mentions/<shard>.parquet
├── articles/<shard>.parquet
├── notices/<shard>.parquet
└── manifest/<shard>.json      written last: its presence means the shard is complete
```

Parquet files are zstd-compressed (level 3). Schemas are defined in
`src/scanner.rs` (`mentions_schema`, `articles_schema`, `notices_schema`).

## mentions

One row per (term, section kind, symbol, sentence) in an article.

| Column | Type | Notes |
|---|---|---|
| `term_id` | int64 | From the terms file. A pattern shared by several terms gives one row per term. |
| `pmcid` | dictionary string | e.g. `PMC1234567` |
| `section` | dictionary string | `title`, `abstract`, `introduction`, `methods`, `results`, `discussion`, `conclusion`, `supplementary`, `figure`, `table`, `other` |
| `symbol` | string | The matched text as it appears in the article |
| `sentence` | string | The sentence containing the match, with trailing whitespace |
| `sentence_offset` | int32 | Character offset of `symbol` within `sentence` |

Polars reads the dictionary columns as `Categorical`.

## articles

One row per article that has **at least one mention**, sorted by `pmcid`.
Articles without mentions aren't written. They are still counted in the
manifest.

| Column | Type | Notes |
|---|---|---|
| `pmcid` | string | |
| `pmid`, `doi` | string, nullable | From `<article-id>` |
| `title` | string, nullable | |
| `abstract` | string, nullable | |
| `authors` | string, nullable | `Surname, Given; Surname, Given; …` for `contrib-type="author"` |
| `journal` | string, nullable | First `<journal-title>` |
| `year` | int32, nullable | Earliest `<pub-date>` year in `<article-meta>` |
| `article_type` | string, nullable | The `article-type` attribute |
| `rna_probability` | float64, nullable | Null if no `filter_model` was given or the article has no abstract |

## notices

Retraction, correction and expression-of-concern links, taken from
`<related-article>` elements in the front matter of **every** article, whether
or not it mentions a term. Sorted by `pmcid`.

| Column | Type | Notes |
|---|---|---|
| `pmcid` | string | The article carrying the link |
| `article_type` | string, nullable | Its `article-type`, e.g. `retraction` |
| `relation` | string | One of `retracted-article`, `corrected-article`, `object-of-concern`, `retraction-forward`, `correction-forward`, `expression-of-concern`, `updated-article`, `addended-article` |
| `target_id_type` | string | `pmcid`, `pmid` or `doi` |
| `target_id` | string | The normalised ID of the other article |

Backward links (`retracted-article`, `corrected-article`, …) sit on the notice
and point at the original. Forward links (`retraction-forward`, …) sit on the
original and point at the notice.

## manifest

```json
{
  "shard": "mini_shard",
  "shard_sha256": "49b2ebce…",
  "terms_sha256": "afb34577…",
  "n_terms": 11101,
  "litscan_version": "0.1.0+d34752e",
  "articles_total": 4,
  "articles_scanned": 4,
  "articles_below_threshold": 0,
  "mentions": 4,
  "notices": 1,
  "started_at": "2026-11-07T09:54:55Z",
  "finished_at": "2026-11-07T09:54:55Z"
}
```

`shard_sha256`, `terms_sha256` and `litscan_version` together identify exactly
which input, vocabulary and code produced the outputs. `articles_total` equals
`articles_scanned` plus `articles_below_threshold`.

## paragraphs

Written by `extract_paragraphs` to a single file. It joins onto mentions by
`(pmcid, section, sentence)`.

| Column | Type |
|---|---|
| `pmcid` | dictionary string |
| `section` | dictionary string |
| `sentence` | string |
| `paragraph` | string |
