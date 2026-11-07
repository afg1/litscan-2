# RNA relevance filter

litscan-2 can score each article's abstract for how likely it is to be about RNA.
The score is used in one of two ways: it is recorded as
`articles.rna_probability`, or, when `min_probability` is set, articles below
the threshold are skipped entirely. The model is a scikit-learn text classifier
that was exported to plain arrays and re-implemented in Rust
(`src/classifier.rs`), so scoring runs inside the scan with no Python involved.

The model shipped in `tests/fixtures/` (`model.safetensors`,
`model_metadata.json`) is believed to be the production model.

## Where it sits in the scan

`search_article_to_batch` (`src/scanner.rs`) decides when to call it:

| `filter_model` | `min_probability` | Behaviour |
|---|---|---|
| unset | unset | No scoring. `rna_probability` is null. |
| set | unset | Scan every article. Articles **with mentions** are scored (the model only runs on rows that will be written). |
| set | set | Score every article first. If the score is below the threshold, skip the scan and count the article in `articles_below_threshold`. |
| unset | set | Error: `min_probability requires a filter_model` |

The model only ever sees the **abstract**. Articles with no abstract (or an
empty one) are never skipped, and get a null `rna_probability`.

## Model type

The arrays and settings describe this model:

1. **TF-IDF features.** A scikit-learn `TfidfVectorizer` (version 1.5.2) over
   word unigrams. It lower-cases the text, uses the default token pattern
   `(?u)\b\w\w+\b` (words of two or more characters), smooth IDF and L2
   normalisation, with no stop words and no accent stripping. The vocabulary
   has 33,946 terms. Words not in the vocabulary are ignored.
2. **Five linear models.** Each fold \(k\) has a weight vector and an intercept,
   which give a decision value
   \(d_k = b_k + \mathbf{w}_k \cdot \mathbf{x}\).
3. **Sigmoid (Platt) calibration for each fold.**
   \(p_k = 1 / (1 + e^{A_k d_k + B_k})\).
4. **Averaging.** \(P(\text{RNA}) = \frac{1}{5}\sum_k p_k\), and the label is
   \(P > 0.5\).

This is the shape scikit-learn's `CalibratedClassifierCV(method="sigmoid")`
produces around a linear estimator with five folds. That conclusion is drawn
from the exported arrays. The type of the underlying linear estimator isn't
recorded in the files.

## How it gets into Rust

The model is two files. Both paths are passed to
`get_filter_model(weights, metadata)`, a thin wrapper around
`TfidfClassifier(weights, metadata)`.

### `model.safetensors`: the numbers

A [safetensors](https://huggingface.co/docs/safetensors) file. Every tensor is
little-endian `F64`:

| Tensor | Shape | Meaning |
|---|---|---|
| `idf` | `[n_features]` | IDF weight for each vocabulary index |
| `coef` | `[n_folds, n_features]` | Row-major: fold \(k\)'s weights start at offset \(k \times n_{features}\) |
| `intercept` | `[n_folds]` | \(b_k\) |
| `sigmoid_a` | `[n_folds]` | \(A_k\) |
| `sigmoid_b` | `[n_folds]` | \(B_k\) |

The shipped model is `coef [5, 33946]`, `idf [33946]`, and `[5]` for the others.
The loader reads raw bytes. It doesn't check dtypes or shapes.

### `model_metadata.json`: the configuration

```json
{
  "sklearn_version": "1.5.2",
  "n_folds": 5,
  "n_features": 33946,
  "classes": [0, 1],
  "vocabulary": { "rna": 27192, "mirna": 19943, "…": "…" },
  "tfidf_config": {
    "lowercase": true,
    "strip_accents": null,
    "token_pattern": "(?u)\\b\\w\\w+\\b",
    "ngram_range": [1, 1],
    "analyzer": "word",
    "stop_words": null,
    "sublinear_tf": false,
    "norm": "l2",
    "smooth_idf": true
  }
}
```

`vocabulary` maps each token to its column index. In the Rust code
(`Tokenizer`, `Vectorizer`):

- Python's `\w` includes Unicode category `No` (superscripts, fractions), and
  Rust's `regex` crate doesn't. When `token_pattern` is exactly the scikit-learn
  default, it is replaced with the equivalent explicit class
  `[\p{L}\p{N}\p{M}\p{Pc}]{2,}` so tokens match Python's. Any other pattern is
  used as written.
- The text is lower-cased with Rust's `to_lowercase`, then term frequency is
  multiplied by IDF and the vector is L2-normalised.

!!! note "Not yet documented"
    The script that exports a fitted scikit-learn model to these two files
    isn't in this repo.

### Parity tests

`tests/fixtures/test_fixtures.json` holds 100 abstracts, each with the outputs
scikit-learn produced for it: tokens, the sparse TF-IDF vector, per-fold
decision values, per-fold probabilities, `predict_proba` and the label. The
tests `tokenizer_parity.rs`, `vectorizer_parity.rs` and `end_to_end_parity.rs`
check that the Rust implementation reproduces each stage. The end-to-end test
requires agreement to within 1e-12. A newly exported model needs new fixtures
from the same scikit-learn version.

## Python API

```python
model = litscan_2.get_filter_model("model.safetensors", "model_metadata.json")
r = model.py_classify("The long non-coding RNA HOTAIR regulates chromatin state.")
```

`py_classify` returns a `ClassificationResult` with these attributes:

| Attribute | Value in the example | Meaning |
|---|---|---|
| `proba` | `[0.00378, 0.99622]` | `[P(not RNA), P(RNA)]`. `rna_probability` is `proba[1]`. |
| `label` | `1` | `1` if `proba[1] > 0.5` |
| `fold_prob` | `[0.9991, 0.9990, 0.9996, 0.9846, 0.9988]` | Calibrated probability for each fold |
| `fold_decisions` | | Raw decision value \(d_k\) for each fold |

Loading the model and building the scanner are independent of each other. One
model instance can be shared by every scan in a process.

## Known limitations

These come from reading the code. They don't affect the shipped model, but they
would affect a different one.

- **The fold count is fixed at 5.** The per-fold probability buffer is a
  hard-coded 5-element vector (`TfidfClassifier::classify`). A model with more
  than 5 folds panics. One with fewer gives correct probabilities, but
  `fold_prob` has trailing zeros.
- **Most `tfidf_config` options are read but not applied.** Only `lowercase`
  and `token_pattern` are used. `ngram_range`, `analyzer`, `sublinear_tf`,
  `strip_accents`, `stop_words` and `norm` are parsed, but the Rust code always
  does word unigrams with raw term counts and L2 normalisation. A model with
  bigrams, `sublinear_tf=True` or accent stripping would load without error and
  give wrong scores. (`smooth_idf` is fine, because it is already baked into
  the exported `idf` values.) The `UnsupportedConfig` and `TensorShape` error
  variants exist for these checks but aren't raised anywhere yet.
- **The method name.** It is exposed to Python as `py_classify`, not
  `classify`.
- **Only binary models.** `proba` is always `[1 − p, p]`, and the second class
  is assumed to be the positive one.

## Toward a swappable filter

Today the filter is specifically this TF-IDF model, and the output column is
named after RNA. To scan for other topics, these are the places that hard-code
it:

| Where | What is fixed |
|---|---|
| `scanner::search_article_to_batch`, `scan_shard_inner`, `scan_shard`, `scan_shards` | They take `Option<&TfidfClassifier>`, a concrete type. |
| `scanner::articles_schema` and `ArticleRow` | The score column is called `rna_probability`. |
| `search_article_to_batch` | The only input is the abstract. |
| `ClassificationResult` | It assumes a binary model with the positive class at index 1. |

What the scanner actually needs from a filter is small: a thread-safe function
from abstract text to a probability in [0, 1]. That's the interface a pluggable
filter would implement. The limitations above are the first things to fix for
any other TF-IDF model.
