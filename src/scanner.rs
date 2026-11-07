use aho_corasick::{AhoCorasickBuilder, AhoCorasick, AhoCorasickKind, Anchored, BuildError, Input, Match, MatchKind, StartKind};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Builder, Int64Builder, RecordBatch, StringBuilder,
    StringDictionaryBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::jats::{JatsArticle, SectionKind};
use crate::classifier::TfidfClassifier;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to build AC automaton: {0}")]
    BuildError(#[from] BuildError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("terms file line {line}: {msg}")]
    Parse { line: usize, msg: String },

    #[error("{0}")]
    Other(String),
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Compiled automaton over every search pattern plus a CSR map from
/// automaton pattern id to the term_ids it reports.
///
/// Normally each pattern carries exactly one term_id. More than one happens
/// only when distinct input terms become the same automaton pattern
/// (NFC-equal, equal under ASCII case folding, or via a Unicode case
/// variant such as ß → SS). Each of those terms really was mentioned, so all
/// of them are reported rather than silently keeping one.
#[pyclass]
pub struct PatternIndex {
    pub automaton: AhoCorasick,
    pub term_offsets: Vec<u32>,
    pub term_ids: Vec<i64>,
    #[pyo3(get)]
    pub n_patterns: u32,
    /// Distinct term_ids in the input file.
    #[pyo3(get)]
    pub n_terms: u64,
    /// Patterns that report more than one term_id.
    #[pyo3(get)]
    pub n_shared_patterns: u64,
    #[pyo3(get)]
    pub terms_sha256: String,
}

#[pyclass]
#[derive(Debug)]
pub struct Hit {
    #[pyo3(get)]
    pub term_id: i64,
    #[pyo3(get)]
    pub section: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub sentence: String,
    #[pyo3(get)]
    pub sentence_offset: i32,
}

impl PatternIndex {
    /// Find the PatternID for a given pattern string by self-searching.
    /// Intended for tests and debugging — too slow for production lookups.
    pub fn pattern_id_for(&self, pattern: &str) -> Option<usize> {
        let pattern: String = pattern.nfc().collect();
        self.automaton.find(pattern.as_str()).and_then(|m| {
            // Confirm it's an exact match, not just a prefix/suffix.
            (m.start() == 0 && m.end() == pattern.len())
                .then(|| m.pattern().as_usize())
        })
    }

    pub fn term_ids_for(&self, pid: usize) -> &[i64] {
        &self.term_ids[self.term_offsets[pid] as usize..self.term_offsets[pid + 1] as usize]
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{:02x}", b)).collect()
}

/// Automaton keys for one input pattern: the ASCII-lowercased NFC form (with
/// ascii_case_insensitive that is exactly what the automaton matches, so the
/// key is used as the pattern), plus Unicode case variants for non-ASCII
/// patterns, since aho-corasick only folds ASCII case.
fn pattern_keys(pattern: &str) -> Vec<String> {
    if pattern.is_ascii() {
        // ASCII is already NFC; this is the common case (all 23.9M LS1 terms).
        return vec![pattern.to_ascii_lowercase()];
    }
    let nfc: String = pattern.nfc().collect();
    let mut keys = vec![
        nfc.to_ascii_lowercase(),
        nfc.to_lowercase().to_ascii_lowercase(),
        nfc.to_uppercase().to_ascii_lowercase(),
    ];
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Build the automaton from a `term_id<TAB>pattern` file (UTF-8, no header).
///
/// Line parsing and normalisation run in parallel on the global rayon pool,
/// and keys are grouped with a parallel sort instead of a hash map. The
/// aho-corasick construction itself is single-threaded.
pub fn build_search_automaton(terms_path: PathBuf) -> Result<PatternIndex, Error> {
    use rayon::prelude::*;

    let bytes = std::fs::read(&terms_path)?;
    let (terms_sha256, text) = rayon::join(
        || sha256_hex(&bytes),
        || std::str::from_utf8(&bytes),
    );
    let text = text.map_err(|e| Error::Other(format!("{} is not UTF-8: {}", terms_path.display(), e)))?;

    let parsed: Vec<(i64, Vec<String>)> = text
        .par_lines()
        .filter_map(|raw| {
            let line = raw.trim_end_matches('\r');
            (!line.is_empty()).then_some((raw, line))
        })
        .map(|(raw, line)| {
            let parse_err = |msg: String| {
                // Line numbers are only needed on error: count newlines before it.
                let offset = raw.as_ptr() as usize - text.as_ptr() as usize;
                Error::Parse { line: text[..offset].bytes().filter(|&b| b == b'\n').count() + 1, msg }
            };
            let (id, pattern) = line
                .split_once('\t')
                .ok_or_else(|| parse_err("expected term_id<TAB>pattern".into()))?;
            let term_id: i64 = id
                .trim()
                .parse()
                .map_err(|e| parse_err(format!("bad term_id {:?}: {}", id, e)))?;
            if pattern.trim().is_empty() {
                return Err(parse_err("empty pattern".into()));
            }
            Ok((term_id, pattern_keys(pattern)))
        })
        .collect::<Result<_, Error>>()?;

    let mut distinct: Vec<i64> = parsed.par_iter().map(|(id, _)| *id).collect();
    distinct.par_sort_unstable();
    distinct.dedup();

    let mut entries: Vec<(String, i64)> = parsed
        .into_par_iter()
        .flat_map_iter(|(id, keys)| keys.into_iter().map(move |k| (k, id)))
        .collect();
    // Sorting by pattern string makes PatternIDs stable across runs. With
    // LeftmostLongest the tiebreak between equal-length patterns is PatternID,
    // so a non-deterministic order would give non-deterministic hits.
    entries.par_sort_unstable();

    let mut term_offsets: Vec<u32> = Vec::with_capacity(entries.len() + 1);
    term_offsets.push(0);
    let mut term_ids: Vec<i64> = Vec::with_capacity(entries.len());
    let mut patterns: Vec<String> = Vec::with_capacity(entries.len());
    let mut n_shared_patterns = 0u64;
    for (key, id) in entries {
        if patterns.last() == Some(&key) {
            // Same pattern: entries are sorted by (key, id), so dedup is a compare with the last id.
            if *term_ids.last().unwrap() != id {
                if term_ids.len() - *term_offsets.last().unwrap() as usize == 1 {
                    n_shared_patterns += 1;
                }
                term_ids.push(id);
            }
            continue;
        }
        if !patterns.is_empty() {
            term_offsets.push(term_ids.len() as u32);
        }
        patterns.push(key);
        term_ids.push(id);
    }
    if !patterns.is_empty() {
        term_offsets.push(term_ids.len() as u32);
    }
    if n_shared_patterns > 0 {
        eprintln!(
            "litscan-2: {} patterns are shared by more than one term_id; each hit reports all of them",
            n_shared_patterns
        );
    }

    let automaton = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .ascii_case_insensitive(true)
        .kind(Some(AhoCorasickKind::ContiguousNFA))
        // Anchored searches let find_mentions retry shorter patterns at the
        // start of a match that failed the word-boundary check.
        .start_kind(StartKind::Both)
        .build(&patterns)?;

    Ok(PatternIndex {
        automaton,
        term_offsets,
        term_ids,
        n_patterns: patterns.len() as u32,
        n_terms: distinct.len() as u64,
        n_shared_patterns,
        terms_sha256,
    })
}

#[inline]
fn is_word_boundary_match(haystack: &str, start: usize, end: usize) -> bool {
    let left_ok = haystack[..start].chars().next_back().map_or(true, |c| !c.is_alphanumeric());
    let right_ok = haystack[end..].chars().next().map_or(true, |c| !c.is_alphanumeric());
    left_ok && right_ok
}

fn segment_text(text: &str) -> Vec<(usize, usize)> {
    text.split_sentence_bound_indices()
    .map(|(start, word)| (start, start+word.len()))
    .collect()
}

/// One term mention, borrowing from the article.
pub struct Mention<'a> {
    pub term_id: i64,
    pub section: SectionKind,
    pub symbol: &'a str,
    pub sentence: &'a str,
    /// Character (Unicode scalar) offset of `symbol` within `sentence`.
    pub sentence_offset: i32,
}

/// Leftmost-longest matches that pass the word-boundary check.
///
/// A plain non-overlapping iterator plus a boundary filter loses real
/// matches: a longer pattern that starts inside the previous word (or runs
/// past the end of the term) wins, is rejected, and the scan resumes after
/// it. Instead, a rejected match with a good left boundary retries the
/// longest shorter pattern at the same start, and if none fits the search
/// resumes one character after the rejected start.
fn boundary_matches<'h>(ac: &'h AhoCorasick, haystack: &'h str) -> impl Iterator<Item = Match> + 'h {
    let len = haystack.len();
    let mut pos = 0;
    std::iter::from_fn(move || {
        while pos < len {
            let first = ac.find(Input::new(haystack).span(pos..len))?;
            let left_ok = haystack[..first.start()].chars().next_back().map_or(true, |c| !c.is_alphanumeric());
            let mut cand = left_ok.then_some(first);
            while let Some(c) = cand {
                if is_word_boundary_match(haystack, c.start(), c.end()) {
                    pos = c.end();
                    return Some(c);
                }
                cand = (c.end() - c.start() > 1)
                    .then(|| ac.find(Input::new(haystack).span(c.start()..c.end() - 1).anchored(Anchored::Yes)))
                    .flatten();
            }
            pos = first.start() + haystack[first.start()..].chars().next().map_or(1, char::len_utf8);
        }
        None
    })
}

/// Find every term mention in an article, deduplicated to one per distinct
/// (section kind, symbol, sentence). Articles often have several sections of
/// one kind (e.g. many `other` sections or tables) repeating a sentence.
pub fn find_mentions<'a>(searcher: &PatternIndex, article: &'a JatsArticle) -> Vec<Mention<'a>> {
    let mut out = Vec::new();
    let mut seen: HashSet<(SectionKind, &str, &str)> = HashSet::new();
    for section in &article.sections {
        let haystack = section.text.as_str();
        let mut bounds: Option<Vec<(usize, usize)>> = None;

        for mat in boundary_matches(&searcher.automaton, haystack) {
            // Segments tile the whole text. The sentence is the segment holding
            // the match start, widened to the segment holding its end if the
            // match crosses a boundary, so `symbol` is always inside `sentence`.
            let bounds = bounds.get_or_insert_with(|| segment_text(haystack));
            let first = bounds.partition_point(|&(_, e)| e <= mat.start());
            let last = bounds.partition_point(|&(_, e)| e < mat.end()).min(bounds.len() - 1);
            let (sent_start, sent_end) = (bounds[first].0, bounds[last].1);
            let sentence = &haystack[sent_start..sent_end];
            let symbol = &haystack[mat.start()..mat.end()];

            if !seen.insert((section.kind, symbol, sentence)) {
                continue;
            }
            let sentence_offset = haystack[sent_start..mat.start()].chars().count() as i32;
            for &term_id in searcher.term_ids_for(mat.pattern().as_usize()) {
                out.push(Mention { term_id, section: section.kind, symbol, sentence, sentence_offset });
            }
        }
    }
    out
}

/// Search a single article with the compiled automaton, returns a Vec of Hit objects to parse
pub fn search_article(searcher: &PatternIndex, article: &JatsArticle) -> Result<Vec<Hit>, Error> {
    Ok(find_mentions(searcher, article)
        .into_iter()
        .map(|m| Hit {
            term_id: m.term_id,
            section: m.section.as_str().to_string(),
            symbol: m.symbol.to_string(),
            sentence: m.sentence.to_string(),
            sentence_offset: m.sentence_offset,
        })
        .collect())
}

pub fn mentions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("term_id", DataType::Int64, false),
        Field::new_dictionary("pmcid", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("section", DataType::UInt32, DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sentence", DataType::Utf8, false),
        Field::new("sentence_offset", DataType::Int32, false),
    ]))
}

pub fn articles_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("pmcid", DataType::Utf8, false),
        Field::new("pmid", DataType::Utf8, true),
        Field::new("doi", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("authors", DataType::Utf8, true),
        Field::new("journal", DataType::Utf8, true),
        Field::new("year", DataType::Int32, true),
        Field::new("article_type", DataType::Utf8, true),
        Field::new("rna_probability", DataType::Float64, true),
    ]))
}

pub fn notices_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("pmcid", DataType::Utf8, false),
        Field::new("article_type", DataType::Utf8, true),
        Field::new("relation", DataType::Utf8, false),
        Field::new("target_id_type", DataType::Utf8, false),
        Field::new("target_id", DataType::Utf8, false),
    ]))
}

/// One `notices/<shard>.parquet` row.
pub struct NoticeRow {
    pub pmcid: String,
    pub article_type: Option<String>,
    pub notice: crate::jats::Notice,
}

pub fn notices_to_batch(rows: &[NoticeRow]) -> Result<RecordBatch, Error> {
    let col = |f: &dyn Fn(&NoticeRow) -> Option<&str>| -> ArrayRef {
        let mut b = StringBuilder::new();
        for r in rows {
            b.append_option(f(r));
        }
        Arc::new(b.finish())
    };
    let arrays: Vec<ArrayRef> = vec![
        col(&|r| Some(r.pmcid.as_str())),
        col(&|r| r.article_type.as_deref()),
        col(&|r| Some(r.notice.relation)),
        col(&|r| Some(r.notice.target_id_type)),
        col(&|r| Some(r.notice.target_id.as_str())),
    ];
    Ok(RecordBatch::try_new(notices_schema(), arrays)?)
}

/// One `articles/<shard>.parquet` row.
pub struct ArticleRow {
    pub pmcid: String,
    pub pmid: Option<String>,
    pub doi: Option<String>,
    pub title: Option<String>,
    pub abstract_text: Option<String>,
    pub authors: Option<String>,
    pub journal: Option<String>,
    pub year: Option<i32>,
    pub article_type: Option<String>,
    pub rna_probability: Option<f64>,
}

pub fn articles_to_batch(rows: &[ArticleRow]) -> Result<RecordBatch, Error> {
    let s = |f: fn(&ArticleRow) -> Option<&str>| -> ArrayRef {
        let mut b = StringBuilder::new();
        for r in rows {
            b.append_option(f(r));
        }
        Arc::new(b.finish())
    };
    let mut year = Int32Builder::new();
    let mut proba = Float64Builder::new();
    for r in rows {
        year.append_option(r.year);
        proba.append_option(r.rna_probability);
    }
    let arrays: Vec<ArrayRef> = vec![
        s(|r| Some(r.pmcid.as_str())),
        s(|r| r.pmid.as_deref()),
        s(|r| r.doi.as_deref()),
        s(|r| r.title.as_deref()),
        s(|r| r.abstract_text.as_deref()),
        s(|r| r.authors.as_deref()),
        s(|r| r.journal.as_deref()),
        Arc::new(year.finish()),
        s(|r| r.article_type.as_deref()),
        Arc::new(proba.finish()),
    ];
    Ok(RecordBatch::try_new(articles_schema(), arrays)?)
}

pub enum ScanOutcome {
    /// `min_probability` was set and the abstract scored below it.
    BelowThreshold,
    /// Scanned; `Some` only when the article had at least one mention.
    Scanned(Option<(RecordBatch, ArticleRow)>),
}

/// Scan one article into a mentions `RecordBatch` plus its articles row.
///
/// The classifier only reports `rna_probability` unless `min_probability` is
/// set, in which case articles scoring below it are skipped. Articles without
/// an abstract are never skipped and get a NULL probability.
pub fn search_article_to_batch(
    searcher: &PatternIndex,
    article: &JatsArticle,
    pmcid: &str,
    schema: &Arc<Schema>,
    classifier: Option<&TfidfClassifier>,
    min_probability: Option<f64>,
) -> Result<ScanOutcome, Error> {
    let abstract_text = article.abstract_text.as_deref().filter(|a| !a.is_empty());
    let probability = || -> Option<f64> {
        let (c, a) = (classifier?, abstract_text?);
        Some(c.classify(a).proba[1])
    };

    let mut rna_probability = None;
    if let Some(threshold) = min_probability {
        rna_probability = probability();
        if rna_probability.map_or(false, |p| p < threshold) {
            return Ok(ScanOutcome::BelowThreshold);
        }
    }

    let mentions = find_mentions(searcher, article);
    if mentions.is_empty() {
        return Ok(ScanOutcome::Scanned(None));
    }
    if min_probability.is_none() {
        // Only pay for the classifier on articles that will be written.
        rna_probability = probability();
    }

    let mut term_b = Int64Builder::with_capacity(mentions.len());
    let mut pmcid_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut section_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut symbol_b = StringBuilder::new();
    let mut sentence_b = StringBuilder::new();
    let mut offset_b = Int32Builder::with_capacity(mentions.len());
    for m in &mentions {
        term_b.append_value(m.term_id);
        pmcid_b.append_value(pmcid);
        section_b.append_value(m.section.as_str());
        symbol_b.append_value(m.symbol);
        sentence_b.append_value(m.sentence);
        offset_b.append_value(m.sentence_offset);
    }
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(term_b.finish()),
        Arc::new(pmcid_b.finish()),
        Arc::new(section_b.finish()),
        Arc::new(symbol_b.finish()),
        Arc::new(sentence_b.finish()),
        Arc::new(offset_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let row = ArticleRow {
        pmcid: pmcid.to_string(),
        pmid: article.pmid.clone(),
        doi: article.doi.clone(),
        title: article.title.clone(),
        abstract_text: article.abstract_text.clone(),
        authors: article.authors_string(),
        journal: article.journal.clone(),
        year: article.year,
        article_type: article.article_type.clone(),
        rna_probability,
    };
    Ok(ScanOutcome::Scanned(Some((batch, row))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jats;
    use std::io::Write;

    fn index(terms: &str) -> PatternIndex {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("litscan2_terms_{}_{}.tsv", std::process::id(), n));
        std::fs::File::create(&path).unwrap().write_all(terms.as_bytes()).unwrap();
        let idx = build_search_automaton(path.clone()).unwrap();
        std::fs::remove_file(path).ok();
        idx
    }

    fn article(body: &str) -> JatsArticle {
        jats::parse(&format!(
            r#"<article><front><article-meta><title-group><article-title>T</article-title></title-group></article-meta></front>
<body><sec sec-type="results"><p>{}</p></sec></body></article>"#,
            body
        ))
        .unwrap()
    }

    fn hits(idx: &PatternIndex, a: &JatsArticle) -> Vec<(i64, String, String)> {
        find_mentions(idx, a)
            .into_iter()
            .map(|m| (m.term_id, m.section.as_str().to_string(), m.symbol.to_string()))
            .collect()
    }

    /// The Postgres invariant: substr(sentence, offset + 1, char_length(symbol)) = symbol.
    fn assert_offset_invariant(m: &Mention) {
        let got: String = m
            .sentence
            .chars()
            .skip(m.sentence_offset as usize)
            .take(m.symbol.chars().count())
            .collect();
        assert_eq!(got, m.symbol);
    }

    #[test]
    fn emits_term_ids() {
        let idx = index("42\tMIR21\n7\tXist\n");
        let a = article("We studied mir21 and XIST.");
        let mut h = hits(&idx, &a);
        h.sort();
        assert_eq!(h, vec![(7, "results".into(), "XIST".into()), (42, "results".into(), "mir21".into())]);
    }

    #[test]
    fn unicode_word_boundary() {
        let idx = index("1\tMIR21\n");
        assert!(hits(&idx, &article("αMIR21 is not a word match.")).is_empty());
        assert!(hits(&idx, &article("MIR21α is not a word match.")).is_empty());
        assert_eq!(hits(&idx, &article("α MIR21 is.")).len(), 1);
    }

    #[test]
    fn unicode_case_variants() {
        let idx = index("5\tΔ-miR\n");
        assert_eq!(hits(&idx, &article("The δ-MIR locus.")), vec![(5, "results".into(), "δ-MIR".into())]);
        assert_eq!(hits(&idx, &article("The Δ-mir locus.")).len(), 1);
    }

    #[test]
    fn nfc_normalises_patterns() {
        // Decomposed e + combining acute in the pattern, precomposed é in the text.
        let idx = index("3\tcafe\u{301}1\n");
        assert_eq!(hits(&idx, &article("The caf\u{e9}1 gene.")).len(), 1);
    }

    #[test]
    fn colliding_terms_all_reported() {
        // Distinct terms that become one automaton pattern both get the mention.
        let idx = index("1\tMIR21\n2\tmir21\n3\tcaf\u{e9}\n4\tcafe\u{301}\n");
        assert_eq!(idx.n_terms, 4);
        // mir21, café, and the upper-case variant cafÉ (É is not ASCII-folded).
        assert_eq!(idx.n_patterns, 3);
        assert_eq!(idx.n_shared_patterns, 3);
        let mut ids: Vec<i64> = hits(&idx, &article("MIR21 in a caf\u{e9}.")).into_iter().map(|h| h.0).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn title_is_scanned() {
        let idx = index("9\tHOTAIR\n");
        let a = jats::parse(
            r#"<article><front><article-meta><title-group><article-title>HOTAIR in cancer</article-title></title-group></article-meta></front></article>"#,
        )
        .unwrap();
        assert_eq!(hits(&idx, &a), vec![(9, "title".into(), "HOTAIR".into())]);
    }

    #[test]
    fn sentence_offset_is_in_chars() {
        let idx = index("1\tMIR21\n");
        let a = article("First sentence. Ünïcödé βeta before MIR21 here.");
        let m = find_mentions(&idx, &a);
        let m: Vec<_> = m.iter().filter(|m| m.section == SectionKind::Results).collect();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].sentence, "Ünïcödé βeta before MIR21 here.");
        assert_eq!(m[0].sentence_offset, 20);
        assert_offset_invariant(m[0]);
    }

    #[test]
    fn match_across_sentence_boundary_keeps_invariant() {
        // "e. coli" as a pattern spans a segment boundary after "e.".
        let idx = index("1\tE. coli\n");
        let a = article("Grown in E. coli cells.");
        for m in find_mentions(&idx, &a) {
            assert!(!m.sentence.is_empty());
            assert_offset_invariant(&m);
        }
    }

    #[test]
    fn rejected_overlapping_match_does_not_hide_term() {
        // "er of SSU" starts inside "number" and is rejected; "SSU rRNA" under it must survive.
        let idx = index("1\tSSU rRNA\n2\ter of SSU\n");
        assert_eq!(hits(&idx, &article("The number of SSU rRNA copies.")), vec![(1, "results".into(), "SSU rRNA".into())]);
        // The longest pattern at a start runs into the next word; a shorter one fits.
        let idx = index("1\tSSU rRNA\n2\tSSU rRNA gene\n");
        assert_eq!(hits(&idx, &article("Two SSU rRNA genes.")), vec![(1, "results".into(), "SSU rRNA".into())]);
        // A match never starts mid-character after a rejection.
        let idx = index("1\tMIR21\n2\tβMIR\n");
        assert_eq!(hits(&idx, &article("αβMIR21 and MIR21.")), vec![(1, "results".into(), "MIR21".into())]);
    }

    #[test]
    fn dedup_across_sections_of_same_kind() {
        let idx = index("1\tMIR21\n");
        let a = jats::parse(concat!(
            "<article><body><sec><title>Notes</title><p>MIR21 again.</p></sec>",
            "<sec><title>More notes</title><p>MIR21 again.</p></sec>",
            r#"<sec sec-type="results"><p>MIR21 again.</p></sec></body></article>"#,
        ))
        .unwrap();
        let mut got: Vec<String> = find_mentions(&idx, &a).iter().map(|m| m.section.as_str().to_string()).collect();
        got.sort();
        assert_eq!(got, vec!["other", "results"]);
    }

    #[test]
    fn dedup_per_sentence() {
        let idx = index("1\tMIR21\n");
        let a = article("MIR21 and MIR21 again. MIR21 once more.");
        assert_eq!(find_mentions(&idx, &a).len(), 2);
    }

    #[test]
    fn bad_line_errors() {
        let path = std::env::temp_dir().join(format!("litscan2_bad_{}.tsv", std::process::id()));
        std::fs::write(&path, "1\tok\nnot-a-number\tx\n").unwrap();
        let err = build_search_automaton(path.clone()).err().unwrap();
        std::fs::remove_file(path).ok();
        assert!(matches!(err, Error::Parse { line: 2, .. }), "{err}");
    }
}
