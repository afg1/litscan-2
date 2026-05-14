use aho_corasick::{AhoCorasickBuilder, AhoCorasick, AhoCorasickKind, MatchKind, BuildError};
use unicode_segmentation::UnicodeSegmentation;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::io::{self, BufRead};
use std::fs::File;
use std::sync::Arc;
use thiserror::Error;

use arrow::array::{ArrayRef, RecordBatch, StringDictionaryBuilder, UInt32Builder};
use arrow::datatypes::{Schema, UInt32Type};

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::jats::JatsArticle;
use crate::classifier::TfidfClassifier;
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to build AC automaton: {0}")]
    BuildError(#[from] BuildError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[pyclass]
pub struct PatternIndex {
    pub automaton: AhoCorasick,
    #[pyo3(get)]
    pub metadata: PatternMetadata
}

#[pyclass]
#[derive(Clone)]
pub struct PatternMetadata {
    pub urs_offsets: Vec<u32>,
    pub urs_indices: Vec<u32>,
    pub urs_table: Vec<String>,
    #[pyo3(get)]
    pub n_patterns: u32,
}

#[pyclass]
#[derive(Debug)]
pub struct Hit {
    #[pyo3(get)]
    pub urs_taxid: String,
    #[pyo3(get)]
    pub section: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub sentence: String,
    #[pyo3(get)]
    pub offset: u32

}


impl PatternIndex {
    /// Find the PatternID for a given pattern string by self-searching.
    /// Intended for tests and debugging — too slow for production lookups.
    pub fn pattern_id_for(&self, pattern: &str) -> Option<usize> {
        self.automaton.find(pattern).and_then(|m| {
            // Confirm it's an exact match, not just a prefix/suffix.
            (m.start() == 0 && m.end() == pattern.len())
                .then(|| m.pattern().as_usize())
        })
    }
}


fn read_lines(filename: PathBuf) -> io::Result<io::Lines<io::BufReader<File>>> {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn build_search_automaton(id_list: PathBuf) -> Result<PatternIndex, Error>  {

    let mut urs_offsets: Vec<u32> = vec![0u32];
    let mut urs_indices: Vec<u32> = Vec::new();
    let mut urs_interner: HashMap<String, u32> = HashMap::new();
    let mut urs_table: Vec<String> = Vec::new();
    let mut pattern_to_urs: HashMap<String, Vec<u32>> = HashMap::new();

    if let Ok(lines) = read_lines(id_list) {
        for line in lines.map_while(Result::ok) {
            if let Some((urs_taxid, id)) = line.split_once('\t')
            {
                let urs_idx = *urs_interner.entry(urs_taxid.to_owned()).or_insert_with(|| {
                    urs_table.push(urs_taxid.to_owned());
                    (urs_table.len() - 1) as u32
                });
                pattern_to_urs.entry(id.to_owned()).or_default().push(urs_idx);
            }
            else {
                
            }
        }
    }

    let mut pattern_strings: Vec<String> = Vec::with_capacity(pattern_to_urs.len());

    // Drain into a Vec and sort by pattern string so PatternIDs are stable across
    // runs. HashMap iteration order is randomized, and with LeftmostLongest the
    // tiebreak between equal-length patterns is PatternID — so a non-deterministic
    // ordering produces non-deterministic hit counts.
    let mut pattern_entries: Vec<(String, Vec<u32>)> = pattern_to_urs.into_iter().collect();
    pattern_entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for (pattern, mut urs_list) in pattern_entries {
        urs_list.sort_unstable();
        urs_list.dedup();
        urs_indices.extend_from_slice(&urs_list);
        urs_offsets.push(urs_indices.len() as u32);
        pattern_strings.push(pattern)
    }
    let automaton = AhoCorasickBuilder::new()
    .match_kind(MatchKind::LeftmostLongest)
    .ascii_case_insensitive(true)
    .kind(Some(AhoCorasickKind::ContiguousNFA))
    .build(&pattern_strings)?;

    Ok(
        PatternIndex { 
            automaton: automaton,
            metadata: PatternMetadata { 
                urs_offsets: urs_offsets, 
                urs_indices: urs_indices, 
                urs_table: urs_table,
                n_patterns: pattern_strings.len() as u32
            } 
        }
    )


}

#[inline]
fn is_word_boundary_match(haystack: &str, start: usize, end: usize) -> bool {
    let bytes = haystack.as_bytes();
    let left_ok  = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
    let right_ok = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
    left_ok && right_ok
}

fn segment_text(text: &str) -> Vec<(usize, usize)> {
    text.split_sentence_bound_indices()
    .map(|(start, word)| (start, start+word.len()))
    .collect()
}

/// Search a single article with the compiled automaton, returns a Vec of Hit objects to parse
pub fn search_article(searcher: &PatternIndex, article: &JatsArticle) -> Result<Vec<Hit>, Error> {
    let mut hits = vec![];
    for section in &article.sections {
        let mut seen_sentences: HashMap<&str, HashSet<&str> > = HashMap::new();
        let haystack= &section.text;
        let sentence_bounds = segment_text(&haystack);
        for mat in searcher.automaton.find_iter(haystack) {
            if !is_word_boundary_match(haystack, mat.start(), mat.end()) {
                continue;
            }

            let hit_sentence_bounds_idx = sentence_bounds.binary_search_by(|&(sent_start, sent_end)| {
            if mat.start() >= sent_start && mat.start() < sent_end {
                std::cmp::Ordering::Equal
            } else if mat.start() < sent_start {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
            });
            let mut sentence: &str = "";
            if let Ok(idx) = hit_sentence_bounds_idx {
                let (start, end) = sentence_bounds[idx];
                if mat.end() <= end {
                    sentence = &haystack[start..end];
                }
            }
            let symbol = &haystack[mat.start()..mat.end()];
            // only insert any given sentence once
            if !seen_sentences.entry(symbol).or_default().insert(sentence) {
                continue;
            }


            
            let pid = mat.pattern().as_usize();
            let urs_range = searcher.metadata.urs_offsets[pid] as usize
                            .. searcher.metadata.urs_offsets[pid + 1] as usize;
                for &idx in &searcher.metadata.urs_indices[urs_range] {
                    hits.push(Hit {
                        urs_taxid: searcher.metadata.urs_table[idx as usize].clone(),
                        symbol: symbol.to_string(),
                        section: section.kind.as_str().to_string(),
                        sentence: sentence.to_string(),
                        offset: mat.start() as u32,
                    });
                }
            }
    }

    Ok(hits)
}

/// Scan an article and append rows directly into Arrow builders, returning a
/// completed `RecordBatch` (or `None` if there were no hits). Avoids the
/// per-urs `Vec<Hit>` allocation that duplicates `sentence` / `symbol` /
/// `urs_taxid` across the fanout.
pub enum ScanOutcome {
    Filtered,
    Scanned(Option<RecordBatch>),
}

pub fn search_article_to_batch(
    searcher: &PatternIndex,
    article: &JatsArticle,
    pmcid: &str,
    schema: &Arc<Schema>,
    filter: &Option<&TfidfClassifier>
) -> Result<ScanOutcome, Error> {

    // Filter based on abstract classification, if the model is passed
    if let Some(classifier) = filter {
        let abstract_text = article.abstract_text.as_deref().unwrap_or("");
        if !abstract_text.is_empty() && classifier.classify(abstract_text).label == 0u32 {
                return Ok(ScanOutcome::Filtered);
        }
    }

    let mut pmcid_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut urs_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut section_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut symbol_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut sentence_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut offset_b = UInt32Builder::new();

    let mut n_rows: usize = 0;

    for section in &article.sections {
        let mut seen_sentences: HashMap<&str, HashSet<&str>> = HashMap::new();
        let haystack = &section.text;
        let sentence_bounds = segment_text(haystack);
        let section_str = section.kind.as_str();

        for mat in searcher.automaton.find_iter(haystack) {
            if !is_word_boundary_match(haystack, mat.start(), mat.end()) {
                continue;
            }

            let hit_sentence_bounds_idx = sentence_bounds.binary_search_by(|&(sent_start, sent_end)| {
                if mat.start() >= sent_start && mat.start() < sent_end {
                    std::cmp::Ordering::Equal
                } else if mat.start() < sent_start {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            });

            let mut sentence: &str = "";
            if let Ok(idx) = hit_sentence_bounds_idx {
                let (start, end) = sentence_bounds[idx];
                if mat.end() <= end {
                    sentence = &haystack[start..end];
                }
            }
            let symbol: &str = &haystack[mat.start()..mat.end()];

            // Dedup: same (symbol, sentence) within a section only fires once.
            if !seen_sentences.entry(symbol).or_default().insert(sentence) {
                continue;
            }

            let pid = mat.pattern().as_usize();
            let urs_range = searcher.metadata.urs_offsets[pid] as usize
                .. searcher.metadata.urs_offsets[pid + 1] as usize;
            let offset = mat.start() as u32;

            for &idx in &searcher.metadata.urs_indices[urs_range] {
                pmcid_b.append_value(pmcid);
                urs_b.append_value(&searcher.metadata.urs_table[idx as usize]);
                section_b.append_value(section_str);
                symbol_b.append_value(symbol);
                sentence_b.append_value(sentence);
                offset_b.append_value(offset);
                n_rows += 1;
            }
        }
    }

    if n_rows == 0 {
        return Ok(ScanOutcome::Scanned(None));
    }

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(pmcid_b.finish()),
        Arc::new(urs_b.finish()),
        Arc::new(section_b.finish()),
        Arc::new(symbol_b.finish()),
        Arc::new(sentence_b.finish()),
        Arc::new(offset_b.finish()),
    ];

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    Ok(ScanOutcome::Scanned(Some(batch)))
}