use aho_corasick::{AhoCorasickBuilder, AhoCorasick, AhoCorasickKind, MatchKind, PatternID, BuildError};
use unicode_segmentation::UnicodeSegmentation;
use std::collections::HashMap;
use std::path::PathBuf;
use std::io::{self, BufRead};
use std::fs::File;
use thiserror::Error;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::jats::JatsArticle;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to build AC automaton: {0}")]
    BuildError(#[from] BuildError)
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

    for (pattern, mut urs_list) in pattern_to_urs {
        urs_list.sort_unstable();
        urs_list.dedup();
        urs_indices.extend_from_slice(&urs_list);
        urs_offsets.push(urs_indices.len() as u32);
        pattern_strings.push(pattern)
    }
    let automaton = AhoCorasickBuilder::new()
    .match_kind(MatchKind::LeftmostLongest)
    .ascii_case_insensitive(true)
    .kind(Some(AhoCorasickKind::DFA))
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

pub fn search_article(searcher: &PatternIndex, article: &JatsArticle) -> Result<Vec<Hit>, Error> {
    
    let mut hits = vec![];
    for section in &article.sections {
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
            let mut sentence: String = "".to_string();
            if let Ok(idx) = hit_sentence_bounds_idx {
                let (start, end) = sentence_bounds[idx];
                if mat.end() <= end {
                    sentence = haystack[start..end].to_string();
                }
            }


            let symbol = &haystack[mat.start()..mat.end()];
            let pid = mat.pattern().as_usize();
            let urs_range = searcher.metadata.urs_offsets[pid] as usize
                            .. searcher.metadata.urs_offsets[pid + 1] as usize;
                for &idx in &searcher.metadata.urs_indices[urs_range] {
                    hits.push(Hit {
                        urs_taxid: searcher.metadata.urs_table[idx as usize].clone(),
                        symbol: symbol.to_string(),
                        section: section.kind.as_str().to_string(),
                        sentence: sentence.clone(),
                        offset: mat.start() as u32,
                    });
                }
            }
    }

    Ok(hits)
}