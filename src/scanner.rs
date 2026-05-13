use aho_corasick::{AhoCorasickBuilder, AhoCorasick, AhoCorasickKind, MatchKind, PatternID, BuildError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::io::{self, BufRead};
use std::fs::File;
use thiserror::Error;

use serde::{Serialize, Deserialize};

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to build AC automaton: {0}")]
    BuildError(#[from] BuildError)
}

pub struct PatternIndex {
    pub automaton: AhoCorasick,
    pub metadata: PatternMetadata
}

pub struct PatternMetadata {
    pub urs_offsets: Vec<u32>,
    pub urs_indices: Vec<u32>,
    pub urs_table: Vec<String>,
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
    println!("{:?}", pattern_strings);
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
                urs_table: urs_table 
            } 
        }
    )


}