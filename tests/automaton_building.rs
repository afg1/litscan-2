mod common;

use common::{test_ids};
use std::path::{Path, PathBuf};
use std::io::{self, BufRead};
use std::fs::File;

use litscan_2::scanner::{build_search_automaton};

fn read_lines(filename: PathBuf) -> io::Result<io::Lines<io::BufReader<File>>> {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn parse_test_data() -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Ok(lines) = read_lines(test_ids()) {
        for line in lines.map_while(Result::ok) {
            if let Some((urs_taxid, id)) = line.split_once('\t')
            {
                pairs.push((id.to_owned(), urs_taxid.to_owned()));
            }
            else {

            }
        }
    }
    pairs
}

#[test]
fn csr_invariants() {
    let pattern_scanner = build_search_automaton(test_ids()).expect("Failed to build pattern matcher");

    // 1. Structural invariants (these should hold for any valid CSR).
    assert_eq!(pattern_scanner.metadata.urs_offsets.len(), pattern_scanner.automaton.patterns_len() + 1);
    assert_eq!(pattern_scanner.metadata.urs_offsets[0], 0);
    assert_eq!(*pattern_scanner.metadata.urs_offsets.last().unwrap() as usize, pattern_scanner.metadata.urs_indices.len());
    for w in pattern_scanner.metadata.urs_offsets.windows(2) {
        assert!(w[0] <= w[1], "offsets must be monotonically non-decreasing");
    }
    for &i in &pattern_scanner.metadata.urs_indices {
        assert!((i as usize) < pattern_scanner.metadata.urs_table.len(), "indices must be in bounds");
    }
}

#[test]
fn round_trip_lookup() {
    use std::collections::{HashMap, HashSet};

    let idx = build_search_automaton(test_ids()).expect("Failed to build pattern matcher");
    let pairs: Vec<(String, String)> = parse_test_data();

    // Group expected URS values by pattern.
    let mut expected: HashMap<String, HashSet<String>> = HashMap::new();
    for (pattern, urs) in pairs {
        expected.entry(pattern).or_default().insert(urs);
    }

    // For each pattern, find its PatternID via self-search and verify
    // the URS values reachable through the CSR match what we expect.
    for (pattern, expected_urs) in expected {
        let pid = idx.pattern_id_for(&pattern)
            .unwrap_or_else(|| panic!("pattern {pattern:?} not found in automaton"));

        let range = idx.metadata.urs_offsets[pid] as usize..idx.metadata.urs_offsets[pid + 1] as usize;
        let actual_urs: HashSet<String> = idx.metadata.urs_indices[range]
            .iter()
            .map(|&i| idx.metadata.urs_table[i as usize].clone())
            .collect();

        assert_eq!(actual_urs, expected_urs,
                   "URS mismatch for pattern {pattern:?}");
    }

}
