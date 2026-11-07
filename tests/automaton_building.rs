mod common;

use common::test_terms;

use litscan_2::scanner::build_search_automaton;

fn parse_test_data() -> Vec<(i64, String)> {
    std::fs::read_to_string(test_terms())
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(id, p)| (id.parse().unwrap(), p.to_owned()))
        .collect()
}

#[test]
fn csr_invariants() {
    let idx = build_search_automaton(test_terms()).expect("Failed to build pattern matcher");
    assert_eq!(idx.term_offsets.len(), idx.automaton.patterns_len() + 1);
    assert_eq!(idx.term_offsets[0], 0);
    assert_eq!(*idx.term_offsets.last().unwrap() as usize, idx.term_ids.len());
    for w in idx.term_offsets.windows(2) {
        assert!(w[0] < w[1], "every pattern reports at least one term_id");
    }
    assert_eq!(idx.n_terms as usize, parse_test_data().len());
}

#[test]
fn round_trip_lookup() {
    let idx = build_search_automaton(test_terms()).expect("Failed to build pattern matcher");
    for (term_id, pattern) in parse_test_data() {
        let pid = idx
            .pattern_id_for(&pattern)
            .unwrap_or_else(|| panic!("pattern {pattern:?} not found in automaton"));
        assert!(
            idx.term_ids_for(pid).contains(&term_id),
            "term_id {term_id} missing for pattern {pattern:?}"
        );
    }
}

#[test]
fn build_is_deterministic() {
    let a = build_search_automaton(test_terms()).unwrap();
    let b = build_search_automaton(test_terms()).unwrap();
    assert_eq!(a.term_ids, b.term_ids);
    assert_eq!(a.term_offsets, b.term_offsets);
    assert_eq!(a.terms_sha256, b.terms_sha256);
}
