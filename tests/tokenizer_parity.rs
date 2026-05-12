mod common;

use common::{fixture_dir, load_fixtures};

use litscan_2::classifier::Tokenizer;



#[test]
fn tokenizer_matches_sklearn() {
    let tokenizer = Tokenizer::new(r"(?u)\b\w\w+\b", true).expect("Failed to build tokenizer");

    for fixture in load_fixtures() {
        assert_eq!(tokenizer.tokenize(&fixture.text), fixture.tokens);
    }
}