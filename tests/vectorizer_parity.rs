mod common;

use common::{load_fixtures, metadata_path, safetensors_path};

use litscan_2::classifier::Vectorizer;

#[test]
fn vectorizer_matches_sklearn() {
    let vectorizer = Vectorizer::new(safetensors_path(), metadata_path()).expect("Failed to build vecotrizer");

       for fixture in load_fixtures() {
        let result = vectorizer.vectorize(&fixture.text);
        assert_eq!(result.indices, fixture.sparse_indices, "index mismatch on {:?}", fixture.text,);

        assert_eq!(result.values.len(), fixture.sparse_values.len());
        for (got, want) in result.values.iter().zip(&fixture.sparse_values) {
            assert!(
                (got - want).abs() < 1e-12,
                "value mismatch on {:?}: got {} want {}",
                fixture.text, got, want,
            );
        }
    }
}
