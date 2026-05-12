
mod common;

use common::{fixture_dir, load_fixtures, metadata_path, safetensors_path};

use litscan_2::classifier::TfidfClassifier;

#[test]
fn predicted_proba_matches_sklearn() {
    let classifier = TfidfClassifier::new(safetensors_path(), metadata_path()).expect("Couldn't create classifier");

    for fixture in load_fixtures() {
        let result = classifier.classify(&fixture.text);


        for (got, want) in result.fold_decisions.iter().zip(&fixture.decision_per_fold) {
        assert!(
            (got - want).abs() < 1e-12,
            "value mismatch on {:?}: got {} want {}",
            fixture.text, got, want,
        );
        }

        for (got, want) in result.fold_prob.iter().zip(&fixture.proba_positive_per_fold) {
        assert!(
            (got - want).abs() < 1e-12,
            "value mismatch on {:?}: got {} want {}",
            fixture.text, got, want,
        );
        }

        for (got, want) in result.proba.iter().zip(&fixture.predict_proba) {
            assert!(
                (got - want).abs() < 1e-12,
                "value mismatch on {:?}: got {} want {}",
                fixture.text, got, want,
            );
        }

        assert_eq!(result.label, fixture.predicted_label);
    }
}