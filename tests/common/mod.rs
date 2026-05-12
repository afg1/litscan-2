use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub text: String,
    pub tokens: Vec<String>,
    pub sparse_indices: Vec<u32>,
    pub sparse_values: Vec<f64>,
    pub decision_per_fold: Vec<f64>,
    pub proba_positive_per_fold: Vec<f64>,
    pub predict_proba: Vec<f64>,
    // sklearn labels can be int or str; keep it generic.
    pub predicted_label: serde_json::Value,
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn load_fixtures() -> Vec<Fixture> {
    let path = fixture_dir().join("test_fixtures.json");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("failed to open {}: {}", path.display(), e));
    serde_json::from_reader(file).expect("failed to parse test_fixtures.json")
}