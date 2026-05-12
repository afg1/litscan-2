use regex::Regex;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;
use std::path::PathBuf;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid regex pattern: {0}")]
    Regex(#[from] regex::Error),

    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("unsupported {field} value: {value:?} (only {expected} is supported)")]
    UnsupportedConfig {
        field: &'static str,
        value: String,
        expected: &'static str,
    },

    #[error("tensor {name:?}: expected {expected} elements, got {got}")]
    TensorShape {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

#[derive(Debug, Deserialize)]
pub struct ModelMetadata {
    pub sklearn_version: String,
    pub n_folds: u32,
    pub n_features: u32,
    pub classes: Vec<serde_json::Value>,
    pub vocabulary: HashMap<String, u32>,
    pub tfidf_config: TfidfConfig,
}

#[derive(Debug, Deserialize)]
pub struct TfidfConfig {
    pub lowercase: bool,
    pub strip_accents: Option<String>,
    pub token_pattern: String,
    pub ngram_range: (usize, usize),
    pub analyzer: String,
    pub stop_words: Option<serde_json::Value>,
    pub sublinear_tf: bool,
    pub norm: Option<String>,
    pub smooth_idf: bool,
}

impl ModelMetadata {
    pub fn from_path(path: &std::path::Path) -> Result<Self, Error> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }
}

fn read_tensors_f64(tensors: &SafeTensors, name: &str) -> Result<Vec<f64>, Error> {
    let view = tensors.tensor(name)?;
    Ok(view
        .data()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

pub struct Tokenizer {
    pattern: Regex,
    lowercase: bool,
}

// sklearn's default `token_pattern` relies on Python `re`'s `\w`, which includes
// Unicode category `No` (superscripts, fractions, Roman numerals). Rust's `regex`
// crate excludes `No` from `\w`, so we substitute an explicit class to preserve
// parity. Only recognises the canonical sklearn default — any other pattern is
// passed through unchanged.
fn translate_sklearn_pattern(pattern: &str) -> &str {
    match pattern {
        r"(?u)\b\w\w+\b" => r"[\p{L}\p{N}\p{M}\p{Pc}]{2,}",
        other => other,
    }
}

impl Tokenizer {
    pub fn new(pattern: &str, lowercase: bool) -> Result<Self, Error> {
        Ok(Self {
            pattern: Regex::new(translate_sklearn_pattern(pattern))?,
            lowercase,
        })
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let input = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_owned()
        };

        self.pattern
            .find_iter(&input)
            .map(|m| m.as_str().to_owned())
            .collect()
    }
}

pub struct Vectorizer {
    tokenizer: Tokenizer,
    vocabulary: HashMap<String, u32>,
    idf: Vec<f64>,
    norm: String,
    sublinear: bool,
}

pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl Vectorizer {
    pub fn new(safetensors_path: PathBuf, metadata_path: PathBuf) -> Result<Self, Error> {
        let metadata: ModelMetadata = ModelMetadata::from_path(&metadata_path)?;
        let bytes: Vec<u8> = std::fs::read(safetensors_path)?;
        let tensors: SafeTensors = SafeTensors::deserialize(&bytes)?;
        let idf: Vec<f64> = read_tensors_f64(&tensors, "idf")?;

        let tok = Tokenizer::new(
            &metadata.tfidf_config.token_pattern,
            metadata.tfidf_config.lowercase,
        )?;

        Ok(Self {
            tokenizer: tok,
            vocabulary: metadata.vocabulary,
            idf: idf,
            norm: metadata.tfidf_config.norm.unwrap_or("l2".to_string()),
            sublinear: metadata.tfidf_config.sublinear_tf,
        })
    }

    pub fn vectorize(&self, text: &str) -> SparseVector {
        let tokens: Vec<String> = self.tokenizer.tokenize(text);

        let mut tf: HashMap<u32, f64> = HashMap::new();
        for tok in &tokens {
            if let Some(&idx) = self.vocabulary.get(tok) {
                *tf.entry(idx).or_insert(0.0) += 1.0;
            }
        };

        let mut entries: Vec<(u32, f64)> = tf
        .into_iter()
        .map(|(idx, count)| (idx, count*self.idf[idx as usize]))
        .collect();
        
        entries.sort_by_key(|&(idx, _)| idx);

        let norm: f64 = entries.iter().map(|&(_, v)| v*v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (_, v) in entries.iter_mut() {
                *v /= norm;
            }
        }
        let (indices, values) = entries.into_iter().unzip();

        SparseVector{indices, values}
    }
}

pub struct TfidfClassifier {
    tokenizer: Tokenizer,
    vectorizer: Vectorizer,
    coef: Vec<f64>,
    intercept: Vec<f64>,
    sig_a: Vec<f64>,
    sig_b: Vec<f64>,
    folds: u32,
    features: u32

}

pub struct ClassificationResult {
    pub proba: Vec<f64>,
    pub fold_decisions: Vec<f64>,
    pub fold_prob: Vec<f64>,
    pub label: u32
}

impl TfidfClassifier {
    pub fn new(safetensors_path: PathBuf, metadata_path: PathBuf ) -> Result<Self, Error> {
        let metadata: ModelMetadata = ModelMetadata::from_path(&metadata_path)?;
        let bytes: Vec<u8> = std::fs::read(&safetensors_path)?;
        let tensors: SafeTensors = SafeTensors::deserialize(&bytes)?;

        let tok = Tokenizer::new(
            &metadata.tfidf_config.token_pattern,
            metadata.tfidf_config.lowercase,
        )?;

        let vectorizer = Vectorizer::new(safetensors_path, metadata_path)?;

        let coef = read_tensors_f64(&tensors, "coef")?;
        let intercept = read_tensors_f64(&tensors, "intercept")?;
        let sigmoid_a = read_tensors_f64(&tensors, "sigmoid_a")?;
        let sigmoid_b = read_tensors_f64(&tensors, "sigmoid_b")?;
        
        Ok(Self {
            tokenizer: tok,
            vectorizer: vectorizer,
            coef: coef,
            intercept: intercept,
            folds: metadata.n_folds,
            features: metadata.n_features,
            sig_a: sigmoid_a,
            sig_b: sigmoid_b
        })

    }

    pub fn classify(&self, text: &str) -> ClassificationResult {

        let feature_vec: SparseVector = self.vectorizer.vectorize(text);

        let mut fold_decisions: Vec<f64> = self.intercept.clone();
        let mut fold_probs: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        for fold in 0 .. self.folds as usize {
            for (&index, &value) in feature_vec.indices.iter().zip(feature_vec.values.iter()) {
                fold_decisions[fold] += self.coef[fold * self.features as usize + index as usize] * value  ;
            }
            fold_probs[fold] = 1.0 / (1.0 + (self.sig_a[fold] * fold_decisions[fold] + self.sig_b[fold]).exp());

        }
        
        let final_prob = fold_probs.iter().sum::<f64>() / self.folds as f64;
        ClassificationResult{
            proba: vec![(1.0 - final_prob), final_prob],
            fold_decisions: fold_decisions,
            fold_prob: fold_probs,
            label: (final_prob > 0.5) as u32
        }
    }
}
