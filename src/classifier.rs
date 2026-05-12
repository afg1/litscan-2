use safetensors;
use regex::{Regex, Error};
use thiserror::Error;

pub struct Tokenizer {
    pattern: Regex,
    lowercase: bool
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("invalid token pattern: {0}")]
    InvalidPattern(#[from] regex::Error),
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
    pub fn new(pattern: &str, lowercase: bool) -> Result<Self, TokenizerError> {
        Ok( Self {
            pattern: Regex::new(translate_sklearn_pattern(pattern))?,
            lowercase,
        })
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let input = if self.lowercase {
            text.to_lowercase()
        }
        else {
            text.to_owned()
        };

        self.pattern
            .find_iter(&input)
            .map(|m| m.as_str().to_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_unicode_other_number() {
        let tok = Tokenizer::new(r"(?u)\b\w\w+\b", true).unwrap();
        assert_eq!(
            tok.tokenize("foo ¹³c bar"),
            vec!["foo".to_string(), "¹³c".to_string(), "bar".to_string()],
        );
    }
}
