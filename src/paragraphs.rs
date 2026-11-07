//! Map scanner mentions back to the full paragraph that contained them.
//!
//! The scanner emits one row per mention with `(pmcid, section, sentence, ...)`.
//! This module re-parses an article and, for a set of `(section, sentence)`
//! requests, returns the paragraph (one `<p>` worth of text) containing the
//! start of each sentence.
//!
//! Correctness relies on `jats::parse_article_node` storing paragraph byte
//! spans in `Section::paragraphs`, built from the same concatenation the
//! scanner sees.

use std::collections::HashMap;

use crate::jats;

/// One request to resolve: the section kind and the sentence the scanner
/// recorded for a mention.
#[derive(Clone, Debug)]
pub struct Request {
    pub section: String,
    pub sentence: String,
}

/// One resolved paragraph for a request. `section` and `sentence` echo the
/// request so output rows can be joined onto mentions on
/// `(pmcid, section, sentence)`.
#[derive(Clone, Debug)]
pub struct ResolvedParagraph {
    pub section: String,
    pub sentence: String,
    pub paragraph: String,
}

/// For each `(section, sentence)` request, find the paragraph containing the
/// sentence's start in the first section of that kind (document order) whose
/// text contains the sentence. Unresolved requests are skipped.
pub fn extract_paragraphs_for_article(
    article_node: roxmltree::Node,
    requests: &[Request],
) -> Vec<ResolvedParagraph> {
    let article = jats::parse_article_node(article_node);

    // Bucket requests by section kind so we only walk sections matching a
    // requested kind.
    let mut by_kind: HashMap<&str, Vec<&Request>> = HashMap::new();
    for req in requests {
        by_kind.entry(req.section.as_str()).or_default().push(req);
    }

    // Track which requests have been resolved so we don't emit a second
    // paragraph when several sections of the same kind contain the sentence.
    let mut resolved_idx: Vec<bool> = vec![false; requests.len()];
    let request_index: HashMap<*const Request, usize> = requests
        .iter()
        .enumerate()
        .map(|(i, r)| (r as *const _, i))
        .collect();

    let mut out = Vec::with_capacity(requests.len());

    for section in &article.sections {
        let kind_str = section.kind.as_str();
        let Some(reqs) = by_kind.get(kind_str) else { continue };

        for req in reqs {
            let idx = request_index[&(*req as *const Request)];
            if resolved_idx[idx] {
                continue;
            }
            if req.sentence.is_empty() {
                continue;
            }
            let Some(offset) = section.text.find(req.sentence.as_str()) else { continue };
            let Some(paragraph) = find_paragraph(&section.text, &section.paragraphs, offset) else {
                continue;
            };
            resolved_idx[idx] = true;
            out.push(ResolvedParagraph {
                section: kind_str.to_string(),
                sentence: req.sentence.clone(),
                paragraph,
            });
        }
    }

    out
}

fn find_paragraph(text: &str, spans: &[(usize, usize)], off: usize) -> Option<String> {
    let idx = spans.binary_search_by(|&(start, end)| {
        if off >= end {
            std::cmp::Ordering::Less
        } else if off < start {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }).ok()?;
    let (s, e) = spans[idx];
    Some(text[s..e].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jats::{parse_document, SectionKind};

    #[test]
    fn finds_paragraph_for_offset_in_section() {
        let xml = r#"<?xml version="1.0"?>
<article>
  <front><article-meta>
    <article-id pub-id-type="pmcid">PMC1</article-id>
  </article-meta></front>
  <body>
    <sec sec-type="results"><title>Results</title>
      <p>First paragraph mentions Airn here.</p>
      <p>Second paragraph mentions Xist later on.</p>
      <p>Third paragraph with Kcnq1ot1.</p>
    </sec>
  </body>
</article>"#;
        let doc = parse_document(xml).unwrap();
        let (_pmcid, node) = jats::article_nodes(&doc).next().unwrap();
        let article = jats::parse_article_node(node);
        let results_section = article
            .sections
            .iter()
            .find(|s| s.kind == SectionKind::Results)
            .unwrap();

        // The section title plus 3 paragraphs.
        assert_eq!(results_section.paragraphs.len(), 4);

        let requests = vec![Request {
            section: "results".to_string(),
            sentence: "Second paragraph mentions Xist later on.".to_string(),
        }];

        let resolved = extract_paragraphs_for_article(node, &requests);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].section, "results");
        assert!(resolved[0].paragraph.contains("Xist"));
        assert!(!resolved[0].paragraph.contains("Airn"));
        assert!(!resolved[0].paragraph.contains("Kcnq1ot1"));
    }

    #[test]
    fn unknown_sentence_is_skipped() {
        let xml = r#"<?xml version="1.0"?>
<article>
  <front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id></article-meta></front>
  <body><sec sec-type="intro"><p>Short.</p></sec></body>
</article>"#;
        let doc = parse_document(xml).unwrap();
        let (_pmcid, node) = jats::article_nodes(&doc).next().unwrap();
        let resolved = extract_paragraphs_for_article(
            node,
            &[Request { section: "introduction".to_string(), sentence: "Not here.".to_string() }],
        );
        assert!(resolved.is_empty());
    }
}
