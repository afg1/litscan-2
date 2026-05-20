//! Map scanner hit offsets back to the full paragraph that contained them.
//!
//! The scanner emits one row per hit with `(pmcid, section, offset, sentence, ...)`
//! where `offset` is a byte offset into the section's concatenated text. This
//! module re-parses an article and, for a set of `(section, offset)` requests,
//! returns the paragraph (one `<p>` worth of text) containing each offset.
//!
//! Correctness relies on `jats::parse_article_node` storing paragraph byte
//! spans in `Section::paragraphs`, built from the same concatenation the
//! scanner sees.

use std::collections::HashMap;

use crate::jats;

/// One request to resolve: which section kind and which byte offset within
/// the section's text. `sentence` is the sentence text the scanner recorded
/// for this hit — used to disambiguate when an article has multiple sections
/// sharing a `SectionKind` (e.g. several `"other"` sections, each with its
/// own offset range). An empty `sentence` disables that check.
#[derive(Clone, Debug)]
pub struct Request {
    pub section: String,
    pub offset: u32,
    pub sentence: String,
}

/// One resolved paragraph for a request. `section`, `offset` and `sentence`
/// echo the request so output rows can be inner-joined onto the hits parquet
/// on any of `(pmcid, section, offset)` or `(pmcid, section, offset, sentence)`.
/// Use the 4-key join when `(pmcid, section, offset)` is non-unique in hits
/// (which happens when an article has multiple sections sharing a kind).
#[derive(Clone, Debug)]
pub struct ResolvedParagraph {
    pub section: String,
    pub offset: u32,
    pub sentence: String,
    pub paragraph: String,
}

/// For each `(section, offset)` request, find the paragraph in the article
/// that contains the offset. Multiple sections can share a `SectionKind`
/// (e.g. several `"other"` sections); we check each in document order and
/// emit one result per section whose paragraph spans contain the offset.
/// Requests that don't resolve to any paragraph are skipped.
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

    // Track which request indices have been resolved so we don't emit a
    // second paragraph for the same `(section, offset, sentence)` when
    // multiple sections of the same kind coincidentally contain the offset.
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
            let Some(paragraph) = find_paragraph(&section.text, &section.paragraphs, req.offset)
            else {
                continue;
            };
            // Disambiguate: if the scanner recorded a sentence, only accept
            // the paragraph if it actually contains that sentence. Sentences
            // can pick up the `\n` paragraph separator at either end when
            // Unicode segmentation runs across a paragraph boundary, so
            // compare against the trimmed sentence.
            if !req.sentence.is_empty() {
                let needle = req.sentence.trim();
                if !needle.is_empty() && !paragraph.contains(needle) {
                    continue;
                }
            }
            resolved_idx[idx] = true;
            out.push(ResolvedParagraph {
                section: kind_str.to_string(),
                offset: req.offset,
                sentence: req.sentence.clone(),
                paragraph,
            });
        }
    }

    out
}

fn find_paragraph(text: &str, spans: &[(usize, usize)], offset: u32) -> Option<String> {
    let off = offset as usize;
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

        // Verify the section actually has 3 paragraph spans.
        assert_eq!(results_section.paragraphs.len(), 3);

        // Pick an offset inside the second paragraph (location of "Xist").
        let xist_offset = results_section.text.find("Xist").unwrap() as u32;
        let requests = vec![Request {
            section: "results".to_string(),
            offset: xist_offset,
            sentence: String::new(),
        }];

        let resolved = extract_paragraphs_for_article(node, &requests);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].section, "results");
        assert_eq!(resolved[0].offset, xist_offset);
        assert!(resolved[0].paragraph.contains("Xist"));
        assert!(!resolved[0].paragraph.contains("Airn"));
        assert!(!resolved[0].paragraph.contains("Kcnq1ot1"));
    }

    #[test]
    fn out_of_range_offset_is_skipped() {
        let xml = r#"<?xml version="1.0"?>
<article>
  <front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id></article-meta></front>
  <body><sec sec-type="intro"><p>Short.</p></sec></body>
</article>"#;
        let doc = parse_document(xml).unwrap();
        let (_pmcid, node) = jats::article_nodes(&doc).next().unwrap();
        let resolved = extract_paragraphs_for_article(
            node,
            &[Request { section: "introduction".to_string(), offset: 9999, sentence: String::new() }],
        );
        assert!(resolved.is_empty());
    }
}
