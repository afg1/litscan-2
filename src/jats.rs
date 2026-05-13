use roxmltree::Document;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use strsim::jaro_winkler;

#[derive(Clone, Debug)]
pub struct Author {
    pub given_names: Option<String>,
    pub surname: Option<String>,
}

impl Author {
    fn new(given_names: Option<String>, surname: Option<String>) -> Self {
        Author {
            given_names,
            surname,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionKind {
    Abstract,
    Introduction,
    Methods,
    Results,
    Discussion,
    Conclusion,
    Supplementary,
    Figure,
    Table,
    Other,
}

impl SectionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SectionKind::Abstract => "abstract",
            SectionKind::Introduction => "introduction",
            SectionKind::Methods => "methods",
            SectionKind::Results => "results",
            SectionKind::Discussion => "discussion",
            SectionKind::Conclusion => "conclusion",
            SectionKind::Supplementary => "supplementary",
            SectionKind::Figure => "figure",
            SectionKind::Table => "table",
            SectionKind::Other => "other",
        }
    }

    pub fn classify(sec_type: Option<&str>, title: Option<&str>) -> SectionKind {
        if let Some(st) = sec_type {
            let st = st.to_lowercase();
            // Normalize separators; sec-type may use `|` for combined values.
            let parts: Vec<&str> = st.split(|c: char| c == '|' || c == ',' || c.is_whitespace())
                .filter(|p| !p.is_empty())
                .collect();
            let has = |needle: &str| parts.iter().any(|p| *p == needle);

            if has("supplementary-material") || st.contains("supplementary-material") {
                return SectionKind::Supplementary;
            }
            if has("abstract") {
                return SectionKind::Abstract;
            }
            if has("intro") || has("introduction") {
                return SectionKind::Introduction;
            }
            // Combined results|discussion → Results wins
            if has("results") {
                return SectionKind::Results;
            }
            if has("discussion") {
                return SectionKind::Discussion;
            }
            if has("methods") || has("materials") || st.contains("materials|methods")
                || st.contains("methods|materials")
            {
                return SectionKind::Methods;
            }
            if has("conclusion") || has("conclusions") {
                return SectionKind::Conclusion;
            }
        }

        if let Some(t) = title {
            let t = t.trim().to_lowercase();
            if t.is_empty() {
                return SectionKind::Other;
            }

            // Supplementary first — catches "Supplementary Methods" etc.
            if t.contains("supplementary")
                || t.contains("supporting information")
                || t.contains("supporting info")
                || t.contains("appendix")
            {
                return SectionKind::Supplementary;
            }
            if t.starts_with("abstract") || t == "summary" {
                return SectionKind::Abstract;
            }
            if t.contains("introduction") || t.contains("background") {
                return SectionKind::Introduction;
            }
            if t.contains("method")
                || t.contains("materials and methods")
                || t.contains("experimental")
                || t.contains("procedure")
            {
                return SectionKind::Methods;
            }
            if t.contains("result") || t.contains("findings") {
                return SectionKind::Results;
            }
            if t.contains("discussion") {
                return SectionKind::Discussion;
            }
            if t.contains("conclusion") {
                return SectionKind::Conclusion;
            }
        }

        SectionKind::Other
    }
}

impl fmt::Display for SectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct Section {
    pub title: Option<String>,
    pub text: String,
    pub kind: SectionKind,
}

impl Section {
    fn new(title: Option<String>, text: String, kind: SectionKind) -> Self {
        Section { title, text, kind }
    }
}

pub struct JatsArticle {
    pub journal: Option<String>,
    pub article_type: Option<String>,
    pub title: Option<String>,
    pub abstract_text: Option<String>,
    pub authors: Vec<Author>,
    pub sections: Vec<Section>,
}

impl JatsArticle {
    fn new() -> Self {
        JatsArticle {
            journal: None,
            article_type: None,
            title: None,
            abstract_text: None,
            authors: Vec::new(),
            sections: Vec::new(),
        }
    }

    pub fn full_text(&self) -> String {
        let mut full_text = String::new();
        for section in &self.sections {
            if let Some(title) = &section.title {
                full_text.push_str(title);
                full_text.push_str("\n\n");
            }
            full_text.push_str(&section.text);
            full_text.push_str("\n\n");
        }
        full_text.trim().to_string()
    }

    pub fn sections_of_kind(&self, kind: SectionKind) -> impl Iterator<Item = &Section> {
        self.sections.iter().filter(move |s| s.kind == kind)
    }

    pub fn get_section(&self, name: &str) -> Option<Section> {
        let name_lower = name.to_lowercase();
        let mut best_match: Option<&Section> = None;
        let mut best_score = 0.0;

        for section in &self.sections {
            if let Some(title) = &section.title {
                let title_lower = title.to_lowercase();

                // Exact match gets highest priority
                if title_lower == name_lower {
                    return Some(section.clone());
                }

                // Substring match gets high priority
                if title_lower.contains(&name_lower) {
                    if best_score < 0.95 {
                        best_match = Some(section);
                        best_score = 0.95;
                    }
                }

                // Fuzzy match
                let score = jaro_winkler(&title_lower, &name_lower);
                if score > best_score && score > 0.8 {
                    best_score = score;
                    best_match = Some(section);
                }
            }
        }

        best_match.cloned()
    }
}

pub fn parse_article_node(article_node: roxmltree::Node) -> JatsArticle {
    let mut article = JatsArticle::new();

    article.article_type = article_node.attribute("article-type").map(String::from);

    if let Some(journal_node) = article_node
        .descendants()
        .find(|n| n.has_tag_name("journal-title"))
    {
        if let Some(text) = journal_node.text() {
            article.journal = Some(text.trim().to_string());
        }
    }

    if let Some(title_group) = article_node
        .descendants()
        .find(|n| n.has_tag_name("title-group"))
    {
        if let Some(title_node) = title_group
            .descendants()
            .find(|n| n.has_tag_name("article-title"))
        {
            article.title = collect_text(title_node).into();
        }
    }

    if let Some(abstract_node) = article_node
        .descendants()
        .find(|n| n.has_tag_name("abstract"))
    {
        article.abstract_text = collect_text(abstract_node);
        if let Some(text) = &article.abstract_text {
            article.sections.push(Section::new(
                Some("Abstract".to_string()),
                text.clone(),
                SectionKind::Abstract,
            ));
        }
    }

    for contrib in article_node
        .descendants()
        .filter(|n| n.has_tag_name("contrib"))
    {
        if contrib.attribute("contrib-type") == Some("author") {
            if let Some(name_node) = contrib.descendants().find(|n| n.has_tag_name("name")) {
                let surname = name_node
                    .descendants()
                    .find(|n| n.has_tag_name("surname"))
                    .and_then(|n| n.text())
                    .map(|s| s.trim().to_string());
                let given_names = name_node
                    .descendants()
                    .find(|n| n.has_tag_name("given-names"))
                    .and_then(|n| n.text())
                    .map(|s| s.trim().to_string());

                article.authors.push(Author {
                    given_names,
                    surname,
                });
            }
        }
    }

    if let Some(body_node) = article_node.descendants().find(|n| n.has_tag_name("body")) {
        for sec_node in body_node
            .descendants()
            .filter(|n| n.has_tag_name("sec") && n.parent() == Some(body_node))
        {
            let title = sec_node
                .children()
                .find(|n| n.has_tag_name("title"))
                .and_then(|n| collect_text(n));
            let sec_type = sec_node.attribute("sec-type");
            let kind = SectionKind::classify(sec_type, title.as_deref());
            let mut text = String::new();
            for p_node in sec_node.descendants().filter(|n| {
                n.has_tag_name("p")
                    && !n.ancestors().any(|a| a.has_tag_name("fig") || a.has_tag_name("table-wrap"))
            }) {
                if let Some(t) = collect_text(p_node) {
                    text.push_str(&t);
                    text.push('\n');
                }
            }
            article.sections.push(Section {
                title,
                text: text.trim().to_string(),
                kind,
            });
        }

        for fig_node in body_node.descendants().filter(|n| n.has_tag_name("fig")) {
            let label = fig_node
                .children()
                .find(|n| n.has_tag_name("label"))
                .and_then(collect_text);
            let caption_node = fig_node.children().find(|n| n.has_tag_name("caption"));
            let caption_title = caption_node
                .and_then(|c| c.children().find(|n| n.has_tag_name("title")))
                .and_then(collect_text);
            let title = match (label, caption_title) {
                (Some(l), Some(t)) => Some(format!("{}: {}", l, t)),
                (Some(l), None) => Some(l),
                (None, Some(t)) => Some(t),
                (None, None) => None,
            };
            let mut text = String::new();
            if let Some(caption) = caption_node {
                for p_node in caption.descendants().filter(|n| n.has_tag_name("p")) {
                    if let Some(t) = collect_text(p_node) {
                        text.push_str(&t);
                        text.push('\n');
                    }
                }
            }
            let text = text.trim().to_string();
            if title.is_some() || !text.is_empty() {
                article.sections.push(Section {
                    title,
                    text,
                    kind: SectionKind::Figure,
                });
            }
        }

        for tw_node in body_node.descendants().filter(|n| n.has_tag_name("table-wrap")) {
            let label = tw_node
                .children()
                .find(|n| n.has_tag_name("label"))
                .and_then(collect_text);
            let caption_node = tw_node.children().find(|n| n.has_tag_name("caption"));
            let caption_title = caption_node
                .and_then(|c| c.children().find(|n| n.has_tag_name("title")))
                .and_then(collect_text);
            let title = match (label, caption_title) {
                (Some(l), Some(t)) => Some(format!("{}: {}", l, t)),
                (Some(l), None) => Some(l),
                (None, Some(t)) => Some(t),
                (None, None) => None,
            };
            let mut text = String::new();
            if let Some(caption) = caption_node {
                for p_node in caption.descendants().filter(|n| n.has_tag_name("p")) {
                    if let Some(t) = collect_text(p_node) {
                        text.push_str(&t);
                        text.push('\n');
                    }
                }
            }
            for foot_node in tw_node.children().filter(|n| n.has_tag_name("table-wrap-foot")) {
                for p_node in foot_node.descendants().filter(|n| n.has_tag_name("p")) {
                    if let Some(t) = collect_text(p_node) {
                        text.push_str(&t);
                        text.push('\n');
                    }
                }
            }
            let text = text.trim().to_string();
            if title.is_some() || !text.is_empty() {
                article.sections.push(Section {
                    title,
                    text,
                    kind: SectionKind::Table,
                });
            }
        }
    }

    article
}

pub fn parse(xml_string: &str) -> Result<JatsArticle, Box<dyn Error>> {
    let parsing_opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = Document::parse_with_options(xml_string, parsing_opts)
        .map_err(|e| format!("Failed to parse XML: {}", e))?;

    if let Some(article_node) = doc.root().descendants().find(|n| n.has_tag_name("article")) {
        Ok(parse_article_node(article_node))
    } else {
        // Fallback: parse entire document if no <article> tag is found
        Ok(parse_article_node(doc.root()))
    }
}

pub fn parse_archive(xml_string: &str) -> Result<HashMap<String, JatsArticle>, Box<dyn Error>> {
    let doc = parse_document(xml_string)?;
    let mut articles = HashMap::new();
    for (pmcid, node) in article_nodes(&doc) {
        articles.insert(pmcid, parse_article_node(node));
    }
    Ok(articles)
}

/// Parse a JATS archive XML into a `Document` without materializing any
/// `JatsArticle`. Pair with `article_nodes` to walk articles one at a time,
/// optionally in parallel via rayon.
pub fn parse_document(xml_string: &str) -> Result<Document, Box<dyn Error>> {
    let parsing_opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    Document::parse_with_options(xml_string, parsing_opts)
        .map_err(|e| format!("Failed to parse XML: {}", e).into())
}

/// Yield `(pmcid, article_node)` pairs for every `<article>` in `doc` that has
/// a pmcid. Articles without a pmcid are skipped. Nodes are cheap `Copy` views
/// into `doc`; pass them to `parse_article_node` to materialize a `JatsArticle`.
///
/// `roxmltree::Node` is `Copy + Send + Sync`, so the typical parallel pattern is:
///   let nodes: Vec<_> = article_nodes(&doc).collect();
///   nodes.par_iter().map(|(pmcid, node)| {
///       let article = parse_article_node(*node);
///       (pmcid.clone(), scanner::search_article(searcher, &article))
///   }).collect()
pub fn article_nodes<'a, 'input>(
    doc: &'a Document<'input>,
) -> impl Iterator<Item = (String, roxmltree::Node<'a, 'input>)> + 'a {
    doc.root()
        .descendants()
        .filter(|n| n.has_tag_name("article"))
        .filter_map(|article_node| extract_pmcid(article_node).map(|id| (id, article_node)))
}

fn extract_pmcid(article_node: roxmltree::Node) -> Option<String> {
    article_node
        .descendants()
        .filter(|n| n.has_tag_name("article-id"))
        .find(|n| n.attribute("pub-id-type") == Some("pmcid"))
        .and_then(|n| n.text())
        .map(|s| s.trim().to_string())
}

fn collect_text(node: roxmltree::Node) -> Option<String> {
    let mut text = String::new();
    for descendant in node.descendants() {
        if descendant.is_text() {
            if let Some(t) = descendant.text() {
                let cleaned_text = t.replace('\n', " ");
                let cleaned_text = cleaned_text.replace("  ", " ");
                text.push_str(&cleaned_text);
            }
        }
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_from_sec_type() {
        assert_eq!(SectionKind::classify(Some("intro"), None), SectionKind::Introduction);
        assert_eq!(SectionKind::classify(Some("introduction"), None), SectionKind::Introduction);
        assert_eq!(SectionKind::classify(Some("materials|methods"), None), SectionKind::Methods);
        assert_eq!(SectionKind::classify(Some("methods"), None), SectionKind::Methods);
        assert_eq!(SectionKind::classify(Some("results"), None), SectionKind::Results);
        assert_eq!(SectionKind::classify(Some("results|discussion"), None), SectionKind::Results);
        assert_eq!(SectionKind::classify(Some("discussion"), None), SectionKind::Discussion);
        assert_eq!(SectionKind::classify(Some("conclusions"), None), SectionKind::Conclusion);
        assert_eq!(SectionKind::classify(Some("supplementary-material"), None), SectionKind::Supplementary);
        assert_eq!(SectionKind::classify(Some("cases"), None), SectionKind::Other);
    }

    #[test]
    fn classify_from_title() {
        assert_eq!(SectionKind::classify(None, Some("Introduction")), SectionKind::Introduction);
        assert_eq!(SectionKind::classify(None, Some("Background")), SectionKind::Introduction);
        assert_eq!(SectionKind::classify(None, Some("Materials and Methods")), SectionKind::Methods);
        assert_eq!(SectionKind::classify(None, Some("Experimental procedure")), SectionKind::Methods);
        assert_eq!(SectionKind::classify(None, Some("Results")), SectionKind::Results);
        assert_eq!(SectionKind::classify(None, Some("Findings")), SectionKind::Results);
        assert_eq!(SectionKind::classify(None, Some("Discussion")), SectionKind::Discussion);
        assert_eq!(SectionKind::classify(None, Some("Conclusions")), SectionKind::Conclusion);
        assert_eq!(SectionKind::classify(None, Some("Abstract")), SectionKind::Abstract);
        assert_eq!(SectionKind::classify(None, Some("Summary")), SectionKind::Abstract);
        assert_eq!(SectionKind::classify(None, Some("Supplementary Methods")), SectionKind::Supplementary);
        assert_eq!(SectionKind::classify(None, Some("Supporting Information")), SectionKind::Supplementary);
        assert_eq!(SectionKind::classify(None, Some("Appendix A")), SectionKind::Supplementary);
        assert_eq!(SectionKind::classify(None, Some("Acknowledgements")), SectionKind::Other);
    }

    #[test]
    fn classify_empty_inputs() {
        assert_eq!(SectionKind::classify(None, None), SectionKind::Other);
        assert_eq!(SectionKind::classify(None, Some("")), SectionKind::Other);
    }

    #[test]
    fn end_to_end_parse_sets_kinds() {
        let xml = r#"<?xml version="1.0"?>
<article article-type="research-article">
  <front>
    <journal-meta><journal-title>Test J</journal-title></journal-meta>
    <article-meta>
      <title-group><article-title>A title</article-title></title-group>
      <abstract><p>An abstract paragraph.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec sec-type="intro"><title>Background</title><p>Intro body.</p></sec>
    <sec><title>Materials and Methods</title><p>Did stuff.</p></sec>
    <sec sec-type="results"><title>Results</title><p>Found stuff.</p></sec>
    <sec><title>Discussion</title><p>Thoughts.</p></sec>
    <sec sec-type="supplementary-material"><title>Supplementary Data</title><p>Extras.</p></sec>
    <sec><title>Appendix</title><p>More extras.</p></sec>
    <sec><title>Weirdly Named Section</title><p>Whatever.</p></sec>
  </body>
</article>"#;
        let article = parse(xml).expect("parse");
        let kinds: Vec<SectionKind> = article.sections.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                SectionKind::Abstract,
                SectionKind::Introduction,
                SectionKind::Methods,
                SectionKind::Results,
                SectionKind::Discussion,
                SectionKind::Supplementary,
                SectionKind::Supplementary,
                SectionKind::Other,
            ]
        );

        let supp_count = article.sections_of_kind(SectionKind::Supplementary).count();
        assert_eq!(supp_count, 2);
    }

    #[test]
    fn figures_are_extracted_separately() {
        let xml = r#"<?xml version="1.0"?>
<article>
  <body>
    <sec sec-type="results"><title>Results</title>
      <p>Body text mentions Airn.</p>
      <fig id="fig1">
        <label>Figure 1</label>
        <caption><title>Airn locus</title><p>Dotted line: TSS of Airn.</p></caption>
      </fig>
    </sec>
    <fig id="fig2">
      <label>Figure 2</label>
      <caption><p>Standalone figure caption mentioning Xist.</p></caption>
    </fig>
  </body>
</article>"#;
        let article = parse(xml).expect("parse");

        let results: Vec<&Section> = article
            .sections_of_kind(SectionKind::Results)
            .collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("Body text mentions Airn"));
        assert!(
            !results[0].text.contains("Dotted line"),
            "figure caption text must not leak into parent section"
        );

        let figs: Vec<&Section> = article
            .sections_of_kind(SectionKind::Figure)
            .collect();
        assert_eq!(figs.len(), 2);
        assert!(figs[0].text.contains("Dotted line: TSS of Airn"));
        assert_eq!(figs[0].title.as_deref(), Some("Figure 1: Airn locus"));
        assert!(figs[1].text.contains("Xist"));
        assert_eq!(figs[1].title.as_deref(), Some("Figure 2"));
    }

    #[test]
    fn tables_are_extracted_separately() {
        let xml = r#"<?xml version="1.0"?>
<article>
  <body>
    <sec sec-type="results"><title>Results</title>
      <p>Body text mentions Airn.</p>
      <table-wrap id="t1">
        <label>Table 1</label>
        <caption><title>Primers</title><p>Primer set used to amplify Airn.</p></caption>
        <table><tr><td>x</td></tr></table>
        <table-wrap-foot><p>Footnote mentioning Kcnq1ot1.</p></table-wrap-foot>
      </table-wrap>
    </sec>
    <table-wrap id="t2">
      <label>Table 2</label>
      <caption><p>Standalone table caption mentioning Xist.</p></caption>
    </table-wrap>
  </body>
</article>"#;
        let article = parse(xml).expect("parse");

        let results: Vec<&Section> = article
            .sections_of_kind(SectionKind::Results)
            .collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("Body text mentions Airn"));
        assert!(
            !results[0].text.contains("Primer set"),
            "table caption text must not leak into parent section"
        );
        assert!(
            !results[0].text.contains("Footnote"),
            "table footnote text must not leak into parent section"
        );

        let tabs: Vec<&Section> = article
            .sections_of_kind(SectionKind::Table)
            .collect();
        assert_eq!(tabs.len(), 2);
        assert!(tabs[0].text.contains("Primer set used to amplify Airn"));
        assert!(tabs[0].text.contains("Footnote mentioning Kcnq1ot1"));
        assert_eq!(tabs[0].title.as_deref(), Some("Table 1: Primers"));
        assert!(tabs[1].text.contains("Xist"));
        assert_eq!(tabs[1].title.as_deref(), Some("Table 2"));
    }
}
