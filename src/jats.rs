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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SectionKind {
    Title,
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
            SectionKind::Title => "title",
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
    /// Byte spans `(start, end)` within `text` for each non-empty `<p>` that
    /// was concatenated to build it. Used by the paragraph-extraction path to
    /// map a hit `offset` back to its containing paragraph. Empty when the
    /// section wasn't built from per-paragraph concatenation (e.g. the
    /// flattened Abstract).
    pub paragraphs: Vec<(usize, usize)>,
}

impl Section {
    fn new(title: Option<String>, text: String, kind: SectionKind) -> Self {
        let paragraphs = if text.is_empty() { Vec::new() } else { vec![(0, text.len())] };
        Section { title, text, kind, paragraphs }
    }
}

/// Concatenate a stream of `<p>` nodes the same way the parser does (each
/// paragraph's `collect_text` output, separated by '\n', final string
/// trimmed) and also return byte spans into the returned text for each
/// non-empty paragraph.
///
/// `collect_text` already trims leading whitespace per paragraph, so the
/// concatenated string never starts with whitespace; `trim_end` is therefore
/// equivalent to `text.trim()` here, and the recorded paragraph spans stay
/// valid against the returned (trimmed) text.
fn build_paragraph_text<'a, 'input: 'a, I>(p_nodes: I) -> (String, Vec<(usize, usize)>)
where
    I: IntoIterator<Item = roxmltree::Node<'a, 'input>>,
{
    build_block_text(p_nodes.into_iter().map(collect_text))
}

/// Same as `build_paragraph_text`, over already-extracted block strings.
fn build_block_text<I: IntoIterator<Item = Option<String>>>(blocks: I) -> (String, Vec<(usize, usize)>) {
    let mut text = String::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for block in blocks {
        if let Some(t) = block {
            let start = text.len();
            text.push_str(&t);
            let end = text.len();
            spans.push((start, end));
            text.push('\n');
        }
    }
    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    spans.retain(|&(s, _)| s < trimmed_len);
    (text, spans)
}

/// Text blocks of a container: each `<p>`, plus the `<title>` of every
/// `<sec>` and `<caption>`, in document order. Block-level elements are kept
/// apart so a title is never glued onto the next paragraph's first word
/// (shard XML has no whitespace between tags).
fn is_text_block(n: &roxmltree::Node) -> bool {
    n.has_tag_name("p")
        || (n.has_tag_name("title")
            && n.parent().map_or(false, |p| p.has_tag_name("sec") || p.has_tag_name("caption")))
}

fn in_float(n: &roxmltree::Node) -> bool {
    n.ancestors().any(|a| a.has_tag_name("fig") || a.has_tag_name("table-wrap"))
}

/// One line per `<tr>`, cells joined with " | " so adjacent cells never fuse
/// into one token.
fn table_rows<'a, 'input: 'a>(tw_node: roxmltree::Node<'a, 'input>) -> impl Iterator<Item = Option<String>> + 'a {
    tw_node.descendants().filter(|n| n.has_tag_name("tr")).map(|tr| {
        let cells: Vec<String> = tr
            .children()
            .filter(|c| c.has_tag_name("td") || c.has_tag_name("th"))
            .filter_map(collect_text)
            .collect();
        (!cells.is_empty()).then(|| cells.join(" | "))
    })
}

pub struct JatsArticle {
    pub pmcid: Option<String>,
    pub pmid: Option<String>,
    pub doi: Option<String>,
    /// Earliest `<pub-date><year>` in `article-meta`.
    pub year: Option<i32>,
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
            pmcid: None,
            pmid: None,
            doi: None,
            year: None,
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

    /// `"Surname, Given; Surname, Given"`, the format the pipeline's
    /// litscan-scan-job.py stored in `litscan_article.author`.
    pub fn authors_string(&self) -> Option<String> {
        let names: Vec<String> = self
            .authors
            .iter()
            .filter_map(|a| match (a.surname.as_deref(), a.given_names.as_deref()) {
                (Some(s), Some(g)) if !s.is_empty() && !g.is_empty() => Some(format!("{}, {}", s, g)),
                (s, g) => {
                    let n = format!("{}{}", s.unwrap_or(""), g.unwrap_or(""));
                    (!n.is_empty()).then_some(n)
                }
            })
            .collect();
        (!names.is_empty()).then(|| names.join("; "))
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
    article.pmcid = extract_article_id(article_node, "pmcid");
    article.pmid = extract_article_id(article_node, "pmid");
    article.doi = extract_article_id(article_node, "doi");
    article.year = article_node
        .descendants()
        .filter(|n| n.has_tag_name("pub-date") && n.ancestors().any(|a| a.has_tag_name("article-meta")))
        .filter_map(|d| d.children().find(|n| n.has_tag_name("year")))
        .filter_map(|y| y.text().and_then(|t| t.trim().parse::<i32>().ok()))
        .min();

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
    if let Some(title) = &article.title {
        article.sections.push(Section::new(None, title.clone(), SectionKind::Title));
    }

    if let Some(abstract_node) = article_node
        .descendants()
        .find(|n| n.has_tag_name("abstract"))
    {
        let (text, _) = build_paragraph_text(abstract_node.descendants().filter(is_text_block));
        article.abstract_text = if text.is_empty() { collect_text(abstract_node) } else { Some(text) };
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
            let (text, paragraphs) =
                build_paragraph_text(sec_node.descendants().filter(|n| is_text_block(n) && !in_float(n)));
            article.sections.push(Section {
                title,
                text,
                kind,
                paragraphs,
            });
        }

        // Body text that sits outside any top-level <sec>.
        let (text, paragraphs) = build_paragraph_text(body_node.descendants().filter(|n| {
            is_text_block(n) && !in_float(n) && !n.ancestors().any(|a| a.has_tag_name("sec"))
        }));
        if !text.is_empty() {
            article.sections.push(Section { title: None, text, kind: SectionKind::Other, paragraphs });
        }
    }

    // Figures and tables anywhere in the article (body, floats-group, back),
    // except inside <sub-article>s, which are separate documents.
    {
        let floats = |tag: &'static str| {
            article_node
                .descendants()
                .filter(move |n| n.has_tag_name(tag) && !n.ancestors().any(|a| a.has_tag_name("sub-article")))
        };
        for fig_node in floats("fig") {
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
            let (text, paragraphs) = if let Some(caption) = caption_node {
                build_paragraph_text(caption.descendants().filter(is_text_block))
            } else {
                (String::new(), Vec::new())
            };
            if title.is_some() || !text.is_empty() {
                article.sections.push(Section {
                    title,
                    text,
                    kind: SectionKind::Figure,
                    paragraphs,
                });
            }
        }

        for tw_node in floats("table-wrap") {
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
            let caption_ps = caption_node
                .into_iter()
                .flat_map(|c| c.descendants().filter(is_text_block))
                .map(collect_text);
            let foot_ps = tw_node
                .children()
                .filter(|n| n.has_tag_name("table-wrap-foot"))
                .flat_map(|f| f.descendants().filter(|n| n.has_tag_name("p")))
                .map(collect_text);
            let (text, paragraphs) = build_block_text(caption_ps.chain(table_rows(tw_node)).chain(foot_ps));
            if title.is_some() || !text.is_empty() {
                article.sections.push(Section {
                    title,
                    text,
                    kind: SectionKind::Table,
                    paragraphs,
                });
            }
        }
    }

    article
}

const XLINK: &str = "http://www.w3.org/1999/xlink";

/// `related-article-type` values emitted to `notices/`. Backward links sit on
/// the notice and point at the original; forward links sit on the original
/// and point at the notice.
pub const NOTICE_RELATIONS: [&str; 8] = [
    "retracted-article",
    "corrected-article",
    "object-of-concern",
    "retraction-forward",
    "correction-forward",
    "expression-of-concern",
    "updated-article",
    "addended-article",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Notice {
    pub relation: &'static str,
    /// `pmcid`, `doi` or `pmid`.
    pub target_id_type: &'static str,
    pub target_id: String,
}

fn normalise_target(link_type: &str, id: &str) -> Option<(&'static str, String)> {
    let id = id.trim();
    match link_type.to_ascii_lowercase().as_str() {
        "pmc" | "pmcid" => {
            let digits = id.trim_start_matches("PMC").trim_start_matches("pmc");
            (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
                .then(|| ("pmcid", format!("PMC{digits}")))
        }
        "pubmed" | "pmid" => {
            (!id.is_empty() && id.bytes().all(|b| b.is_ascii_digit())).then(|| ("pmid", id.to_string()))
        }
        "doi" => {
            let lower = id.to_ascii_lowercase();
            let cut = ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"]
                .iter()
                .find(|p| lower.starts_with(*p))
                .map_or(0, |p| p.len());
            let doi = id[cut..].trim();
            doi.starts_with("10.").then(|| ("doi", doi.to_string()))
        }
        _ => None,
    }
}

/// Correction / retraction / expression-of-concern links in the article's
/// own front matter (not `<sub-article>`s, not the body). The target comes
/// from the element's `ext-link-type` + `xlink:href`, else a nested
/// `<ext-link>`, else a nested `<pub-id>`.
// ponytail: links with an unrecognised id type (e.g. a bare URL) are dropped.
pub fn notices(article_node: roxmltree::Node) -> Vec<Notice> {
    let Some(front) = article_node.children().find(|n| n.has_tag_name("front")) else {
        return Vec::new();
    };
    let mut out: Vec<Notice> = front
        .descendants()
        .filter(|n| n.has_tag_name("related-article"))
        .filter_map(|ra| {
            let rel = ra.attribute("related-article-type")?;
            let relation = *NOTICE_RELATIONS.iter().find(|r| **r == rel)?;
            let own = ra.attribute("ext-link-type").zip(ra.attribute((XLINK, "href")));
            let ext = || {
                ra.descendants()
                    .filter(|n| n.has_tag_name("ext-link"))
                    .find_map(|e| e.attribute("ext-link-type").zip(e.attribute((XLINK, "href"))))
            };
            let pub_id = || {
                ra.descendants()
                    .filter(|n| n.has_tag_name("pub-id"))
                    .find_map(|e| e.attribute("pub-id-type").zip(e.text()))
            };
            let (t, id) = [own, ext(), pub_id()]
                .into_iter()
                .flatten()
                .find_map(|(t, id)| normalise_target(t, id))?;
            Some(Notice { relation, target_id_type: t, target_id: id })
        })
        .collect();
    out.sort();
    out.dedup();
    out
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

/// A shard split into its articles, so they can be parsed in parallel.
pub struct ShardPieces<'a> {
    text: &'a str,
    /// The root start tag, e.g. `<articles xmlns:xlink="…">`, re-applied to
    /// each piece so namespace declarations on the root still resolve.
    root_open: &'a str,
    root_name: &'a str,
    /// Per top-level article: its byte span, and the spans of `<article>`s
    /// nested inside it, which are cut out.
    articles: Vec<((usize, usize), Vec<(usize, usize)>)>,
}

impl ShardPieces<'_> {
    pub fn len(&self) -> usize {
        self.articles.len()
    }

    /// Article `i` as a standalone document.
    pub fn document_source(&self, i: usize) -> String {
        let ((start, end), cuts) = &self.articles[i];
        let mut src = String::with_capacity(self.root_open.len() + end - start + self.root_name.len() + 3);
        src.push_str(self.root_open);
        let mut pos = *start;
        for &(cs, ce) in cuts {
            src.push_str(&self.text[pos..cs]);
            pos = ce;
        }
        src.push_str(&self.text[pos..*end]);
        src.push_str("</");
        src.push_str(self.root_name);
        src.push('>');
        src
    }
}

fn is_article_open(text: &str, at: usize) -> bool {
    matches!(text.as_bytes().get(at + "<article".len()), Some(b' ' | b'\t' | b'\n' | b'\r' | b'>'))
}

/// Split `<root …><article …>…</article>…</root>` into its top-level
/// articles by scanning for `<article` / `</article>` tags.
///
/// Some notices embed a full copy of the article they concern inside
/// `<related-article>`. That copy is cut out of the notice (the
/// `<related-article>` link itself stays), so the embedded paper is neither
/// scanned twice nor mixed into the notice's own title/body.
///
/// Returns `None` for anything else (a single `<article>` file, a DOCTYPE
/// that may define entities, unbalanced tags); the caller then parses the
/// whole document.
// ponytail: a literal "<article " or "</article>" inside a comment or CDATA
// section would mis-split; the piece then fails to parse and the shard
// errors loudly rather than producing wrong output.
pub fn split_articles(text: &str) -> Option<ShardPieces<'_>> {
    // Skip the XML declaration, comments and processing instructions.
    let mut pos = 0;
    let root_start = loop {
        let lt = pos + text[pos..].find('<')?;
        match text.as_bytes().get(lt + 1) {
            Some(b'?') => pos = lt + text[lt..].find("?>")? + 2,
            Some(b'!') if text[lt..].starts_with("<!--") => pos = lt + text[lt..].find("-->")? + 3,
            Some(b'!') => return None, // DOCTYPE: entities would be lost per piece
            _ => break lt,
        }
    };
    let root_end = root_start + text[root_start..].find('>')? + 1;
    let root_open = &text[root_start..root_end];
    let root_name = root_open[1..].split(|c: char| c.is_whitespace() || c == '>' || c == '/').next()?;
    if root_name == "article" || root_open.ends_with("/>") {
        return None;
    }

    let body = &text.as_bytes()[root_end..];
    let mut opens = memchr::memmem::find_iter(body, b"<article")
        .map(|i| i + root_end)
        .filter(|&i| is_article_open(text, i))
        .peekable();
    let mut closes = memchr::memmem::find_iter(body, b"</article>").map(|i| i + root_end).peekable();

    let mut articles = Vec::new();
    let (mut depth, mut start, mut cut_start) = (0usize, 0usize, 0usize);
    let mut cuts = Vec::new();
    loop {
        let take_open = match (opens.peek(), closes.peek()) {
            (Some(o), Some(c)) => o < c,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_open {
            let o = opens.next().unwrap();
            if depth == 0 {
                start = o;
            } else if depth == 1 {
                cut_start = o;
            }
            depth += 1;
        } else {
            let end = closes.next().unwrap() + "</article>".len();
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                articles.push(((start, end), std::mem::take(&mut cuts)));
            } else if depth == 1 {
                cuts.push((cut_start, end));
            }
        }
    }
    (depth == 0).then_some(ShardPieces { text, root_open, root_name, articles })
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
        // Skip copies of other articles embedded in a notice's <related-article>.
        .filter(|n| !n.ancestors().skip(1).any(|a| a.has_tag_name("article")))
        .filter_map(|article_node| extract_pmcid(article_node).map(|id| (id, article_node)))
}

fn extract_pmcid(article_node: roxmltree::Node) -> Option<String> {
    extract_article_id(article_node, "pmcid")
}

/// First non-empty `<article-id pub-id-type="{kind}">` in document order, so
/// the article's own `front/article-meta` wins over any `<sub-article>`.
fn extract_article_id(article_node: roxmltree::Node, kind: &str) -> Option<String> {
    article_node
        .descendants()
        .filter(|n| n.has_tag_name("article-id"))
        .filter(|n| n.attribute("pub-id-type") == Some(kind))
        .filter_map(|n| n.text())
        .map(|s| s.trim().to_string())
        .find(|s| !s.is_empty())
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
                SectionKind::Title,
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
        assert_eq!(article.sections[0].text, "A title");
    }

    #[test]
    fn titles_body_text_and_table_cells_are_scanned() {
        // No whitespace between tags, as in the real shards.
        let xml = concat!(
            r#"<article><front><article-meta>"#,
            r#"<abstract><sec><title>Background</title><p>MIR21 matters.</p></sec></abstract>"#,
            r#"</article-meta></front><body>"#,
            r#"<p>Loose body text with HOTAIR.</p>"#,
            r#"<sec><title>Results</title><sec><title>XIST knockdown</title><p>Body.</p></sec></sec>"#,
            r#"<fig><label>Figure 1</label><caption><title>TERC levels</title><p>Cap.</p></caption></fig>"#,
            r#"</body><floats-group><table-wrap><label>Table 1</label><table>"#,
            r#"<tr><th>Gene</th><th>Fold</th></tr><tr><td>MCM5</td><td>2.1</td></tr>"#,
            r#"</table></table-wrap></floats-group></article>"#,
        );
        let a = parse(xml).expect("parse");
        let text_of = |k: SectionKind| -> Vec<String> { a.sections_of_kind(k).map(|s| s.text.clone()).collect() };
        assert_eq!(a.abstract_text.as_deref(), Some("Background\nMIR21 matters."));
        assert_eq!(text_of(SectionKind::Other), vec!["Loose body text with HOTAIR."]);
        assert_eq!(text_of(SectionKind::Results), vec!["Results\nXIST knockdown\nBody."]);
        assert_eq!(text_of(SectionKind::Figure), vec!["TERC levels\nCap."]);
        assert_eq!(text_of(SectionKind::Table), vec!["Gene | Fold\nMCM5 | 2.1"]);
    }

    #[test]
    fn notices_from_front_matter() {
        let xml = r#"<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="retraction"><front><article-meta>
<related-article related-article-type="retracted-article" ext-link-type="pmc" xlink:href="PMC7519587"/>
<related-article related-article-type="corrected-article" ext-link-type="doi" xlink:href="https://doi.org/10.1021/X.1"/>
<related-article related-article-type="object-of-concern">See <ext-link ext-link-type="pmcid" xlink:href="PMC8890747">this</ext-link></related-article>
<related-article related-article-type="correction-forward"><pub-id pub-id-type="pmid">42</pub-id></related-article>
<related-article related-article-type="reviewed-article" ext-link-type="pmc" xlink:href="PMC1"/>
<related-article related-article-type="retracted-article" ext-link-type="uri" xlink:href="https://example.org"/>
</article-meta></front>
<sub-article><front-stub><related-article related-article-type="retracted-article" ext-link-type="pmc" xlink:href="PMC9"/></front-stub></sub-article>
</article>"#;
        let doc = parse_document(xml).unwrap();
        let node = doc.root().descendants().find(|n| n.has_tag_name("article")).unwrap();
        let got: Vec<(&str, &str, String)> =
            notices(node).into_iter().map(|n| (n.relation, n.target_id_type, n.target_id)).collect();
        assert_eq!(
            got,
            vec![
                ("corrected-article", "doi", "10.1021/X.1".to_string()),
                ("correction-forward", "pmid", "42".to_string()),
                ("object-of-concern", "pmcid", "PMC8890747".to_string()),
                ("retracted-article", "pmcid", "PMC7519587".to_string()),
            ]
        );
    }

    #[test]
    fn splits_shard_into_articles() {
        let shard = r#"<?xml version="1.0"?><!-- c --><articles xmlns:xlink="http://www.w3.org/1999/xlink"><article a="1"><front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id></article-meta></front><body><sub-article/></body></article><article-ish/><article><front><article-meta><article-id pub-id-type="pmcid">PMC2</article-id><related-article xlink:href="PMC1"/></article-meta></front></article></articles>"#;
        let p = split_articles(shard).expect("splittable");
        assert_eq!(p.root_name, "articles");
        assert_eq!(p.len(), 2);
        for i in 0..2 {
            let src = p.document_source(i);
            let doc = parse_document(&src).expect("piece parses, xlink prefix resolves");
            let ids: Vec<String> = article_nodes(&doc).map(|(id, _)| id).collect();
            assert_eq!(ids, vec![format!("PMC{}", i + 1)]);
        }
        assert!(split_articles("<article><front/></article>").is_none());
        assert!(split_articles("<!DOCTYPE articles><articles><article></article></articles>").is_none());
        assert!(split_articles("<articles><article></article></article></articles>").is_none());
        assert!(split_articles("<articles><article><front/></articles>").is_none());
    }

    #[test]
    fn embedded_article_copy_is_cut_from_notice() {
        let shard = concat!(
            r#"<articles><article article-type="correction"><front><article-meta>"#,
            r#"<article-id pub-id-type="pmcid">PMC2</article-id>"#,
            r#"<related-article related-article-type="corrected-article" ext-link-type="pmc" xlink:href="PMC1" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
            r#"<article><front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id>"#,
            r#"<title-group><article-title>Embedded copy</article-title></title-group></article-meta></front></article>"#,
            r#"</related-article><title-group><article-title>Correction notice</article-title></title-group>"#,
            r#"</article-meta></front></article>"#,
            r#"<article><front><article-meta><article-id pub-id-type="pmcid">PMC1</article-id></article-meta></front></article></articles>"#,
        );
        let p = split_articles(shard).expect("splittable");
        assert_eq!(p.len(), 2);
        let src = p.document_source(0);
        assert!(!src.contains("Embedded copy"));
        let doc = parse_document(&src).unwrap();
        let (id, node) = article_nodes(&doc).next().unwrap();
        assert_eq!(id, "PMC2");
        assert_eq!(parse_article_node(node).title.as_deref(), Some("Correction notice"));
        assert_eq!(notices(node).len(), 1, "the related-article link survives the cut");

        // The whole-document path yields only top-level articles.
        let doc = parse_document(shard).unwrap();
        let ids: Vec<String> = article_nodes(&doc).map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["PMC2", "PMC1"]);
    }

    #[test]
    fn metadata_ids_year_authors() {
        let xml = r#"<?xml version="1.0"?>
<article article-type="research-article">
  <front><article-meta>
    <article-id pub-id-type="pmcid">PMC9</article-id>
    <article-id pub-id-type="pmid">123</article-id>
    <article-id pub-id-type="doi">10.1/x</article-id>
    <contrib-group>
      <contrib contrib-type="author"><name><surname>Smith</surname><given-names>Jo</given-names></name></contrib>
      <contrib contrib-type="author"><name><surname>Onlysurname</surname></name></contrib>
    </contrib-group>
    <pub-date pub-type="ppub"><year>2021</year></pub-date>
    <pub-date pub-type="epub"><year>2020</year></pub-date>
  </article-meta></front>
</article>"#;
        let a = parse(xml).expect("parse");
        assert_eq!(a.pmcid.as_deref(), Some("PMC9"));
        assert_eq!(a.pmid.as_deref(), Some("123"));
        assert_eq!(a.doi.as_deref(), Some("10.1/x"));
        assert_eq!(a.year, Some(2020));
        assert_eq!(a.authors_string().as_deref(), Some("Smith, Jo; Onlysurname"));
        assert!(a.title.is_none());
        assert!(a.sections.is_empty());
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
