use roxmltree::Document;
use std::collections::HashMap;
use std::error::Error;
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

#[derive(Clone, Debug)]
pub struct Section {
    pub title: Option<String>,
    pub text: String,
}

impl Section {
    fn new(title: Option<String>, text: String) -> Self {
        Section { title, text }
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

    fn full_text(&self) -> String {
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

    fn get_section(&self, name: &str) -> Option<Section> {
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

fn parse_article_node(article_node: roxmltree::Node) -> JatsArticle {
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
        article.abstract_text = collect_text(abstract_node).into();
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
            let mut text = String::new();
            for p_node in sec_node.descendants().filter(|n| n.has_tag_name("p")) {
                if let Some(t) = collect_text(p_node) {
                    text.push_str(&t);
                    text.push('\n');
                }
            }
            article.sections.push(Section {
                title,
                text: text.trim().to_string(),
            });
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
    let parsing_opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = Document::parse_with_options(xml_string, parsing_opts)
        .map_err(|e| format!("Failed to parse XML: {}", e))?;

    let mut articles = HashMap::new();

    for article_node in doc
        .root()
        .descendants()
        .filter(|n| n.has_tag_name("article"))
    {
        let mut pmcid = None;

        // Find the pmcid within this article node
        for id_node in article_node
            .descendants()
            .filter(|n| n.has_tag_name("article-id"))
        {
            if id_node.attribute("pub-id-type") == Some("pmcid") {
                if let Some(text) = id_node.text() {
                    pmcid = Some(text.trim().to_string());
                    break;
                }
            }
        }

        if let Some(id) = pmcid {
            articles.insert(id, parse_article_node(article_node));
        } else {
            // If we can't find a pmcid, skip this article or use a fallback identifier?
            // User requested mapping PMCID: Article, so we'll skip if no PMCID is found.
            // Could also fall back to DOI or generic id if PMCID isn't strictly required
            // but the prompt explicitly asked for dict mapping PMCID: Article.
        }
    }

    Ok(articles)
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
