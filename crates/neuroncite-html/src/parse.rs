// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// HTML parsing, text extraction, section splitting, and metadata extraction.
//
// Provides functions to extract visible text from HTML documents, apply a
// readability algorithm to remove boilerplate (navigation, ads, sidebars),
// split extracted text into heading-based sections (H1/H2), extract metadata
// from HTML <head> elements and HTTP response headers, and convert sections
// to PageText records for the chunking pipeline.

use std::collections::HashSet;
use std::path::Path;

use scraper::{Html, Selector};

use neuroncite_core::{ExtractionBackend, PageText};

use crate::error::HtmlError;
use crate::types::{HtmlSection, WebMetadata};

/// Elements whose text content is invisible to the user and should be excluded
/// from text extraction (script code, style rules, template markup).
const INVISIBLE_ELEMENTS: &[&str] = &["script", "style", "noscript", "template", "svg"];

/// CSS class/id substrings that indicate boilerplate content. Nodes containing
/// these substrings in their class or id attributes are penalized by the
/// readability scoring algorithm.
const BOILERPLATE_INDICATORS: &[&str] = &[
    "nav",
    "sidebar",
    "footer",
    "header",
    "menu",
    "ad",
    "comment",
    "social",
    "share",
    "cookie",
    "banner",
    "popup",
    "modal",
    "toolbar",
    "breadcrumb",
    "widget",
    "related",
    "sponsored",
];

/// CSS class/id substrings that indicate main article content. Nodes containing
/// these substrings receive a scoring bonus in the readability algorithm.
const CONTENT_INDICATORS: &[&str] = &[
    "article",
    "content",
    "post",
    "entry",
    "main",
    "body",
    "text",
    "story",
    "prose",
    "document",
    "page-content",
    "blog",
];

/// Extracts all visible text from an HTML document, preserving structural
/// whitespace (paragraph breaks, headings). Removes all HTML tags, scripts,
/// styles, and other non-visible elements. This is the "raw" extraction mode
/// that includes navigation, footer, and sidebar text.
///
/// Block-level elements (`<p>`, `<div>`, `<h1>`-`<h6>`, `<br>`, `<hr>`, `<li>`)
/// introduce line breaks in the output. Inline elements are concatenated
/// without extra whitespace.
pub fn extract_visible_text(html: &str) -> String {
    if html.is_empty() {
        return String::new();
    }

    let document = Html::parse_document(html);
    let raw = extract_text_from_document(&document);
    let cleaned = collapse_whitespace(&raw);
    cleaned.trim().to_string()
}

/// Extracts the main article content using a readability-style algorithm.
/// Removes navigation, sidebars, footers, ads, and other boilerplate content.
/// Falls back to full visible text extraction if no high-scoring content node
/// is found.
///
/// The algorithm scores DOM nodes by text density (text length relative to
/// child element count), penalizes nodes with boilerplate class/id indicators,
/// and boosts nodes with content class/id indicators. The highest-scoring
/// subtree's text is returned.
///
/// # Errors
///
/// Returns `HtmlError::Parse` if the HTML cannot be parsed.
pub fn extract_main_content(html: &str) -> Result<String, HtmlError> {
    if html.is_empty() {
        return Ok(String::new());
    }

    let document = Html::parse_document(html);

    // Score candidate container elements and extract text from the best one.
    let best_text = extract_best_candidate_text(&document);

    match best_text {
        Some(text) if !text.is_empty() => {
            let cleaned = collapse_whitespace(&text);
            Ok(cleaned.trim().to_string())
        }
        _ => {
            // Fallback: return all visible text when no clear article content is found.
            Ok(extract_visible_text(html))
        }
    }
}

/// Extracts metadata from HTML `<head>` elements and HTTP response headers.
/// Parses `<title>`, `<meta>` tags (description, author, og:* properties),
/// `<link rel="canonical">`, and the `<html lang>` attribute.
pub fn extract_metadata(
    html: &str,
    response_url: &str,
    http_status: u16,
    content_type: Option<&str>,
) -> WebMetadata {
    let document = Html::parse_document(html);

    let title = select_text(&document, "title");
    let meta_description = select_meta_content(&document, "description");
    let author = select_meta_content(&document, "author");

    // Open Graph tags use property="og:*" instead of name="*".
    let og_title = select_meta_property(&document, "og:title");
    let og_description = select_meta_property(&document, "og:description");
    let og_image = select_meta_property(&document, "og:image");

    // Canonical URL from <link rel="canonical" href="...">.
    let canonical_url = select_link_href(&document, "canonical");

    // Language from <html lang="..."> attribute.
    let language = select_html_lang(&document);

    // Published date from <meta property="article:published_time">.
    let published_date = select_meta_property(&document, "article:published_time");

    // Domain from the response URL.
    let domain = url::Url::parse(response_url)
        .ok()
        .and_then(|u| u.host_str().map(String::from))
        .unwrap_or_default();

    WebMetadata {
        url: response_url.to_string(),
        canonical_url,
        title,
        meta_description,
        language,
        og_image,
        og_title,
        og_description,
        author,
        published_date,
        domain,
        fetch_timestamp: 0, // Set by the caller (fetch module).
        http_status,
        content_type: content_type.map(String::from),
    }
}

/// Splits extracted text into sections based on H1/H2 headings. Content before
/// the first heading becomes section 0 (the preamble). Each subsequent H1/H2
/// heading starts a new section. H3-H6 headings do not trigger section breaks
/// and are treated as inline content within their parent section.
///
/// When `strip_boilerplate` is true, the readability algorithm is applied before
/// splitting, removing navigation, ads, and sidebar content. When false, all
/// visible text is included.
///
/// # Errors
///
/// Returns `HtmlError::Parse` if the HTML cannot be parsed.
pub fn split_into_sections(
    html: &str,
    strip_boilerplate: bool,
) -> Result<Vec<HtmlSection>, HtmlError> {
    if html.is_empty() {
        return Ok(Vec::new());
    }

    // Extract the text content (with or without boilerplate removal).
    let text = if strip_boilerplate {
        extract_main_content(html)?
    } else {
        extract_visible_text(html)
    };

    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Parse the HTML to find H1/H2 heading text for section boundary detection.
    let document = Html::parse_document(html);
    let headings = extract_heading_positions(&document, &text);

    // If no headings are found, return the entire text as a single section.
    if headings.is_empty() {
        return Ok(vec![HtmlSection {
            section_index: 0,
            heading: None,
            heading_level: None,
            content: text.clone(),
            byte_offset_start: 0,
            byte_offset_end: text.len(),
        }]);
    }

    // Split the text at heading boundaries.
    let mut sections = Vec::new();

    // Preamble section (content before the first heading), if non-empty.
    let first_heading_offset = headings[0].2;
    if first_heading_offset > 0 {
        let preamble = text[..first_heading_offset].trim();
        if !preamble.is_empty() {
            sections.push(HtmlSection {
                section_index: 0,
                heading: None,
                heading_level: None,
                content: preamble.to_string(),
                byte_offset_start: 0,
                byte_offset_end: first_heading_offset,
            });
        }
    }

    // Each heading starts a new section that extends to the next heading
    // (or to the end of the text).
    for (i, (heading_text, heading_level, start_offset)) in headings.iter().enumerate() {
        let end_offset = if i + 1 < headings.len() {
            headings[i + 1].2
        } else {
            text.len()
        };

        let section_content = text[*start_offset..end_offset].trim();
        if section_content.is_empty() {
            continue;
        }

        sections.push(HtmlSection {
            section_index: sections.len(),
            heading: Some(heading_text.clone()),
            heading_level: Some(*heading_level),
            content: section_content.to_string(),
            byte_offset_start: *start_offset,
            byte_offset_end: end_offset,
        });
    }

    // Re-index sections after filtering empty ones.
    for (i, section) in sections.iter_mut().enumerate() {
        section.section_index = i;
    }

    Ok(sections)
}

/// Converts a slice of `HtmlSection` values into `Vec<PageText>` for the chunking pipeline.
/// Section indices (0-based) are mapped to page numbers (1-based). The backend
/// is set based on whether boilerplate removal was applied.
pub fn sections_to_pages(
    sections: &[HtmlSection],
    source_file: &Path,
    backend: ExtractionBackend,
) -> Vec<PageText> {
    sections
        .iter()
        .map(|section| PageText {
            page_number: section.section_index + 1,
            content: section.content.clone(),
            source_file: source_file.to_path_buf(),
            backend,
        })
        .collect()
}

/// Extracts all links (`<a href>`) from an HTML document, resolving relative
/// URLs against the base URL. Deduplicates links and filters out javascript:,
/// mailto:, tel:, and data: URLs.
pub fn extract_links(html: &str, base_url: &str) -> Vec<String> {
    let document = Html::parse_document(html);

    let selector = match Selector::parse("a[href]") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let base = match url::Url::parse(base_url) {
        Ok(u) => u,
        Err(_) => return Vec::new(),
    };

    let mut seen = HashSet::new();
    let mut links = Vec::new();

    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            let href = href.trim();

            // Skip non-HTTP schemes.
            if href.starts_with("javascript:")
                || href.starts_with("mailto:")
                || href.starts_with("tel:")
                || href.starts_with("data:")
                || href.starts_with('#')
            {
                continue;
            }

            // Resolve relative URLs against the base.
            let resolved = match base.join(href) {
                Ok(u) => u.to_string(),
                Err(_) => continue,
            };

            // Remove fragment identifiers for deduplication.
            let without_fragment = if let Some(pos) = resolved.find('#') {
                &resolved[..pos]
            } else {
                &resolved
            };

            if seen.insert(without_fragment.to_string()) {
                links.push(without_fragment.to_string());
            }
        }
    }

    links
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Extracts text from the document by walking all text nodes in document order
/// and filtering out those inside invisible elements (script, style, noscript,
/// template, svg). Block-level parent elements produce newlines in the output.
fn extract_text_from_document(document: &Html) -> String {
    let mut text = String::new();

    // Walk all nodes in document order and collect text from visible elements.
    // Each text node's ancestors are checked against the INVISIBLE_ELEMENTS list
    // to filter out content inside script, style, noscript, template, and svg tags.
    for node in document.tree.nodes() {
        if let scraper::Node::Text(text_node) = node.value() {
            let t = text_node.text.trim();
            if t.is_empty() {
                continue;
            }

            // Check if any ancestor is an invisible element by walking up the tree.
            let is_invisible = node.ancestors().any(|ancestor| {
                if let scraper::Node::Element(el) = ancestor.value() {
                    INVISIBLE_ELEMENTS.contains(&el.name())
                } else {
                    false
                }
            });

            if is_invisible {
                continue;
            }

            // Determine separator: newline for block elements, space for inline.
            if !text.is_empty()
                && let Some(parent) = node.parent()
            {
                if is_block_element(parent.value()) {
                    text.push('\n');
                } else {
                    text.push(' ');
                }
            }
            text.push_str(t);
        }
    }

    text
}

/// Returns true if a DOM node represents a block-level HTML element.
fn is_block_element(node: &scraper::Node) -> bool {
    match node {
        scraper::Node::Element(el) => {
            matches!(
                el.name(),
                "p" | "div"
                    | "h1"
                    | "h2"
                    | "h3"
                    | "h4"
                    | "h5"
                    | "h6"
                    | "br"
                    | "hr"
                    | "li"
                    | "ul"
                    | "ol"
                    | "table"
                    | "tr"
                    | "td"
                    | "th"
                    | "blockquote"
                    | "pre"
                    | "section"
                    | "article"
                    | "aside"
                    | "nav"
                    | "header"
                    | "footer"
                    | "main"
                    | "figure"
                    | "figcaption"
                    | "details"
                    | "summary"
                    | "dt"
                    | "dd"
            )
        }
        _ => false,
    }
}

/// Collapses runs of whitespace: multiple blank lines become at most two
/// newlines, and trailing whitespace on each line is removed.
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut consecutive_newlines = 0u32;

    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                result.push('\n');
            }
        } else {
            if consecutive_newlines > 0
                && !result.is_empty()
                && consecutive_newlines > 1
                && !result.ends_with("\n\n")
            {
                result.push('\n');
            }
            consecutive_newlines = 0;
            if !result.is_empty() && !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(trimmed);
        }
    }

    result
}

/// Scores candidate container elements for the readability algorithm and
/// returns the extracted text from the highest-scoring candidate. Returns
/// None if no suitable candidate is found.
fn extract_best_candidate_text(document: &Html) -> Option<String> {
    let container_selector = Selector::parse("div, article, section, main, td").ok()?;

    let mut best_score: f64 = 0.0;
    let mut best_text: Option<String> = None;

    for element in document.select(&container_selector) {
        let el = element.value();

        // Collect text from this element's subtree.
        let element_text: String = element.text().collect::<Vec<_>>().join(" ");
        let text_len = element_text.trim().len();

        if text_len < 50 {
            continue;
        }

        let mut score = text_len as f64;

        // Count direct child elements.
        let child_element_count = element
            .children()
            .filter(|c| matches!(c.value(), scraper::Node::Element(_)))
            .count();

        // Text density penalty: elements with many children relative to text
        // are likely navigation lists.
        if child_element_count > 0 {
            let density = text_len as f64 / child_element_count as f64;
            if density < 20.0 {
                score *= 0.3;
            }
        }

        // Check class and id attributes for boilerplate/content indicators.
        let class = el.attr("class").unwrap_or("");
        let id = el.attr("id").unwrap_or("");
        let attrs = format!("{class} {id}").to_lowercase();

        for indicator in BOILERPLATE_INDICATORS {
            if attrs.contains(indicator) {
                score *= 0.5;
            }
        }

        for indicator in CONTENT_INDICATORS {
            if attrs.contains(indicator) {
                score *= 1.5;
            }
        }

        // Boost article and main elements.
        match el.name() {
            "article" => score *= 2.0,
            "main" => score *= 1.8,
            _ => {}
        }

        if score > best_score {
            best_score = score;
            best_text = Some(element_text);
        }
    }

    best_text
}

/// Extracts H1/H2 heading positions from the HTML document. Returns a list
/// of (heading_text, heading_level, byte_offset_in_extracted_text) tuples.
/// The byte offsets correspond to positions within the extracted text (not the
/// raw HTML), enabling section boundary detection.
fn extract_heading_positions(document: &Html, extracted_text: &str) -> Vec<(String, u8, usize)> {
    let h1_sel = Selector::parse("h1").ok();
    let h2_sel = Selector::parse("h2").ok();

    let mut headings = Vec::new();

    // Collect all H1 headings with their byte offsets in the extracted text.
    if let Some(ref sel) = h1_sel {
        for element in document.select(sel) {
            let text: String = element
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            if !text.is_empty()
                && let Some(offset) = extracted_text.find(&text)
            {
                headings.push((text, 1u8, offset));
            }
        }
    }

    // Collect all H2 headings with their byte offsets in the extracted text.
    if let Some(ref sel) = h2_sel {
        for element in document.select(sel) {
            let text: String = element
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            if !text.is_empty()
                && let Some(offset) = extracted_text.find(&text)
            {
                headings.push((text, 2u8, offset));
            }
        }
    }

    // Sort by byte offset to get document order.
    headings.sort_by_key(|h| h.2);

    // Deduplicate headings at the same offset.
    headings.dedup_by_key(|h| h.2);

    headings
}

/// Selects the text content of the first element matching the given CSS selector.
fn select_text(document: &Html, selector_str: &str) -> Option<String> {
    let sel = Selector::parse(selector_str).ok()?;
    let element = document.select(&sel).next()?;
    let text: String = element
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Selects the content attribute of a `<meta name="...">` tag.
fn select_meta_content(document: &Html, name: &str) -> Option<String> {
    let selector_str = format!("meta[name=\"{name}\"]");
    let sel = Selector::parse(&selector_str).ok()?;
    let element = document.select(&sel).next()?;
    element
        .value()
        .attr("content")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Selects the content attribute of a `<meta property="...">` tag (Open Graph).
fn select_meta_property(document: &Html, property: &str) -> Option<String> {
    let selector_str = format!("meta[property=\"{property}\"]");
    let sel = Selector::parse(&selector_str).ok()?;
    let element = document.select(&sel).next()?;
    element
        .value()
        .attr("content")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Selects the href attribute of a `<link rel="...">` tag.
fn select_link_href(document: &Html, rel: &str) -> Option<String> {
    let selector_str = format!("link[rel=\"{rel}\"]");
    let sel = Selector::parse(&selector_str).ok()?;
    let element = document.select(&sel).next()?;
    element
        .value()
        .attr("href")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Selects the lang attribute from the `<html>` element.
fn select_html_lang(document: &Html) -> Option<String> {
    let sel = Selector::parse("html").ok()?;
    let element = document.select(&sel).next()?;
    element
        .value()
        .attr("lang")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-HTML-P-001: extract_visible_text extracts text from simple HTML.
    #[test]
    fn t_html_p_001_simple_text() {
        let html = "<html><body><p>Hello World</p></body></html>";
        let text = extract_visible_text(html);
        assert!(
            text.contains("Hello World"),
            "must contain 'Hello World': {text}"
        );
    }

    /// T-HTML-P-002: extract_visible_text preserves paragraph breaks.
    #[test]
    fn t_html_p_002_paragraph_breaks() {
        let html = "<html><body><p>Paragraph one.</p><p>Paragraph two.</p></body></html>";
        let text = extract_visible_text(html);
        assert!(
            text.contains("Paragraph one.") && text.contains("Paragraph two."),
            "must contain both paragraphs: {text}"
        );
    }

    /// T-HTML-P-003: extract_visible_text ignores script and style content.
    #[test]
    fn t_html_p_003_ignores_scripts_styles() {
        let html = r#"<html><head><style>body { color: red; }</style></head>
            <body><script>alert('xss')</script><p>Visible text</p></body></html>"#;
        let text = extract_visible_text(html);
        assert!(text.contains("Visible text"), "must contain 'Visible text'");
        assert!(!text.contains("alert"), "must not contain script content");
        assert!(!text.contains("color"), "must not contain style content");
    }

    /// T-HTML-P-004: extract_visible_text handles HTML entities.
    #[test]
    fn t_html_p_004_html_entities() {
        let html = "<html><body><p>Tom &amp; Jerry &lt;3&gt;</p></body></html>";
        let text = extract_visible_text(html);
        assert!(
            text.contains("Tom & Jerry"),
            "must decode &amp; entity: {text}"
        );
        assert!(
            text.contains("<3>"),
            "must decode &lt; and &gt; entities: {text}"
        );
    }

    /// T-HTML-P-005: extract_visible_text handles empty HTML.
    #[test]
    fn t_html_p_005_empty_html() {
        let text = extract_visible_text("");
        assert!(text.is_empty(), "empty HTML must produce empty text");
    }

    /// T-HTML-P-006: extract_main_content extracts article content.
    #[test]
    fn t_html_p_006_main_content() {
        let html = r#"<html><body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <article class="content">
                <h1>Article Title</h1>
                <p>This is a long paragraph with enough text content to score well in the readability algorithm.
                It contains multiple sentences that form a coherent article body. The text density here is
                high relative to the number of child elements, which boosts the readability score.</p>
            </article>
            <footer>Copyright 2025</footer>
        </body></html>"#;
        let result = extract_main_content(html).expect("extract_main_content failed");
        assert!(
            result.contains("Article Title"),
            "must contain article title: {result}"
        );
        assert!(
            result.contains("readability"),
            "must contain article body text: {result}"
        );
    }

    /// T-HTML-P-008: extract_metadata extracts title from <title> tag.
    #[test]
    fn t_html_p_008_metadata_title() {
        let html = "<html><head><title>My Page Title</title></head><body></body></html>";
        let meta = extract_metadata(html, "https://neuroncite.com", 200, Some("text/html"));
        assert_eq!(meta.title.as_deref(), Some("My Page Title"));
    }

    /// T-HTML-P-009: extract_metadata extracts Open Graph tags.
    #[test]
    fn t_html_p_009_metadata_og_tags() {
        let html = r#"<html><head>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Desc">
            <meta property="og:image" content="https://neuroncite.com/img.jpg">
        </head><body></body></html>"#;
        let meta = extract_metadata(html, "https://neuroncite.com", 200, None);
        assert_eq!(meta.og_title.as_deref(), Some("OG Title"));
        assert_eq!(meta.og_description.as_deref(), Some("OG Desc"));
        assert_eq!(
            meta.og_image.as_deref(),
            Some("https://neuroncite.com/img.jpg")
        );
    }

    /// T-HTML-P-010: extract_metadata extracts language from <html lang>.
    #[test]
    fn t_html_p_010_metadata_language() {
        let html = r#"<html lang="de"><head></head><body></body></html>"#;
        let meta = extract_metadata(html, "https://neuroncite.com", 200, None);
        assert_eq!(meta.language.as_deref(), Some("de"));
    }

    /// T-HTML-P-011: extract_metadata extracts author from <meta name="author">.
    #[test]
    fn t_html_p_011_metadata_author() {
        let html =
            r#"<html><head><meta name="author" content="Jane Smith"></head><body></body></html>"#;
        let meta = extract_metadata(html, "https://neuroncite.com", 200, None);
        assert_eq!(meta.author.as_deref(), Some("Jane Smith"));
    }

    /// T-HTML-P-012: extract_metadata extracts canonical URL.
    #[test]
    fn t_html_p_012_metadata_canonical() {
        let html = r#"<html><head><link rel="canonical" href="https://neuroncite.com/canonical"></head><body></body></html>"#;
        let meta = extract_metadata(html, "https://neuroncite.com/page?ref=abc", 200, None);
        assert_eq!(
            meta.canonical_url.as_deref(),
            Some("https://neuroncite.com/canonical")
        );
    }

    /// T-HTML-P-013: split_into_sections creates one section per H1/H2 heading.
    #[test]
    fn t_html_p_013_section_split_headings() {
        let html = r#"<html><body>
            <h1>First Section</h1><p>Content A</p>
            <h2>Second Section</h2><p>Content B</p>
            <h2>Third Section</h2><p>Content C</p>
        </body></html>"#;
        let sections = split_into_sections(html, false).expect("split failed");
        assert!(
            sections.len() >= 3,
            "must have at least 3 sections, got {}: {:?}",
            sections.len(),
            sections
                .iter()
                .map(|s| s.heading.as_deref())
                .collect::<Vec<_>>()
        );
    }

    /// T-HTML-P-015: split_into_sections handles page with no headings.
    #[test]
    fn t_html_p_015_no_headings() {
        let html = "<html><body><p>Just some text without any headings.</p></body></html>";
        let sections = split_into_sections(html, false).expect("split failed");
        assert_eq!(
            sections.len(),
            1,
            "must have exactly 1 section (no headings)"
        );
        assert!(
            sections[0].heading.is_none(),
            "single section must have no heading"
        );
    }

    /// T-HTML-P-019: sections_to_pages maps section indices to 1-based page numbers.
    #[test]
    fn t_html_p_019_sections_to_pages() {
        let sections = vec![
            HtmlSection {
                section_index: 0,
                heading: None,
                heading_level: None,
                content: "Preamble".into(),
                byte_offset_start: 0,
                byte_offset_end: 8,
            },
            HtmlSection {
                section_index: 1,
                heading: Some("Heading 1".into()),
                heading_level: Some(1),
                content: "Heading 1\nContent".into(),
                byte_offset_start: 8,
                byte_offset_end: 25,
            },
        ];

        let pages = sections_to_pages(
            &sections,
            Path::new("/cache/page.html"),
            ExtractionBackend::HtmlRaw,
        );
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[1].page_number, 2);
        assert_eq!(pages[0].backend, ExtractionBackend::HtmlRaw);
    }

    /// T-HTML-P-020: sections_to_pages sets correct ExtractionBackend.
    #[test]
    fn t_html_p_020_backend_variant() {
        let sections = vec![HtmlSection {
            section_index: 0,
            heading: None,
            heading_level: None,
            content: "Content".into(),
            byte_offset_start: 0,
            byte_offset_end: 7,
        }];

        let pages = sections_to_pages(
            &sections,
            Path::new("/cache/page.html"),
            ExtractionBackend::HtmlReadability,
        );
        assert_eq!(pages[0].backend, ExtractionBackend::HtmlReadability);
    }

    /// T-HTML-P-021: extract_links resolves relative URLs against base URL.
    #[test]
    fn t_html_p_021_extract_links_relative() {
        let html = r#"<html><body><a href="/about">About</a><a href="https://other.com">Other</a></body></html>"#;
        let links = extract_links(html, "https://neuroncite.com/page");
        assert!(
            links.contains(&"https://neuroncite.com/about".to_string()),
            "must resolve relative URL: {links:?}"
        );
        // url::Url normalizes bare domains to include a trailing slash.
        assert!(
            links.contains(&"https://other.com/".to_string()),
            "must include absolute URL: {links:?}"
        );
    }

    /// T-HTML-P-022: extract_links deduplicates links.
    #[test]
    fn t_html_p_022_extract_links_dedup() {
        let html = r#"<html><body>
            <a href="/page">Link 1</a>
            <a href="/page">Link 2</a>
            <a href="/page#section">Link 3</a>
        </body></html>"#;
        let links = extract_links(html, "https://neuroncite.com");
        let page_links: Vec<_> = links.iter().filter(|l| l.contains("/page")).collect();
        assert_eq!(
            page_links.len(),
            1,
            "duplicate links must be deduplicated: {links:?}"
        );
    }

    /// T-HTML-P-023: extract_links ignores javascript: and mailto: URLs.
    #[test]
    fn t_html_p_023_extract_links_filters() {
        let html = r#"<html><body>
            <a href="javascript:void(0)">JS</a>
            <a href="mailto:test@neuroncite.com">Email</a>
            <a href="tel:+1234567890">Phone</a>
            <a href="https://neuroncite.com/real">Real</a>
        </body></html>"#;
        let links = extract_links(html, "https://neuroncite.com");
        assert_eq!(links.len(), 1, "must only include HTTP links: {links:?}");
        assert!(links[0].contains("real"));
    }

    /// T-HTML-P-025: extract_visible_text handles <br> and <hr> as line breaks.
    #[test]
    fn t_html_p_025_br_hr_line_breaks() {
        let html = "<html><body><p>Line one<br>Line two</p><hr><p>After HR</p></body></html>";
        let text = extract_visible_text(html);
        assert!(
            text.contains("Line one") && text.contains("Line two") && text.contains("After HR"),
            "must contain all text segments: {text}"
        );
    }
}
