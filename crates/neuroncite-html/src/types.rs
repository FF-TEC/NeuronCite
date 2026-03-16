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

// Data types for the HTML scraping and parsing pipeline.
//
// Defines the metadata, fetch result, section, and crawl configuration
// structures used throughout the neuroncite-html crate. These types flow
// through the fetch -> parse -> section-split -> chunk pipeline and are
// stored in the database via the neuroncite-store web_source table.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata extracted from an HTML page's `<head>` section and HTTP response
/// headers. Stored alongside the indexed file record in the web_source table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebMetadata {
    /// The URL that was fetched (after any redirects).
    pub url: String,
    /// Canonical URL from `<link rel="canonical">`, if present.
    pub canonical_url: Option<String>,
    /// Page title from the `<title>` tag.
    pub title: Option<String>,
    /// Content of `<meta name="description">`.
    pub meta_description: Option<String>,
    /// Language code from `<html lang="...">` or Content-Language header.
    pub language: Option<String>,
    /// Open Graph image URL from `<meta property="og:image">`.
    pub og_image: Option<String>,
    /// Open Graph title from `<meta property="og:title">`.
    pub og_title: Option<String>,
    /// Open Graph description from `<meta property="og:description">`.
    pub og_description: Option<String>,
    /// Author from `<meta name="author">`.
    pub author: Option<String>,
    /// Published date from `<meta name="article:published_time">` or similar.
    pub published_date: Option<String>,
    /// Domain name extracted from the URL (e.g., "example.com").
    pub domain: String,
    /// Unix timestamp (seconds) when the page was fetched.
    pub fetch_timestamp: i64,
    /// HTTP status code of the response (e.g., 200, 301, 404).
    pub http_status: u16,
    /// Content-Type header value from the HTTP response.
    pub content_type: Option<String>,
}

/// Result of fetching a single URL. Combines the raw HTML content, extracted
/// metadata, and the local cache file path where the raw HTML is stored.
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// The URL that was fetched (after redirects).
    pub url: String,
    /// The raw HTML content as a UTF-8 string.
    pub raw_html: String,
    /// Metadata extracted from the HTTP response and HTML head section.
    pub metadata: WebMetadata,
    /// Path to the local cache file where the raw HTML is written.
    pub cache_path: PathBuf,
}

/// A logical section extracted from an HTML document. Each H1/H2 heading
/// starts a new section, analogous to how PDF pages divide a document.
/// Content before the first heading becomes section 0 (converted to page
/// number 1 when mapped to PageText).
#[derive(Debug, Clone)]
pub struct HtmlSection {
    /// 0-indexed section position within the document. Converted to 1-indexed
    /// page_number when mapped to the PageText pipeline type.
    pub section_index: usize,
    /// The H1/H2 heading text that starts this section. None for the preamble
    /// section (content before the first heading).
    pub heading: Option<String>,
    /// The heading level: 1 for H1, 2 for H2. None for the preamble section.
    pub heading_level: Option<u8>,
    /// Full text content of this section, including the heading line if present.
    pub content: String,
    /// Byte offset of this section's start within the full extracted text.
    pub byte_offset_start: usize,
    /// Byte offset of this section's end within the full extracted text.
    pub byte_offset_end: usize,
}

/// Configuration for a website crawl operation. Controls link-following depth,
/// domain restrictions, URL filtering, rate limiting, and content extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlConfig {
    /// Starting URL for the crawl.
    pub start_url: String,
    /// Maximum link-following depth. 0 = only the start URL, 1 = start URL +
    /// pages linked from it, etc.
    pub max_depth: usize,
    /// When true, only follow links to the same domain as the start URL.
    pub same_domain_only: bool,
    /// Optional regex pattern to filter which URLs are followed. Only URLs
    /// matching this pattern are fetched. Applied after same_domain filtering.
    pub url_pattern: Option<String>,
    /// Maximum total pages to fetch before stopping the crawl.
    pub max_pages: usize,
    /// Delay in milliseconds between consecutive HTTP requests. Prevents
    /// overwhelming the target server.
    pub delay_ms: u64,
    /// When true, fetch and parse sitemap.xml for URL discovery instead of
    /// following links in page content.
    pub use_sitemap: bool,
    /// When true, apply the readability algorithm to remove navigation, ads,
    /// sidebars, and other boilerplate content from extracted text.
    pub strip_boilerplate: bool,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            start_url: String::new(),
            max_depth: 1,
            same_domain_only: true,
            url_pattern: None,
            max_pages: 50,
            delay_ms: 500,
            use_sitemap: false,
            strip_boilerplate: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CrawlConfig::default() produces the documented default values.
    #[test]
    fn crawl_config_defaults() {
        let cfg = CrawlConfig::default();
        assert!(cfg.start_url.is_empty());
        assert_eq!(cfg.max_depth, 1);
        assert!(cfg.same_domain_only);
        assert!(cfg.url_pattern.is_none());
        assert_eq!(cfg.max_pages, 50);
        assert_eq!(cfg.delay_ms, 500);
        assert!(!cfg.use_sitemap);
        assert!(cfg.strip_boilerplate);
    }
}
