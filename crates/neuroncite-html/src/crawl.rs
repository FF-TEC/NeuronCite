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

// Website crawling operations.
//
// Provides configurable website crawling with BFS link-following, same-domain
// filtering, URL pattern matching, sitemap-based discovery, and rate limiting.
// Each fetched page is written to the HTML cache directory and returned with
// its metadata for downstream indexing.
//
// All outbound HTTP requests are guarded by SSRF validation (see ssrf.rs).
// The crawl entry point validates the start URL before beginning the BFS
// traversal, and each individual page fetch goes through fetch_url which
// performs its own SSRF check. The fetch_sitemap function validates the
// constructed sitemap URL before making a direct HTTP request.

use std::collections::{HashSet, VecDeque};
use std::path::Path;

use regex::RegexBuilder;
use tracing::{debug, info, warn};

use crate::error::HtmlError;
use crate::fetch::{build_http_client, fetch_url};
use crate::parse::extract_links;
use crate::ssrf::validate_url_no_ssrf;
use crate::types::{CrawlConfig, FetchResult};

/// Crawls a website starting from the given URL according to the CrawlConfig.
/// Uses breadth-first search (BFS) to discover linked pages up to the configured
/// depth and page limit. Each fetched page is cached to disk and returned with
/// its metadata.
///
/// The start URL is validated against SSRF protection before crawling begins.
/// Each individual page fetch (via `fetch_url`) performs its own SSRF check,
/// so discovered links that point to private/loopback/link-local addresses
/// are rejected at fetch time.
///
/// When `use_sitemap` is true, the crawl fetches `sitemap.xml` from the
/// start URL's domain and uses it for URL discovery instead of link-following.
///
/// # Arguments
///
/// * `config` - Crawl parameters (depth, domain filter, URL pattern, limits).
/// * `cache_dir` - Directory where fetched HTML pages are cached.
///
/// # Errors
///
/// Returns `HtmlError::Ssrf` if the start URL resolves to a blocked IP range.
/// Returns `HtmlError::InvalidArgument` for invalid configuration (e.g., empty
/// start URL), `HtmlError::Http` for HTTP failures, or `HtmlError::Crawl` for
/// crawl-specific issues.
pub async fn crawl(config: &CrawlConfig, cache_dir: &Path) -> Result<Vec<FetchResult>, HtmlError> {
    if config.start_url.is_empty() {
        return Err(HtmlError::InvalidArgument(
            "start_url must not be empty".to_string(),
        ));
    }

    if config.max_pages == 0 {
        return Err(HtmlError::InvalidArgument(
            "max_pages must be greater than 0".to_string(),
        ));
    }

    // SSRF protection: validate the start URL before initiating any requests.
    // Individual page fetches also perform SSRF checks via fetch_url, but
    // checking here provides an early rejection with a clear error message.
    validate_url_no_ssrf(&config.start_url)?;

    let client = build_http_client()?;

    // Parse the start URL to extract the domain for same-domain filtering.
    let start_parsed = url::Url::parse(&config.start_url)?;
    let start_domain = start_parsed.host_str().unwrap_or("").to_string();

    // Compile the URL pattern regex if provided. The NFA size is capped at
    // 10,000 states to prevent catastrophic backtracking (ReDoS) from
    // user-supplied patterns that produce exponential-time NFA execution.
    // Standard URL-matching patterns (e.g., ".*\\.pdf$", "/blog/.*") compile
    // well within this limit; only adversarial patterns are rejected.
    let url_regex = match &config.url_pattern {
        Some(pattern) => Some(RegexBuilder::new(pattern).size_limit(10_000).build()?),
        None => None,
    };

    if config.use_sitemap {
        // Sitemap-based crawl: fetch sitemap.xml and process its URLs.
        return crawl_via_sitemap(
            &client,
            &start_parsed,
            &start_domain,
            config,
            &url_regex,
            cache_dir,
        )
        .await;
    }

    // BFS link-following crawl.
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut results: Vec<FetchResult> = Vec::new();

    // Seed the queue with the start URL at depth 0.
    let normalized_start = normalize_url(&config.start_url)?;
    visited.insert(normalized_start.clone());
    queue.push_back((config.start_url.clone(), 0));

    info!(
        start_url = config.start_url.as_str(),
        max_depth = config.max_depth,
        max_pages = config.max_pages,
        same_domain = config.same_domain_only,
        "starting BFS crawl"
    );

    while let Some((url, depth)) = queue.pop_front() {
        if results.len() >= config.max_pages {
            info!(
                fetched = results.len(),
                max = config.max_pages,
                "reached max_pages limit"
            );
            break;
        }

        // Rate limiting: delay between consecutive requests.
        if !results.is_empty() && config.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
        }

        debug!(url = url.as_str(), depth = depth, "fetching page");

        match fetch_url(&client, &url, cache_dir, config.strip_boilerplate).await {
            Ok(result) => {
                // Extract links from the fetched page for BFS expansion.
                if depth < config.max_depth {
                    let links = extract_links(&result.raw_html, &result.url);

                    for link in links {
                        let normalized = match normalize_url(&link) {
                            Ok(n) => n,
                            Err(_) => continue,
                        };

                        // Skip already-visited URLs.
                        if visited.contains(&normalized) {
                            continue;
                        }

                        // Same-domain filter.
                        if config.same_domain_only {
                            let link_domain = url::Url::parse(&link)
                                .ok()
                                .and_then(|u| u.host_str().map(String::from))
                                .unwrap_or_default();
                            if link_domain != start_domain {
                                continue;
                            }
                        }

                        // URL pattern filter: skip links that do not match the regex.
                        if let Some(regex) = &url_regex
                            && !regex.is_match(&link)
                        {
                            continue;
                        }

                        visited.insert(normalized);
                        queue.push_back((link, depth + 1));
                    }
                }

                results.push(result);
            }
            Err(e) => {
                warn!(url = url.as_str(), error = %e, "failed to fetch page during crawl");
            }
        }
    }

    info!(
        fetched = results.len(),
        visited = visited.len(),
        "crawl completed"
    );

    Ok(results)
}

/// Fetches and parses a sitemap.xml file, returning all URLs found in it.
/// Handles both standard sitemap XML format and sitemap index files
/// (which contain references to other sitemaps).
///
/// The constructed sitemap URL is validated against SSRF protection before
/// the HTTP request is made.
///
/// # Arguments
///
/// * `client` - A pre-configured reqwest::Client.
/// * `base_url` - The base URL of the website (e.g., `https://neuroncite.com`).
///
/// # Errors
///
/// Returns `HtmlError::Ssrf` if the sitemap URL resolves to a blocked IP range.
/// Returns `HtmlError::Http` on fetch failures or `HtmlError::Parse` if
/// the sitemap XML is malformed.
pub async fn fetch_sitemap(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<String>, HtmlError> {
    let parsed = url::Url::parse(base_url)?;
    let sitemap_url = format!(
        "{}://{}/sitemap.xml",
        parsed.scheme(),
        parsed.host_str().unwrap_or("")
    );

    // SSRF protection: validate the sitemap URL before making the HTTP request.
    // This function constructs the sitemap URL from the base URL, so the SSRF
    // check prevents crawling sitemaps from private/internal hosts.
    validate_url_no_ssrf(&sitemap_url)?;

    debug!(url = sitemap_url.as_str(), "fetching sitemap.xml");

    let response = client.get(&sitemap_url).send().await?;
    if !response.status().is_success() {
        return Err(HtmlError::Crawl(format!(
            "sitemap.xml returned HTTP {}",
            response.status()
        )));
    }

    let body = response.text().await?;
    let urls = parse_sitemap_xml(&body);

    debug!(urls_found = urls.len(), "parsed sitemap.xml");

    Ok(urls)
}

/// Normalizes a URL for deduplication: lowercases the scheme and host, removes
/// the fragment identifier, removes the trailing slash (unless the path is "/"),
/// and sorts query parameters alphabetically.
///
/// # Errors
///
/// Returns `HtmlError::UrlParse` if the URL cannot be parsed.
pub fn normalize_url(url_str: &str) -> Result<String, HtmlError> {
    let mut parsed = url::Url::parse(url_str)?;

    // Remove fragment.
    parsed.set_fragment(None);

    // Sort query parameters.
    let query_pairs: Vec<(String, String)> = parsed
        .query_pairs()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    if query_pairs.is_empty() {
        parsed.set_query(None);
    } else {
        let mut sorted = query_pairs;
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let query_string: String = sorted
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    k.clone()
                } else {
                    format!("{k}={v}")
                }
            })
            .collect::<Vec<_>>()
            .join("&");
        parsed.set_query(Some(&query_string));
    }

    let mut result = parsed.to_string();

    // Remove trailing slash (unless path is exactly "/").
    if result.ends_with('/') && parsed.path() != "/" {
        result.pop();
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Crawls via sitemap.xml: fetches the sitemap, filters URLs, and fetches
/// each matching URL sequentially with delay.
async fn crawl_via_sitemap(
    client: &reqwest::Client,
    start_parsed: &url::Url,
    start_domain: &str,
    config: &CrawlConfig,
    url_regex: &Option<regex::Regex>,
    cache_dir: &Path,
) -> Result<Vec<FetchResult>, HtmlError> {
    let urls = fetch_sitemap(client, start_parsed.as_str()).await?;

    let mut results = Vec::new();

    for url in urls {
        if results.len() >= config.max_pages {
            break;
        }

        // Same-domain filter.
        if config.same_domain_only {
            let link_domain = url::Url::parse(&url)
                .ok()
                .and_then(|u| u.host_str().map(String::from))
                .unwrap_or_default();
            if link_domain != start_domain {
                continue;
            }
        }

        // URL pattern filter: skip URLs that do not match the regex.
        if let Some(regex) = &url_regex
            && !regex.is_match(&url)
        {
            continue;
        }

        // Rate limiting.
        if !results.is_empty() && config.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
        }

        match fetch_url(client, &url, cache_dir, config.strip_boilerplate).await {
            Ok(result) => results.push(result),
            Err(e) => {
                warn!(url = url.as_str(), error = %e, "failed to fetch sitemap URL");
            }
        }
    }

    Ok(results)
}

/// Parses a sitemap XML document and extracts all `<loc>` URL values.
/// Handles both `<urlset>` (standard sitemap) and `<sitemapindex>` formats.
fn parse_sitemap_xml(xml: &str) -> Vec<String> {
    let mut urls = Vec::new();

    // Simple XML parsing: extract text content of all <loc> elements.
    // This avoids a full XML parser dependency. The pattern matches
    // <loc>...</loc> across the sitemap format.
    let mut remaining = xml;
    while let Some(start) = remaining.find("<loc>") {
        let after_tag = &remaining[start + 5..];
        if let Some(end) = after_tag.find("</loc>") {
            let url = after_tag[..end].trim();
            if !url.is_empty() {
                urls.push(url.to_string());
            }
            remaining = &after_tag[end + 6..];
        } else {
            break;
        }
    }

    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-HTML-C-001: normalize_url handles various URL edge cases.
    #[test]
    fn t_html_c_001_normalize_url() {
        // Strips fragment.
        let normalized = normalize_url("https://neuroncite.com/page#section").unwrap();
        assert!(
            !normalized.contains('#'),
            "must strip fragment: {normalized}"
        );

        // Removes trailing slash.
        let normalized = normalize_url("https://neuroncite.com/page/").unwrap();
        assert!(
            !normalized.ends_with('/'),
            "must remove trailing slash: {normalized}"
        );

        // Preserves root path slash.
        let normalized = normalize_url("https://neuroncite.com/").unwrap();
        assert!(
            normalized.ends_with('/'),
            "must preserve root path slash: {normalized}"
        );

        // Sorts query parameters.
        let normalized = normalize_url("https://neuroncite.com/page?z=1&a=2").unwrap();
        assert!(
            normalized.contains("a=2&z=1"),
            "must sort query params: {normalized}"
        );
    }

    /// T-HTML-C-002: parse_sitemap_xml extracts URLs from valid sitemap XML.
    #[test]
    fn t_html_c_002_parse_sitemap() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://neuroncite.com/page1</loc></url>
            <url><loc>https://neuroncite.com/page2</loc></url>
        </urlset>"#;

        let urls = parse_sitemap_xml(xml);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://neuroncite.com/page1");
        assert_eq!(urls[1], "https://neuroncite.com/page2");
    }

    /// T-HTML-C-003: CrawlConfig::default has correct default values.
    #[test]
    fn t_html_c_003_crawl_config_default() {
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

    /// normalize_url lowercases scheme and host.
    #[test]
    fn normalize_url_lowercases() {
        let normalized = normalize_url("HTTPS://NEURONCITE.COM/Page").unwrap();
        assert!(
            normalized.starts_with("https://neuroncite.com"),
            "must lowercase scheme and host: {normalized}"
        );
    }

    /// normalize_url returns error for invalid URLs.
    #[test]
    fn normalize_url_invalid() {
        let result = normalize_url("not a url");
        assert!(result.is_err(), "must return error for invalid URL");
    }

    /// parse_sitemap_xml handles empty XML.
    #[test]
    fn parse_sitemap_empty() {
        let urls = parse_sitemap_xml("");
        assert!(urls.is_empty());
    }

    /// parse_sitemap_xml handles sitemap index format.
    #[test]
    fn parse_sitemap_index() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://neuroncite.com/sitemap1.xml</loc></sitemap>
            <sitemap><loc>https://neuroncite.com/sitemap2.xml</loc></sitemap>
        </sitemapindex>"#;

        let urls = parse_sitemap_xml(xml);
        assert_eq!(urls.len(), 2);
    }

    /// T-HTML-C-004: crawl rejects a start URL that resolves to a private IP.
    #[tokio::test]
    async fn t_html_c_004_crawl_ssrf_rejection() {
        let config = CrawlConfig {
            start_url: "http://192.168.1.1/".to_string(),
            max_depth: 1,
            same_domain_only: true,
            url_pattern: None,
            max_pages: 10,
            delay_ms: 0,
            use_sitemap: false,
            strip_boilerplate: true,
        };
        let result = crawl(&config, std::path::Path::new("/tmp/cache")).await;
        assert!(result.is_err(), "crawl with private IP start URL must fail");
        let err = result.unwrap_err();
        assert!(
            matches!(err, HtmlError::Ssrf(_)),
            "error must be HtmlError::Ssrf, got: {err:?}"
        );
    }

    /// T-SEC-008: A standard URL-matching pattern compiles within the NFA size
    /// limit imposed by RegexBuilder. This verifies that typical user-supplied
    /// patterns for crawl filtering are not rejected by the ReDoS protection.
    #[test]
    fn t_sec_008_normal_pattern_accepted() {
        let normal = r".*\.pdf$";
        let result = RegexBuilder::new(normal).size_limit(10_000).build();
        assert!(
            result.is_ok(),
            "standard URL pattern must compile within the 10,000 state limit"
        );
    }
}
