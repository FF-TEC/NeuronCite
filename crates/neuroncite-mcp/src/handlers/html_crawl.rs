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

//! Handler for the `neuroncite_html_crawl` MCP tool.
//!
//! Crawls a website starting from a given URL using either breadth-first
//! link-following or sitemap-based URL discovery. Each discovered page is
//! fetched via HTTP GET and cached to disk. Returns metadata for all fetched
//! pages, enabling subsequent indexing via the `neuroncite_html_index` tool.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_html::CrawlConfig;

/// Crawls a website and returns metadata for all fetched pages.
///
/// # SSRF Protection
///
/// The start URL and every discovered page URL are validated against the SSRF
/// protection in neuroncite_html::ssrf before any HTTP request is made. URLs
/// that resolve to private, loopback, link-local, or cloud metadata IP addresses
/// are rejected. See neuroncite_html::ssrf::validate_url_no_ssrf for details.
///
/// # Parameters (from MCP tool call)
///
/// - `start_url` (required): The seed URL where the crawl begins.
/// - `max_depth` (optional): Maximum link-following depth from the start URL.
///   0 means only the start URL itself. Defaults to 1.
/// - `same_domain_only` (optional): When true, only follow links to the same
///   domain as the start URL. Defaults to true.
/// - `url_pattern` (optional): Regex pattern to filter which discovered URLs
///   are fetched. Applied after same-domain filtering.
/// - `max_pages` (optional): Maximum total pages to fetch before stopping.
///   Defaults to 50.
/// - `delay_ms` (optional): Milliseconds between consecutive HTTP requests.
///   Defaults to 500.
/// - `use_sitemap` (optional): When true, fetch sitemap.xml for URL discovery
///   instead of following links in page content. Defaults to false.
/// - `strip_boilerplate` (optional): Whether to apply readability-based
///   boilerplate removal. Defaults to true.
pub async fn handle(
    _state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let start_url = params["start_url"]
        .as_str()
        .ok_or("missing required parameter: start_url")?;

    let config = CrawlConfig {
        start_url: start_url.to_string(),
        max_depth: params["max_depth"].as_u64().unwrap_or(1) as usize,
        same_domain_only: params["same_domain_only"].as_bool().unwrap_or(true),
        url_pattern: params["url_pattern"].as_str().map(String::from),
        max_pages: params["max_pages"].as_u64().unwrap_or(50) as usize,
        delay_ms: params["delay_ms"].as_u64().unwrap_or(500),
        use_sitemap: params["use_sitemap"].as_bool().unwrap_or(false),
        strip_boilerplate: params["strip_boilerplate"].as_bool().unwrap_or(true),
    };

    let cache_dir = neuroncite_html::default_cache_dir();

    let results = neuroncite_html::crawl(&config, &cache_dir)
        .await
        .map_err(|e| format!("crawl failed: {e}"))?;

    // Build the response array with metadata for each fetched page.
    let page_items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "url": r.url,
                "cache_path": r.cache_path.display().to_string(),
                "http_status": r.metadata.http_status,
                "title": r.metadata.title,
                "domain": r.metadata.domain,
                "content_type": r.metadata.content_type,
                "language": r.metadata.language,
                "author": r.metadata.author,
                "meta_description": r.metadata.meta_description,
                "fetch_timestamp": r.metadata.fetch_timestamp,
                "html_bytes": r.raw_html.len(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "start_url": start_url,
        "pages_fetched": page_items.len(),
        "max_depth": config.max_depth,
        "same_domain_only": config.same_domain_only,
        "use_sitemap": config.use_sitemap,
        "results": page_items,
    }))
}
