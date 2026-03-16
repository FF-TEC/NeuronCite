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

//! Handler for the `neuroncite_html_fetch` MCP tool.
//!
//! Fetches one or more web pages via HTTP GET and caches the raw HTML to disk.
//! Accepts either a single URL string or an array of URLs. Each fetched page
//! is cached using a SHA-256-based filename in the default HTML cache directory.
//! Returns a JSON array containing the metadata and cache path for each URL.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Fetches web pages via HTTP GET, caches the raw HTML to disk, and returns
/// per-URL metadata including title, domain, HTTP status, and cache file path.
///
/// # SSRF Protection
///
/// All URLs are validated against the SSRF protection in neuroncite_html::ssrf
/// before any HTTP request is made. URLs that resolve to private networks
/// (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), loopback (127.0.0.0/8),
/// link-local (169.254.0.0/16 including cloud metadata at 169.254.169.254),
/// or non-HTTP schemes (file://, ftp://) are rejected with an error.
///
/// # Parameters (from MCP tool call)
///
/// - `urls` (optional): JSON array of URL strings to fetch. When both `url`
///   and `urls` are provided, the single `url` is merged into the `urls` list
///   (deduplicated).
/// - `url` (optional): Single URL string to fetch. When `urls` is also provided,
///   this URL is appended to the array if not already present.
///   At least one of `url` or `urls` must be provided.
/// - `strip_boilerplate` (optional): Whether to apply readability-based boilerplate
///   removal during metadata extraction. Defaults to true.
/// - `delay_ms` (optional): Delay in milliseconds between consecutive HTTP
///   requests when fetching multiple URLs. Defaults to 500.
pub async fn handle(
    _state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    // Resolve the URL list from either the `urls` array parameter or the
    // single `url` string parameter.
    let url_list: Vec<String> = if let Some(arr) = params["urls"].as_array() {
        let mut list: Vec<String> = arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| format!("urls[{i}] is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // When both `url` and `urls` are provided, merge the single URL into
        // the list rather than silently discarding it. Deduplication prevents
        // fetching the same URL twice if it appears in both parameters.
        if let Some(single) = params["url"].as_str()
            && !list.iter().any(|u| u == single)
        {
            list.push(single.to_string());
        }
        list
    } else if let Some(single) = params["url"].as_str() {
        vec![single.to_string()]
    } else {
        return Err(
            "missing required parameter: provide either 'url' (string) or 'urls' (array)"
                .to_string(),
        );
    };

    if url_list.is_empty() {
        return Err("urls array must not be empty".to_string());
    }

    let strip_boilerplate = params["strip_boilerplate"].as_bool().unwrap_or(true);
    let delay_ms = params["delay_ms"].as_u64().unwrap_or(500);

    let client = neuroncite_html::build_http_client()
        .map_err(|e| format!("HTTP client initialization failed: {e}"))?;
    let cache_dir = neuroncite_html::default_cache_dir();

    // Build a slice-of-str references for fetch_urls.
    let url_refs: Vec<&str> = url_list.iter().map(String::as_str).collect();

    let results =
        neuroncite_html::fetch_urls(&client, &url_refs, &cache_dir, delay_ms, strip_boilerplate)
            .await;

    // Build the response array with one entry per URL. Failed fetches are
    // reported as inline error objects rather than failing the entire batch.
    let mut response_items: Vec<serde_json::Value> = Vec::with_capacity(results.len());

    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(fetch_result) => {
                response_items.push(serde_json::json!({
                    "url": fetch_result.url,
                    "cache_path": fetch_result.cache_path.display().to_string(),
                    "http_status": fetch_result.metadata.http_status,
                    "title": fetch_result.metadata.title,
                    "domain": fetch_result.metadata.domain,
                    "content_type": fetch_result.metadata.content_type,
                    "language": fetch_result.metadata.language,
                    "author": fetch_result.metadata.author,
                    "meta_description": fetch_result.metadata.meta_description,
                    "fetch_timestamp": fetch_result.metadata.fetch_timestamp,
                    "html_bytes": fetch_result.raw_html.len(),
                }));
            }
            Err(e) => {
                response_items.push(serde_json::json!({
                    "url": url_list[i],
                    "error": format!("{e}"),
                }));
            }
        }
    }

    Ok(serde_json::json!({
        "fetched": response_items.iter().filter(|v| v.get("error").is_none()).count(),
        "failed": response_items.iter().filter(|v| v.get("error").is_some()).count(),
        "results": response_items,
    }))
}
