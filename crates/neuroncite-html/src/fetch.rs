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

// HTTP fetch operations for the HTML scraping pipeline.
//
// Provides functions to fetch web pages via HTTP GET, write raw HTML to a
// disk cache, and extract metadata from HTTP response headers. The cache
// uses SHA-256-hashed filenames to avoid filesystem special character issues
// and to enable deduplication.
//
// All outbound HTTP requests are guarded by SSRF validation (see ssrf.rs)
// which blocks URLs that resolve to private, loopback, link-local, or
// cloud metadata IP addresses.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::error::HtmlError;
use crate::parse;
use crate::ssrf::validate_url_no_ssrf;
use crate::types::{FetchResult, WebMetadata};

/// User-Agent header value sent with all HTTP requests. Identifies the
/// NeuronCite scraper to web servers for robots.txt compliance and logging.
const USER_AGENT: &str = "NeuronCite/0.1 (semantic-search-engine; +https://neuroncite.com/)";

/// HTTP request timeout in seconds. Pages that take longer than this to
/// respond are treated as fetch failures.
const TIMEOUT_SECS: u64 = 30;

/// Maximum number of HTTP redirects to follow before failing.
const MAX_REDIRECTS: usize = 10;

/// Builds a reqwest::Client with a caller-specified timeout. Configured with
/// the NeuronCite User-Agent, the given timeout, and a 10-redirect limit.
/// TLS uses rustls for cross-platform compatibility without OpenSSL
/// dependencies.
///
/// System proxy support: reqwest reads HTTP_PROXY, HTTPS_PROXY, ALL_PROXY,
/// and NO_PROXY environment variables by default. Corporate environments
/// behind a proxy are supported without additional configuration. Do NOT add
/// `.no_proxy()` to this builder -- it would disable automatic proxy detection.
///
/// # Errors
///
/// Returns `HtmlError::Http` if the client builder fails (typically due to
/// TLS backend initialization issues).
pub fn build_http_client_with_timeout(timeout_secs: u64) -> Result<reqwest::Client, HtmlError> {
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
        .build()?;
    Ok(client)
}

/// Builds a default reqwest::Client with a 30-second timeout.
/// Convenience wrapper around `build_http_client_with_timeout` for callers
/// that do not need configurable timeout values.
///
/// See `build_http_client_with_timeout` for proxy support documentation.
pub fn build_http_client() -> Result<reqwest::Client, HtmlError> {
    build_http_client_with_timeout(TIMEOUT_SECS)
}

/// Fetches a single URL via HTTP GET, writes the raw HTML to the cache
/// directory, and returns the raw HTML content, extracted metadata, and
/// cache file path.
///
/// Before making any HTTP request, the URL is validated against SSRF
/// protection rules (see `ssrf::validate_url_no_ssrf`). URLs that resolve
/// to private networks, loopback, link-local, or cloud metadata addresses
/// are rejected with `HtmlError::Ssrf`.
///
/// The cache filename is derived from the SHA-256 hash of the URL to avoid
/// filesystem special character issues and enable deterministic lookups.
///
/// # Arguments
///
/// * `client` - A pre-configured reqwest::Client (created via `build_http_client`).
/// * `url` - The URL to fetch.
/// * `cache_dir` - Directory where raw HTML cache files are stored.
/// * `strip_boilerplate` - When true, metadata extraction notes the intended
///   extraction mode, but the actual text extraction happens in the parse module.
///
/// # Errors
///
/// Returns `HtmlError::Ssrf` when the URL resolves to a blocked IP range.
/// Returns `HtmlError::Http` on HTTP request failures, `HtmlError::Io` on
/// cache file write failures, or `HtmlError::UrlParse` for invalid URLs.
pub async fn fetch_url(
    client: &reqwest::Client,
    url: &str,
    cache_dir: &Path,
    _strip_boilerplate: bool,
) -> Result<FetchResult, HtmlError> {
    // SSRF protection: reject URLs that resolve to private/loopback/link-local IPs.
    validate_url_no_ssrf(url)?;

    debug!(url = url, "fetching web page");

    // Validate the URL before making the request.
    let parsed_url = url::Url::parse(url)?;
    let domain = parsed_url.host_str().unwrap_or("unknown").to_string();

    // Send the HTTP GET request.
    let response = client.get(url).send().await?;
    let http_status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    // Extract the final URL after any redirects.
    let final_url = response.url().to_string();

    if !response.status().is_success() {
        warn!(
            url = url,
            status = http_status,
            "non-success HTTP status code"
        );
    }

    // Read the response body as text.
    let raw_html = response.text().await?;

    // Compute the fetch timestamp.
    let fetch_timestamp = neuroncite_core::unix_timestamp_secs();

    // Extract metadata from the HTML head section and response headers.
    let metadata =
        parse::extract_metadata(&raw_html, &final_url, http_status, content_type.as_deref());

    // Write the raw HTML to the cache directory.
    let cache_path = cache_path_for_url(cache_dir, &final_url);
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        HtmlError::Cache(format!(
            "failed to create cache directory {}: {e}",
            cache_dir.display()
        ))
    })?;
    // Use an atomic write (temp file + rename) so readers never observe a
    // partially written cache file. Cache write failures are non-fatal.
    write_cache_atomic(&cache_path, raw_html.as_bytes()).ok();

    // When the original URL differs from the final URL (redirect occurred),
    // write an additional cache file under the original URL's hash. This
    // ensures that both cache_path_for_url(dir, original_url) and
    // cache_path_for_url(dir, final_url) resolve to valid cached content.
    // Typical case: DOI URLs (https://doi.org/10.1234/...) redirect to the
    // publisher's page; callers who later look up the DOI URL directly will
    // find the cached file.
    let original_cache_path = cache_path_for_url(cache_dir, url);
    if original_cache_path != cache_path {
        if let Err(e) = write_cache_atomic(&original_cache_path, raw_html.as_bytes()) {
            warn!(
                original_url = url,
                path = original_cache_path.display().to_string().as_str(),
                "failed to write redirect alias cache file: {e}"
            );
        } else {
            debug!(
                original_url = url,
                final_url = final_url.as_str(),
                "wrote redirect alias cache file for original URL"
            );
        }
    }

    debug!(
        url = final_url.as_str(),
        cache_path = cache_path.display().to_string().as_str(),
        status = http_status,
        bytes = raw_html.len(),
        "page fetched and cached"
    );

    Ok(FetchResult {
        url: final_url,
        raw_html,
        metadata: WebMetadata {
            url: metadata.url,
            canonical_url: metadata.canonical_url,
            title: metadata.title,
            meta_description: metadata.meta_description,
            language: metadata.language,
            og_image: metadata.og_image,
            og_title: metadata.og_title,
            og_description: metadata.og_description,
            author: metadata.author,
            published_date: metadata.published_date,
            domain,
            fetch_timestamp,
            http_status,
            content_type,
        },
        cache_path,
    })
}

/// Fetches multiple URLs sequentially with a configurable delay between
/// requests to avoid overwhelming the target server. Returns a Vec of
/// results, one per URL. Failed fetches are included as Err variants.
///
/// Each URL is individually validated against SSRF protection before the
/// HTTP request is made (via `fetch_url`).
///
/// # Arguments
///
/// * `client` - A pre-configured reqwest::Client.
/// * `urls` - The URLs to fetch in order.
/// * `cache_dir` - Directory for raw HTML cache files.
/// * `delay_ms` - Delay in milliseconds between consecutive requests.
/// * `strip_boilerplate` - Forwarded to `fetch_url`.
pub async fn fetch_urls(
    client: &reqwest::Client,
    urls: &[&str],
    cache_dir: &Path,
    delay_ms: u64,
    strip_boilerplate: bool,
) -> Vec<Result<FetchResult, HtmlError>> {
    let mut results = Vec::with_capacity(urls.len());

    for (i, url) in urls.iter().enumerate() {
        // Delay between requests (skip delay before the first request).
        if i > 0 && delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }

        let result = fetch_url(client, url, cache_dir, strip_boilerplate).await;
        results.push(result);
    }

    results
}

/// Writes `data` to `path` using a temp file in the same directory followed by
/// a rename. On POSIX systems `rename(2)` is atomic, so readers never observe a
/// partially written file. On Windows, `std::fs::rename` replaces the target
/// atomically when the source and destination are on the same volume, which is
/// the normal case when both are under the same cache directory. If any step
/// fails the partially written temp file is left in the directory but the
/// target file is not corrupted.
///
/// On Unix, the temp file is also given mode 0o600 (owner read-write only) so
/// the cache file is not world-readable on shared systems.
fn write_cache_atomic(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    // The temp file is placed in the same directory as the target so that the
    // rename does not cross filesystem boundaries (which would make it non-atomic
    // on POSIX and would fail on Windows with ERROR_NOT_SAME_DEVICE).
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Restrict the temp file to owner read-write before rename so the final
        // file also has the restricted mode. chmod after rename would leave a
        // window where the file is world-readable.
        std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Computes the cache file path for a URL. The URL is first normalized by
/// stripping known tracking and error query parameters (UTM, Cloudflare
/// cookie-error codes, etc.) before SHA-256 hashing. This ensures that URLs
/// differing only in transient query parameters (e.g., Nature/Springer
/// `?error=cookies_not_supported&code=<uuid>`) map to the same cache file.
pub fn cache_path_for_url(cache_dir: &Path, url: &str) -> PathBuf {
    let normalized = normalize_cache_url(url);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let hash = hasher.finalize();
    let hex = format!("{hash:x}");
    cache_dir.join(format!("{hex}.html"))
}

/// Normalizes a URL for cache key computation by stripping known tracking
/// and error query parameters that vary between requests to the same page.
/// Publishers like Nature and Springer append dynamic parameters such as
/// `error=cookies_not_supported&code=<uuid>` to redirect URLs, causing
/// cache misses on re-fetch. Common analytics parameters (UTM, fbclid,
/// gclid) are also stripped for consistency.
///
/// When all query parameters are removed, the trailing `?` is also dropped.
/// Fragment identifiers (#...) are always removed because they identify
/// client-side anchors that do not affect the served content.
fn normalize_cache_url(url: &str) -> String {
    let mut parsed = match url::Url::parse(url) {
        Ok(u) => u,
        Err(_) => return url.to_string(),
    };

    // Remove fragment identifiers (#section) which are client-side only.
    parsed.set_fragment(None);

    // Query parameter names to strip (lowercase). Sorted alphabetically.
    const STRIP_PARAMS: &[&str] = &[
        "code",
        "error",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "utm_campaign",
        "utm_content",
        "utm_medium",
        "utm_source",
        "utm_term",
    ];

    // If the URL has no query string, return as-is after fragment removal.
    if parsed.query().is_none() {
        return parsed.to_string();
    }

    let filtered_pairs: Vec<(String, String)> = parsed
        .query_pairs()
        .filter(|(key, _)| {
            let key_lower = key.to_lowercase();
            !STRIP_PARAMS.contains(&key_lower.as_str())
        })
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    if filtered_pairs.is_empty() {
        parsed.set_query(None);
    } else {
        let qs = filtered_pairs
            .iter()
            .map(|(k, v)| {
                format!(
                    "{}={}",
                    url::form_urlencoded::byte_serialize(k.as_bytes()).collect::<String>(),
                    url::form_urlencoded::byte_serialize(v.as_bytes()).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("&");
        parsed.set_query(Some(&qs));
    }

    parsed.to_string()
}

/// Returns the default HTML cache directory: `~/.neuroncite/html_cache/`.
/// The directory is created on first use by the fetch functions.
pub fn default_cache_dir() -> PathBuf {
    let home = dirs_path();
    home.join(".neuroncite").join("html_cache")
}

/// Returns the user's home directory path. Falls back to the current
/// directory if the home directory cannot be determined.
fn dirs_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-HTML-F-001: cache_path_for_url produces deterministic SHA-256-based paths.
    /// The same URL always maps to the same cache file path.
    #[test]
    fn t_html_f_001_cache_path_deterministic() {
        let dir = Path::new("/cache");
        let path1 = cache_path_for_url(dir, "https://neuroncite.com/page");
        let path2 = cache_path_for_url(dir, "https://neuroncite.com/page");
        assert_eq!(path1, path2, "same URL must produce same cache path");
        assert!(
            path1.to_string_lossy().ends_with(".html"),
            "cache path must end with .html"
        );
    }

    /// T-HTML-F-002: cache_path_for_url produces different paths for different URLs.
    #[test]
    fn t_html_f_002_cache_path_different_urls() {
        let dir = Path::new("/cache");
        let path1 = cache_path_for_url(dir, "https://neuroncite.com/page1");
        let path2 = cache_path_for_url(dir, "https://neuroncite.com/page2");
        assert_ne!(
            path1, path2,
            "different URLs must produce different cache paths"
        );
    }

    /// T-HTML-F-003: default_cache_dir returns a path ending with html_cache.
    #[test]
    fn t_html_f_003_default_cache_dir() {
        let dir = default_cache_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("html_cache"),
            "default cache dir must contain 'html_cache': {dir_str}"
        );
    }

    /// T-HTML-F-008: build_http_client returns a client without error.
    #[test]
    fn t_html_f_008_build_http_client() {
        let result = build_http_client();
        assert!(result.is_ok(), "build_http_client must succeed");
    }

    // --- Bug 1: redirect alias cache ---

    /// T-HTML-F-009: When the original URL differs from the final (redirected)
    /// URL, cache_path_for_url produces different paths for each. Both must
    /// be written by fetch_url to enable lookups via either URL.
    #[test]
    fn t_html_f_009_redirect_produces_different_cache_paths() {
        let dir = Path::new("/cache");
        let original = cache_path_for_url(dir, "https://doi.org/10.1234/abc");
        let final_url = cache_path_for_url(dir, "https://publisher.com/articles/abc");
        assert_ne!(
            original, final_url,
            "original and final URLs must produce different cache paths"
        );
    }

    // --- Bug 3: URL normalization for cache keys ---

    /// T-HTML-F-010: cache_path_for_url produces the same path for URLs that
    /// differ only in stripped tracking parameters (UTM, error/code).
    #[test]
    fn t_html_f_010_cache_path_ignores_tracking_params() {
        let dir = Path::new("/cache");
        let base = "https://www.nature.com/articles/nphys1170";
        let with_params =
            "https://www.nature.com/articles/nphys1170?error=cookies_not_supported&code=abc123";
        let path_base = cache_path_for_url(dir, base);
        let path_params = cache_path_for_url(dir, with_params);
        assert_eq!(
            path_base, path_params,
            "URLs differing only in stripped tracking params must produce the same cache path"
        );
    }

    /// T-HTML-F-011: cache_path_for_url preserves meaningful query parameters
    /// that are not in the strip list.
    #[test]
    fn t_html_f_011_cache_path_preserves_meaningful_params() {
        let dir = Path::new("/cache");
        let without = "https://example.com/article";
        let with_page = "https://example.com/article?page=2";
        let path1 = cache_path_for_url(dir, without);
        let path2 = cache_path_for_url(dir, with_page);
        assert_ne!(
            path1, path2,
            "URLs with meaningful query params must produce different cache paths"
        );
    }

    /// T-HTML-F-012: normalize_cache_url strips UTM parameters but preserves
    /// other query parameters.
    #[test]
    fn t_html_f_012_normalize_strips_utm() {
        let url = "https://example.com/page?utm_source=twitter&utm_medium=social&id=42";
        let normalized = normalize_cache_url(url);
        assert!(
            normalized.contains("id=42"),
            "meaningful param 'id' must be preserved: {normalized}"
        );
        assert!(
            !normalized.contains("utm_source"),
            "utm_source must be stripped: {normalized}"
        );
        assert!(
            !normalized.contains("utm_medium"),
            "utm_medium must be stripped: {normalized}"
        );
    }

    /// T-HTML-F-013: normalize_cache_url removes the query string entirely
    /// when all parameters are in the strip list.
    #[test]
    fn t_html_f_013_normalize_removes_empty_query() {
        let url = "https://example.com/page?error=cookies_not_supported&code=uuid123";
        let normalized = normalize_cache_url(url);
        assert_eq!(
            normalized, "https://example.com/page",
            "URL must have no query string after all params are stripped"
        );
    }

    /// T-HTML-F-014: normalize_cache_url removes fragment identifiers since
    /// they are client-side only and do not affect the served content.
    #[test]
    fn t_html_f_014_normalize_removes_fragment() {
        let url = "https://example.com/page#section-3";
        let normalized = normalize_cache_url(url);
        assert_eq!(
            normalized, "https://example.com/page",
            "fragment identifier must be removed"
        );
    }

    /// T-HTML-F-015: normalize_cache_url handles URLs without query or fragment.
    #[test]
    fn t_html_f_015_normalize_plain_url_unchanged() {
        let url = "https://example.com/article/123";
        let normalized = normalize_cache_url(url);
        // url::Url always appends a trailing slash for path-only URLs, so
        // we just check the essential content is preserved.
        assert!(
            normalized.starts_with("https://example.com/article/123"),
            "plain URL must be preserved: {normalized}"
        );
    }

    /// T-HTML-F-016: normalize_cache_url handles multiple cookie-error and
    /// tracking params from Nature/Springer combined URLs.
    #[test]
    fn t_html_f_016_normalize_combined_nature_springer() {
        let url = "https://www.nature.com/articles/s41586-020-2314-9?error=cookies_not_supported&code=a1b2c3d4&utm_source=google&utm_medium=organic";
        let normalized = normalize_cache_url(url);
        assert_eq!(
            normalized, "https://www.nature.com/articles/s41586-020-2314-9",
            "all tracking and error params must be stripped"
        );
    }
}
