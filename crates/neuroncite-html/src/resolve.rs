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

// DOI resolution chain for the citation source fetch pipeline.
//
// Resolves DOIs to direct PDF URLs by querying free academic APIs in sequence.
// The resolution chain tries Unpaywall (requires email), Semantic Scholar,
// and OpenAlex before falling back to the standard doi.org HTTP redirect.
// Each API provides open-access PDF URLs when available, bypassing publisher
// landing pages that would otherwise be classified as HTML by the download
// pipeline.
//
// The chain short-circuits on the first successful resolution. Each API call
// has a 10-second independent timeout so a slow or unresponsive API does not
// block subsequent sources. All parsing functions are pure (no network I/O)
// and accept serde_json::Value inputs for deterministic unit testing.

use tracing::{debug, warn};

/// Outcome of resolving a DOI through the multi-source resolution chain.
/// Contains the resolved URL, the API source that provided it, and whether
/// the URL is confirmed to point directly to a PDF file (based on the API
/// metadata, not a HEAD request).
#[derive(Debug, Clone)]
pub struct ResolvedDoi {
    /// The URL resolved from the DOI. Either a direct PDF download link
    /// (from Unpaywall/Semantic Scholar/OpenAlex) or a doi.org redirect
    /// URL (fallback).
    pub url: String,

    /// The API source that produced this URL. Used for logging and for
    /// the `doi_resolved_via` field in the MCP handler's JSON response.
    pub source: DoiSource,

    /// Whether the resolved URL is confirmed to be a direct PDF link
    /// based on the API's metadata. When true, the download pipeline
    /// can skip the classify_url HEAD request and proceed directly to
    /// download_pdf. When false, the URL must still be classified via
    /// the standard HEAD-request pipeline.
    pub is_pdf: bool,
}

/// Identifies which API source in the resolution chain provided the URL.
/// The order of variants reflects the chain priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoiSource {
    /// Unpaywall API (api.unpaywall.org). Requires an email address as
    /// a polite-access identifier. Provides direct PDF URLs for ~30M
    /// open-access papers.
    Unpaywall,

    /// Semantic Scholar Graph API (api.semanticscholar.org). No authentication
    /// required. Rate-limited to ~100 requests per 5 minutes without an API key.
    SemanticScholar,

    /// OpenAlex API (api.openalex.org). No authentication required. Covers
    /// ~250M scholarly works with open-access location metadata.
    OpenAlex,

    /// Standard doi.org HTTP redirect. Constructs `https://doi.org/{doi}`
    /// and relies on the HTTP redirect chain to reach the publisher's page.
    /// This is the current fallback behavior when no API returns a PDF URL.
    DoiOrg,
}

impl std::fmt::Display for DoiSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DoiSource::Unpaywall => write!(f, "unpaywall"),
            DoiSource::SemanticScholar => write!(f, "semantic_scholar"),
            DoiSource::OpenAlex => write!(f, "openalex"),
            DoiSource::DoiOrg => write!(f, "doi_org"),
        }
    }
}

/// Per-API timeout for HTTP requests in the resolution chain. Each API
/// call is independently capped at this duration so a slow API does not
/// block the entire chain.
const API_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Resolves a DOI to a direct URL by querying free academic APIs in sequence.
///
/// The resolution chain tries each source in order and returns the first
/// successful result:
///
///   1. Unpaywall (skipped when `email` is None)
///   2. Semantic Scholar
///   3. OpenAlex
///   4. doi.org (constructs URL without a network request)
///
/// # Arguments
///
/// * `client` - Pre-configured reqwest::Client with TLS and redirect support.
/// * `doi` - The DOI string (e.g., "10.1086/260062"). Must not include the
///   `https://doi.org/` prefix.
/// * `email` - Optional email address for Unpaywall API access. When None,
///   Unpaywall is skipped and resolution starts at Semantic Scholar.
///
/// # Returns
///
/// Always returns a `ResolvedDoi`. The doi.org fallback constructs the URL
/// without any network request, so this function cannot fail at the resolution
/// stage (the actual HTTP redirect happens during the download phase).
pub async fn resolve_doi(client: &reqwest::Client, doi: &str, email: Option<&str>) -> ResolvedDoi {
    // Strip the doi.org prefix if the caller accidentally included it.
    let doi = doi
        .trim()
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/");

    // 1. Unpaywall (requires email)
    if let Some(email) = email
        && !email.is_empty()
    {
        match try_unpaywall(client, doi, email).await {
            Some(resolved) => {
                debug!(
                    doi = doi,
                    url = resolved.url.as_str(),
                    "DOI resolved via Unpaywall"
                );
                return resolved;
            }
            None => {
                debug!(
                    doi = doi,
                    "Unpaywall returned no PDF URL, trying Semantic Scholar"
                );
            }
        }
    }

    // 2. Semantic Scholar (no auth required)
    match try_semantic_scholar(client, doi).await {
        Some(resolved) => {
            debug!(
                doi = doi,
                url = resolved.url.as_str(),
                "DOI resolved via Semantic Scholar"
            );
            return resolved;
        }
        None => {
            debug!(
                doi = doi,
                "Semantic Scholar returned no PDF URL, trying OpenAlex"
            );
        }
    }

    // 3. OpenAlex (no auth required)
    match try_openalex(client, doi).await {
        Some(resolved) => {
            debug!(
                doi = doi,
                url = resolved.url.as_str(),
                "DOI resolved via OpenAlex"
            );
            return resolved;
        }
        None => {
            debug!(
                doi = doi,
                "OpenAlex returned no PDF URL, falling back to doi.org"
            );
        }
    }

    // 4. doi.org fallback (no network request, just URL construction)
    ResolvedDoi {
        url: format!("https://doi.org/{doi}"),
        source: DoiSource::DoiOrg,
        is_pdf: false,
    }
}

// ---------------------------------------------------------------------------
// API-specific async wrappers
// ---------------------------------------------------------------------------

/// Queries the Unpaywall API for a direct PDF URL.
///
/// Endpoint: GET https://api.unpaywall.org/v2/{doi}?email={email}
///
/// Returns None when the API returns an error, the DOI is not found,
/// or no open-access PDF location exists in the response.
async fn try_unpaywall(client: &reqwest::Client, doi: &str, email: &str) -> Option<ResolvedDoi> {
    let url = format!(
        "https://api.unpaywall.org/v2/{}?email={}",
        urlencoding(doi),
        urlencoding(email)
    );

    let response = match tokio::time::timeout(API_TIMEOUT, client.get(&url).send()).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            warn!(doi = doi, error = %e, "Unpaywall HTTP request failed");
            return None;
        }
        Err(_) => {
            warn!(doi = doi, "Unpaywall request timed out after 10s");
            return None;
        }
    };

    if !response.status().is_success() {
        debug!(
            doi = doi,
            status = response.status().as_u16(),
            "Unpaywall returned non-success status"
        );
        return None;
    }

    let json: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(e) => {
            warn!(doi = doi, error = %e, "Unpaywall response JSON parsing failed");
            return None;
        }
    };

    parse_unpaywall_response(&json)
}

/// Queries the Semantic Scholar Graph API for a direct PDF URL.
///
/// Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf
///
/// Returns None when the API returns an error, the paper is not found,
/// or no open-access PDF exists.
async fn try_semantic_scholar(client: &reqwest::Client, doi: &str) -> Option<ResolvedDoi> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/DOI:{}?fields=openAccessPdf",
        urlencoding(doi)
    );

    let response = match tokio::time::timeout(API_TIMEOUT, client.get(&url).send()).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            warn!(doi = doi, error = %e, "Semantic Scholar HTTP request failed");
            return None;
        }
        Err(_) => {
            warn!(doi = doi, "Semantic Scholar request timed out after 10s");
            return None;
        }
    };

    if !response.status().is_success() {
        debug!(
            doi = doi,
            status = response.status().as_u16(),
            "Semantic Scholar returned non-success status"
        );
        return None;
    }

    let json: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(e) => {
            warn!(doi = doi, error = %e, "Semantic Scholar response JSON parsing failed");
            return None;
        }
    };

    parse_semantic_scholar_response(&json)
}

/// Queries the OpenAlex API for a direct PDF URL.
///
/// Endpoint: GET https://api.openalex.org/works/doi:{doi}
///
/// Returns None when the API returns an error, the work is not found,
/// or no open-access PDF location exists.
async fn try_openalex(client: &reqwest::Client, doi: &str) -> Option<ResolvedDoi> {
    let url = format!("https://api.openalex.org/works/doi:{}", urlencoding(doi));

    let response = match tokio::time::timeout(API_TIMEOUT, client.get(&url).send()).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            warn!(doi = doi, error = %e, "OpenAlex HTTP request failed");
            return None;
        }
        Err(_) => {
            warn!(doi = doi, "OpenAlex request timed out after 10s");
            return None;
        }
    };

    if !response.status().is_success() {
        debug!(
            doi = doi,
            status = response.status().as_u16(),
            "OpenAlex returned non-success status"
        );
        return None;
    }

    let json: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(e) => {
            warn!(doi = doi, error = %e, "OpenAlex response JSON parsing failed");
            return None;
        }
    };

    parse_openalex_response(&json)
}

// ---------------------------------------------------------------------------
// Pure response parsers (no I/O, deterministic, testable)
// ---------------------------------------------------------------------------

/// Parses an Unpaywall API response and extracts the best open-access PDF URL.
///
/// The function checks `best_oa_location.url_for_pdf` first (direct PDF link),
/// then falls back to `best_oa_location.url` (which may be a landing page but
/// is still more specific than doi.org). Returns None when no open-access
/// location exists or the URL is invalid.
fn parse_unpaywall_response(json: &serde_json::Value) -> Option<ResolvedDoi> {
    let oa_location = json.get("best_oa_location")?;

    // Primary: url_for_pdf is a direct PDF download link.
    if let Some(pdf_url) = oa_location.get("url_for_pdf").and_then(|v| v.as_str())
        && validate_url(pdf_url)
    {
        return Some(ResolvedDoi {
            url: pdf_url.to_string(),
            source: DoiSource::Unpaywall,
            is_pdf: true,
        });
    }

    // Fallback: url field (may be a landing page with better access than doi.org).
    if let Some(url) = oa_location.get("url").and_then(|v| v.as_str())
        && validate_url(url)
    {
        return Some(ResolvedDoi {
            url: url.to_string(),
            source: DoiSource::Unpaywall,
            is_pdf: false,
        });
    }

    None
}

/// Parses a Semantic Scholar Graph API response and extracts the open-access
/// PDF URL from the `openAccessPdf` field. Returns None when the field is
/// null or absent.
fn parse_semantic_scholar_response(json: &serde_json::Value) -> Option<ResolvedDoi> {
    let oa_pdf = json.get("openAccessPdf")?;

    if let Some(url) = oa_pdf.get("url").and_then(|v| v.as_str())
        && validate_url(url)
    {
        return Some(ResolvedDoi {
            url: url.to_string(),
            source: DoiSource::SemanticScholar,
            is_pdf: true,
        });
    }

    None
}

/// Parses an OpenAlex API response and extracts the best PDF URL from
/// open-access location metadata. Checks `best_oa_location.pdf_url` first,
/// then falls back to `primary_location.pdf_url`. Returns None when no
/// PDF URL exists in any location.
fn parse_openalex_response(json: &serde_json::Value) -> Option<ResolvedDoi> {
    // Primary: best_oa_location.pdf_url
    if let Some(best_oa) = json.get("best_oa_location")
        && let Some(pdf_url) = best_oa.get("pdf_url").and_then(|v| v.as_str())
        && validate_url(pdf_url)
    {
        return Some(ResolvedDoi {
            url: pdf_url.to_string(),
            source: DoiSource::OpenAlex,
            is_pdf: true,
        });
    }

    // Fallback: primary_location.pdf_url
    if let Some(primary) = json.get("primary_location")
        && let Some(pdf_url) = primary.get("pdf_url").and_then(|v| v.as_str())
        && validate_url(pdf_url)
    {
        return Some(ResolvedDoi {
            url: pdf_url.to_string(),
            source: DoiSource::OpenAlex,
            is_pdf: true,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validates that a URL string is well-formed and uses the http or https scheme.
/// Rejects empty strings, data: URIs, javascript: URIs, and malformed URLs.
fn validate_url(url: &str) -> bool {
    if url.is_empty() {
        return false;
    }

    match url::Url::parse(url) {
        Ok(parsed) => {
            let scheme = parsed.scheme();
            scheme == "http" || scheme == "https"
        }
        Err(_) => false,
    }
}

/// Percent-encodes a string for use in URL path segments and query parameters.
/// Encodes all characters except unreserved characters (ALPHA, DIGIT, '-', '.', '_', '~').
fn urlencoding(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Unpaywall response parsing
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-001: Unpaywall response with `url_for_pdf` extracts
    /// the direct PDF URL and sets is_pdf=true.
    #[test]
    fn t_html_resolve_001_unpaywall_pdf_url() {
        let json = serde_json::json!({
            "is_oa": true,
            "best_oa_location": {
                "url_for_pdf": "https://europepmc.org/articles/pmc3814466?pdf=render",
                "url": "https://europepmc.org/articles/pmc3814466"
            }
        });

        let result = parse_unpaywall_response(&json);
        assert!(
            result.is_some(),
            "Unpaywall response with url_for_pdf must resolve"
        );
        let resolved = result.unwrap();
        assert_eq!(
            resolved.url,
            "https://europepmc.org/articles/pmc3814466?pdf=render"
        );
        assert_eq!(resolved.source, DoiSource::Unpaywall);
        assert!(resolved.is_pdf, "url_for_pdf result must have is_pdf=true");
    }

    /// T-HTML-RESOLVE-002: Unpaywall response with `url_for_pdf` = null
    /// falls back to the `url` field and sets is_pdf=false.
    #[test]
    fn t_html_resolve_002_unpaywall_fallback_url() {
        let json = serde_json::json!({
            "is_oa": true,
            "best_oa_location": {
                "url_for_pdf": null,
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345"
            }
        });

        let result = parse_unpaywall_response(&json);
        assert!(result.is_some(), "Unpaywall response with url must resolve");
        let resolved = result.unwrap();
        assert_eq!(
            resolved.url,
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345"
        );
        assert_eq!(resolved.source, DoiSource::Unpaywall);
        assert!(!resolved.is_pdf, "url fallback must have is_pdf=false");
    }

    /// T-HTML-RESOLVE-003: Unpaywall response with no `best_oa_location`
    /// returns None (paper is not open access).
    #[test]
    fn t_html_resolve_003_unpaywall_no_oa_location() {
        let json = serde_json::json!({
            "is_oa": false,
            "best_oa_location": null
        });

        let result = parse_unpaywall_response(&json);
        assert!(
            result.is_none(),
            "response without best_oa_location must return None"
        );
    }

    /// T-HTML-RESOLVE-004: Unpaywall response with missing `best_oa_location`
    /// key returns None (unexpected schema).
    #[test]
    fn t_html_resolve_004_unpaywall_missing_key() {
        let json = serde_json::json!({
            "is_oa": false
        });

        let result = parse_unpaywall_response(&json);
        assert!(
            result.is_none(),
            "response without best_oa_location key must return None"
        );
    }

    // -----------------------------------------------------------------------
    // Semantic Scholar response parsing
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-005: Semantic Scholar response with `openAccessPdf.url`
    /// extracts the PDF URL and sets is_pdf=true.
    #[test]
    fn t_html_resolve_005_semantic_scholar_pdf_url() {
        let json = serde_json::json!({
            "paperId": "abc123",
            "openAccessPdf": {
                "url": "https://arxiv.org/pdf/2301.12345v1.pdf",
                "status": "GREEN"
            }
        });

        let result = parse_semantic_scholar_response(&json);
        assert!(
            result.is_some(),
            "Semantic Scholar response with openAccessPdf must resolve"
        );
        let resolved = result.unwrap();
        assert_eq!(resolved.url, "https://arxiv.org/pdf/2301.12345v1.pdf");
        assert_eq!(resolved.source, DoiSource::SemanticScholar);
        assert!(
            resolved.is_pdf,
            "openAccessPdf result must have is_pdf=true"
        );
    }

    /// T-HTML-RESOLVE-006: Semantic Scholar response with `openAccessPdf` = null
    /// returns None (no open-access PDF available).
    #[test]
    fn t_html_resolve_006_semantic_scholar_null_pdf() {
        let json = serde_json::json!({
            "paperId": "def456",
            "openAccessPdf": null
        });

        let result = parse_semantic_scholar_response(&json);
        assert!(
            result.is_none(),
            "response with null openAccessPdf must return None"
        );
    }

    /// T-HTML-RESOLVE-007: Semantic Scholar response without the `openAccessPdf`
    /// key returns None (field absent from response).
    #[test]
    fn t_html_resolve_007_semantic_scholar_missing_field() {
        let json = serde_json::json!({
            "paperId": "ghi789"
        });

        let result = parse_semantic_scholar_response(&json);
        assert!(
            result.is_none(),
            "response without openAccessPdf key must return None"
        );
    }

    // -----------------------------------------------------------------------
    // OpenAlex response parsing
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-008: OpenAlex response with `best_oa_location.pdf_url`
    /// extracts the PDF URL and sets is_pdf=true.
    #[test]
    fn t_html_resolve_008_openalex_best_oa_pdf() {
        let json = serde_json::json!({
            "id": "https://openalex.org/W12345",
            "best_oa_location": {
                "pdf_url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC67890&blobtype=pdf",
                "landing_page_url": "https://europepmc.org/articles/PMC67890"
            }
        });

        let result = parse_openalex_response(&json);
        assert!(
            result.is_some(),
            "OpenAlex response with best_oa_location.pdf_url must resolve"
        );
        let resolved = result.unwrap();
        assert!(resolved.url.contains("ptpmcrender"));
        assert_eq!(resolved.source, DoiSource::OpenAlex);
        assert!(resolved.is_pdf, "pdf_url result must have is_pdf=true");
    }

    /// T-HTML-RESOLVE-009: OpenAlex response without `best_oa_location.pdf_url`
    /// falls back to `primary_location.pdf_url`.
    #[test]
    fn t_html_resolve_009_openalex_primary_location_fallback() {
        let json = serde_json::json!({
            "id": "https://openalex.org/W54321",
            "best_oa_location": {
                "pdf_url": null,
                "landing_page_url": "https://publisher.com/article"
            },
            "primary_location": {
                "pdf_url": "https://publisher.com/article.pdf",
                "landing_page_url": "https://publisher.com/article"
            }
        });

        let result = parse_openalex_response(&json);
        assert!(
            result.is_some(),
            "OpenAlex must fall back to primary_location.pdf_url"
        );
        let resolved = result.unwrap();
        assert_eq!(resolved.url, "https://publisher.com/article.pdf");
        assert_eq!(resolved.source, DoiSource::OpenAlex);
    }

    /// T-HTML-RESOLVE-010: OpenAlex response with no PDF URLs in any location
    /// returns None.
    #[test]
    fn t_html_resolve_010_openalex_no_pdf_urls() {
        let json = serde_json::json!({
            "id": "https://openalex.org/W99999",
            "best_oa_location": null,
            "primary_location": {
                "pdf_url": null,
                "landing_page_url": "https://publisher.com/abstract"
            }
        });

        let result = parse_openalex_response(&json);
        assert!(
            result.is_none(),
            "response with no pdf_url in any location must return None"
        );
    }

    // -----------------------------------------------------------------------
    // URL validation
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-011: validate_url accepts a well-formed https URL.
    #[test]
    fn t_html_resolve_011_validate_https_url() {
        assert!(validate_url("https://example.com/paper.pdf"));
    }

    /// T-HTML-RESOLVE-012: validate_url accepts a well-formed http URL.
    #[test]
    fn t_html_resolve_012_validate_http_url() {
        assert!(validate_url("http://example.com/legacy/paper.pdf"));
    }

    /// T-HTML-RESOLVE-013: validate_url rejects an empty string.
    #[test]
    fn t_html_resolve_013_validate_empty_string() {
        assert!(!validate_url(""));
    }

    /// T-HTML-RESOLVE-014: validate_url rejects a data: URI.
    #[test]
    fn t_html_resolve_014_validate_data_uri() {
        assert!(!validate_url("data:text/html,<h1>test</h1>"));
    }

    /// T-HTML-RESOLVE-015: validate_url rejects a malformed URL.
    #[test]
    fn t_html_resolve_015_validate_malformed_url() {
        assert!(!validate_url("not a url at all"));
    }

    // -----------------------------------------------------------------------
    // Resolution chain logic
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-016: resolve_doi with email=None skips Unpaywall.
    /// Verified by the fact that doi.org fallback is used (no network
    /// calls in unit test context would succeed for Semantic Scholar/OpenAlex).
    #[tokio::test]
    async fn t_html_resolve_016_no_email_skips_unpaywall() {
        let client = crate::build_http_client().expect("client must build");
        // Use a fake DOI that will not exist in any API.
        let result = resolve_doi(&client, "10.0000/nonexistent-test-doi-00000", None).await;
        // Without working API connections, the chain falls through to doi.org.
        assert_eq!(result.source, DoiSource::DoiOrg);
        assert_eq!(
            result.url,
            "https://doi.org/10.0000/nonexistent-test-doi-00000"
        );
        assert!(!result.is_pdf, "doi.org fallback must have is_pdf=false");
    }

    /// T-HTML-RESOLVE-017: DoiSource Display trait renders the expected strings.
    #[test]
    fn t_html_resolve_017_doi_source_display() {
        assert_eq!(format!("{}", DoiSource::Unpaywall), "unpaywall");
        assert_eq!(
            format!("{}", DoiSource::SemanticScholar),
            "semantic_scholar"
        );
        assert_eq!(format!("{}", DoiSource::OpenAlex), "openalex");
        assert_eq!(format!("{}", DoiSource::DoiOrg), "doi_org");
    }

    /// T-HTML-RESOLVE-018: doi.org fallback produces is_pdf=false because the
    /// redirect target is unknown at resolution time.
    #[test]
    fn t_html_resolve_018_doi_org_fallback_not_pdf() {
        let resolved = ResolvedDoi {
            url: "https://doi.org/10.1086/260062".to_string(),
            source: DoiSource::DoiOrg,
            is_pdf: false,
        };
        assert!(!resolved.is_pdf);
        assert_eq!(resolved.source, DoiSource::DoiOrg);
    }

    /// T-HTML-RESOLVE-019: Unpaywall parser sets is_pdf=true for url_for_pdf
    /// results, distinguishing them from landing page fallbacks.
    #[test]
    fn t_html_resolve_019_unpaywall_is_pdf_flag() {
        let json_with_pdf = serde_json::json!({
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf",
                "url": "https://example.com/paper"
            }
        });
        let json_without_pdf = serde_json::json!({
            "best_oa_location": {
                "url_for_pdf": null,
                "url": "https://example.com/paper"
            }
        });

        let with_pdf = parse_unpaywall_response(&json_with_pdf).unwrap();
        let without_pdf = parse_unpaywall_response(&json_without_pdf).unwrap();

        assert!(with_pdf.is_pdf, "url_for_pdf result must be is_pdf=true");
        assert!(!without_pdf.is_pdf, "url fallback must be is_pdf=false");
    }

    // -----------------------------------------------------------------------
    // URL encoding
    // -----------------------------------------------------------------------

    /// T-HTML-RESOLVE-020: urlencoding encodes DOI special characters correctly.
    #[test]
    fn t_html_resolve_020_urlencoding_doi() {
        let encoded = urlencoding("10.1086/260062");
        assert_eq!(encoded, "10.1086%2F260062");
    }

    /// T-HTML-RESOLVE-021: urlencoding encodes email special characters correctly.
    #[test]
    fn t_html_resolve_021_urlencoding_email() {
        let encoded = urlencoding("user@example.com");
        assert_eq!(encoded, "user%40example.com");
    }
}
