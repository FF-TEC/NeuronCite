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

// File download operations for the HTML/web acquisition pipeline.
//
// Provides functions to download files from URLs, classify whether a URL
// points to a PDF or an HTML page, and sanitize filenames for local storage.
// The download function follows HTTP redirects (covering DOI resolver URLs
// like https://doi.org/10.1234/...) and validates the Content-Type header
// to confirm the response is a PDF before writing to disk.
//
// All outbound HTTP requests are guarded by SSRF validation (see ssrf.rs)
// which blocks URLs that resolve to private, loopback, link-local, or
// cloud metadata IP addresses.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tracing::{debug, warn};

use crate::error::HtmlError;
use crate::ssrf::validate_url_no_ssrf;

/// Classification of a URL target based on HTTP response headers.
/// Determined by sending an HTTP HEAD request and inspecting the
/// Content-Type header, with a fallback to URL path extension analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlSourceType {
    /// The URL serves a PDF file (Content-Type: application/pdf or .pdf extension).
    Pdf,
    /// The URL serves an HTML page or any non-PDF content.
    Html,
}

/// Classifies a URL as PDF or HTML by sending an HTTP HEAD request and
/// inspecting the Content-Type response header. Recognizes `application/pdf`
/// directly and also accepts `application/octet-stream` when the URL path
/// ends in `.pdf` (many academic servers serve PDFs with this generic
/// Content-Type). Falls back to URL path extension analysis on the final
/// URL after following redirects.
///
/// Before making the HEAD request, the URL is validated against SSRF
/// protection rules (see `ssrf::validate_url_no_ssrf`). URLs that resolve
/// to private networks, loopback, link-local, or cloud metadata addresses
/// cause this function to return `UrlSourceType::Html` as a safe default
/// (the caller will then attempt to fetch and encounter the SSRF block).
///
/// # Arguments
///
/// * `client` - A pre-configured reqwest::Client (created via `build_http_client`).
/// * `url` - The URL to classify.
///
/// # Returns
///
/// `UrlSourceType::Pdf` when the server responds with `application/pdf`,
/// with `application/octet-stream` and a `.pdf` URL path, or when the
/// final URL (after redirects) ends with `.pdf`. `UrlSourceType::Html`
/// in all other cases. Network errors are treated as Html (the caller
/// should attempt to fetch and will see the actual error).
pub async fn classify_url(client: &reqwest::Client, url: &str) -> UrlSourceType {
    // Check the URL path extension first as a fast heuristic that avoids
    // a network round-trip for obvious .pdf URLs.
    if url_path_ends_with_pdf(url) {
        return UrlSourceType::Pdf;
    }

    // SSRF protection: reject URLs that resolve to private/loopback/link-local IPs.
    // On SSRF violation, return Html as a safe default. The caller will encounter
    // the SSRF block when it tries to fetch or download the URL.
    if validate_url_no_ssrf(url).is_err() {
        warn!(
            url = url,
            "classify_url: URL blocked by SSRF protection, returning Html"
        );
        return UrlSourceType::Html;
    }

    // Send an HTTP HEAD request to inspect the Content-Type header without
    // downloading the full response body.
    match client.head(url).send().await {
        Ok(response) => {
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");

            // Apply the same Content-Type classification logic as download_pdf():
            // accept application/pdf directly, accept application/octet-stream
            // when the URL path ends in .pdf (many academic servers serve PDFs
            // with this generic Content-Type), and fall back to URL path analysis
            // after following redirects.
            let final_url = response.url().as_str();
            let is_pdf = content_type.contains("application/pdf")
                || (content_type.contains("application/octet-stream")
                    && url_path_ends_with_pdf(final_url))
                || url_path_ends_with_pdf(final_url);

            if is_pdf {
                UrlSourceType::Pdf
            } else {
                UrlSourceType::Html
            }
        }
        Err(e) => {
            warn!(url = url, error = %e, "HEAD request failed, treating URL as HTML");
            UrlSourceType::Html
        }
    }
}

/// Downloads a PDF from the given URL and writes it to the output directory.
/// The file is saved as `{filename}.pdf` within `output_dir`. Follows HTTP
/// redirects (DOI resolver URLs redirect to the actual PDF host).
///
/// Before making the HTTP request, the URL is validated against SSRF
/// protection rules (see `ssrf::validate_url_no_ssrf`). URLs that resolve
/// to private networks, loopback, link-local, or cloud metadata addresses
/// are rejected with `HtmlError::Ssrf`.
///
/// # Arguments
///
/// * `client` - A pre-configured reqwest::Client.
/// * `url` - The URL to download the PDF from.
/// * `output_dir` - Directory where the downloaded file is stored.
/// * `filename` - Base filename without extension. Callers should provide a
///   descriptive name derived from BibTeX metadata (e.g., "Title (Author, Year)")
///   so the file is recognizable and matches the token-overlap algorithm used
///   by the citation verification pipeline for PDF-to-citation matching.
///
/// # Returns
///
/// The absolute path to the downloaded file on success.
///
/// # Errors
///
/// Returns `HtmlError::Ssrf` when the URL resolves to a blocked IP range.
/// Returns `HtmlError::Download` when the server responds with a non-PDF
/// Content-Type or a non-success HTTP status code. Returns `HtmlError::Http`
/// on network failures and `HtmlError::Io` on filesystem write failures.
pub async fn download_pdf(
    client: &reqwest::Client,
    url: &str,
    output_dir: &Path,
    filename: &str,
) -> Result<PathBuf, HtmlError> {
    // SSRF protection: reject URLs that resolve to private/loopback/link-local IPs.
    validate_url_no_ssrf(url)?;

    debug!(url = url, filename = filename, "downloading PDF");

    let response = client.get(url).send().await?;
    let status = response.status();
    let final_url = response.url().to_string();

    if !status.is_success() {
        return Err(HtmlError::Download(format!(
            "HTTP {} for URL: {final_url}",
            status.as_u16()
        )));
    }

    // Verify Content-Type is PDF. Some servers return application/octet-stream
    // for PDF files, so also accept that when the URL path ends in .pdf.
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let is_pdf = content_type.contains("application/pdf")
        || (content_type.contains("application/octet-stream")
            && url_path_ends_with_pdf(&final_url))
        || url_path_ends_with_pdf(&final_url);

    if !is_pdf {
        return Err(HtmlError::Download(format!(
            "URL did not serve a PDF (Content-Type: {content_type}): {final_url}"
        )));
    }

    // Stream the response body directly to disk instead of buffering the
    // entire PDF in memory. This handles large files (100+ MB) without
    // excessive RAM consumption. The first 5 bytes are captured during
    // streaming to validate the %PDF- magic header.
    let sanitized = sanitize_filename(filename);
    let file_path = output_dir.join(format!("{sanitized}.pdf"));

    std::fs::create_dir_all(output_dir)?;

    let mut file = tokio::fs::File::create(&file_path).await.map_err(|e| {
        HtmlError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to create file {}: {e}", file_path.display()),
        ))
    })?;

    let mut stream = response.bytes_stream();
    let mut total_bytes: u64 = 0;
    // Captures the first 5 bytes for PDF magic validation (%PDF-).
    let mut header_buf = [0u8; 5];
    let mut header_filled: usize = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        // Capture the first 5 bytes from the stream for magic validation.
        if header_filled < 5 {
            let needed = 5 - header_filled;
            let available = chunk.len().min(needed);
            header_buf[header_filled..header_filled + available]
                .copy_from_slice(&chunk[..available]);
            header_filled += available;
        }

        file.write_all(&chunk).await.map_err(|e| {
            HtmlError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to write to {}: {e}", file_path.display()),
            ))
        })?;
        total_bytes += chunk.len() as u64;
    }

    file.flush().await.map_err(|e| {
        HtmlError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to flush {}: {e}", file_path.display()),
        ))
    })?;
    drop(file);

    // Validate the PDF magic bytes (%PDF- header). If validation fails,
    // remove the partially-written file to prevent corrupt files from
    // persisting on disk.
    if header_filled < 5 || &header_buf[..5] != b"%PDF-" {
        let _ = tokio::fs::remove_file(&file_path).await;
        return Err(HtmlError::Download(format!(
            "downloaded content is not a valid PDF (missing %%PDF- header): {final_url}"
        )));
    }

    debug!(
        url = final_url.as_str(),
        path = file_path.display().to_string().as_str(),
        bytes = total_bytes,
        "PDF downloaded (streamed to disk)"
    );

    Ok(file_path)
}

/// Checks whether the URL path component ends with ".pdf" (case-insensitive).
/// Ignores query parameters and fragment identifiers.
fn url_path_ends_with_pdf(url: &str) -> bool {
    url::Url::parse(url)
        .ok()
        .map(|u| u.path().to_lowercase().ends_with(".pdf"))
        .unwrap_or(false)
}

/// Builds a descriptive filename for a downloaded source PDF from BibTeX
/// metadata. The resulting format is `Title (Author, Year)` which provides
/// sufficient tokens for the citation verification pipeline's token-overlap
/// matching algorithm.
///
/// Author extraction: Takes the first author's family name from the BibTeX
/// author field (handles both "Family, Given" and "Given Family" formats).
/// When the author string contains "and" separators, only the first author
/// is used, followed by "et al." if there are additional authors.
///
/// Fallback: When title is empty, falls back to the cite_key.
///
/// # Arguments
///
/// * `title` - The BibTeX title field.
/// * `author` - The BibTeX author field (may contain multiple authors).
/// * `year` - The BibTeX year field (optional).
/// * `cite_key` - The BibTeX cite-key used as fallback when title is empty.
///
/// # Returns
///
/// A sanitized filename string without the .pdf extension.
pub fn build_source_filename(
    title: &str,
    author: &str,
    year: Option<&str>,
    cite_key: &str,
) -> String {
    let title_trimmed = title.trim();
    if title_trimmed.is_empty() {
        return sanitize_filename(cite_key);
    }

    // Extract the first author's family name from BibTeX author format.
    // BibTeX uses "and" as separator between multiple authors.
    let first_author_family = extract_first_author_family(author);

    // Build the parenthetical suffix: "(Author, Year)" or "(Author)" or "(Year)".
    let suffix = match (first_author_family.as_deref(), year) {
        (Some(family), Some(y)) if !y.is_empty() => {
            let et_al = if author.contains(" and ") {
                " et al."
            } else {
                ""
            };
            format!(" ({}{}, {})", family, et_al, y.trim())
        }
        (Some(family), _) => {
            let et_al = if author.contains(" and ") {
                " et al."
            } else {
                ""
            };
            format!(" ({}{})", family, et_al)
        }
        (None, Some(y)) if !y.is_empty() => format!(" ({})", y.trim()),
        _ => String::new(),
    };

    let raw = format!("{}{}", title_trimmed, suffix);
    sanitize_filename(&raw)
}

/// Extracts the first author's family name from a BibTeX author string.
/// Handles "Family, Given" format (returns "Family") and "Given Family"
/// format (returns "Family" as the last whitespace-delimited token).
/// Returns None when the author string is empty.
fn extract_first_author_family(author: &str) -> Option<String> {
    let trimmed = author.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Split on " and " to isolate the first author.
    let first = trimmed.split(" and ").next().unwrap_or(trimmed).trim();

    if first.contains(',') {
        // "Family, Given" format: family name is before the comma.
        let family = first.split(',').next().unwrap_or(first).trim();
        if family.is_empty() {
            None
        } else {
            Some(strip_latex_braces(family))
        }
    } else {
        // "Given Family" format: family name is the last token.
        let tokens: Vec<&str> = first.split_whitespace().collect();
        tokens.last().map(|t| strip_latex_braces(t))
    }
}

/// Removes surrounding LaTeX braces from a string (e.g., "{M\"uller}" -> "M\"uller").
/// Single-level brace stripping only; nested braces are preserved.
fn strip_latex_braces(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') && trimmed.len() > 2 {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Sanitizes a string for use as a filename by replacing characters that are
/// invalid on Windows or problematic on Unix with underscores. Limits the
/// result to 200 characters to stay within filesystem path length limits.
fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();

    // Truncate to 200 characters to prevent path-length issues.
    if sanitized.len() > 200 {
        sanitized[..200].to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-HTML-DL-001: URL path ending in .pdf is detected as PDF.
    #[test]
    fn t_html_dl_001_url_path_pdf_extension() {
        assert!(url_path_ends_with_pdf(
            "https://example.com/papers/fama1970.pdf"
        ));
    }

    /// T-HTML-DL-002: URL path ending in .pdf with query parameters is detected.
    #[test]
    fn t_html_dl_002_url_path_pdf_with_query() {
        assert!(url_path_ends_with_pdf(
            "https://example.com/download.pdf?token=abc123"
        ));
    }

    /// T-HTML-DL-003: URL path not ending in .pdf is not detected.
    #[test]
    fn t_html_dl_003_url_path_html() {
        assert!(!url_path_ends_with_pdf("https://example.com/article/12345"));
    }

    /// T-HTML-DL-004: URL path ending in .PDF (uppercase) is detected.
    #[test]
    fn t_html_dl_004_url_path_uppercase_pdf() {
        assert!(url_path_ends_with_pdf("https://example.com/PAPER.PDF"));
    }

    /// T-HTML-DL-005: DOI resolver URL (no .pdf extension) is not detected as PDF.
    #[test]
    fn t_html_dl_005_doi_url_not_pdf() {
        assert!(!url_path_ends_with_pdf("https://doi.org/10.1086/260062"));
    }

    /// T-HTML-DL-006: Invalid URL is not detected as PDF.
    #[test]
    fn t_html_dl_006_invalid_url_not_pdf() {
        assert!(!url_path_ends_with_pdf("not a url"));
    }

    /// T-HTML-DL-007: sanitize_filename replaces invalid characters with underscores.
    #[test]
    fn t_html_dl_007_sanitize_invalid_chars() {
        let result = sanitize_filename("file/name:with*bad?chars");
        assert_eq!(result, "file_name_with_bad_chars");
    }

    /// T-HTML-DL-008: sanitize_filename preserves valid characters.
    #[test]
    fn t_html_dl_008_sanitize_valid_chars() {
        let result = sanitize_filename("fama1970");
        assert_eq!(result, "fama1970");
    }

    /// T-HTML-DL-009: sanitize_filename truncates long names to 200 characters.
    #[test]
    fn t_html_dl_009_sanitize_truncation() {
        let long_name = "a".repeat(300);
        let result = sanitize_filename(&long_name);
        assert_eq!(result.len(), 200);
    }

    /// T-HTML-DL-010: sanitize_filename handles empty string.
    #[test]
    fn t_html_dl_010_sanitize_empty() {
        let result = sanitize_filename("");
        assert_eq!(result, "");
    }

    /// T-HTML-DL-011: sanitize_filename replaces control characters.
    #[test]
    fn t_html_dl_011_sanitize_control_chars() {
        let result = sanitize_filename("file\x00name\x01test");
        assert_eq!(result, "file_name_test");
    }

    /// T-HTML-DL-012: URL path ending in .pdf with fragment identifier is detected.
    #[test]
    fn t_html_dl_012_url_path_pdf_with_fragment() {
        assert!(url_path_ends_with_pdf(
            "https://example.com/paper.pdf#page=5"
        ));
    }

    /// T-HTML-DL-013: classify_url returns Pdf for obvious .pdf URLs without
    /// making a network request (fast path via URL path analysis).
    #[tokio::test]
    async fn t_html_dl_013_classify_url_fast_path() {
        let client = crate::build_http_client().expect("client must build");
        // This URL does not exist, but the fast path should return Pdf
        // before making any network request based on the .pdf extension.
        let result = classify_url(&client, "https://nonexistent.example.com/paper.pdf").await;
        assert_eq!(result, UrlSourceType::Pdf);
    }

    /// T-HTML-DL-014: build_source_filename produces a descriptive filename
    /// from BibTeX metadata with "Title (Author, Year)" format.
    #[test]
    fn t_html_dl_014_source_filename_full_metadata() {
        let result = build_source_filename(
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            "Fama, Eugene F.",
            Some("1970"),
            "fama1970",
        );
        assert_eq!(
            result,
            "Efficient Capital Markets_ A Review of Theory and Empirical Work (Fama, 1970)"
        );
    }

    /// T-HTML-DL-015: build_source_filename handles multiple authors with
    /// "et al." suffix when "and" separator is present.
    #[test]
    fn t_html_dl_015_source_filename_multiple_authors() {
        let result = build_source_filename(
            "The Pricing of Options and Corporate Liabilities",
            "Black, Fischer and Scholes, Myron",
            Some("1973"),
            "black1973",
        );
        assert_eq!(
            result,
            "The Pricing of Options and Corporate Liabilities (Black et al., 1973)"
        );
    }

    /// T-HTML-DL-016: build_source_filename falls back to cite_key when
    /// the title is empty.
    #[test]
    fn t_html_dl_016_source_filename_empty_title_fallback() {
        let result = build_source_filename("", "Some Author", Some("2020"), "mykey2020");
        assert_eq!(result, "mykey2020");
    }

    /// T-HTML-DL-017: build_source_filename handles missing year.
    #[test]
    fn t_html_dl_017_source_filename_no_year() {
        let result = build_source_filename("Some Title", "Author, First", None, "author_nodate");
        assert_eq!(result, "Some Title (Author)");
    }

    /// T-HTML-DL-018: build_source_filename handles missing author and year.
    #[test]
    fn t_html_dl_018_source_filename_no_author_no_year() {
        let result = build_source_filename("Orphan Title", "", None, "orphan");
        assert_eq!(result, "Orphan Title");
    }

    /// T-HTML-DL-019: build_source_filename handles "Given Family" author
    /// format (no comma), extracting the last token as family name.
    #[test]
    fn t_html_dl_019_source_filename_given_family_format() {
        let result = build_source_filename("Some Paper", "John Smith", Some("2000"), "smith2000");
        assert_eq!(result, "Some Paper (Smith, 2000)");
    }

    /// T-HTML-DL-020: build_source_filename strips LaTeX braces from
    /// author names. The inner backslash-quote sequence from LaTeX accent
    /// commands gets sanitized (double-quotes are invalid in filenames).
    #[test]
    fn t_html_dl_020_source_filename_latex_braces() {
        let result = build_source_filename(
            "Risk Management",
            "M{\\\"u}ller, Hans",
            Some("2015"),
            "muller2015",
        );
        // The sanitizer replaces " with _ and strips LaTeX brace pairs.
        assert!(result.starts_with("Risk Management (M"), "got: {result}");
        assert!(result.contains("2015"), "must contain year, got: {result}");
    }

    /// T-HTML-DL-021: classify_url returns Html for a URL targeting a private IP
    /// (SSRF protection blocks the HEAD request and falls back to Html).
    #[tokio::test]
    async fn t_html_dl_021_classify_url_ssrf_blocked() {
        let client = crate::build_http_client().expect("client must build");
        let result = classify_url(&client, "http://192.168.1.1/secret.html").await;
        assert_eq!(
            result,
            UrlSourceType::Html,
            "SSRF-blocked URL must return Html as safe default"
        );
    }
}
