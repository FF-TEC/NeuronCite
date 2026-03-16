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

// Error types for the neuroncite-html crate.
//
// HtmlError covers HTTP request failures, URL parsing errors, HTML parsing
// issues, crawl-specific problems, file I/O errors during cache operations,
// SSRF protection violations, and invalid argument conditions.

/// Represents all error conditions that can occur within the HTML scraping
/// and parsing layer. Each variant maps to a distinct failure category.
#[derive(Debug, thiserror::Error)]
pub enum HtmlError {
    /// An HTTP request failed (timeout, connection refused, DNS failure, etc.).
    /// Wraps the underlying reqwest error.
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A URL could not be parsed or resolved against a base URL.
    #[error("url parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    /// A file system operation failed during HTML cache read/write.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The HTML document could not be parsed or contained unexpected structure.
    #[error("html parse error: {0}")]
    Parse(String),

    /// A crawl operation failed due to configuration issues or runtime errors.
    #[error("crawl error: {0}")]
    Crawl(String),

    /// A cache directory operation failed (creation, read, or write).
    #[error("cache error: {0}")]
    Cache(String),

    /// A function argument was invalid (empty URL list, negative depth, etc.).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A regular expression pattern failed to compile (used by URL pattern
    /// filtering during crawl operations).
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),

    /// A file download failed due to content-type mismatch, missing PDF
    /// header, or other validation errors during the download pipeline.
    #[error("download error: {0}")]
    Download(String),

    /// A URL was blocked by SSRF (Server-Side Request Forgery) protection
    /// because it resolves to a private, loopback, link-local, or otherwise
    /// non-routable IP address. The contained string describes the specific
    /// blocked address range and the original URL.
    #[error("ssrf blocked: {0}")]
    Ssrf(String),
}
