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

//! neuroncite-html: HTML web scraping, parsing, section splitting, and crawling
//! for the NeuronCite workspace.
//!
//! This crate provides:
//!
//! - **Fetch** (`fetch`) -- HTTP GET requests with caching, metadata extraction,
//!   and configurable rate limiting. Raw HTML is cached to disk using SHA-256-based
//!   filenames for deduplication.
//!
//! - **Parse** (`parse`) -- HTML text extraction with optional readability-based
//!   boilerplate removal. Splits extracted text into heading-based sections (H1/H2)
//!   that map to the existing PageText pipeline. Extracts metadata from `<head>`
//!   elements and HTTP response headers.
//!
//! - **Crawl** (`crawl`) -- BFS website crawling with configurable depth, same-domain
//!   filtering, URL pattern matching, sitemap-based discovery, and rate limiting.
//!
//! - **SSRF** (`ssrf`) -- Server-Side Request Forgery protection. Validates URLs
//!   against private IP ranges, loopback addresses, link-local addresses, and
//!   cloud metadata endpoints before any outbound HTTP request is made.
//!
//! The extracted sections integrate with the existing NeuronCite chunking and
//! embedding pipeline via `PageText` conversion, enabling unified search across
//! both PDF and HTML content within the same sessions.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_panics_doc
)]

pub mod crawl;
pub mod download;
pub mod error;
pub mod fetch;
pub mod parse;
pub mod resolve;
pub mod ssrf;
pub mod types;

// Re-export the primary error type at crate root.
pub use error::HtmlError;

// Re-export primary types at crate root for ergonomic imports.
pub use types::{CrawlConfig, FetchResult, HtmlSection, WebMetadata};

// Re-export fetch functions.
pub use fetch::{
    build_http_client, build_http_client_with_timeout, cache_path_for_url, default_cache_dir,
    fetch_url, fetch_urls,
};

// Re-export parse functions.
pub use parse::{
    extract_links, extract_main_content, extract_metadata, extract_visible_text, sections_to_pages,
    split_into_sections,
};

// Re-export crawl functions.
pub use crawl::{crawl, fetch_sitemap, normalize_url};

// Re-export download functions.
pub use download::{UrlSourceType, build_source_filename, classify_url, download_pdf};

// Re-export DOI resolution types and functions.
pub use resolve::{DoiSource, ResolvedDoi, resolve_doi};

// Re-export the SSRF validation function.
pub use ssrf::validate_url_no_ssrf;
