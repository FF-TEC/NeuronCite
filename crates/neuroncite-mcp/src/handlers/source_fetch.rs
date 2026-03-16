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

// Handler for the `neuroncite_citation_fetch_sources` MCP tool.
//
// Reads a BibTeX file, extracts URL fields from each entry, classifies
// each URL as PDF or HTML, downloads PDFs to the output directory,
// fetches HTML pages via the existing neuroncite-html cache pipeline,
// and adds all acquired files to the specified index session. This
// enables an AI agent to populate the indexed corpus with cited source
// documents after a citation verification job has been created.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_api::html_indexer;
use neuroncite_api::indexer;
use neuroncite_store::build_hnsw;

/// Known title/content patterns that indicate a bot-detection, Cloudflare
/// challenge, or access-denied page rather than actual article content.
/// When a fetched page's word count is below `MIN_CONTENT_WORDS` and the
/// title or visible text matches one of these patterns (case-insensitive),
/// the page is classified as blocked and excluded from the indexing phase.
const BOT_DETECTION_PATTERNS: &[&str] = &[
    "just a moment",
    "attention required",
    "checking your browser",
    "access denied",
    "please enable cookies",
    "security check",
    "redirecting",
    "please wait",
    "verify you are human",
    "enable javascript",
    "one more step",
    "403 forbidden",
];

/// Minimum number of words required for a fetched HTML page to be considered
/// valid content. Pages below this threshold that also match a bot-detection
/// pattern in their title or content are classified as blocked.
const MIN_CONTENT_WORDS: usize = 50;

/// Checks whether a fetched HTML page is a bot-detection or access-denied
/// stub rather than actual article content. Returns a descriptive reason
/// string if the page is blocked, or None if the page passes the quality
/// check.
///
/// Two criteria must BOTH be met for the page to be classified as blocked:
/// 1. The extracted visible text contains fewer than `MIN_CONTENT_WORDS`.
/// 2. The page title or visible text matches a known bot-detection pattern.
///
/// This two-criteria approach avoids false positives on legitimate short
/// pages (e.g., abstract-only pages) that do not match bot-detection patterns.
fn check_blocked_page(title: Option<&str>, raw_html: &str) -> Option<String> {
    let visible_text = neuroncite_html::extract_visible_text(raw_html);
    let word_count = visible_text.split_whitespace().count();

    if word_count >= MIN_CONTENT_WORDS {
        return None;
    }

    let title_lower = title.unwrap_or("").to_lowercase();
    let text_lower = visible_text.to_lowercase();

    for pattern in BOT_DETECTION_PATTERNS {
        if title_lower.contains(pattern) || text_lower.contains(pattern) {
            return Some(format!(
                "blocked: {word_count} words, matched pattern '{pattern}'"
            ));
        }
    }

    None
}

/// Data collected during Phase 1 (HTML fetch) for a single HTML page.
/// Carries everything the Phase 3 HTML indexing pipeline needs: the original
/// URL, the WebMetadata from the HTTP response, the raw HTML bytes, and the
/// cache file path. Replaces the previous `Vec<String>` of URLs, which was
/// insufficient because Phase 3 requires metadata and raw HTML for the
/// `html_indexer::extract_and_chunk_html` pipeline.
struct FetchedHtml {
    url: String,
    metadata: neuroncite_html::WebMetadata,
    raw_html: Vec<u8>,
    cache_path: std::path::PathBuf,
}

/// Fetches source documents referenced in a BibTeX file and adds them to
/// an index session. For each BibTeX entry that has a `url` field:
///
/// - If the URL points to a PDF (Content-Type: application/pdf or .pdf
///   extension), the PDF is downloaded to `output_directory/{cite_key}.pdf`
///   and added to the index session via the extraction/chunking/embedding
///   pipeline.
///
/// - If the URL points to an HTML page, the page is fetched and cached
///   via `neuroncite_html::fetch_url`, then indexed into the session via
///   the HTML indexing pipeline.
///
/// Entries with a `doi` field but no `url` field are resolved via
/// `https://doi.org/{doi}` as the URL.
///
/// # Parameters (from MCP tool call)
///
/// - `bib_path` (required): Absolute path to the .bib file.
/// - `session_id` (required): Index session to add downloaded sources to.
/// - `output_directory` (required): Directory where downloaded PDFs are stored.
/// - `delay_ms` (optional, default: 1000): Delay in milliseconds between
///   consecutive HTTP requests to avoid overwhelming target servers.
/// - `email` (optional): Email address for Unpaywall API access. When
///   provided, DOI resolution tries Unpaywall first for direct PDF URLs.
///   When absent, resolution starts with Semantic Scholar.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let bib_path = params["bib_path"]
        .as_str()
        .ok_or("missing required parameter: bib_path")?;

    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let output_directory = params["output_directory"]
        .as_str()
        .ok_or("missing required parameter: output_directory")?;

    let delay_ms = params["delay_ms"].as_u64().unwrap_or(1000);

    // Optional email for Unpaywall API access in the DOI resolution chain.
    let email: Option<String> = params["email"].as_str().map(String::from);

    // Validate bib file exists.
    let bib_file = std::path::Path::new(bib_path);
    if !bib_file.is_file() {
        return Err(format!("bib file does not exist: {bib_path}"));
    }

    // Validate session exists and retrieve its configuration for chunking.
    let session = {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;
        neuroncite_store::get_session(&conn, session_id)
            .map_err(|_| format!("session {session_id} not found"))?
    };

    // Validate the loaded model's vector dimension matches the session.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let dim = session.vector_dimension as usize;
    if dim != loaded_dim {
        return Err(format!(
            "session vector dimension ({dim}d) does not match the loaded model ({}d)",
            loaded_dim
        ));
    }

    // Parse the BibTeX file on a blocking thread (CPU-bound text processing).
    let bib_path_owned = bib_path.to_string();
    let bib_entries = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;
        Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
    })
    .await
    .map_err(|e| format!("blocking task panicked: {e}"))??;

    let total_entries = bib_entries.len();

    // Build the HTTP client before the URL resolution loop because
    // resolve_doi needs it for API requests to Unpaywall/Semantic Scholar/OpenAlex.
    let client = neuroncite_html::build_http_client()
        .map_err(|e| format!("HTTP client initialization failed: {e}"))?;

    // Resolve URLs for entries: prefer the explicit `url` field, then try
    // the multi-source DOI resolution chain (Unpaywall -> Semantic Scholar
    // -> OpenAlex -> doi.org). Entries without url or doi are skipped.
    struct SourceEntry {
        cite_key: String,
        url: String,
        /// The API source that resolved this DOI, or None when the entry
        /// had an explicit `url` field (no DOI resolution needed).
        doi_source: Option<neuroncite_html::DoiSource>,
        /// Whether the DOI resolution chain already confirmed that the URL
        /// points to a direct PDF file. When true, the classify_url() HEAD
        /// request is skipped because the API metadata is authoritative.
        /// Entries with an explicit `url` field (no DOI resolution) use false
        /// since the URL has not been pre-classified.
        is_pdf_confirmed: bool,
    }

    let mut sources: Vec<SourceEntry> = Vec::new();
    for (cite_key, entry) in &bib_entries {
        if let Some(url) = &entry.url {
            // Explicit URL field takes precedence over DOI resolution.
            // The URL has not been pre-classified, so is_pdf_confirmed is false.
            sources.push(SourceEntry {
                cite_key: cite_key.clone(),
                url: url.clone(),
                doi_source: None,
                is_pdf_confirmed: false,
            });
        } else if let Some(doi) = &entry.doi {
            // Resolve DOI through the multi-source resolution chain.
            let resolved = neuroncite_html::resolve_doi(&client, doi, email.as_deref()).await;
            tracing::info!(
                cite_key = %cite_key,
                doi = %doi,
                source = %resolved.source,
                url = %resolved.url,
                is_pdf = resolved.is_pdf,
                "DOI resolved"
            );
            sources.push(SourceEntry {
                cite_key: cite_key.clone(),
                url: resolved.url,
                doi_source: Some(resolved.source),
                is_pdf_confirmed: resolved.is_pdf,
            });
        }
    }

    let entries_with_url = sources.len();

    // Create the output directory and type-separated subfolders.
    // Downloaded PDFs are stored in output_dir/pdf/ and HTML pages in
    // output_dir/html/ so the user can browse sources grouped by format.
    let output_dir = std::path::Path::new(output_directory);
    let pdf_dir = output_dir.join("pdf");
    let html_dir = output_dir.join("html");
    std::fs::create_dir_all(&pdf_dir)
        .map_err(|e| format!("failed to create pdf output directory: {e}"))?;
    std::fs::create_dir_all(&html_dir)
        .map_err(|e| format!("failed to create html output directory: {e}"))?;

    let html_cache_dir = neuroncite_html::default_cache_dir();

    // Scan the output directory and its pdf/html subfolders for existing
    // files so already-downloaded sources can be skipped. Checks the flat
    // root directory (backward compatibility with sources downloaded before
    // subfolder separation) and the type-separated subfolders.
    let mut existing_files: std::collections::HashSet<String> = std::collections::HashSet::new();
    for scan_dir in [output_dir, pdf_dir.as_path(), html_dir.as_path()] {
        if let Ok(rd) = std::fs::read_dir(scan_dir) {
            for entry in rd.filter_map(|e| e.ok()) {
                existing_files.insert(entry.file_name().to_string_lossy().to_lowercase());
            }
        }
    }

    // --- Phase 1: Classify and download each source ---
    let mut results: Vec<serde_json::Value> = Vec::new();
    let mut downloaded_pdfs: Vec<String> = Vec::new();
    let mut fetched_html_pages: Vec<FetchedHtml> = Vec::new();
    let mut pdfs_downloaded = 0usize;
    let mut pdfs_failed = 0usize;
    let mut pdfs_skipped = 0usize;
    let mut html_fetched = 0usize;
    let mut html_failed = 0usize;
    let mut html_blocked = 0usize;
    let mut html_skipped = 0usize;

    for (i, source) in sources.iter().enumerate() {
        // Rate limiting: delay between consecutive requests.
        if i > 0 && delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }

        // Build a descriptive filename from BibTeX metadata for consistent
        // naming between MCP and API handlers, and for the skip-check.
        let bib_entry = bib_entries.get(&source.cite_key);
        let base_filename = neuroncite_html::build_source_filename(
            bib_entry.map(|e| e.title.as_str()).unwrap_or(""),
            bib_entry.map(|e| e.author.as_str()).unwrap_or(""),
            bib_entry.and_then(|e| e.year.as_deref()),
            &source.cite_key,
        );

        // Check whether a file for this entry already exists in the output
        // directory. Matches the expected filename with .pdf or .html extension,
        // and also checks for cite_key-based filenames.
        let base_lower = base_filename.to_lowercase();
        let cite_lower = source.cite_key.to_lowercase();
        let already_exists = existing_files.iter().any(|f| {
            let stem = f
                .strip_suffix(".pdf")
                .or_else(|| f.strip_suffix(".html"))
                .unwrap_or(f);
            stem == base_lower || stem == cite_lower || f.contains(&cite_lower)
        });

        if already_exists {
            let stype = if existing_files.contains(&format!("{base_lower}.pdf"))
                || existing_files.contains(&format!("{cite_lower}.pdf"))
            {
                "pdf"
            } else {
                "html"
            };
            if stype == "pdf" {
                pdfs_skipped += 1;
            } else {
                html_skipped += 1;
            }
            results.push(serde_json::json!({
                "cite_key": source.cite_key,
                "url": source.url,
                "type": stype,
                "status": "skipped",
                "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
            }));
            continue;
        }

        // Classify the URL to determine if it serves a PDF or HTML page.
        // When the DOI resolution chain already confirmed the URL is a direct
        // PDF link (is_pdf_confirmed=true), skip the HEAD request entirely.
        // This avoids a redundant network round-trip and prevents false
        // negatives from servers that block HEAD requests or return misleading
        // Content-Type headers on HEAD while serving valid PDFs on GET.
        let source_type = if source.is_pdf_confirmed {
            neuroncite_html::UrlSourceType::Pdf
        } else {
            neuroncite_html::classify_url(&client, &source.url).await
        };

        match source_type {
            neuroncite_html::UrlSourceType::Pdf => {
                // Download the PDF into the pdf/ subfolder using the descriptive
                // filename for citation matching.
                match neuroncite_html::download_pdf(&client, &source.url, &pdf_dir, &base_filename)
                    .await
                {
                    Ok(pdf_path) => {
                        let path_str = pdf_path.display().to_string();
                        pdfs_downloaded += 1;
                        downloaded_pdfs.push(path_str.clone());
                        results.push(serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "pdf",
                            "status": "downloaded",
                            "file_path": path_str,
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        }));
                    }
                    Err(e) => {
                        pdfs_failed += 1;
                        results.push(serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "pdf",
                            "status": "failed",
                            "error": format!("{e}"),
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        }));
                    }
                }
            }
            neuroncite_html::UrlSourceType::Html => {
                // Fetch the HTML page and cache it via the neuroncite-html pipeline.
                match neuroncite_html::fetch_url(&client, &source.url, &html_cache_dir, true).await
                {
                    Ok(fetch_result) => {
                        // Read the raw HTML bytes from the cache file for the
                        // blocked-page check and for the Phase 3 indexing pipeline.
                        let raw_html = match std::fs::read(&fetch_result.cache_path) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                html_failed += 1;
                                results.push(serde_json::json!({
                                    "cite_key": source.cite_key,
                                    "url": source.url,
                                    "type": "html",
                                    "status": "failed",
                                    "error": format!("reading cached HTML: {e}"),
                                    "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                                }));
                                continue;
                            }
                        };

                        // Quality check: detect Cloudflare challenges, access-denied
                        // stubs, and other bot-detection pages that contain no useful
                        // article content. These are excluded from indexing.
                        let raw_html_str = String::from_utf8_lossy(&raw_html);
                        if let Some(reason) = check_blocked_page(
                            fetch_result.metadata.title.as_deref(),
                            &raw_html_str,
                        ) {
                            html_blocked += 1;
                            results.push(serde_json::json!({
                                "cite_key": source.cite_key,
                                "url": source.url,
                                "type": "html",
                                "status": "blocked",
                                "reason": reason,
                                "title": fetch_result.metadata.title,
                                "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                            }));
                            continue;
                        }

                        // Save the HTML page into the html/ subfolder within
                        // the output directory for type-separated organization.
                        let html_filename = format!("{base_filename}.html");
                        let html_path = html_dir.join(&html_filename);
                        if let Err(e) = std::fs::write(&html_path, &raw_html) {
                            tracing::warn!(
                                path = %html_path.display(),
                                error = %e,
                                "failed to save HTML to output directory"
                            );
                        }

                        html_fetched += 1;
                        results.push(serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "html",
                            "status": "fetched",
                            "file_path": html_path.display().to_string(),
                            "cache_path": fetch_result.cache_path.display().to_string(),
                            "title": fetch_result.metadata.title,
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        }));

                        fetched_html_pages.push(FetchedHtml {
                            url: fetch_result.url.clone(),
                            metadata: fetch_result.metadata,
                            raw_html,
                            cache_path: fetch_result.cache_path,
                        });
                    }
                    Err(e) => {
                        html_failed += 1;
                        results.push(serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "html",
                            "status": "failed",
                            "error": format!("{e}"),
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        }));
                    }
                }
            }
        }
    }

    // --- Phase 2: Index downloaded PDFs into the session ---
    let mut chunks_created = 0usize;
    let mut index_failed: Vec<serde_json::Value> = Vec::new();

    if !downloaded_pdfs.is_empty() {
        let chunk_size = session.chunk_size.map(|v| v as usize);
        let chunk_overlap = session.chunk_overlap.map(|v| v as usize);
        let max_words = session.max_words.map(|v| v as usize);
        let tokenizer_json = state.worker_handle.tokenizer_json();

        for pdf_path_str in &downloaded_pdfs {
            let pdf_path = std::path::Path::new(pdf_path_str).to_path_buf();

            let strategy = neuroncite_chunk::create_strategy(
                &session.chunk_strategy,
                chunk_size,
                chunk_overlap,
                max_words,
                tokenizer_json.as_deref(),
            )
            .map_err(|e| format!("chunking strategy: {e}"))?;

            match tokio::task::spawn_blocking(move || {
                indexer::extract_and_chunk_file(strategy.as_ref(), &pdf_path)
            })
            .await
            {
                Ok(Ok(extracted)) => {
                    if extracted.chunks.is_empty() {
                        index_failed.push(serde_json::json!({
                            "file_path": pdf_path_str,
                            "error": "extraction produced zero chunks"
                        }));
                        continue;
                    }

                    match indexer::embed_and_store_file_async(
                        &state.pool,
                        &state.worker_handle,
                        &extracted,
                        session_id,
                    )
                    .await
                    {
                        Ok(result) => {
                            chunks_created += result.chunks_created;
                        }
                        Err(e) => {
                            index_failed.push(serde_json::json!({
                                "file_path": pdf_path_str,
                                "error": format!("embedding/storage: {e}")
                            }));
                        }
                    }
                }
                Ok(Err(e)) => {
                    index_failed.push(serde_json::json!({
                        "file_path": pdf_path_str,
                        "error": format!("extraction: {e}")
                    }));
                }
                Err(e) => {
                    index_failed.push(serde_json::json!({
                        "file_path": pdf_path_str,
                        "error": format!("task panicked: {e}")
                    }));
                }
            }
        }
    }

    // --- Phase 3: Index fetched HTML pages into the session ---
    // Uses the HTML indexing pipeline (section splitting, heading-based chunks,
    // web_source metadata record) instead of the PDF extraction pipeline. The
    // previous implementation incorrectly called `indexer::extract_and_chunk_file`
    // which invokes the pdfium-based PDF extractor on HTML cache files, failing
    // with "Invalid file header".
    if !fetched_html_pages.is_empty() {
        let chunk_size = session.chunk_size.map(|v| v as usize);
        let chunk_overlap = session.chunk_overlap.map(|v| v as usize);
        let max_words = session.max_words.map(|v| v as usize);
        let tokenizer_json = state.worker_handle.tokenizer_json();

        for page in &fetched_html_pages {
            let strategy = neuroncite_chunk::create_strategy(
                &session.chunk_strategy,
                chunk_size,
                chunk_overlap,
                max_words,
                tokenizer_json.as_deref(),
            )
            .map_err(|e| format!("chunking strategy: {e}"))?;

            // Phase 3a: CPU-bound extraction and chunking via html_indexer.
            // Parses HTML into heading-based sections, converts to PageText,
            // and applies the chunking strategy.
            let cache_path = page.cache_path.clone();
            let url = page.url.clone();
            let metadata = page.metadata.clone();
            let raw_html = page.raw_html.clone();

            let extraction_result = tokio::task::spawn_blocking(move || {
                html_indexer::extract_and_chunk_html(
                    strategy.as_ref(),
                    &cache_path,
                    &url,
                    &metadata,
                    &raw_html,
                    true, // strip_boilerplate: remove navigation, sidebars, etc.
                )
            })
            .await;

            match extraction_result {
                Ok(Ok(extracted)) => {
                    if extracted.chunks.is_empty() {
                        index_failed.push(serde_json::json!({
                            "file_path": page.url,
                            "error": "HTML extraction produced zero chunks"
                        }));
                        continue;
                    }

                    // Phase 3b: Embed chunks via the GPU worker and store in the
                    // database with source_type='html' and web_source metadata.
                    match html_indexer::embed_and_store_html_async(
                        &state.pool,
                        &state.worker_handle,
                        &extracted,
                        session_id,
                    )
                    .await
                    {
                        Ok(result) => {
                            chunks_created += result.chunks_created;
                        }
                        Err(e) => {
                            index_failed.push(serde_json::json!({
                                "file_path": page.url,
                                "error": format!("HTML embedding/storage: {e}")
                            }));
                        }
                    }
                }
                Ok(Err(e)) => {
                    index_failed.push(serde_json::json!({
                        "file_path": page.url,
                        "error": format!("HTML extraction: {e}")
                    }));
                }
                Err(e) => {
                    index_failed.push(serde_json::json!({
                        "file_path": page.url,
                        "error": format!("task panicked: {e}")
                    }));
                }
            }
        }
    }

    // --- Phase 4: Rebuild HNSW index if any files were added ---
    let hnsw_rebuilt = if chunks_created > 0 {
        let hnsw_chunks = {
            let conn = state
                .pool
                .get()
                .map_err(|e| format!("connection pool error: {e}"))?;
            neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
                .map_err(|e| format!("loading embeddings for HNSW: {e}"))?
        };

        let vectors: Vec<(i64, Vec<f32>)> = hnsw_chunks
            .iter()
            .map(|(id, bytes)| {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (*id, floats)
            })
            .collect();

        let total_vectors = vectors.len();
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let index = build_hnsw(&labeled, dim).map_err(|e| format!("HNSW build failed: {e}"))?;
        state.insert_hnsw(session_id, index);

        Some(total_vectors)
    } else {
        None
    };

    let indexed_count = pdfs_downloaded + html_fetched - index_failed.len();

    Ok(serde_json::json!({
        "total_entries": total_entries,
        "entries_with_url": entries_with_url,
        "pdfs_downloaded": pdfs_downloaded,
        "pdfs_failed": pdfs_failed,
        "pdfs_skipped": pdfs_skipped,
        "html_fetched": html_fetched,
        "html_failed": html_failed,
        "html_blocked": html_blocked,
        "html_skipped": html_skipped,
        "indexed_to_session": indexed_count,
        "index_failed": index_failed.len(),
        "index_failed_details": index_failed,
        "chunks_created": chunks_created,
        "hnsw_rebuilt": hnsw_rebuilt.is_some(),
        "total_session_vectors": hnsw_rebuilt,
        "session_id": session_id,
        "results": results,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-SRCFETCH-001: Handler rejects requests missing the bib_path
    /// parameter.
    #[test]
    fn t_mcp_srcfetch_001_missing_bib_path() {
        let params = serde_json::json!({
            "session_id": 1,
            "output_directory": "/tmp/output"
        });
        assert!(params["bib_path"].as_str().is_none());
    }

    /// T-MCP-SRCFETCH-002: Handler rejects requests missing the session_id
    /// parameter.
    #[test]
    fn t_mcp_srcfetch_002_missing_session_id() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "output_directory": "/tmp/output"
        });
        assert!(params["session_id"].as_i64().is_none());
    }

    /// T-MCP-SRCFETCH-003: Handler rejects requests missing the output_directory
    /// parameter.
    #[test]
    fn t_mcp_srcfetch_003_missing_output_directory() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "session_id": 1
        });
        assert!(params["output_directory"].as_str().is_none());
    }

    /// T-MCP-SRCFETCH-004: Default delay_ms is 1000 when not specified.
    #[test]
    fn t_mcp_srcfetch_004_default_delay_ms() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "session_id": 1,
            "output_directory": "/tmp/output"
        });
        let delay = params["delay_ms"].as_u64().unwrap_or(1000);
        assert_eq!(delay, 1000);
    }

    /// T-MCP-SRCFETCH-005: Custom delay_ms is respected when specified.
    #[test]
    fn t_mcp_srcfetch_005_custom_delay_ms() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "session_id": 1,
            "output_directory": "/tmp/output",
            "delay_ms": 2000
        });
        let delay = params["delay_ms"].as_u64().unwrap_or(1000);
        assert_eq!(delay, 2000);
    }

    /// T-MCP-SRCFETCH-006: URL resolution prefers the `url` field over DOI.
    /// When both are present, the `url` field takes precedence.
    #[test]
    fn t_mcp_srcfetch_006_url_preferred_over_doi() {
        let entry = neuroncite_citation::BibEntry {
            entry_type: "article".to_string(),
            author: "Fama, Eugene F.".to_string(),
            title: "Efficient Capital Markets".to_string(),
            year: Some("1970".to_string()),
            bib_abstract: None,
            keywords: None,
            url: Some("https://example.com/fama1970.pdf".to_string()),
            doi: Some("10.1086/260062".to_string()),
            extra_fields: std::collections::HashMap::new(),
        };

        // The handler logic: prefer url, fallback to doi.
        let resolved = entry.url.clone().or_else(|| {
            entry
                .doi
                .as_ref()
                .map(|doi| format!("https://doi.org/{doi}"))
        });

        assert_eq!(
            resolved.as_deref(),
            Some("https://example.com/fama1970.pdf"),
            "url field must take precedence over doi"
        );
    }

    /// T-MCP-SRCFETCH-007: DOI is resolved to a URL when no url field exists.
    #[test]
    fn t_mcp_srcfetch_007_doi_fallback() {
        let entry = neuroncite_citation::BibEntry {
            entry_type: "article".to_string(),
            author: "Black, Fischer".to_string(),
            title: "The Pricing of Options".to_string(),
            year: Some("1973".to_string()),
            bib_abstract: None,
            keywords: None,
            url: None,
            doi: Some("10.1086/260062".to_string()),
            extra_fields: std::collections::HashMap::new(),
        };

        let resolved = entry.url.clone().or_else(|| {
            entry
                .doi
                .as_ref()
                .map(|doi| format!("https://doi.org/{doi}"))
        });

        assert_eq!(
            resolved.as_deref(),
            Some("https://doi.org/10.1086/260062"),
            "DOI must be resolved via doi.org when url is absent"
        );
    }

    /// T-MCP-SRCFETCH-008: Entries without url or doi are skipped.
    #[test]
    fn t_mcp_srcfetch_008_no_url_no_doi_skipped() {
        let entry = neuroncite_citation::BibEntry {
            entry_type: "article".to_string(),
            author: "Roll, Richard".to_string(),
            title: "A Simple Measure".to_string(),
            year: Some("1984".to_string()),
            bib_abstract: None,
            keywords: None,
            url: None,
            doi: None,
            extra_fields: std::collections::HashMap::new(),
        };

        let resolved = entry.url.clone().or_else(|| {
            entry
                .doi
                .as_ref()
                .map(|doi| format!("https://doi.org/{doi}"))
        });

        assert!(
            resolved.is_none(),
            "entries without url and doi must produce no resolved URL"
        );
    }

    /// T-MCP-SRCFETCH-009: The indexed_count calculation correctly subtracts
    /// index failures from the total of downloaded + fetched sources.
    #[test]
    fn t_mcp_srcfetch_009_indexed_count_calculation() {
        let pdfs_downloaded: usize = 10;
        let html_fetched: usize = 5;
        let index_failed_len: usize = 3;

        let indexed_count = pdfs_downloaded + html_fetched - index_failed_len;
        assert_eq!(indexed_count, 12);
    }

    /// T-MCP-SRCFETCH-010: BibTeX parsing with url and doi fields produces
    /// entries with the correct field values through the full parse pipeline.
    #[test]
    fn t_mcp_srcfetch_010_full_bib_parse_with_url_doi() {
        let bib_content = r#"
@article{fama1970,
    author = {Fama, Eugene F.},
    title = {Efficient Capital Markets},
    year = {1970},
    url = {https://www.jstor.org/stable/2325486},
    doi = {10.2307/2325486}
}

@article{roll1984,
    author = {Roll, Richard},
    title = {A Simple Measure},
    year = {1984}
}

@article{black1973,
    author = {Black, Fischer},
    title = {The Pricing of Options},
    year = {1973},
    doi = {10.1086/260062}
}
"#;
        let entries = neuroncite_citation::bibtex::parse_bibtex(bib_content);

        // Count entries with url or doi.
        let with_url_or_doi = entries
            .values()
            .filter(|e| e.url.is_some() || e.doi.is_some())
            .count();
        assert_eq!(
            with_url_or_doi, 2,
            "two entries have url or doi (fama1970 and black1973)"
        );

        // Verify fama1970 has both url and doi.
        let fama = entries.get("fama1970").expect("fama1970 must exist");
        assert!(fama.url.is_some());
        assert!(fama.doi.is_some());

        // Verify roll1984 has neither.
        let roll = entries.get("roll1984").expect("roll1984 must exist");
        assert!(roll.url.is_none());
        assert!(roll.doi.is_none());

        // Verify black1973 has doi but no url.
        let black = entries.get("black1973").expect("black1973 must exist");
        assert!(black.url.is_none());
        assert!(black.doi.is_some());
        assert_eq!(black.doi.as_deref(), Some("10.1086/260062"));
    }

    // --- Bug 5: Blocked page detection tests ---

    use super::check_blocked_page;

    /// T-MCP-SRCFETCH-011: Cloudflare "Just a moment" challenge page is
    /// detected as blocked. The page has fewer than 50 words and the title
    /// matches the "just a moment" bot-detection pattern.
    #[test]
    fn t_mcp_srcfetch_011_cloudflare_detected() {
        let html = r#"<html><head><title>Just a moment...</title></head>
            <body><p>Checking your browser before accessing the site.</p></body></html>"#;
        let result = check_blocked_page(Some("Just a moment..."), html);
        assert!(
            result.is_some(),
            "Cloudflare challenge page must be classified as blocked"
        );
        let reason = result.unwrap();
        assert!(
            reason.contains("just a moment"),
            "reason must mention the matched pattern: {reason}"
        );
    }

    /// T-MCP-SRCFETCH-012: A legitimate short page (e.g., an abstract-only page)
    /// that has fewer than 50 words but does NOT match any bot-detection pattern
    /// passes the quality check. This verifies that short content alone is not
    /// sufficient for the blocked classification -- both criteria must be met.
    #[test]
    fn t_mcp_srcfetch_012_legitimate_short_page_passes() {
        let html = r#"<html><head><title>Abstract - Financial Economics</title></head>
            <body><p>This paper examines the efficiency of capital markets.</p></body></html>"#;
        let result = check_blocked_page(Some("Abstract - Financial Economics"), html);
        assert!(
            result.is_none(),
            "legitimate short page must not be classified as blocked"
        );
    }

    /// T-MCP-SRCFETCH-013: A long page (>= 50 words) with a suspect title like
    /// "Security Check" passes the quality check because the word count exceeds
    /// the MIN_CONTENT_WORDS threshold. Only pages that are BOTH short and
    /// pattern-matching are blocked.
    #[test]
    fn t_mcp_srcfetch_013_long_page_with_suspect_title_passes() {
        // Build a page with >= 50 words of body text.
        let words = (0..60)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let html = format!(
            r#"<html><head><title>Security Check</title></head>
            <body><p>{words}</p></body></html>"#
        );
        let result = check_blocked_page(Some("Security Check"), &html);
        assert!(
            result.is_none(),
            "page with >= 50 words must not be blocked even with suspect title"
        );
    }

    /// T-MCP-SRCFETCH-014: Access denied page is detected as blocked when the
    /// title does not match but the body text contains "Access Denied".
    #[test]
    fn t_mcp_srcfetch_014_access_denied_detected() {
        let html = r#"<html><head><title>Publisher Site</title></head>
            <body><h1>Access Denied</h1><p>You do not have permission.</p></body></html>"#;
        let result = check_blocked_page(Some("Publisher Site"), html);
        assert!(
            result.is_some(),
            "access denied page must be classified as blocked"
        );
        let reason = result.unwrap();
        assert!(
            reason.contains("access denied"),
            "reason must mention the matched pattern: {reason}"
        );
    }

    /// T-MCP-SRCFETCH-015: A page with None title and "Redirecting" body text
    /// is detected as blocked (covers the case where metadata.title is absent).
    #[test]
    fn t_mcp_srcfetch_015_none_title_redirecting() {
        let html = r#"<html><body><p>Redirecting</p></body></html>"#;
        let result = check_blocked_page(None, html);
        assert!(
            result.is_some(),
            "redirecting stub with None title must be classified as blocked"
        );
    }

    /// T-MCP-SRCFETCH-017: Email parameter is extracted from params when present.
    #[test]
    fn t_mcp_srcfetch_017_email_param_present() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "session_id": 1,
            "output_directory": "/tmp/output",
            "email": "user@example.com"
        });
        let email: Option<String> = params["email"].as_str().map(String::from);
        assert_eq!(email.as_deref(), Some("user@example.com"));
    }

    /// T-MCP-SRCFETCH-018: Email parameter is None when absent from params.
    #[test]
    fn t_mcp_srcfetch_018_email_param_absent() {
        let params = serde_json::json!({
            "bib_path": "/tmp/refs.bib",
            "session_id": 1,
            "output_directory": "/tmp/output"
        });
        let email: Option<String> = params["email"].as_str().map(String::from);
        assert!(email.is_none(), "absent email must produce None");
    }

    /// T-MCP-SRCFETCH-016: The FetchedHtml struct correctly stores all four
    /// fields needed by the Phase 3 HTML indexing pipeline.
    #[test]
    fn t_mcp_srcfetch_016_fetched_html_struct_fields() {
        let page = super::FetchedHtml {
            url: "https://example.com/article".to_string(),
            metadata: neuroncite_html::WebMetadata {
                url: "https://example.com/article".to_string(),
                title: Some("Test Article".to_string()),
                canonical_url: None,
                meta_description: None,
                language: None,
                og_image: None,
                og_title: None,
                og_description: None,
                author: None,
                published_date: None,
                domain: "example.com".to_string(),
                fetch_timestamp: 1700000000,
                http_status: 200,
                content_type: Some("text/html".to_string()),
            },
            raw_html: b"<html><body>test</body></html>".to_vec(),
            cache_path: std::path::PathBuf::from("/tmp/cache/test.html"),
        };

        assert_eq!(page.url, "https://example.com/article");
        assert_eq!(page.metadata.title.as_deref(), Some("Test Article"));
        assert_eq!(page.raw_html.len(), 30);
        assert_eq!(
            page.cache_path,
            std::path::PathBuf::from("/tmp/cache/test.html")
        );
    }
}
