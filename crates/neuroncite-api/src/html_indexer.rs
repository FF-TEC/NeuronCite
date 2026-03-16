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

//! HTML indexing pipeline for the NeuronCite workspace.
//!
//! This module provides a two-phase pipeline for indexing HTML web pages,
//! analogous to the PDF indexer in `indexer.rs`:
//!
//! - **Phase 1** (CPU-bound): Read cached HTML from disk, parse into
//!   heading-based sections (H1/H2), convert sections to `PageText`,
//!   and apply the chunking strategy.
//! - **Phase 2** (sequential, GPU-bound): Embed each file's chunks via
//!   the embedding backend and bulk-insert them into the SQLite database.
//!   Creates the `indexed_file` record with `source_type='html'` and
//!   the `web_source` metadata record.
//!
//! Two variants of Phase 2 are provided:
//!
//! - `embed_and_store_html` -- synchronous, takes `&dyn EmbeddingBackend`
//!   directly.
//! - `embed_and_store_html_async` -- asynchronous, routes embedding through
//!   the `WorkerHandle` priority channels.

use std::path::{Path, PathBuf};

use neuroncite_core::{Chunk, ChunkStrategy, EmbeddingBackend, ExtractionBackend, PageText};
use neuroncite_html::WebMetadata;

use crate::indexer::{
    FileIndexResult, compute_batch_size, f32_slice_to_bytes, length_sorted_permutation,
    restore_original_order,
};
use crate::worker::WorkerHandle;

// ---------------------------------------------------------------------------
// Pipeline data types
// ---------------------------------------------------------------------------

/// Result of the CPU-bound extraction and chunking phase for a single HTML file.
/// Produced by `extract_and_chunk_html`, consumed by `embed_and_store_html` or
/// `embed_and_store_html_async`.
pub struct ExtractedHtml {
    /// Path to the cached HTML file on disk.
    pub cache_path: PathBuf,
    /// Original URL of the fetched web page.
    pub url: String,
    /// Extracted `PageText` records (one per H1/H2 section).
    pub pages: Vec<PageText>,
    /// Chunks produced by the chunking strategy from the sections.
    pub chunks: Vec<Chunk>,
    /// SHA-256 hash of the cached HTML file bytes.
    pub content_hash: String,
    /// Unix timestamp (seconds) of the HTTP fetch.
    pub fetch_timestamp: i64,
    /// Size of the cached HTML file in bytes.
    pub file_size: i64,
    /// Metadata extracted from the HTML document's `<head>` and HTTP headers.
    pub metadata: WebMetadata,
    /// Raw HTML bytes for storage in the `web_source` table.
    pub raw_html: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Phase 1: CPU-bound extraction and chunking
// ---------------------------------------------------------------------------

/// Reads a cached HTML file from disk, parses it into heading-based sections
/// (H1/H2), converts sections to `PageText`, and applies the chunking strategy.
/// This function performs no I/O to the database or GPU, so multiple HTML files
/// can be processed concurrently.
///
/// When `strip_boilerplate` is true, the readability algorithm removes navigation,
/// sidebars, footers, and other non-content elements before section splitting.
/// When false, all visible text is included.
pub fn extract_and_chunk_html(
    chunk_strategy: &dyn ChunkStrategy,
    cache_path: &Path,
    url: &str,
    metadata: &WebMetadata,
    raw_html: &[u8],
    strip_boilerplate: bool,
) -> Result<ExtractedHtml, String> {
    let html_str =
        std::str::from_utf8(raw_html).map_err(|e| format!("HTML is not valid UTF-8: {e}"))?;

    // Split the HTML into heading-based sections (one section per H1/H2 heading).
    let sections = neuroncite_html::split_into_sections(html_str, strip_boilerplate)
        .map_err(|e| format!("section splitting: {e}"))?;

    if sections.is_empty() {
        tracing::warn!(url = url, "HTML produced zero sections");
        return Ok(ExtractedHtml {
            cache_path: cache_path.to_path_buf(),
            url: url.to_string(),
            pages: Vec::new(),
            chunks: Vec::new(),
            content_hash: String::new(),
            fetch_timestamp: metadata.fetch_timestamp,
            file_size: raw_html.len() as i64,
            metadata: metadata.clone(),
            raw_html: raw_html.to_vec(),
        });
    }

    // Select the extraction backend variant based on whether boilerplate
    // was removed. This is recorded in the `page` table for provenance.
    let backend = if strip_boilerplate {
        ExtractionBackend::HtmlReadability
    } else {
        ExtractionBackend::HtmlRaw
    };

    // Convert HtmlSection records to PageText for the chunking pipeline.
    // Section indices (0-based) are mapped to page numbers (1-based).
    let pages = neuroncite_html::sections_to_pages(&sections, cache_path, backend);

    let total_text_bytes: usize = pages.iter().map(|p| p.content.len()).sum();
    let total_non_ws: usize = pages
        .iter()
        .map(|p| p.content.chars().filter(|c| !c.is_whitespace()).count())
        .sum();

    tracing::debug!(
        url = url,
        sections = sections.len(),
        text_bytes = total_text_bytes,
        non_whitespace_chars = total_non_ws,
        "HTML section extraction complete"
    );

    if total_non_ws == 0 {
        tracing::warn!(
            url = url,
            sections = sections.len(),
            "HTML extraction produced only whitespace text"
        );
    }

    // Compute the SHA-256 hash of the raw HTML bytes.
    let content_hash = Chunk::compute_file_hash(cache_path).unwrap_or_default();

    let file_size = raw_html.len() as i64;

    let chunks = chunk_strategy
        .chunk(&pages)
        .map_err(|e| format!("chunking: {e}"))?;

    tracing::debug!(url = url, chunks = chunks.len(), "HTML chunking complete");

    Ok(ExtractedHtml {
        cache_path: cache_path.to_path_buf(),
        url: url.to_string(),
        pages,
        chunks,
        content_hash,
        fetch_timestamp: metadata.fetch_timestamp,
        file_size,
        metadata: metadata.clone(),
        raw_html: raw_html.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Phase 2 (synchronous): Embed and store via direct backend access
// ---------------------------------------------------------------------------

/// Inserts the indexed_file record (with source_type='html'), page records,
/// embeds chunks via the given embedding backend, bulk-inserts chunks with
/// embedding blobs, and creates the web_source metadata record.
///
/// Uses the same length-sorted batching strategy as the PDF indexer to
/// minimize padding and GPU memory usage.
pub fn embed_and_store_html(
    conn: &rusqlite::Connection,
    backend: &dyn EmbeddingBackend,
    extracted: &ExtractedHtml,
    session_id: i64,
) -> Result<FileIndexResult, String> {
    if extracted.chunks.is_empty() {
        return Ok(FileIndexResult { chunks_created: 0 });
    }

    // Remove any stale file record from a previous indexing run.
    let file_path_str = extracted.url.as_str();
    let deleted = neuroncite_store::delete_file_by_session_path(conn, session_id, file_path_str)
        .map_err(|e| format!("stale file cleanup: {e}"))?;
    if deleted > 0 {
        tracing::info!(
            session_id,
            url = file_path_str,
            "removed stale HTML file record from previous indexing run"
        );
    }

    // Insert the indexed_file record with source_type='html'.
    // The file_path column stores the URL for HTML sources.
    let file_id = neuroncite_store::insert_file_with_source_type(
        conn,
        session_id,
        file_path_str,
        &extracted.content_hash,
        extracted.fetch_timestamp,
        extracted.file_size,
        extracted.pages.len() as i64,
        None, // pdf_page_count is not applicable for HTML
        "html",
    )
    .map_err(|e| format!("file insert: {e}"))?;

    // Insert page records (one per HTML section).
    let backend_names: Vec<String> = extracted
        .pages
        .iter()
        .map(|p| p.backend.to_string())
        .collect();
    let page_tuples: Vec<(i64, &str, &str)> = extracted
        .pages
        .iter()
        .zip(backend_names.iter())
        .map(|(p, bn)| (p.page_number as i64, p.content.as_str(), bn.as_str()))
        .collect();

    if let Err(e) = neuroncite_store::bulk_insert_pages(conn, file_id, &page_tuples) {
        tracing::warn!(file_id, "page insert error: {e}");
    }

    // Compute the embedding batch size from the model's maximum sequence length.
    let batch_size = compute_batch_size(backend.max_sequence_length());
    let sorted_indices = length_sorted_permutation(&extracted.chunks);

    tracing::debug!(
        batch_size,
        max_seq_len = backend.max_sequence_length(),
        total_chunks = extracted.chunks.len(),
        "HTML embedding with length-sorted batching"
    );

    let mut sorted_embeddings: Vec<Vec<f32>> = Vec::with_capacity(extracted.chunks.len());

    for batch_start in (0..sorted_indices.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(sorted_indices.len());
        let batch_texts: Vec<&str> = sorted_indices[batch_start..batch_end]
            .iter()
            .map(|&idx| extracted.chunks[idx].content.as_str())
            .collect();

        match backend.embed_batch(&batch_texts) {
            Ok(embeddings) => sorted_embeddings.extend(embeddings),
            Err(e) => {
                if let Err(del_err) = neuroncite_store::delete_file(conn, file_id) {
                    tracing::warn!(
                        file_id,
                        "failed to clean up file record after embedding error: {del_err}"
                    );
                }
                return Err(format!("embedding: {e}"));
            }
        }
    }

    let all_embeddings = restore_original_order(sorted_embeddings, &sorted_indices);

    let result =
        insert_html_chunks_with_embeddings(conn, extracted, session_id, file_id, all_embeddings)?;

    // Insert the web_source metadata record after the file and chunks are stored.
    insert_web_source_record(conn, file_id, extracted)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Phase 2 (asynchronous): Embed via WorkerHandle, store in database
// ---------------------------------------------------------------------------

/// Asynchronous variant that routes embedding requests through the GPU worker's
/// low-priority channel, so interactive search queries remain responsive during
/// batch HTML indexing.
pub async fn embed_and_store_html_async(
    pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    worker: &WorkerHandle,
    extracted: &ExtractedHtml,
    session_id: i64,
) -> Result<FileIndexResult, String> {
    if extracted.chunks.is_empty() {
        return Ok(FileIndexResult { chunks_created: 0 });
    }

    let conn = pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let file_path_str = extracted.url.as_str();
    let deleted = neuroncite_store::delete_file_by_session_path(&conn, session_id, file_path_str)
        .map_err(|e| format!("stale file cleanup: {e}"))?;
    if deleted > 0 {
        tracing::info!(
            session_id,
            url = file_path_str,
            "removed stale HTML file record from previous indexing run"
        );
    }

    let file_id = neuroncite_store::insert_file_with_source_type(
        &conn,
        session_id,
        file_path_str,
        &extracted.content_hash,
        extracted.fetch_timestamp,
        extracted.file_size,
        extracted.pages.len() as i64,
        None,
        "html",
    )
    .map_err(|e| format!("file insert: {e}"))?;

    // Insert page records.
    let backend_names: Vec<String> = extracted
        .pages
        .iter()
        .map(|p| p.backend.to_string())
        .collect();
    let page_tuples: Vec<(i64, &str, &str)> = extracted
        .pages
        .iter()
        .zip(backend_names.iter())
        .map(|(p, bn)| (p.page_number as i64, p.content.as_str(), bn.as_str()))
        .collect();

    if let Err(e) = neuroncite_store::bulk_insert_pages(&conn, file_id, &page_tuples) {
        tracing::warn!(file_id, "page insert error: {e}");
    }

    let batch_size = compute_batch_size(worker.max_sequence_length());
    let sorted_indices = length_sorted_permutation(&extracted.chunks);

    tracing::debug!(
        batch_size,
        max_seq_len = worker.max_sequence_length(),
        total_chunks = extracted.chunks.len(),
        "async HTML embedding with length-sorted batching"
    );

    let mut sorted_embeddings: Vec<Vec<f32>> = Vec::with_capacity(extracted.chunks.len());

    for batch_start in (0..sorted_indices.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(sorted_indices.len());
        let batch_texts: Vec<String> = sorted_indices[batch_start..batch_end]
            .iter()
            .map(|&idx| extracted.chunks[idx].content.clone())
            .collect();

        match worker.embed_batch(batch_texts).await {
            Ok(embeddings) => sorted_embeddings.extend(embeddings),
            Err(e) => {
                if let Err(del_err) = neuroncite_store::delete_file(&conn, file_id) {
                    tracing::warn!(
                        file_id,
                        "failed to clean up file record after embedding error: {del_err}"
                    );
                }
                return Err(format!("embedding via worker: {e}"));
            }
        }
    }

    // Release the connection before re-acquiring for the insert.
    drop(conn);

    let conn = pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let all_embeddings = restore_original_order(sorted_embeddings, &sorted_indices);

    let result =
        insert_html_chunks_with_embeddings(&conn, extracted, session_id, file_id, all_embeddings)?;

    insert_web_source_record(&conn, file_id, extracted)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Shared insertion helpers
// ---------------------------------------------------------------------------

/// Converts embedding vectors to byte blobs, computes SimHash fingerprints,
/// and bulk-inserts chunks into the database. Shared by both sync and async
/// HTML embed-and-store variants.
fn insert_html_chunks_with_embeddings(
    conn: &rusqlite::Connection,
    extracted: &ExtractedHtml,
    session_id: i64,
    file_id: i64,
    all_embeddings: Vec<Vec<f32>>,
) -> Result<FileIndexResult, String> {
    let embedding_bytes: Vec<Vec<u8>> = all_embeddings
        .into_iter()
        .map(|emb| f32_slice_to_bytes(&emb))
        .collect();

    let chunk_inserts: Vec<neuroncite_store::ChunkInsert<'_>> = extracted
        .chunks
        .iter()
        .zip(embedding_bytes.iter())
        .map(|(chunk, emb_bytes)| {
            let simhash = neuroncite_search::compute_simhash(&chunk.content);
            neuroncite_store::ChunkInsert {
                file_id,
                session_id,
                page_start: chunk.page_start as i64,
                page_end: chunk.page_end as i64,
                chunk_index: chunk.chunk_index as i64,
                doc_text_offset_start: chunk.doc_text_offset_start as i64,
                doc_text_offset_end: chunk.doc_text_offset_end as i64,
                content: &chunk.content,
                embedding: Some(emb_bytes.as_slice()),
                ext_offset: None,
                ext_length: None,
                content_hash: &chunk.content_hash,
                simhash: Some(simhash as i64),
            }
        })
        .collect();

    let ids = neuroncite_store::bulk_insert_chunks(conn, &chunk_inserts)
        .map_err(|e| format!("chunk insert: {e}"))?;

    Ok(FileIndexResult {
        chunks_created: ids.len(),
    })
}

/// Inserts the web_source metadata record into the database, linking
/// the HTML page's metadata (URL, title, OG tags, etc.) to its
/// indexed_file record.
fn insert_web_source_record(
    conn: &rusqlite::Connection,
    file_id: i64,
    extracted: &ExtractedHtml,
) -> Result<(), String> {
    let ws = neuroncite_store::WebSourceInsert {
        file_id,
        url: &extracted.url,
        canonical_url: extracted.metadata.canonical_url.as_deref(),
        title: extracted.metadata.title.as_deref(),
        meta_description: extracted.metadata.meta_description.as_deref(),
        language: extracted.metadata.language.as_deref(),
        og_image: extracted.metadata.og_image.as_deref(),
        og_title: extracted.metadata.og_title.as_deref(),
        og_description: extracted.metadata.og_description.as_deref(),
        author: extracted.metadata.author.as_deref(),
        published_date: extracted.metadata.published_date.as_deref(),
        domain: &extracted.metadata.domain,
        fetch_timestamp: extracted.metadata.fetch_timestamp,
        http_status: extracted.metadata.http_status as i64,
        content_type: extracted.metadata.content_type.as_deref(),
        raw_html: Some(&extracted.raw_html),
    };

    neuroncite_store::insert_web_source(conn, &ws)
        .map_err(|e| format!("web_source insert: {e}"))?;

    Ok(())
}
