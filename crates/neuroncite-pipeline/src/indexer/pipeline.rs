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

//! Core extraction and embedding pipeline functions.
//!
//! This module contains:
//! - Phase 1 (CPU-bound): `extract_and_chunk_file` for single-file extraction
//!   (PDF or HTML, detected by extension) and chunking, and
//!   `run_extraction_phase` for parallel multi-file extraction with stdout
//!   suppression and a silent panic hook.
//! - Phase 2 (synchronous): `embed_and_store_file` for direct backend access.
//! - Phase 2 (asynchronous): `embed_and_store_file_async` for WorkerHandle-routed
//!   embedding via priority channels.
//! - Shared insertion logic: `insert_chunks_with_embeddings` used by both variants.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use neuroncite_core::{Chunk, ChunkStrategy, EmbeddingBackend};

use crate::worker::WorkerHandle;

use super::batch::{
    compute_batch_size_with_caps, f32_slice_to_bytes, length_sorted_permutation,
    restore_original_order,
};
use super::stdio::StdoutSuppressionGuard;
use super::{ExtractedFile, ExtractionResult, FileIndexResult};

// ---------------------------------------------------------------------------
// Phase 1: CPU-bound extraction and chunking
// ---------------------------------------------------------------------------

/// Extracts content from a document file (PDF or HTML), computes file metadata,
/// and applies the chunking strategy. Detects the file type by extension and
/// delegates to the appropriate extraction backend:
///
/// - `.pdf` files: extracted via `neuroncite_pdf::extract_pages` with panic
///   recovery (the pdf-extract crate panics on certain malformed PDFs).
/// - `.html`/`.htm` files: parsed via `neuroncite_html::split_into_sections`
///   with readability-based boilerplate removal.
///
/// This function performs no I/O to the database or GPU, so multiple files
/// can be processed concurrently via rayon's par_iter.
pub fn extract_and_chunk_file(
    chunk_strategy: &dyn ChunkStrategy,
    file_path: &std::path::Path,
) -> Result<ExtractedFile, String> {
    if neuroncite_pdf::has_html_extension(file_path) {
        extract_and_chunk_html_file(chunk_strategy, file_path)
    } else {
        extract_and_chunk_pdf_file(chunk_strategy, file_path)
    }
}

/// Extracts pages from a PDF file, computes file metadata, and applies the
/// chunking strategy. The entire extraction is wrapped in `catch_unwind`
/// because the pdf-extract crate contains unwrap() calls that panic on
/// malformed PDFs (e.g., broken CIDRange tables, invalid Type1 encoding data).
/// Without this guard, a single corrupt PDF crashes the entire rayon thread pool.
fn extract_and_chunk_pdf_file(
    chunk_strategy: &dyn ChunkStrategy,
    pdf_path: &std::path::Path,
) -> Result<ExtractedFile, String> {
    let extraction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        neuroncite_pdf::extract_pages(pdf_path, None, None)
    }))
    .map_err(|panic| {
        let msg = if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = panic.downcast_ref::<&str>() {
            (*s).to_string()
        } else {
            "unknown panic during PDF extraction".to_string()
        };
        format!("pdf-extract panic: {msg}")
    })?
    .map_err(|e| format!("page extraction: {e}"))?;

    let pdf_page_count = extraction
        .pdf_page_count
        .map(|c| i64::try_from(c).unwrap_or(i64::MAX));
    let pages = extraction.pages;

    if pages.is_empty() {
        tracing::warn!(
            file = %pdf_path.display(),
            "extraction returned zero pages"
        );
        return Ok(ExtractedFile {
            pdf_path: pdf_path.to_path_buf(),
            pages: Vec::new(),
            chunks: Vec::new(),
            file_hash: String::new(),
            mtime: 0,
            file_size: 0,
            pdf_page_count,
        });
    }

    // Diagnostic: compute total text length and non-whitespace character count
    // across all extracted pages. This helps identify PDFs where the extraction
    // backend produced empty or garbled text (a common issue with CID fonts,
    // custom Type1 encodings, or image-only pages).
    let total_text_bytes: usize = pages.iter().map(|p| p.content.len()).sum();
    let total_non_ws: usize = pages
        .iter()
        .map(|p| p.content.chars().filter(|c| !c.is_whitespace()).count())
        .sum();

    tracing::debug!(
        file = %pdf_path.display(),
        pages = pages.len(),
        text_bytes = total_text_bytes,
        non_whitespace_chars = total_non_ws,
        "page extraction complete"
    );

    if total_non_ws == 0 {
        tracing::warn!(
            file = %pdf_path.display(),
            pages = pages.len(),
            "extraction produced only whitespace text (font mapping may have failed)"
        );
    }

    // Compute the SHA-256 hash of the raw file bytes. This is the authoritative
    // content fingerprint used by check_file_changed for Stage-2 hash verification.
    // Reading from the PDF path is safe because extract_pages already opened the
    // same file successfully above. On I/O failure (e.g., file removed between
    // extraction and hashing), the hash falls back to an empty string, which
    // always compares as ContentChanged against any stored hash, forcing a full
    // re-index on the next run rather than silently skipping the file.
    let file_hash = Chunk::compute_file_hash(pdf_path).unwrap_or_default();
    // Single metadata() call for both mtime and file_size to avoid a
    // redundant filesystem stat syscall.
    let meta = std::fs::metadata(pdf_path).ok();
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0_i64, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    let file_size = meta
        .as_ref()
        .map_or(0_i64, |m| i64::try_from(m.len()).unwrap_or(i64::MAX));

    let chunks = chunk_strategy
        .chunk(&pages)
        .map_err(|e| format!("chunking: {e}"))?;

    tracing::debug!(
        file = %pdf_path.display(),
        chunks = chunks.len(),
        "chunking complete"
    );

    if chunks.is_empty() && total_non_ws > 0 {
        tracing::warn!(
            file = %pdf_path.display(),
            non_whitespace_chars = total_non_ws,
            "chunking produced zero chunks despite non-empty text"
        );
    }

    Ok(ExtractedFile {
        pdf_path: pdf_path.to_path_buf(),
        pages,
        chunks,
        file_hash,
        mtime,
        file_size,
        pdf_page_count,
    })
}

/// Reads a local HTML file from disk, parses it into heading-based sections
/// (H1/H2) using the readability algorithm for boilerplate removal, converts
/// sections to `PageText`, and applies the chunking strategy.
///
/// This function handles HTML files discovered on the local filesystem during
/// directory indexing. It produces an `ExtractedFile` compatible with the
/// same embedding pipeline used for PDFs. The `pdf_page_count` field is set
/// to `None` since HTML files have no structural page tree.
fn extract_and_chunk_html_file(
    chunk_strategy: &dyn ChunkStrategy,
    html_path: &std::path::Path,
) -> Result<ExtractedFile, String> {
    let raw_bytes = std::fs::read(html_path)
        .map_err(|e| format!("failed to read HTML file {}: {e}", html_path.display()))?;

    let html_str = std::str::from_utf8(&raw_bytes)
        .map_err(|e| format!("HTML file {} is not valid UTF-8: {e}", html_path.display()))?;

    // Split the HTML into heading-based sections using the readability algorithm
    // to remove navigation, sidebars, footers, and other boilerplate content.
    let sections = neuroncite_html::split_into_sections(html_str, true)
        .map_err(|e| format!("HTML section splitting: {e}"))?;

    let pages = neuroncite_html::sections_to_pages(
        &sections,
        html_path,
        neuroncite_core::ExtractionBackend::HtmlReadability,
    );

    if pages.is_empty() {
        tracing::warn!(
            file = %html_path.display(),
            "HTML extraction returned zero sections"
        );
        return Ok(ExtractedFile {
            pdf_path: html_path.to_path_buf(),
            pages: Vec::new(),
            chunks: Vec::new(),
            file_hash: String::new(),
            mtime: 0,
            file_size: 0,
            pdf_page_count: None,
        });
    }

    let total_text_bytes: usize = pages.iter().map(|p| p.content.len()).sum();
    let total_non_ws: usize = pages
        .iter()
        .map(|p| p.content.chars().filter(|c| !c.is_whitespace()).count())
        .sum();

    tracing::debug!(
        file = %html_path.display(),
        sections = pages.len(),
        text_bytes = total_text_bytes,
        non_whitespace_chars = total_non_ws,
        "HTML extraction complete"
    );

    let file_hash = Chunk::compute_file_hash(html_path).unwrap_or_default();
    let meta = std::fs::metadata(html_path).ok();
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0_i64, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    let file_size = meta
        .as_ref()
        .map_or(0_i64, |m| i64::try_from(m.len()).unwrap_or(i64::MAX));

    let chunks = chunk_strategy
        .chunk(&pages)
        .map_err(|e| format!("chunking: {e}"))?;

    tracing::debug!(
        file = %html_path.display(),
        chunks = chunks.len(),
        "HTML chunking complete"
    );

    Ok(ExtractedFile {
        pdf_path: html_path.to_path_buf(),
        pages,
        chunks,
        file_hash,
        mtime,
        file_size,
        pdf_page_count: None,
    })
}

/// Runs the parallel document extraction and chunking phase with suppressed
/// panic and stdout output. Handles both PDF and HTML files -- the file type
/// is detected by extension inside `extract_and_chunk_file`.
///
/// Third-party crates used during PDF extraction produce unwanted console output:
///
/// - `pdf-extract` (v0.7.12): leftover `println!` debug statements that
///   produce "Unicode mismatch" output during font encoding processing.
/// - `type1-encoding-parser` (v0.1.0): panics on malformed Type1 font data,
///   which the default panic hook prints to stderr before `catch_unwind`
///   handles it.
///
/// This function wraps the rayon-parallel extraction in a clean environment
/// by redirecting stdout to NUL and installing a silent panic hook. Both
/// are restored after extraction completes. HTML files do not produce
/// unwanted output, but the suppression is harmless for them.
///
/// A static Mutex serializes access to the process-global resources that
/// this function modifies: stdout (fd 1 via suppress/restore) and the panic
/// hook (via take_hook/set_hook). Without this guard, concurrent extraction
/// phases from different callers (e.g., API server and MCP server) could
/// interleave their stdout redirections or overwrite each other's panic
/// hooks, leading to corrupted transport output or lost panic information.
///
/// Stdout suppression uses a `StdoutSuppressionGuard` (RAII drop guard) to
/// guarantee that stdout is restored even if the rayon pool build or the
/// extraction phase panics. Without the guard, a panic between
/// `suppress_stdout()` and the manual `restore_stdout()` call would leave
/// fd 1 permanently redirected to NUL.
///
/// The optional `on_file_done` callback is invoked after each file completes
/// extraction (success or failure). Arguments are `(files_done, files_total)`.
/// The callback is called from rayon worker threads, so it must be Send+Sync.
/// This enables real-time progress reporting during the extraction phase.
pub fn run_extraction_phase(
    chunk_strategy: &dyn ChunkStrategy,
    files: &[PathBuf],
    on_file_done: Option<&(dyn Fn(usize, usize) + Send + Sync)>,
) -> Result<Vec<ExtractionResult>, String> {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    /// Serializes access to the global panic hook and stdout redirection.
    /// Only one extraction phase may modify these process-global resources
    /// at a time. The Mutex is held for the entire extraction duration to
    /// prevent interleaving of suppress/restore sequences.
    static EXTRACTION_LOCK: Mutex<()> = Mutex::new(());

    let _guard = EXTRACTION_LOCK.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("extraction lock was poisoned by a previous panic, recovering guard");
        poisoned.into_inner()
    });

    // Install a silent panic hook that redirects panic output to tracing
    // debug level instead of printing to stderr. The catch_unwind in
    // extract_and_chunk_file captures the panic for error reporting;
    // the default hook's stderr output is just noise for the end user.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        if let Some(loc) = info.location() {
            tracing::debug!(
                "{}:{}: panic during PDF extraction (suppressed)",
                loc.file(),
                loc.line()
            );
        }
    }));

    // Redirect stdout to NUL to suppress pdf-extract debug output.
    // This is process-global and serialized by EXTRACTION_LOCK above.
    //
    // The `StdoutSuppressionGuard` ensures stdout is restored even if the
    // code below panics (e.g., rayon pool build failure). On normal
    // completion, the guard is dropped at the end of this function scope,
    // calling `restore_stdout` automatically.
    //
    // When running in MCP mode, the MCP server uses a saved fd obtained
    // via writer_from_saved_fd() BEFORE this suppression takes place, so
    // the JSON-RPC transport writes to the original stdout pipe while the
    // pdf-extract debug messages go to NUL.
    //
    // TODO(upstream): File an issue on the pdf-extract crate to remove the
    // leftover println! calls in font/CMap processing. Once fixed upstream,
    // this suppression can be removed entirely.
    let stdout_guard = StdoutSuppressionGuard::new();

    // Atomic counter for tracking completed extractions across rayon threads.
    // Each thread increments this after finishing one file, enabling the
    // progress callback to report accurate (files_done, files_total) values.
    let done_counter = AtomicUsize::new(0);
    let total = files.len();

    // Build a dedicated rayon thread pool with an 8 MB per-thread stack size.
    // The pdf-extract crate's recursive CMap/font parser can exceed the default
    // 2 MB stack on complex PDFs (especially in debug builds where stack frames
    // are larger due to absent optimizations). HTML files do not require the
    // extra stack, but the overhead is negligible. Using a local pool avoids
    // affecting the global rayon pool or other parallel workloads.
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build()
        .map_err(|e| format!("failed to build rayon extraction thread pool: {e}"))?;

    let results = pool.install(|| {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|file_path| {
                let result = extract_and_chunk_file(chunk_strategy, file_path)
                    .map_err(|e| (file_path.clone(), e));

                // Increment the atomic counter and invoke the progress callback.
                // Ordering::Relaxed is sufficient because the counter is only
                // used for display purposes, not for synchronization.
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(cb) = on_file_done {
                    cb(done, total);
                }

                result
            })
            .collect()
    });

    // Drop the stdout guard explicitly to restore stdout before restoring
    // the panic hook. The guard's Drop implementation calls `restore_stdout`
    // with the saved fd.
    drop(stdout_guard);

    // Restore the previous panic hook.
    std::panic::set_hook(prev_hook);

    Ok(results)
}

// ---------------------------------------------------------------------------
// Phase 2 (synchronous): Embed and store via direct backend access
// ---------------------------------------------------------------------------

/// Inserts file and page records into the database, embeds chunks via the
/// given embedding backend in adaptively-sized batches, and bulk-inserts
/// the chunks with their embedding blobs.
///
/// Chunks are sorted by content byte length before batching so that
/// similar-length chunks are grouped together. This minimizes padding
/// within each batch, reducing GPU memory usage and computation for the
/// attention mechanism (which scales as O(seq_len^2)). After embedding,
/// the results are reordered back to document order for database insertion.
///
/// The batch size is computed from `backend.max_sequence_length()` to avoid
/// GPU memory exhaustion with models that have long context windows (e.g.,
/// Qwen3-Embedding with 8192 tokens uses batch_size=2, while BAAI/bge-small
/// with 512 tokens uses batch_size=32).
///
/// This synchronous variant takes a borrowed `&dyn EmbeddingBackend` directly.
/// It is used by the GUI worker (which holds an `Arc<Mutex<Backend>>` and
/// passes the locked guard) and the CLI indexer (which holds a `Box<Backend>`).
pub fn embed_and_store_file(
    conn: &rusqlite::Connection,
    backend: &dyn EmbeddingBackend,
    extracted: &ExtractedFile,
    session_id: i64,
) -> Result<FileIndexResult, String> {
    if extracted.chunks.is_empty() {
        return Ok(FileIndexResult { chunks_created: 0 });
    }

    // Remove any stale file record from a previous indexing run on this
    // session. When a session is reused (same directory, model, chunk config),
    // file records from the previous run remain in the database even if
    // embedding failed partway through. Without this cleanup, the INSERT
    // below hits the UNIQUE(session_id, file_path) constraint and fails,
    // producing total_chunks=0 for the entire session.
    // ON DELETE CASCADE removes dependent page and chunk rows.
    let file_path_str = extracted.pdf_path.to_string_lossy();
    let deleted = neuroncite_store::delete_file_by_session_path(conn, session_id, &file_path_str)
        .map_err(|e| format!("stale file cleanup: {e}"))?;
    if deleted > 0 {
        tracing::info!(
            session_id,
            file = %file_path_str,
            "removed stale file record from previous indexing run"
        );
    }

    let file_id = neuroncite_store::insert_file(
        conn,
        session_id,
        &file_path_str,
        &extracted.file_hash,
        extracted.mtime,
        extracted.file_size,
        i64::try_from(extracted.pages.len()).unwrap_or(i64::MAX),
        extracted.pdf_page_count,
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
        .map(|(p, bn)| {
            let page_num = i64::try_from(p.page_number).unwrap_or(i64::MAX);
            (page_num, p.content.as_str(), bn.as_str())
        })
        .collect();

    if let Err(e) = neuroncite_store::bulk_insert_pages(conn, file_id, &page_tuples) {
        tracing::warn!(file_id, "page insert error: {e}");
    }

    // Compute the embedding batch size adapted to both the model's sequence
    // length and the platform's hardware capabilities (unified memory, VRAM).
    let batch_size = compute_batch_size_with_caps(
        backend.max_sequence_length(),
        &backend.inference_capabilities(),
    );

    // Sort chunks by content byte length (ascending) before batching.
    // Byte length correlates strongly with token count for plain text, so
    // this groups similar-length chunks into the same batch, minimizing
    // the padding added by the tokenizer to match the longest sequence.
    //
    // For Qwen3 (batch_size=2), this prevents the pathological case where
    // a 150-token chunk is paired with an 800-token chunk, which would
    // pad the short chunk to 800 tokens and waste ~4x the GPU memory
    // on attention computation.
    let sorted_indices = length_sorted_permutation(&extracted.chunks);

    tracing::debug!(
        batch_size,
        max_seq_len = backend.max_sequence_length(),
        total_chunks = extracted.chunks.len(),
        min_content_bytes = sorted_indices
            .first()
            .map(|&i| extracted.chunks[i].content.len()),
        max_content_bytes = sorted_indices
            .last()
            .map(|&i| extracted.chunks[i].content.len()),
        "embedding with length-sorted batching"
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
                // The file and page rows are already committed to the database.
                // Delete the file record (ON DELETE CASCADE removes pages) so
                // that change detection on the next indexing run does not see
                // this file as "unchanged" and skip re-indexing it.
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

    // Restore original document order so embeddings align with
    // extracted.chunks for the database insert.
    let all_embeddings = restore_original_order(sorted_embeddings, &sorted_indices);

    insert_chunks_with_embeddings(conn, extracted, session_id, file_id, all_embeddings)
}

// ---------------------------------------------------------------------------
// Phase 2 (asynchronous): Embed via WorkerHandle, store in database
// ---------------------------------------------------------------------------

/// Inserts file and page records into the database, embeds chunks via the
/// `WorkerHandle` priority channels in adaptively-sized batches, and
/// bulk-inserts the chunks with their embedding blobs.
///
/// Uses the same length-sorted batching strategy as the synchronous variant
/// to minimize padding and GPU memory usage. See `embed_and_store_file` for
/// a detailed explanation of why sorting by content length matters.
///
/// The batch size is computed from `worker.max_sequence_length()` using the
/// same formula as the synchronous variant.
///
/// This asynchronous variant routes embedding requests through the GPU worker's
/// low-priority channel, so interactive search queries (high-priority) remain
/// responsive during batch indexing. Used by the background job executor.
///
/// Composes [`prepare_and_embed_file_async`] (Phase 2a: embedding) with
/// [`store_embedded_file`] (Phase 2b: database insertion). When pipelined
/// execution is needed (overlapping DB writes with embedding of the next file),
/// callers can invoke these two phases separately.
pub async fn embed_and_store_file_async(
    pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    worker: &WorkerHandle,
    extracted: &ExtractedFile,
    session_id: i64,
) -> Result<FileIndexResult, String> {
    let embedded = prepare_and_embed_file_async(pool, worker, extracted, session_id).await?;
    store_embedded_file(pool, embedded)
}

/// Intermediate result from Phase 2a (embedding) that carries all data needed
/// for Phase 2b (database insertion). Splitting the pipeline into two phases
/// allows the executor to overlap DB writes for file N with embedding of
/// file N+1, keeping both the ANE/GPU and the database writer busy.
pub struct EmbeddedFile {
    /// Database row ID of the file record (from `insert_file`).
    pub file_id: i64,
    /// Session this file belongs to.
    pub session_id: i64,
    /// Chunk metadata cloned from the `ExtractedFile`. Needed for the
    /// `bulk_insert_chunks` call in Phase 2b.
    pub chunks: Vec<neuroncite_core::Chunk>,
    /// Embedding vectors in the original document order (already restored
    /// from length-sorted order). One vector per chunk.
    pub embeddings: Vec<Vec<f32>>,
    /// Source file path for logging and error messages.
    pub source_file: std::path::PathBuf,
}

/// Phase 2a: Embed file chunks via WorkerHandle without writing to the
/// database. Inserts the file and page metadata records (needed for the
/// file_id foreign key), computes embeddings via the GPU worker, and
/// returns an [`EmbeddedFile`] ready for Phase 2b.
///
/// On embedding failure, the already-inserted file record is cleaned up
/// (ON DELETE CASCADE removes associated page rows).
pub async fn prepare_and_embed_file_async(
    pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    worker: &WorkerHandle,
    extracted: &ExtractedFile,
    session_id: i64,
) -> Result<EmbeddedFile, String> {
    if extracted.chunks.is_empty() {
        return Ok(EmbeddedFile {
            file_id: 0,
            session_id,
            chunks: Vec::new(),
            embeddings: Vec::new(),
            source_file: extracted.pdf_path.clone(),
        });
    }

    let conn = pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let file_path_str = extracted.pdf_path.to_string_lossy();
    let deleted = neuroncite_store::delete_file_by_session_path(&conn, session_id, &file_path_str)
        .map_err(|e| format!("stale file cleanup: {e}"))?;
    if deleted > 0 {
        tracing::info!(
            session_id,
            file = %file_path_str,
            "removed stale file record from previous indexing run"
        );
    }

    let file_id = neuroncite_store::insert_file(
        &conn,
        session_id,
        &file_path_str,
        &extracted.file_hash,
        extracted.mtime,
        extracted.file_size,
        i64::try_from(extracted.pages.len()).unwrap_or(i64::MAX),
        extracted.pdf_page_count,
    )
    .map_err(|e| format!("file insert: {e}"))?;

    let backend_names: Vec<String> = extracted
        .pages
        .iter()
        .map(|p| p.backend.to_string())
        .collect();
    let page_tuples: Vec<(i64, &str, &str)> = extracted
        .pages
        .iter()
        .zip(backend_names.iter())
        .map(|(p, bn)| {
            let page_num = i64::try_from(p.page_number).unwrap_or(i64::MAX);
            (page_num, p.content.as_str(), bn.as_str())
        })
        .collect();

    if let Err(e) = neuroncite_store::bulk_insert_pages(&conn, file_id, &page_tuples) {
        tracing::warn!(file_id, "page insert error: {e}");
    }

    // Drop connection before async embedding to avoid holding it across awaits.
    drop(conn);

    let batch_size = compute_batch_size_with_caps(
        worker.max_sequence_length(),
        &worker.inference_capabilities(),
    );

    let sorted_indices = length_sorted_permutation(&extracted.chunks);

    tracing::debug!(
        batch_size,
        max_seq_len = worker.max_sequence_length(),
        total_chunks = extracted.chunks.len(),
        min_content_bytes = sorted_indices
            .first()
            .map(|&i| extracted.chunks[i].content.len()),
        max_content_bytes = sorted_indices
            .last()
            .map(|&i| extracted.chunks[i].content.len()),
        "async embedding with length-sorted batching"
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
                // Clean up the file record so change detection does not skip
                // this file on the next indexing run.
                if let Some(cc) = pool.get().ok()
                    && let Err(del_err) = neuroncite_store::delete_file(&cc, file_id)
                {
                    tracing::warn!(
                        file_id,
                        "failed to clean up file record after embedding error: {del_err}"
                    );
                }
                return Err(format!("embedding via worker: {e}"));
            }
        }
    }

    let all_embeddings = restore_original_order(sorted_embeddings, &sorted_indices);

    Ok(EmbeddedFile {
        file_id,
        session_id,
        chunks: extracted.chunks.clone(),
        embeddings: all_embeddings,
        source_file: extracted.pdf_path.clone(),
    })
}

/// Phase 2b: Store pre-embedded chunks to SQLite. Blocking (designed for
/// `tokio::task::spawn_blocking`). Takes ownership of the [`EmbeddedFile`]
/// so that the embedding vectors are freed after conversion to byte blobs.
pub fn store_embedded_file(
    pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    embedded: EmbeddedFile,
) -> Result<FileIndexResult, String> {
    if embedded.chunks.is_empty() {
        return Ok(FileIndexResult { chunks_created: 0 });
    }

    let conn = pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Build a temporary ExtractedFile-like view for insert_chunks_with_embeddings.
    // We only need chunks, source path, and the IDs — pages are already inserted
    // in Phase 2a.
    let proxy = ExtractedFile {
        pdf_path: embedded.source_file,
        pages: Vec::new(),
        chunks: embedded.chunks,
        file_hash: String::new(),
        mtime: 0,
        file_size: 0,
        pdf_page_count: None,
    };

    insert_chunks_with_embeddings(
        &conn,
        &proxy,
        embedded.session_id,
        embedded.file_id,
        embedded.embeddings,
    )
}

// ---------------------------------------------------------------------------
// Shared chunk insertion logic
// ---------------------------------------------------------------------------

/// Converts embedding vectors to byte blobs, computes SimHash fingerprints,
/// and bulk-inserts chunks into the database. Shared by both sync and async
/// embed-and-store variants. Consumes the embedding vectors via `into_iter()`
/// so that the f32 representation is freed as each embedding is converted to
/// bytes, avoiding dual storage of both f32 and u8 representations.
fn insert_chunks_with_embeddings(
    conn: &rusqlite::Connection,
    extracted: &ExtractedFile,
    session_id: i64,
    file_id: i64,
    all_embeddings: Vec<Vec<f32>>,
) -> Result<FileIndexResult, String> {
    // Convert embeddings to byte blobs for SQLite storage. Uses into_iter()
    // to consume the Vec<Vec<f32>> so each f32 vector is freed after
    // conversion, avoiding holding both representations simultaneously.
    let embedding_bytes: Vec<Vec<u8>> = all_embeddings
        .into_iter()
        .map(|emb| f32_slice_to_bytes(&emb))
        .collect();

    let chunk_inserts: Vec<neuroncite_store::ChunkInsert<'_>> = extracted
        .chunks
        .iter()
        .zip(embedding_bytes.iter())
        .map(|(chunk, emb_bytes)| {
            // Precompute the 64-bit SimHash fingerprint at insertion time so the
            // dedup module can read it directly from the database instead of
            // loading the full chunk content at every search.
            let simhash = neuroncite_search::compute_simhash(&chunk.content);
            neuroncite_store::ChunkInsert {
                file_id,
                session_id,
                page_start: i64::try_from(chunk.page_start).unwrap_or(i64::MAX),
                page_end: i64::try_from(chunk.page_end).unwrap_or(i64::MAX),
                chunk_index: i64::try_from(chunk.chunk_index).unwrap_or(i64::MAX),
                doc_text_offset_start: i64::try_from(chunk.doc_text_offset_start)
                    .unwrap_or(i64::MAX),
                doc_text_offset_end: i64::try_from(chunk.doc_text_offset_end).unwrap_or(i64::MAX),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::batch::bytes_to_f32_vec;
    use crate::test_support::StubBackend;
    use neuroncite_core::{ModelInfo, NeuronCiteError, PageText};

    /// Creates an in-memory database with migrations applied.
    fn setup_db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().expect("open db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable fk");
        neuroncite_store::migrate(&conn).expect("migrate");
        conn
    }

    /// Creates a minimal valid PDF file in a temporary directory and returns
    /// the file path.
    fn create_test_pdf(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let pdf_content = format!(
            "%PDF-1.0\n\
             1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
             2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
             3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n\
             4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n\
             5 0 obj<</Length {}>>stream\nBT /F1 12 Tf 100 700 Td ({}) Tj ET\nendstream\nendobj\n\
             xref\n0 6\n\
             0000000000 65535 f \n\
             0000000009 00000 n \n\
             0000000058 00000 n \n\
             0000000115 00000 n \n\
             0000000266 00000 n \n\
             0000000340 00000 n \n\
             trailer<</Size 6/Root 1 0 R>>\nstartxref\n{}\n%%EOF",
            content.len() + 42,
            content,
            400 + content.len()
        );
        let path = dir.join(name);
        std::fs::write(&path, pdf_content).expect("write test pdf");
        path
    }

    // -----------------------------------------------------------------------
    // T-IDX-007: embed_and_store_file handles empty chunks list
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_007_embed_and_store_empty_chunks() {
        let conn = setup_db();
        let backend = StubBackend::with_dimension(4);

        let extracted = ExtractedFile {
            pdf_path: PathBuf::from("/test/empty.pdf"),
            pages: Vec::new(),
            chunks: Vec::new(),
            file_hash: String::new(),
            mtime: 0,
            file_size: 0,
            pdf_page_count: None,
        };

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let result = embed_and_store_file(&conn, &backend, &extracted, session_id)
            .expect("embed_and_store_file should succeed for empty chunks");

        assert_eq!(result.chunks_created, 0);
    }

    // -----------------------------------------------------------------------
    // T-IDX-003: embed_and_store_file inserts file, pages, and chunks
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_003_embed_and_store_inserts_records() {
        let conn = setup_db();
        let backend = StubBackend::with_dimension(4);

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let pdf_path = PathBuf::from("/test/paper.pdf");
        let extracted = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![
                PageText {
                    source_file: pdf_path.clone(),
                    page_number: 1,
                    content: "First page about statistics".to_string(),
                    backend: neuroncite_core::ExtractionBackend::PdfExtract,
                },
                PageText {
                    source_file: pdf_path.clone(),
                    page_number: 2,
                    content: "Second page about regression".to_string(),
                    backend: neuroncite_core::ExtractionBackend::PdfExtract,
                },
            ],
            chunks: vec![
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 0,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: 27,
                    content: "First page about statistics".to_string(),
                    content_hash: Chunk::compute_content_hash("First page about statistics"),
                },
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 2,
                    page_end: 2,
                    chunk_index: 1,
                    doc_text_offset_start: 27,
                    doc_text_offset_end: 55,
                    content: "Second page about regression".to_string(),
                    content_hash: Chunk::compute_content_hash("Second page about regression"),
                },
            ],
            file_hash: "testhash123".to_string(),
            mtime: 1_700_000_000,
            file_size: 4096,
            pdf_page_count: None,
        };

        let result = embed_and_store_file(&conn, &backend, &extracted, session_id)
            .expect("embed_and_store_file should succeed");

        assert_eq!(result.chunks_created, 2);

        // Verify embeddings are stored in the database with correct dimension.
        let embeddings =
            neuroncite_store::load_embeddings_for_hnsw(&conn, session_id).expect("load embeddings");
        assert_eq!(embeddings.len(), 2, "two embeddings stored in database");
        for (chunk_id, emb_bytes) in &embeddings {
            assert!(*chunk_id > 0, "chunk_id must be positive");
            let f32_vec = bytes_to_f32_vec(emb_bytes);
            assert_eq!(f32_vec.len(), 4, "vector dimension must match backend");
        }

        // Verify the file record exists in the database.
        let files = neuroncite_store::list_files_by_session(&conn, session_id).expect("list files");
        assert_eq!(files.len(), 1);
        assert!(files[0].file_path.contains("paper.pdf"));

        // Verify page records exist.
        let pages = neuroncite_store::get_pages_by_file(&conn, files[0].id).expect("get pages");
        assert_eq!(pages.len(), 2);

        // Verify chunk records exist.
        let chunks =
            neuroncite_store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert_eq!(chunks.len(), 2);
    }

    // -----------------------------------------------------------------------
    // T-IDX-004: embed_and_store_file_async produces same results as sync
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_idx_004_async_embed_matches_sync() {
        use std::sync::Arc;

        // Create an in-memory pool for the async variant.
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend::with_dimension(4));
        let worker = crate::worker::spawn_worker(backend, None);

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let conn = pool.get().expect("conn");
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");
        drop(conn);

        let pdf_path = PathBuf::from("/test/paper.pdf");
        let extracted = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "Test page content".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks: vec![Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 17,
                content: "Test page content".to_string(),
                content_hash: Chunk::compute_content_hash("Test page content"),
            }],
            file_hash: "asynchash".to_string(),
            mtime: 1_700_000_000,
            file_size: 2048,
            pdf_page_count: None,
        };

        let result = embed_and_store_file_async(&pool, &worker, &extracted, session_id)
            .await
            .expect("async embed should succeed");

        assert_eq!(result.chunks_created, 1);

        // Verify the embedding is stored in the database.
        let conn = pool.get().expect("conn for verification");
        let embeddings =
            neuroncite_store::load_embeddings_for_hnsw(&conn, session_id).expect("load embeddings");
        assert_eq!(embeddings.len(), 1, "one embedding stored in database");
        let f32_vec = bytes_to_f32_vec(&embeddings[0].1);
        assert_eq!(f32_vec.len(), 4, "vector dimension from stub");
    }

    // -----------------------------------------------------------------------
    // T-IDX-001: extract_and_chunk_file produces ExtractedFile from test PDF
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_001_extract_and_chunk_file_produces_result() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let pdf_path = create_test_pdf(tmp.path(), "test.pdf", "Hello World from NeuronCite");

        let strategy = neuroncite_chunk::create_strategy("word", Some(300), Some(50), None, None)
            .expect("create strategy");

        let result = extract_and_chunk_file(strategy.as_ref(), &pdf_path);

        // The minimal test PDF may or may not produce extractable text depending
        // on the pdf-extract crate's ability to parse the minimal structure.
        // The function should not panic or return an error in either case.
        match result {
            Ok(ef) => {
                assert_eq!(ef.pdf_path, pdf_path);
                // File metadata should be populated regardless of text content.
                assert!(ef.file_size > 0, "file_size should be positive");
            }
            Err(e) => {
                // pdf-extract may fail on minimal PDFs -- this is acceptable.
                // The key test is that it doesn't panic.
                tracing::debug!("expected extraction failure on minimal PDF: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // T-IDX-002: run_extraction_phase processes multiple PDFs in parallel
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_002_run_extraction_phase_parallel() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let pdf1 = create_test_pdf(tmp.path(), "doc1.pdf", "First document");
        let pdf2 = create_test_pdf(tmp.path(), "doc2.pdf", "Second document");
        let pdfs = vec![pdf1, pdf2];

        let strategy = neuroncite_chunk::create_strategy("word", Some(300), Some(50), None, None)
            .expect("create strategy");

        let results = run_extraction_phase(strategy.as_ref(), &pdfs, None)
            .expect("rayon pool build must succeed");

        // Both PDFs should produce a result (success or failure).
        assert_eq!(
            results.len(),
            2,
            "extraction phase must return one result per input PDF"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-006: extract_and_chunk_file handles corrupted PDF without panic
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_006_corrupted_pdf_no_panic() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let bad_pdf = tmp.path().join("corrupt.pdf");
        std::fs::write(&bad_pdf, b"this is not a valid PDF file at all")
            .expect("write corrupt file");

        let strategy = neuroncite_chunk::create_strategy("word", Some(300), Some(50), None, None)
            .expect("create strategy");

        // Must not panic -- should return Err.
        let result = extract_and_chunk_file(strategy.as_ref(), &bad_pdf);
        assert!(
            result.is_err(),
            "corrupted PDF should return an error, not panic"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-009: Re-indexing the same file in a reused session replaces the
    //            stale record instead of failing with a UNIQUE constraint.
    // -----------------------------------------------------------------------

    /// Verifies that calling `embed_and_store_file` twice for the same PDF
    /// path and session ID succeeds on the second call. The first call
    /// inserts a file record; the second call must delete the stale record
    /// (with cascade to pages and chunks) before inserting fresh data.
    #[test]
    fn t_idx_009_reindex_same_file_replaces_stale_record() {
        let conn = setup_db();
        let backend = StubBackend::with_dimension(4);

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let pdf_path = PathBuf::from("/test/reindex.pdf");

        // Build the first ExtractedFile with one chunk.
        let extracted_v1 = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "Version one content".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks: vec![Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 19,
                content: "Version one content".to_string(),
                content_hash: Chunk::compute_content_hash("Version one content"),
            }],
            file_hash: "hash_v1".to_string(),
            mtime: 1_700_000_000,
            file_size: 1024,
            pdf_page_count: None,
        };

        // First indexing run succeeds.
        let result_v1 = embed_and_store_file(&conn, &backend, &extracted_v1, session_id)
            .expect("first indexing must succeed");
        assert_eq!(result_v1.chunks_created, 1);

        // Build the second ExtractedFile with two chunks (simulating file change).
        let extracted_v2 = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![
                PageText {
                    source_file: pdf_path.clone(),
                    page_number: 1,
                    content: "Version two page one".to_string(),
                    backend: neuroncite_core::ExtractionBackend::PdfExtract,
                },
                PageText {
                    source_file: pdf_path.clone(),
                    page_number: 2,
                    content: "Version two page two".to_string(),
                    backend: neuroncite_core::ExtractionBackend::PdfExtract,
                },
            ],
            chunks: vec![
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 0,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: 20,
                    content: "Version two page one".to_string(),
                    content_hash: Chunk::compute_content_hash("Version two page one"),
                },
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 2,
                    page_end: 2,
                    chunk_index: 1,
                    doc_text_offset_start: 20,
                    doc_text_offset_end: 40,
                    content: "Version two page two".to_string(),
                    content_hash: Chunk::compute_content_hash("Version two page two"),
                },
            ],
            file_hash: "hash_v2".to_string(),
            mtime: 1_700_100_000,
            file_size: 2048,
            pdf_page_count: None,
        };

        // Second indexing run with the same session and file path must
        // succeed, replacing the stale record from the first run.
        let result_v2 = embed_and_store_file(&conn, &backend, &extracted_v2, session_id)
            .expect("re-indexing must succeed (stale record replaced)");
        assert_eq!(result_v2.chunks_created, 2);

        // The session must contain exactly one file record (the replaced one).
        let files = neuroncite_store::list_files_by_session(&conn, session_id).expect("list files");
        assert_eq!(
            files.len(),
            1,
            "re-indexing must not create duplicate files"
        );
        assert_eq!(files[0].file_hash, "hash_v2");

        // The session must contain exactly two chunks (from v2, not v1+v2).
        let chunks =
            neuroncite_store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert_eq!(
            chunks.len(),
            2,
            "stale chunks from v1 must be cascade-deleted"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-010: Re-indexing via async variant also replaces stale records.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_idx_010_reindex_async_replaces_stale_record() {
        use std::sync::Arc;

        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend::with_dimension(4));
        let worker = crate::worker::spawn_worker(backend, None);

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let conn = pool.get().expect("conn");
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");
        drop(conn);

        let pdf_path = PathBuf::from("/test/async_reindex.pdf");

        // First indexing run via async.
        let extracted_v1 = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "Async version one".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks: vec![Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 17,
                content: "Async version one".to_string(),
                content_hash: Chunk::compute_content_hash("Async version one"),
            }],
            file_hash: "async_hash_v1".to_string(),
            mtime: 1_700_000_000,
            file_size: 512,
            pdf_page_count: None,
        };

        let result_v1 = embed_and_store_file_async(&pool, &worker, &extracted_v1, session_id)
            .await
            .expect("first async indexing must succeed");
        assert_eq!(result_v1.chunks_created, 1);

        // Second indexing run with the same file path.
        let extracted_v2 = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "Async version two".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks: vec![
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 0,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: 17,
                    content: "Async version two".to_string(),
                    content_hash: Chunk::compute_content_hash("Async version two"),
                },
                Chunk {
                    source_file: pdf_path.clone(),
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 1,
                    doc_text_offset_start: 17,
                    doc_text_offset_end: 34,
                    content: "Extra chunk added".to_string(),
                    content_hash: Chunk::compute_content_hash("Extra chunk added"),
                },
            ],
            file_hash: "async_hash_v2".to_string(),
            mtime: 1_700_100_000,
            file_size: 1024,
            pdf_page_count: None,
        };

        let result_v2 = embed_and_store_file_async(&pool, &worker, &extracted_v2, session_id)
            .await
            .expect("async re-indexing must succeed (stale record replaced)");
        assert_eq!(result_v2.chunks_created, 2);

        // Verify: one file, two chunks (from v2).
        let conn = pool.get().expect("conn");
        let files = neuroncite_store::list_files_by_session(&conn, session_id).expect("list files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_hash, "async_hash_v2");

        let chunks =
            neuroncite_store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert_eq!(chunks.len(), 2);
    }

    // -----------------------------------------------------------------------
    // T-IDX-015: StubBackend returns default max_sequence_length (512).
    //            Verifies that the default trait method works for backends
    //            that do not override it.
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_015_stub_backend_default_max_sequence_length() {
        let backend = StubBackend::with_dimension(4);
        assert_eq!(
            backend.max_sequence_length(),
            512,
            "default trait method must return 512"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-022: embed_and_store_file with length-sorted batching produces
    //            correct chunk-to-embedding alignment in the database.
    //            Uses a RecordingBackend that returns content-dependent
    //            embeddings to verify that each chunk's embedding matches
    //            its content after the sort/unsort round-trip.
    // -----------------------------------------------------------------------

    /// Backend that produces content-dependent embeddings by hashing the input
    /// text. This allows verifying that the sort/unsort round-trip in
    /// embed_and_store_file preserves the correct chunk-to-embedding mapping.
    struct ContentHashBackend {
        dimension: usize,
    }

    impl EmbeddingBackend for ContentHashBackend {
        fn name(&self) -> &str {
            "content-hash"
        }

        fn vector_dimension(&self) -> usize {
            self.dimension
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
            Ok(())
        }

        /// Produces a deterministic embedding based on the byte sum of each
        /// input text. The first element of the vector is the sum of all bytes
        /// in the text modulo 256, cast to f32. Remaining elements are zero.
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
            Ok(texts
                .iter()
                .map(|text| {
                    let byte_sum: u32 = text.bytes().map(u32::from).sum();
                    let mut v = vec![0.0_f32; self.dimension];
                    v[0] = (byte_sum % 256) as f32;
                    v
                })
                .collect())
        }

        fn supports_gpu(&self) -> bool {
            false
        }

        fn available_models(&self) -> Vec<ModelInfo> {
            vec![]
        }

        fn loaded_model_id(&self) -> String {
            "content-hash-model".to_string()
        }
    }

    #[test]
    fn t_idx_022_sorted_batching_preserves_chunk_embedding_alignment() {
        let conn = setup_db();
        let backend = ContentHashBackend { dimension: 4 };

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "content-hash-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let pdf_path = PathBuf::from("/test/sorted.pdf");

        // Create chunks with deliberately varying lengths so the sort
        // reorders them relative to document order.
        let chunk_contents = [
            "A medium length text segment here", // 33 bytes
            "Tiny",                              // 4 bytes
            "This is a significantly longer text that spans many more words than any of the other chunks", // 91 bytes
            "Short text",                       // 10 bytes
            "Another medium segment of text x", // 32 bytes
        ];

        let chunks: Vec<Chunk> = chunk_contents
            .iter()
            .enumerate()
            .map(|(i, &text)| Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: 0,
                doc_text_offset_end: text.len(),
                content: text.to_string(),
                content_hash: Chunk::compute_content_hash(text),
            })
            .collect();

        let extracted = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "placeholder".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks,
            file_hash: "sorted_test_hash".to_string(),
            mtime: 1_700_000_000,
            file_size: 4096,
            pdf_page_count: None,
        };

        let result = embed_and_store_file(&conn, &backend, &extracted, session_id)
            .expect("embed_and_store_file with sorted batching");

        assert_eq!(result.chunks_created, 5);

        // Verify that each embedding matches the content-hash of its
        // corresponding chunk by reading them back from the database.
        // This confirms the sort/unsort round-trip correctly aligned
        // embeddings with their chunks.
        // list_chunks_by_session excludes embedding blobs (performance
        // optimization), so we fetch chunk IDs first, then retrieve each
        // full row via get_chunk which includes the embedding column.
        let chunk_ids =
            neuroncite_store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert_eq!(chunk_ids.len(), 5);

        for (i, summary) in chunk_ids.iter().enumerate() {
            let full_chunk =
                neuroncite_store::get_chunk(&conn, summary.id).expect("get_chunk by id");
            let emb_bytes = full_chunk.embedding.as_ref().expect("embedding stored");
            let embedding = bytes_to_f32_vec(emb_bytes);
            let expected_byte_sum: u32 = chunk_contents[i].bytes().map(u32::from).sum();
            let expected_first = (expected_byte_sum % 256) as f32;
            assert!(
                (embedding[0] - expected_first).abs() < f32::EPSILON,
                "chunk {} embedding mismatch: expected first element {}, got {}. \
                 The sort/unsort round-trip broke chunk-to-embedding alignment.",
                i,
                expected_first,
                embedding[0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-IDX-023: async variant also preserves chunk-embedding alignment
    //            with length-sorted batching.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_idx_023_async_sorted_batching_preserves_alignment() {
        use std::sync::Arc;

        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(ContentHashBackend { dimension: 4 });
        let worker = crate::worker::spawn_worker(backend, None);

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "content-hash-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let conn = pool.get().expect("conn");
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");
        drop(conn);

        let pdf_path = PathBuf::from("/test/async_sorted.pdf");
        let chunk_contents = [
            "A medium length text segment", // 29 bytes
            "XY",                           // 2 bytes
            "Significantly longer text that has many more characters than the short ones", // 75 bytes
        ];

        let chunks: Vec<Chunk> = chunk_contents
            .iter()
            .enumerate()
            .map(|(i, &text)| Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: 0,
                doc_text_offset_end: text.len(),
                content: text.to_string(),
                content_hash: Chunk::compute_content_hash(text),
            })
            .collect();

        let extracted = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: "placeholder".to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks,
            file_hash: "async_sorted_hash".to_string(),
            mtime: 1_700_000_000,
            file_size: 2048,
            pdf_page_count: None,
        };

        let result = embed_and_store_file_async(&pool, &worker, &extracted, session_id)
            .await
            .expect("async embed with sorted batching");

        assert_eq!(result.chunks_created, 3);

        // Verify alignment by reading embeddings from the database.
        // list_chunks_by_session excludes embedding blobs (performance
        // optimization), so we fetch chunk IDs first, then retrieve each
        // full row via get_chunk which includes the embedding column.
        let conn = pool.get().expect("conn for verification");
        let chunk_ids =
            neuroncite_store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert_eq!(chunk_ids.len(), 3);

        for (i, summary) in chunk_ids.iter().enumerate() {
            let full_chunk =
                neuroncite_store::get_chunk(&conn, summary.id).expect("get_chunk by id");
            let emb_bytes = full_chunk.embedding.as_ref().expect("embedding stored");
            let embedding = bytes_to_f32_vec(emb_bytes);
            let expected_byte_sum: u32 = chunk_contents[i].bytes().map(u32::from).sum();
            let expected_first = (expected_byte_sum % 256) as f32;
            assert!(
                (embedding[0] - expected_first).abs() < f32::EPSILON,
                "async chunk {} embedding mismatch: sort/unsort alignment broken",
                i,
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-IDX-026: embed_and_store_file cleans up the file record when
    //            the embedding backend returns an error.
    //
    // Prior to the fix, embed_and_store_file wrote the file and page rows
    // to the database before calling embed_batch. On embedding failure only
    // a warning was logged and the function returned an error, leaving orphan
    // file and page rows. On subsequent runs change detection found the
    // existing record with the same hash and treated it as Unchanged, so the
    // file was silently skipped forever and its chunks were never indexed.
    //
    // After the fix, embed_and_store_file calls delete_file (which cascades
    // to pages and chunks via ON DELETE CASCADE) when embed_batch returns an
    // error, restoring the invariant that a file row in the database always
    // has a corresponding set of chunk embeddings.
    // -----------------------------------------------------------------------

    /// Backend that always returns an error from embed_batch, simulating a
    /// model or hardware failure during the embedding phase.
    struct FailingBackend;

    impl EmbeddingBackend for FailingBackend {
        fn name(&self) -> &str {
            "failing"
        }

        fn vector_dimension(&self) -> usize {
            4
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
            Ok(())
        }

        /// Returns an error unconditionally to simulate an embedding failure.
        fn embed_batch(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
            Err(NeuronCiteError::Embed(
                "simulated embedding failure".to_string(),
            ))
        }

        fn supports_gpu(&self) -> bool {
            false
        }

        fn available_models(&self) -> Vec<ModelInfo> {
            vec![]
        }

        fn loaded_model_id(&self) -> String {
            "failing-model".to_string()
        }
    }

    #[test]
    fn t_idx_006_embedding_failure_cleans_up_file_record() {
        let conn = setup_db();
        let backend = FailingBackend;

        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from("/test"),
            model_name: "failing-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let pdf_path = PathBuf::from("/test/failing.pdf");
        let chunk_text = "Some content that triggers embedding";

        let extracted = ExtractedFile {
            pdf_path: pdf_path.clone(),
            pages: vec![PageText {
                source_file: pdf_path.clone(),
                page_number: 1,
                content: chunk_text.to_string(),
                backend: neuroncite_core::ExtractionBackend::PdfExtract,
            }],
            chunks: vec![Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: chunk_text.len(),
                content: chunk_text.to_string(),
                content_hash: Chunk::compute_content_hash(chunk_text),
            }],
            file_hash: "failing_hash_001".to_string(),
            mtime: 1_700_000_000,
            file_size: 512,
            pdf_page_count: None,
        };

        // The function must return an error because the backend fails.
        let result = embed_and_store_file(&conn, &backend, &extracted, session_id);
        assert!(
            result.is_err(),
            "embed_and_store_file must return Err when the backend fails"
        );

        // The file record must have been deleted by the cleanup path so that
        // a subsequent run does not treat this file as Unchanged and skip it.
        let files = neuroncite_store::list_files_by_session(&conn, session_id)
            .expect("list_files_by_session must not fail");
        assert!(
            files.is_empty(),
            "file record must be deleted after embedding failure; found {} record(s)",
            files.len()
        );

        // Chunk records are deleted via ON DELETE CASCADE from the file row,
        // so no orphan chunks must remain in the database.
        let chunks = neuroncite_store::list_chunks_by_session(&conn, session_id)
            .expect("list_chunks_by_session must not fail");
        assert!(
            chunks.is_empty(),
            "chunk records must be deleted after embedding failure; found {} record(s)",
            chunks.len()
        );
    }
}
