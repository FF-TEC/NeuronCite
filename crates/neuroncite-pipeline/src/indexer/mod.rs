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

//! Shared indexing pipeline for the NeuronCite workspace.
//!
//! This module contains the two-phase indexing pipeline used by all entry
//! points (GUI, CLI, API server, MCP server):
//!
//! - **Phase 1** (CPU-bound, parallelizable): Extract content from documents
//!   (PDF pages via pdf-extract/pdfium, HTML sections via readability parser)
//!   and split them into chunks using rayon's parallel iterator.
//! - **Phase 2** (sequential, GPU-bound): Embed each file's chunks via the
//!   embedding backend and bulk-insert them into the SQLite database.
//!
//! Two variants of Phase 2 are provided:
//!
//! - `embed_and_store_file` -- synchronous, takes `&dyn EmbeddingBackend`
//!   directly. Used by the GUI worker (which holds an `Arc<Mutex<Backend>>`)
//!   and the CLI indexer (which holds a `Box<Backend>`).
//! - `embed_and_store_file_async` -- asynchronous, routes embedding through
//!   the `WorkerHandle` priority channels. Used by the background job executor
//!   so that interactive search queries retain priority over batch indexing.
//!
//! The module is split into three sub-modules for maintainability:
//!
//! - `batch` -- Adaptive batch size computation, length-sorted permutation,
//!   and byte conversion utilities.
//! - `pipeline` -- Core extraction, embedding, and database insertion functions.
//! - `stdio` -- Stdout suppression/restoration for safe PDF extraction in the
//!   presence of noisy third-party crates.

use std::path::PathBuf;

use neuroncite_core::{Chunk, PageText};

pub mod batch;
pub mod pipeline;
pub mod stdio;

// ---------------------------------------------------------------------------
// Re-exports: all items that were previously public in the monolithic
// indexer.rs are re-exported here so that callers (executor.rs, agent.rs,
// handlers/index.rs, main.rs, neuroncite-mcp handlers) continue to work
// without import path changes.
// ---------------------------------------------------------------------------

// From batch module: byte conversion and batch computation utilities.
pub use batch::{
    bytes_to_f32_vec, compute_batch_size, compute_batch_size_with_caps, f32_slice_to_bytes,
    length_sorted_permutation, restore_original_order,
};

// From pipeline module: extraction and embedding pipeline functions.
pub use pipeline::{
    EmbeddedFile, embed_and_store_file, embed_and_store_file_async, extract_and_chunk_file,
    prepare_and_embed_file_async, run_extraction_phase, store_embedded_file,
};

// From stdio module: stdout suppression utilities and the MCP writer flag.
pub use stdio::{
    STDOUT_MCP_WRITER_INITIALIZED, StdoutSuppressionGuard, restore_stdout, suppress_stdout,
    writer_from_saved_fd,
};

// ---------------------------------------------------------------------------
// Pipeline data types
// ---------------------------------------------------------------------------

/// Result of extracting a single document: either the extracted content or
/// the file path and an error message describing what went wrong.
pub type ExtractionResult = Result<ExtractedFile, (PathBuf, String)>;

/// Result of the CPU-bound extraction and chunking phase for a single document
/// (PDF or HTML). Produced in parallel by `extract_and_chunk_file`, consumed
/// sequentially by `embed_and_store_file` or `embed_and_store_file_async`.
pub struct ExtractedFile {
    /// Path to the source document on disk (PDF or HTML file).
    pub pdf_path: PathBuf,
    pub pages: Vec<PageText>,
    pub chunks: Vec<Chunk>,
    pub file_hash: String,
    pub mtime: i64,
    pub file_size: i64,
    /// Structural page count from the PDF page tree (lopdf). `None` for HTML
    /// files. Independent of the number of pages that produced extractable
    /// text. Used by the MCP `neuroncite_files` handler to report extraction
    /// completeness.
    pub pdf_page_count: Option<i64>,
}

/// Result of the sequential embed-and-store phase for a single document.
/// Embedding vectors are persisted to SQLite and are not kept in memory.
/// The caller reads them back from the database after all files have been
/// processed, using `load_embeddings_for_hnsw`, to build the HNSW index.
pub struct FileIndexResult {
    /// Number of chunks created from this PDF file.
    pub chunks_created: usize,
}
