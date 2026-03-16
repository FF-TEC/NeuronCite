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

// Shared test utilities for integration tests of the neuroncite binary crate.
//
// This module provides reusable helpers that multiple integration test files
// depend on: minimal PDF file creation for filesystem-based tests, in-memory
// SQLite connection pool construction with schema migrations applied, and
// stub implementations of the EmbeddingBackend and Reranker traits that
// produce deterministic output without requiring ML model files.
//
// Not every integration test submodule uses every utility in this module.
// The allow(dead_code) attributes prevent warnings for utilities that are
// only consumed by a subset of test files.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;

use neuroncite_core::{EmbeddingBackend, ModelInfo, NeuronCiteError, Reranker};

// ---------------------------------------------------------------------------
// Global rayon thread pool initialization
// ---------------------------------------------------------------------------

/// Configures the global rayon thread pool to use 8 MB per-thread stacks.
/// Called via `std::sync::Once` to ensure it runs at most once per process.
///
/// The pdf-extract crate's recursive CMap and Type1 font parsers consume
/// 1-2 MB of stack per extraction call. When the test harness runs multiple
/// extraction-heavy tests concurrently with rayon `par_iter()`, the default
/// ~2 MB rayon worker stacks overflow. Increasing to 8 MB provides sufficient
/// headroom for the recursive parsing paths in debug builds.
///
/// Must be called before any rayon `par_iter()` usage. Subsequent calls
/// are no-ops (the global pool can only be configured once).
#[allow(dead_code)]
pub fn init_rayon_pool() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(8 * 1024 * 1024)
            .build_global()
            .ok();
    });
}

// ---------------------------------------------------------------------------
// Heavy test serialization
// ---------------------------------------------------------------------------

/// Global mutex that serializes real-PDF integration tests.
///
/// The Rust test harness runs test functions concurrently across threads.
/// When multiple real_pdf tests execute in parallel, each spawning rayon
/// `par_iter()` over PDF files, the combined CPU and memory pressure from
/// dozens of concurrent pdf-extract and pdfium invocations causes extraction
/// timeouts and resource exhaustion. Individual tests pass; the contention
/// arises from simultaneous execution.
///
/// Each heavy test acquires this lock at entry, ensuring that at most one
/// resource-intensive PDF extraction test runs at a time. The lock is held
/// for the duration of the test via the returned `MutexGuard`.
static HEAVY_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquires the global heavy-test lock and initializes the rayon thread pool.
/// Returns a `MutexGuard` that must be held (bound to a variable) for the
/// entire test duration. Dropping the guard releases the lock, allowing the
/// next heavy test to proceed.
///
/// If a previous test panicked while holding the lock, the mutex is poisoned.
/// This function recovers from poisoning to avoid cascading test failures.
#[allow(dead_code)]
pub fn acquire_heavy_test_lock() -> MutexGuard<'static, ()> {
    init_rayon_pool();
    HEAVY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// PDF file creation
// ---------------------------------------------------------------------------

/// Creates a minimal valid PDF file at `dir/<name>`. The file contains the
/// bare minimum PDF structure (header, single page, catalog, cross-reference
/// table, trailer) with the given content string embedded as a text stream
/// on the first page.
///
/// Returns the absolute path to the created file.
///
/// The produced file is a syntactically valid PDF 1.0 document. It is
/// sufficient for the pdf-extract crate to parse and extract text from,
/// though the text recovery from such a minimal stream is not guaranteed
/// to be perfect. For integration tests that verify file discovery and
/// pipeline structure rather than extraction fidelity, this is adequate.
///
/// # Panics
///
/// Panics if the file cannot be written (test infrastructure failure).
pub fn create_test_pdf(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);

    // Construct a minimal PDF 1.0 document. The stream operator BT/ET
    // wraps text rendering operators. Tf selects font, Td positions the
    // cursor, and Tj renders the string.
    let stream_body = format!("BT /F1 12 Tf 100 700 Td ({content}) Tj ET");
    let stream_length = stream_body.len();

    let pdf_content = format!(
        "%PDF-1.0\n\
         1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
         2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
         3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\
         4 0 obj\n<< /Length {stream_length} >>\nstream\n{stream_body}\nendstream\nendobj\n\
         5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\
         xref\n0 6\n\
         0000000000 65535 f \n\
         0000000009 00000 n \n\
         0000000058 00000 n \n\
         0000000115 00000 n \n\
         0000000266 00000 n \n\
         0000000360 00000 n \n\
         trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n431\n%%EOF\n"
    );

    std::fs::write(&path, pdf_content.as_bytes()).expect("failed to write test PDF file");

    path
}

// ---------------------------------------------------------------------------
// SQLite pool creation
// ---------------------------------------------------------------------------

/// Creates an in-memory SQLite connection pool with the specified maximum
/// connection count and with schema migrations applied.
///
/// The pool uses `r2d2_sqlite::SqliteConnectionManager::memory()` with foreign
/// key enforcement enabled via the connection initializer. Schema migrations
/// from `neuroncite_store::migrate` are applied on the first connection
/// obtained from the pool.
///
/// The `max_connections` parameter controls the r2d2 pool ceiling. Tests
/// that need more than one concurrent connection must pass a value >= 2.
///
/// # Panics
///
/// Panics if pool creation or schema migration fails (test infrastructure failure).
pub fn create_test_pool(max_connections: u32) -> Pool<SqliteConnectionManager> {
    let manager = SqliteConnectionManager::memory().with_init(|conn| {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(())
    });

    let pool = Pool::builder()
        .max_size(max_connections)
        .build(manager)
        .expect("failed to build in-memory SQLite pool");

    // Apply the schema migrations on a connection from the pool.
    {
        let conn = pool
            .get()
            .expect("failed to obtain connection for migration");
        neuroncite_store::migrate(&conn).expect("schema migration failed");
    }

    pool
}

/// Creates a test connection pool with the default size of 2 connections.
///
/// This is a convenience wrapper around `create_test_pool` for the common
/// case where tests do not need to customize the pool size.
#[allow(dead_code)]
pub fn create_default_test_pool() -> Pool<SqliteConnectionManager> {
    create_test_pool(2)
}

// ---------------------------------------------------------------------------
// Stub embedding backend
// ---------------------------------------------------------------------------

/// Deterministic stub implementation of `EmbeddingBackend` that returns
/// fixed 4-dimensional vectors without loading any ML model files.
///
/// Each call to `embed_batch` produces vectors of the form `[1.0, 0.0, 0.0, 0.0]`
/// for every input text, enabling tests to verify pipeline structure and
/// data flow without requiring GPU hardware or cached model artifacts.
#[allow(dead_code)]
pub struct StubEmbedder;

impl EmbeddingBackend for StubEmbedder {
    /// Returns the backend identifier "stub".
    fn name(&self) -> &str {
        "stub"
    }

    /// Returns 4, the fixed vector dimensionality of this stub backend.
    fn vector_dimension(&self) -> usize {
        4
    }

    /// No-op model loading. Always succeeds regardless of the model identifier.
    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }

    /// Returns one `[1.0, 0.0, 0.0, 0.0]` vector per input text.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
    }

    /// Returns false because this stub does not use GPU acceleration.
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Returns a single model entry describing the stub model.
    fn available_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "stub-model".to_string(),
            display_name: "Stub Model".to_string(),
            vector_dimension: 4,
            backend: "stub".to_string(),
        }]
    }

    /// Returns "stub-model" as the loaded model identifier.
    fn loaded_model_id(&self) -> String {
        "stub-model".to_string()
    }
}

// ---------------------------------------------------------------------------
// Stub reranker
// ---------------------------------------------------------------------------

/// Deterministic stub implementation of `Reranker` that returns index-based
/// scores without loading any cross-encoder model.
///
/// For a batch of N candidates, returns scores `[0.0, 1.0, 2.0, ..., (N-1)]`,
/// making the score equal to the candidate's zero-based position index.
/// This allows tests to verify that reranking integration works and that
/// scores are correctly propagated through the pipeline.
#[allow(dead_code)]
pub struct StubReranker;

impl Reranker for StubReranker {
    /// Returns the reranker identifier "stub-reranker".
    fn name(&self) -> &str {
        "stub-reranker"
    }

    /// Returns one f64 score per candidate, where the score equals the
    /// candidate's zero-based index in the input slice.
    fn rerank_batch(&self, _query: &str, candidates: &[&str]) -> Result<Vec<f64>, NeuronCiteError> {
        Ok((0..candidates.len()).map(|i| i as f64).collect())
    }

    /// No-op model loading. Always succeeds regardless of the model identifier.
    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }
}
