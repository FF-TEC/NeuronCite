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

// Shared test utility module for the neuroncite-api crate.
//
// This module consolidates stub implementations, helper constructors, and
// convenience functions used across multiple test modules (lib.rs, executor.rs,
// worker.rs, handlers/verify.rs, indexer.rs). All intra-crate test modules
// import from this module instead of maintaining their own duplicate copies.
//
// Addresses audit findings #57 (duplicated StubBackend across 5 test modules)
// and #58 (duplicated setup_test_state across 3 test modules). The M-014 fix
// completed the migration by replacing all local copies in lib.rs, verify.rs,
// and worker.rs with imports from this module.

// All items in this module are provided for use by test modules across the
// crate. The dead_code lint is suppressed at the module level because not
// every test module imports every helper function, and unused items would
// trigger false positives from #![deny(warnings)].
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use neuroncite_core::{
    AppConfig, EmbeddingBackend, IndexConfig, ModelInfo, NeuronCiteError, Reranker, StorageMode,
};
use neuroncite_store::{self as store, ChunkInsert};

use crate::state::AppState;
use crate::worker::spawn_worker;

// ---------------------------------------------------------------------------
// Stub embedding backend
// ---------------------------------------------------------------------------

/// Deterministic embedding backend for tests. Produces vectors of a configurable
/// dimensionality where the first element is 1.0 and the remaining elements are
/// 0.0. This makes distance calculations predictable without requiring a real
/// ONNX Runtime session.
///
/// The `dimension` field controls the length of the returned embedding vectors.
/// The `model_id` field controls the string returned by `loaded_model_id()`,
/// which the index handler uses to compare against the requested model name.
///
/// Common configurations used throughout the test suite:
///
/// - `StubBackend::default()` -- 4-dimensional vectors, model ID "stub-model".
///   Suitable for most unit tests that only need a functional backend.
///
/// - `StubBackend::with_dimension(384)` -- 384-dimensional vectors matching
///   the BGE-small model. Used for tests that validate dimension-aware logic.
///
/// - `StubBackend::matching_default_model()` -- 4-dimensional vectors, model ID
///   "BAAI/bge-small-en-v1.5". Used by integration tests that exercise the full
///   API router, where the index handler's model guard checks against
///   `AppConfig::default_model`.
pub struct StubBackend {
    /// Number of dimensions in the embedding vectors produced by this backend.
    pub dimension: usize,
    /// Hugging Face model identifier reported by `loaded_model_id()`.
    /// The index handler and executor compare this against the requested model
    /// to detect model mismatches.
    pub model_id: String,
}

impl Default for StubBackend {
    /// Creates a 4-dimensional stub backend with the model ID "stub-model".
    /// This is the configuration used by the majority of API integration tests.
    fn default() -> Self {
        Self {
            dimension: 4,
            model_id: "stub-model".to_string(),
        }
    }
}

impl StubBackend {
    /// Creates a stub backend with the specified vector dimensionality and
    /// the default model ID "stub-model".
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "stub-model".to_string(),
        }
    }

    /// Creates a stub backend with the specified vector dimensionality and
    /// a custom model ID string. Used by tests that validate model-mismatch
    /// detection logic.
    #[must_use]
    pub fn with_dimension_and_model(dimension: usize, model_id: impl Into<String>) -> Self {
        Self {
            dimension,
            model_id: model_id.into(),
        }
    }

    /// Creates a 4-dimensional stub backend whose `loaded_model_id()` returns
    /// "BAAI/bge-small-en-v1.5". This matches the default model name used by
    /// `AppConfig::default()`, which prevents the index handler's model guard
    /// from rejecting requests when no explicit model is specified.
    #[must_use]
    pub fn matching_default_model() -> Self {
        Self {
            dimension: 4,
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }
}

impl EmbeddingBackend for StubBackend {
    fn name(&self) -> &str {
        "stub"
    }

    fn vector_dimension(&self) -> usize {
        self.dimension
    }

    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }

    /// Returns one deterministic embedding vector per input text. Each vector
    /// has `self.dimension` elements: the first is 1.0, the rest are 0.0.
    /// This produces a valid unit-length vector for dimension=1 and a vector
    /// with known L2 norm (1.0) for all dimensions, which simplifies assertions
    /// in distance-based tests.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        Ok(texts
            .iter()
            .map(|_| {
                let mut v = vec![0.0_f32; self.dimension];
                v[0] = 1.0;
                v
            })
            .collect())
    }

    fn supports_gpu(&self) -> bool {
        false
    }

    /// Returns a single stub model entry. The vector dimension matches the
    /// backend's configured dimension so that model catalog lookups are
    /// consistent with the actual embedding output.
    fn available_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: self.model_id.clone(),
            display_name: "Stub Model".to_string(),
            vector_dimension: self.dimension,
            backend: "stub".to_string(),
        }]
    }

    fn loaded_model_id(&self) -> String {
        self.model_id.clone()
    }
}

// ---------------------------------------------------------------------------
// Stub reranker
// ---------------------------------------------------------------------------

/// Deterministic cross-encoder reranker for tests. Returns the candidate's
/// zero-based index as its relevance score (candidate 0 scores 0.0, candidate 1
/// scores 1.0, etc.). This produces a predictable ascending score order that
/// tests can assert against without requiring a real cross-encoder model.
pub struct StubReranker;

impl Reranker for StubReranker {
    fn name(&self) -> &str {
        "stub-reranker"
    }

    /// Scores each candidate by its position index. The first candidate receives
    /// score 0.0, the second 1.0, and so on. This means the last candidate
    /// always ranks highest, which tests can rely on for deterministic assertions.
    fn rerank_batch(&self, _query: &str, candidates: &[&str]) -> Result<Vec<f64>, NeuronCiteError> {
        Ok((0..candidates.len()).map(|i| i as f64).collect())
    }

    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Vector construction helpers
// ---------------------------------------------------------------------------

/// Creates a vector of `dim` elements, each initialized to `fill`.
/// Provides a concise alternative to `vec![fill; dim]` that reads clearly
/// in test assertions and HNSW index construction.
///
/// # Examples
///
/// ```ignore
/// let v = make_vector(4, 1.0);
/// assert_eq!(v, vec![1.0_f32, 1.0, 1.0, 1.0]);
/// ```
#[must_use]
pub fn make_vector(dim: usize, fill: f32) -> Vec<f32> {
    vec![fill; dim]
}

/// Creates a unit vector of `dim` elements where only the element at `axis`
/// is 1.0 and all others are 0.0. Useful for constructing orthogonal test
/// vectors in HNSW index tests where distinct directions matter for nearest
/// neighbor correctness.
///
/// # Panics
///
/// Panics if `axis >= dim`.
#[must_use]
pub fn make_unit_vector(dim: usize, axis: usize) -> Vec<f32> {
    assert!(axis < dim, "axis {axis} must be less than dim {dim}");
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    v
}

/// Serializes a slice of f32 values to little-endian bytes. This is the
/// encoding format used by neuroncite-store for storing embedding blobs
/// in SQLite. Tests that insert chunks with pre-computed embeddings need
/// this conversion to build the embedding byte representation.
#[must_use]
pub fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// AppState construction helpers
// ---------------------------------------------------------------------------

/// Creates an in-memory `AppState` with a stub embedding backend and no
/// reranker. The database is a shared in-memory SQLite instance with foreign
/// keys enabled and all migrations applied.
///
/// The stub backend produces 4-dimensional vectors with model ID "stub-model".
/// The worker is spawned with no reranker. This is the simplest test harness
/// suitable for tests that do not exercise reranking or require a specific
/// model identity.
///
/// Returns `Arc<AppState>` ready for use with axum handler tests or direct
/// state manipulation.
pub fn setup_test_state() -> Arc<AppState> {
    setup_test_state_inner(StubBackend::default(), None, None)
}

/// Creates an in-memory `AppState` with a stub embedding backend, a stub
/// reranker, and no bearer token. The presence of the reranker enables the
/// `reranker_available()` flag on the `WorkerHandle`, which the search handler
/// checks before applying cross-encoder rescoring.
pub fn setup_test_state_with_reranker() -> Arc<AppState> {
    let reranker: Arc<dyn Reranker> = Arc::new(StubReranker);
    setup_test_state_inner(StubBackend::default(), Some(reranker), None)
}

/// Creates an in-memory `AppState` with a bearer token set for LAN
/// authentication testing. The auth middleware hashes incoming tokens with
/// SHA-256 and compares against the stored hash. Pass the plaintext token
/// here; it will be hashed during `AppState::new()`.
pub fn setup_test_state_with_token(token: String) -> Arc<AppState> {
    setup_test_state_inner(StubBackend::default(), None, Some(token))
}

/// Creates an in-memory `AppState` with a stub reranker and a bearer token.
/// Combines both authentication and reranking test capabilities.
pub fn setup_test_state_with_reranker_and_token(token: String) -> Arc<AppState> {
    let reranker: Arc<dyn Reranker> = Arc::new(StubReranker);
    setup_test_state_inner(StubBackend::default(), Some(reranker), Some(token))
}

/// Creates an in-memory `AppState` with a stub backend that reports
/// "BAAI/bge-small-en-v1.5" as the loaded model ID and includes a stub
/// reranker. This configuration matches what `AppConfig::default()` expects,
/// preventing the index handler's model guard from rejecting requests that
/// do not specify a model.
///
/// This is the configuration used by the integration tests in lib.rs that
/// exercise the full API router (health, index, search, verify, etc.).
pub fn setup_test_state_for_api() -> Arc<AppState> {
    let reranker: Arc<dyn Reranker> = Arc::new(StubReranker);
    setup_test_state_inner(StubBackend::matching_default_model(), Some(reranker), None)
}

/// Creates an in-memory `AppState` that matches the default model identity
/// ("BAAI/bge-small-en-v1.5"), includes a stub reranker, and has a bearer
/// token set for authentication. Combines `setup_test_state_for_api()` with
/// bearer token authentication. Used by integration tests that exercise both
/// the model guard and the auth middleware simultaneously.
pub fn setup_test_state_for_api_with_token(token: String) -> Arc<AppState> {
    let reranker: Arc<dyn Reranker> = Arc::new(StubReranker);
    setup_test_state_inner(
        StubBackend::matching_default_model(),
        Some(reranker),
        Some(token),
    )
}

/// Creates an in-memory `AppState` with a custom `StubBackend` configuration.
/// Allows tests to control the vector dimensionality and model identity when
/// validating dimension-mismatch or model-guard logic.
pub fn setup_test_state_with_backend(backend: StubBackend) -> Arc<AppState> {
    setup_test_state_inner(backend, None, None)
}

/// Internal constructor that all `setup_test_state_*` variants delegate to.
/// Builds an r2d2 connection pool over a shared in-memory SQLite database,
/// applies schema migrations, spawns the GPU worker with the given backend
/// and optional reranker, and constructs the `AppState`.
fn setup_test_state_inner(
    backend: StubBackend,
    reranker: Option<Arc<dyn Reranker>>,
    bearer_token: Option<String>,
) -> Arc<AppState> {
    let dimension = backend.dimension;

    let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(())
    });
    let pool = r2d2::Pool::builder()
        .max_size(2)
        .build(manager)
        .expect("test pool build failed");

    // Apply database schema migrations on a pool connection.
    {
        let conn = pool.get().expect("get conn for migration");
        store::migrate(&conn).expect("migration failed on test pool");
    }

    let backend_arc: Arc<dyn EmbeddingBackend> = Arc::new(backend);
    let handle = spawn_worker(backend_arc, reranker);
    let config = AppConfig::default();

    AppState::new(pool, handle, config, true, bearer_token, dimension)
        .expect("test AppState construction must succeed")
}

// ---------------------------------------------------------------------------
// Session and job creation helpers
// ---------------------------------------------------------------------------

/// Creates a test session in the database with word-based chunking at 300/50
/// word/overlap, pointing at the given directory path. Returns the session ID.
///
/// The session uses the default stub model ("stub-model") with 4-dimensional
/// vectors and `SqliteBlob` storage mode.
pub fn create_test_session(state: &Arc<AppState>, directory: &str) -> i64 {
    create_test_session_with_config(state, directory, "word", Some(300), Some(50), None)
}

/// Creates a test session with a configurable chunking strategy. Supports all
/// strategies recognized by the chunking pipeline: "word", "sentence", "page",
/// and "token".
///
/// # Arguments
///
/// * `state` - Application state containing the database pool.
/// * `directory` - Directory path for the session's index scope.
/// * `strategy` - Chunking strategy name ("word", "sentence", "page", "token").
/// * `chunk_size` - Number of words (or sentences/pages) per chunk.
/// * `chunk_overlap` - Number of overlapping words between consecutive chunks.
/// * `max_words` - Maximum words per page-strategy chunk (only used with "page" strategy).
pub fn create_test_session_with_config(
    state: &Arc<AppState>,
    directory: &str,
    strategy: &str,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    max_words: Option<usize>,
) -> i64 {
    let conn = state.pool.get().expect("get conn for session creation");
    let config = IndexConfig {
        directory: PathBuf::from(directory),
        model_name: "stub-model".to_string(),
        chunk_strategy: strategy.to_string(),
        chunk_size,
        chunk_overlap,
        max_words,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    store::create_session(&conn, &config, "0.1.0").expect("create test session")
}

/// Creates a queued job record in the database for the specified session.
/// Returns the generated job ID (a UUID v4 string).
///
/// The job starts in the "queued" state. Tests can transition it through
/// the lifecycle states (Running, Completed, Failed, Canceled) via
/// `neuroncite_store::update_job_state`.
pub fn create_test_job(state: &Arc<AppState>, kind: &str, session_id: Option<i64>) -> String {
    let conn = state.pool.get().expect("get conn for job creation");
    let job_id = uuid::Uuid::new_v4().to_string();
    store::create_job(&conn, &job_id, kind, session_id).expect("create test job");
    job_id
}

// ---------------------------------------------------------------------------
// Test data insertion helpers
// ---------------------------------------------------------------------------

/// Inserts a file, two pages, and two chunks (without embeddings) into the
/// database for the given session. Returns the vector of chunk IDs.
///
/// The test data consists of:
/// - One file: "/test/docs/paper.pdf" with hash "hash123"
/// - Page 1: "First page of the document about statistics"
/// - Page 2: "Second page covers regression analysis methods"
/// - Chunk 0 (page 1): Same text as page 1
/// - Chunk 1 (page 2): Same text as page 2
///
/// Chunks are inserted without embedding vectors. Tests that need embedded
/// chunks should use `create_test_chunks_with_embeddings` instead.
pub fn create_test_chunks(state: &Arc<AppState>, session_id: i64) -> Vec<i64> {
    let conn = state.pool.get().expect("get conn for chunk insertion");
    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/docs/paper.pdf",
        "hash123",
        1_700_000_000,
        4096,
        2,
        None,
    )
    .expect("insert test file");

    let pages = vec![
        (
            1_i64,
            "First page of the document about statistics",
            "pdf-extract",
        ),
        (
            2,
            "Second page covers regression analysis methods",
            "pdf-extract",
        ),
    ];
    store::bulk_insert_pages(&conn, file_id, &pages).expect("insert test pages");

    let chunks = vec![
        ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "First page of the document about statistics",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_chunk_0",
            simhash: None,
        },
        ChunkInsert {
            file_id,
            session_id,
            page_start: 2,
            page_end: 2,
            chunk_index: 1,
            doc_text_offset_start: 50,
            doc_text_offset_end: 100,
            content: "Second page covers regression analysis methods",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_chunk_1",
            simhash: None,
        },
    ];
    store::bulk_insert_chunks(&conn, &chunks).expect("insert test chunks")
}

/// Inserts a file, pages, and chunks with pre-computed embedding vectors into
/// the database for the given session. Returns the vector of chunk IDs.
///
/// Each chunk receives a 4-dimensional embedding where the first element is
/// 1.0 and the rest are 0.0 (matching `StubBackend::default()` output). This
/// enables tests that exercise HNSW index building and search without requiring
/// a real embedding pipeline run.
pub fn create_test_chunks_with_embeddings(state: &Arc<AppState>, session_id: i64) -> Vec<i64> {
    let conn = state
        .pool
        .get()
        .expect("get conn for embedded chunk insertion");
    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/docs/paper.pdf",
        "hash_emb_123",
        1_700_000_000,
        4096,
        1,
        None,
    )
    .expect("insert test file for embedded chunks");

    let pages = vec![(1_i64, "Test page content for HNSW", "pdf-extract")];
    store::bulk_insert_pages(&conn, file_id, &pages).expect("insert test pages");

    let embedding_bytes = f32_slice_to_bytes(&[1.0_f32, 0.0, 0.0, 0.0]);

    let chunks = vec![ChunkInsert {
        file_id,
        session_id,
        page_start: 1,
        page_end: 1,
        chunk_index: 0,
        doc_text_offset_start: 0,
        doc_text_offset_end: 20,
        content: "Test page content for HNSW",
        embedding: Some(&embedding_bytes),
        ext_offset: None,
        ext_length: None,
        content_hash: "hash_chunk_emb_0",
        simhash: None,
    }];
    store::bulk_insert_chunks(&conn, &chunks).expect("insert embedded test chunks")
}

// ---------------------------------------------------------------------------
// Axum test request helpers
// ---------------------------------------------------------------------------

/// Sends a single HTTP request through the test router and returns the
/// response. Constructs the full axum router from the provided `AppState`,
/// dispatches the request via tower's `oneshot` method, and returns the
/// response without consuming the state (it is moved into the router).
///
/// This is the primary entry point for API integration tests that exercise
/// handler logic through the full middleware stack (auth, CORS, etc.).
pub async fn send_request(
    state: Arc<AppState>,
    req: axum::http::Request<axum::body::Body>,
) -> axum::response::Response {
    use tower::ServiceExt;
    let app = crate::router::build_router(state);
    app.oneshot(req)
        .await
        .expect("test request dispatch failed")
}

/// Reads the full response body and deserializes it as a JSON value.
/// Panics if the body cannot be read or if it is not valid JSON.
/// Intended for use in test assertions after `send_request`.
pub async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&bytes).expect("parse response body as JSON")
}
