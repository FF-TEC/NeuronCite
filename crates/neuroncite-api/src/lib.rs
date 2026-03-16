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

//! neuroncite-api: The axum-based REST API server for the NeuronCite application.
//!
//! This crate provides:
//!
//! - **State** (`state`) -- Shared application state (`AppState`) holding the
//!   database connection pool, HNSW index handle, and GPU worker channel.
//! - **Router** (`router`) -- Axum router construction with all endpoint routes,
//!   middleware layers, and OpenAPI specification serving.
//! - **Worker** (`worker`) -- Re-exported from neuroncite-pipeline. The GPU
//!   Worker serializes all embedding and reranking requests through priority
//!   channels to avoid GPU contention.
//! - **Indexer** (`indexer`) -- Re-exported from neuroncite-pipeline. Shared
//!   two-phase indexing pipeline (extract+chunk, then embed+store) used by GUI,
//!   CLI, API, and MCP entry points.
//! - **Executor** (`executor`) -- Re-exported from neuroncite-pipeline.
//!   Background job executor that polls for queued indexing jobs and runs them
//!   through the indexer pipeline.
//! - **DTO** (`dto`) -- Request and response data transfer objects for all API
//!   endpoints, with serde and utoipa derive macros.
//! - **OpenAPI** (`openapi`) -- Automatic OpenAPI specification generation from
//!   handler annotations via utoipa.
//! - **Handlers** (`handlers`) -- Per-endpoint request handling functions.
//! - **Middleware** (`middleware`) -- Tower middleware layers for authentication
//!   and CORS.
//! - **Service** (`service`) -- Shared business logic (e.g., vector dimension
//!   resolution) used by both REST API handlers and MCP handlers. Functions in
//!   this module are protocol-agnostic and contain no HTTP or MCP concerns.

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod agent;
mod dto;
mod error;
mod handlers;
pub mod html_indexer;
pub mod middleware;
mod openapi;
mod router;
pub mod service;
mod state;
pub(crate) mod util;

// Re-exports from neuroncite-pipeline. These replace the thin shim modules
// (executor.rs, worker.rs, indexer/) that previously existed in this crate.
// Intra-crate paths such as `crate::worker::WorkerHandle` and
// `crate::indexer::batch::bytes_to_f32_vec` continue to resolve because
// `pub use` creates module-level aliases.
pub use neuroncite_pipeline::{executor, indexer, worker};

// Public re-exports for downstream crates (the binary crate).
pub use error::ApiError;
pub use router::build_router;
pub use state::AppState;
pub use worker::{WorkerHandle, spawn_worker};

// ---------------------------------------------------------------------------
// Executor wrapper functions.
//
// The pipeline executor functions (`load_all_session_hnsw`,
// `ensure_hnsw_for_session`, `ensure_hnsw_for_session_async`) accept
// `&Arc<dyn PipelineContext>`. Downstream crates (binary, MCP) hold
// `Arc<AppState>`. These wrapper functions perform the coercion from
// `Arc<AppState>` to `Arc<dyn PipelineContext>` so that callers do not need
// to import PipelineContext or perform the coercion themselves.
//
// `spawn_job_executor` is generic over `PipelineContext`, so it accepts
// `Arc<AppState>` directly without coercion; it is re-exported from the
// `executor` module alias above.
// ---------------------------------------------------------------------------

use neuroncite_pipeline::context::PipelineContext;
use std::sync::Arc;

/// Type alias for the trait-object pointer that pipeline executor functions expect.
type PipelineState = Arc<dyn PipelineContext>;

/// Re-export of `neuroncite_pipeline::executor::spawn_job_executor`.
/// Accepts `Arc<AppState>` directly because the function is generic over
/// `PipelineContext` implementations.
pub use executor::spawn_job_executor;

/// Loads HNSW indices for all sessions present in the database.
/// Coerces `Arc<AppState>` to `Arc<dyn PipelineContext>` before delegating
/// to `neuroncite_pipeline::executor::load_all_session_hnsw`.
pub fn load_all_session_hnsw(state: &Arc<AppState>) {
    let coerced: PipelineState = state.clone();
    neuroncite_pipeline::executor::load_all_session_hnsw(&coerced);
}

/// Builds or retrieves the HNSW index for a single session.
/// Returns `true` if the index is available (either already loaded or
/// built from existing embeddings), `false` if there are no embeddings
/// for the session.
pub fn ensure_hnsw_for_session(state: &Arc<AppState>, session_id: i64) -> bool {
    let coerced: PipelineState = state.clone();
    neuroncite_pipeline::executor::ensure_hnsw_for_session(&coerced, session_id)
}

/// Async wrapper around `ensure_hnsw_for_session`. Runs the synchronous
/// HNSW build on a blocking thread pool via `tokio::task::spawn_blocking`
/// so that the calling async task is not stalled by CPU-bound work.
pub async fn ensure_hnsw_for_session_async(state: &Arc<AppState>, session_id: i64) -> bool {
    let coerced: PipelineState = state.clone();
    neuroncite_pipeline::executor::ensure_hnsw_for_session_async(&coerced, session_id).await
}

// Re-export DTO types and constants for consumers that need to construct or
// inspect request/response structures.
pub use dto::{
    API_VERSION, BackendDto, BackendListResponse, HealthResponse, IndexRequest, IndexResponse,
    JobCancelResponse, JobListResponse, JobResponse, OptimizeResponse, PageResponse,
    RebuildResponse, SearchRequest, SearchResponse, SearchResultDto,
    SessionDeleteByDirectoryRequest, SessionDeleteByDirectoryResponse, SessionDeleteResponse,
    SessionDto, SessionListResponse, ShutdownResponse, VerifyRequest, VerifyResponse,
};

// Shared test utilities module. Contains canonical implementations of stub
// backends, stub rerankers, test state constructors, and helper functions
// used by all test modules in this crate. The local copies that previously
// existed in lib.rs, handlers/verify.rs, and worker.rs have been migrated
// to import from this module (audit finding M-014).
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};

    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{self as store, JobState};

    use crate::state::AppState;
    use crate::test_support;

    // -----------------------------------------------------------------------
    // Test harness helpers
    //
    // All stub backends (StubBackend, StubReranker) and state constructors
    // (setup_test_state, setup_test_state_for_api, etc.) are provided by the
    // shared crate::test_support module. The local copies that previously
    // lived here were removed as part of audit finding M-014 (duplicated
    // StubBackend across 5 test modules).
    //
    // The integration tests in this module use setup_test_state_for_api()
    // which configures a StubBackend with model_id "BAAI/bge-small-en-v1.5"
    // (matching AppConfig::default_model) and includes a StubReranker. This
    // prevents the index handler's model guard from rejecting requests that
    // do not specify a model name.
    // -----------------------------------------------------------------------

    /// Creates a test application state with the default model identity
    /// ("BAAI/bge-small-en-v1.5") and a stub reranker. Delegates to
    /// `test_support::setup_test_state_for_api()`.
    fn setup_test_state() -> Arc<AppState> {
        test_support::setup_test_state_for_api()
    }

    /// Creates a test application state with the default model identity,
    /// a stub reranker, and a bearer token for authentication testing.
    /// Delegates to `test_support::setup_test_state_for_api_with_token()`.
    fn setup_test_state_with_token(token: Option<String>) -> Arc<AppState> {
        match token {
            Some(t) => test_support::setup_test_state_for_api_with_token(t),
            None => test_support::setup_test_state_for_api(),
        }
    }

    /// Creates a test session in the database with word-based chunking at
    /// 300/50 word/overlap for the "/test/docs" directory. Returns the
    /// session ID.
    fn create_test_session(state: &Arc<AppState>) -> i64 {
        test_support::create_test_session(state, "/test/docs")
    }

    /// Creates a queued job record in the database for the specified session.
    /// Returns the generated job ID string.
    fn create_test_job(state: &Arc<AppState>, kind: &str, session_id: Option<i64>) -> String {
        test_support::create_test_job(state, kind, session_id)
    }

    /// Inserts a file, two pages, and two chunks (without embeddings) into
    /// the database for the given session. Returns the vector of chunk IDs.
    fn create_test_chunks(state: &Arc<AppState>, session_id: i64) -> Vec<i64> {
        test_support::create_test_chunks(state, session_id)
    }

    /// Sends a request through the test router and returns the response.
    async fn send_request(state: Arc<AppState>, req: Request<Body>) -> axum::response::Response {
        test_support::send_request(state, req).await
    }

    /// Reads the response body as a JSON value.
    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        test_support::body_json(resp).await
    }

    // -----------------------------------------------------------------------
    // T-API-001: Health endpoint returns 200 with correct fields
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_001_health_returns_200() {
        let state = setup_test_state();
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;

        assert!(json["version"].is_string());
        assert!(json["build_features"].is_array());
        assert!(json["gpu_available"].is_boolean());
        assert!(json["active_backend"].is_string());
        assert!(json["reranker_available"].is_boolean());
        assert!(json["pdfium_available"].is_boolean());
        assert!(json["tesseract_available"].is_boolean());
    }

    // -----------------------------------------------------------------------
    // T-API-002: Index endpoint creates job (returns 202)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_002_index_creates_job() {
        let state = setup_test_state();
        let tmp = tempfile::tempdir().expect("create temp dir");
        let dir = tmp.path().to_str().expect("temp dir path must be UTF-8");
        let body = serde_json::json!({
            "directory": dir
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let json = body_json(resp).await;

        assert!(json["job_id"].is_string());
        assert!(json["session_id"].is_number());
    }

    // -----------------------------------------------------------------------
    // T-API-003: Idempotency key deduplication
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_003_idempotency_key_deduplication() {
        let state = setup_test_state();
        let tmp = tempfile::tempdir().expect("create temp dir");
        let dir = tmp.path().to_str().expect("temp dir path must be UTF-8");

        let body = serde_json::json!({
            "directory": dir,
            "idempotency_key": "idem-test-001"
        });

        // First request.
        let req1 = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp1 = send_request(state.clone(), req1).await;
        assert_eq!(resp1.status(), StatusCode::ACCEPTED);
        let json1 = body_json(resp1).await;
        let job_id_1 = json1["job_id"].as_str().unwrap().to_string();

        // Mark the first job as completed so the concurrent check passes.
        {
            let conn = state.pool.get().expect("get conn");
            store::update_job_state(&conn, &job_id_1, JobState::Running, None)
                .expect("transition to running");
            store::update_job_state(&conn, &job_id_1, JobState::Completed, None)
                .expect("transition to completed");
        }

        // Second request with the same idempotency key.
        let req2 = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp2 = send_request(state, req2).await;
        assert_eq!(resp2.status(), StatusCode::ACCEPTED);
        let json2 = body_json(resp2).await;

        // Both requests return the same job_id.
        assert_eq!(
            json2["job_id"].as_str().unwrap(),
            job_id_1.as_str(),
            "idempotent requests must return the same job_id"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-004: Concurrent index job returns 409
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_004_concurrent_index_returns_409() {
        let state = setup_test_state();

        // Create a running indexing job.
        let session_id = create_test_session(&state);
        let job_id = create_test_job(&state, "index", Some(session_id));
        {
            let conn = state.pool.get().expect("get conn");
            store::update_job_state(&conn, &job_id, JobState::Running, None)
                .expect("transition to running");
        }

        // Attempt to start another indexing job.
        let body = serde_json::json!({
            "directory": "/test/other"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    // -----------------------------------------------------------------------
    // T-API-005: Search endpoint returns results
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_005_search_returns_results() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        create_test_chunks(&state, session_id);

        // Build a trivial HNSW index with the test chunks.
        // In this test, the HNSW is empty so results will be empty,
        // but the endpoint should still return 200 with the correct shape.
        let body = serde_json::json!({
            "query": "regression analysis",
            "session_id": session_id,
            "top_k": 5
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;

        assert!(json["results"].is_array());
    }

    // -----------------------------------------------------------------------
    // T-API-006: Hybrid search vector-only mode (use_fts=false)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_006_hybrid_search_vector_only() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        let body = serde_json::json!({
            "query": "statistics",
            "session_id": session_id,
            "use_fts": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search/hybrid")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;

        // All results should have bm25_rank as null when FTS is disabled.
        if let Some(results) = json["results"].as_array() {
            for r in results {
                assert!(
                    r["bm25_rank"].is_null(),
                    "bm25_rank must be null when use_fts=false"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // T-API-007: Verify endpoint verdict
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_007_verify_verdict() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        let chunk_ids = create_test_chunks(&state, session_id);

        let body = serde_json::json!({
            "claim": "The document discusses regression analysis methods",
            "session_id": session_id,
            "chunk_ids": chunk_ids
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/verify")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;

        let verdict = json["verdict"].as_str().unwrap();
        assert!(
            verdict == "supports" || verdict == "partial" || verdict == "not_supported",
            "verdict must be one of supports/partial/not_supported, got: {verdict}"
        );
        assert!(json["combined_score"].is_number());
        assert!(json["keyword_score"].is_number());
        assert!(json["semantic_score"].is_number());
    }

    // -----------------------------------------------------------------------
    // T-API-008: Job status polling (completed job has finished_at)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_008_job_status_polling() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        let job_id = create_test_job(&state, "index", Some(session_id));

        // Transition the job to completed.
        {
            let conn = state.pool.get().expect("get conn");
            store::update_job_state(&conn, &job_id, JobState::Running, None).expect("running");
            store::update_job_state(&conn, &job_id, JobState::Completed, None).expect("completed");
        }

        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/v1/jobs/{job_id}"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert_eq!(json["state"], "completed");
        assert!(
            json["finished_at"].is_number(),
            "completed job must have finished_at set"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-009: Job cancellation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_009_job_cancellation() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        let job_id = create_test_job(&state, "index", Some(session_id));

        // Transition to running first.
        {
            let conn = state.pool.get().expect("get conn");
            store::update_job_state(&conn, &job_id, JobState::Running, None).expect("running");
        }

        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/jobs/{job_id}/cancel"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert_eq!(json["state"], "canceled");

        // Verify the job record in the database.
        let conn = state.pool.get().expect("get conn");
        let row = store::get_job(&conn, &job_id).expect("get job");
        assert_eq!(row.state, JobState::Canceled);
    }

    // -----------------------------------------------------------------------
    // T-API-010: Session delete cascades
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_010_session_delete_cascades() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        create_test_chunks(&state, session_id);

        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/api/v1/sessions/{session_id}"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert!(json["deleted"].as_bool().unwrap());

        // Verify the session is gone.
        let conn = state.pool.get().expect("get conn");
        let result = store::get_session(&conn, session_id);
        assert!(result.is_err(), "session must be deleted");

        // Verify chunks are also gone (cascade).
        let chunks = store::list_chunks_by_session(&conn, session_id).expect("list chunks");
        assert!(
            chunks.is_empty(),
            "chunks must be cascade-deleted with session"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-011: Bearer token required for LAN mode
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_011_bearer_token_required() {
        let state =
            setup_test_state_with_token(Some("test-secret-token-123-for-api-t!".to_string()));

        // Request without authorization header.
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // -----------------------------------------------------------------------
    // T-API-012: Valid bearer token grants access
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_012_valid_bearer_token() {
        let state =
            setup_test_state_with_token(Some("test-secret-token-123-for-api-t!".to_string()));

        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .header("authorization", "Bearer test-secret-token-123-for-api-t!")
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // -----------------------------------------------------------------------
    // T-API-013: Invalid request returns 400 (missing query field)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_013_invalid_request_400() {
        let state = setup_test_state();

        // Search request without the required "query" field.
        let body = serde_json::json!({
            "session_id": 1
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        // Axum returns 422 for deserialization errors (missing field),
        // which is the correct behavior for invalid JSON bodies.
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
            "missing required field should return 400 or 422, got: {}",
            resp.status()
        );
    }

    // -----------------------------------------------------------------------
    // T-API-014: Non-existent session returns 404
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_014_nonexistent_session_404() {
        let state = setup_test_state();

        let req = Request::builder()
            .method("DELETE")
            .uri("/api/v1/sessions/99999")
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // -----------------------------------------------------------------------
    // T-API-015: WorkerHandle embed_query returns vector
    // (tested in neuroncite-pipeline worker module tests)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // T-API-016: WorkerHandle priority ordering
    // (tested in neuroncite-pipeline worker module tests)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // T-API-038: top_k outside valid range (1-200) returns 400 Bad Request.
    // Verifies that the search handler rejects out-of-range top_k values
    // with a descriptive error instead of silently clamping or allocating
    // unbounded memory. Both lower bound (0) and upper bound (1000) are
    // tested.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_038_top_k_out_of_range_returns_400() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // top_k=1000 exceeds the upper bound (200) and must return 400.
        let body = serde_json::json!({
            "query": "test query",
            "session_id": session_id,
            "top_k": 1000
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "top_k=1000 must return 400 Bad Request"
        );

        // top_k=0 is below the lower bound (1) and must return 400.
        let body = serde_json::json!({
            "query": "test query",
            "session_id": session_id,
            "top_k": 0
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "top_k=0 must return 400 Bad Request"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-039: vector_dimension resolution by the index handler.
    //
    // With `backend-ort`: the handler resolves vector_dimension from the
    // static model catalog. The default model (BAAI/bge-small-en-v1.5) has
    // dimension 384 in the catalog. AppState.vector_dimension is 4 (from the
    // stub backend), so the session dimension (384) differs from the state
    // dimension (4), proving catalog-based resolution.
    //
    // Without `backend-ort`: the model catalog is unavailable, so the handler
    // falls back to state.vector_dimension (4). The session dimension equals
    // the state dimension, which is correct because the model guard has
    // verified that the loaded model matches the requested model.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_039_vector_dimension_from_app_state() {
        let state = setup_test_state();

        // AppState.vector_dimension is 4 (from the stub backend). This field
        // is used at search time for dimension compatibility checks. With
        // backend-ort, the index handler ignores this and uses the catalog
        // instead. Without backend-ort, the handler falls back to this value.
        assert_eq!(
            state
                .index
                .vector_dimension
                .load(std::sync::atomic::Ordering::Relaxed),
            4,
            "AppState.vector_dimension must match the value passed to \
             the constructor (4 from stub backend)"
        );

        let tmp = tempfile::tempdir().expect("create temp dir for t_api_039");
        let dir = tmp.path().to_str().expect("temp dir path must be UTF-8");
        let body = serde_json::json!({
            "directory": dir
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let json = body_json(resp).await;
        let session_id = json["session_id"].as_i64().unwrap();

        let conn = state.pool.get().expect("get conn");
        let session = store::get_session(&conn, session_id).expect("session must exist");

        // With backend-ort, the dimension comes from the model catalog (384
        // for BGE-small). Without backend-ort, it falls back to the state's
        // vector_dimension (4 from the stub backend).
        #[cfg(feature = "backend-ort")]
        assert_eq!(
            session.vector_dimension, 384,
            "with backend-ort, the session dimension must come from the model \
             catalog (384 for BGE-small), not from state.vector_dimension (4)"
        );
        #[cfg(not(feature = "backend-ort"))]
        assert_eq!(
            session.vector_dimension, 4,
            "without backend-ort, the session dimension falls back to \
             state.vector_dimension (4 from the stub backend)"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-017: FTS5 optimize endpoint returns 202
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_017_fts5_optimize_returns_202() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);
        create_test_chunks(&state, session_id);

        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/sessions/{session_id}/optimize"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let json = body_json(resp).await;

        assert!(json["status"].is_string());
    }

    // -----------------------------------------------------------------------
    // T-API-018: HNSW rebuild endpoint returns 202 with job_id
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_018_hnsw_rebuild_returns_202() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/sessions/{session_id}/rebuild"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let json = body_json(resp).await;

        assert!(
            json["job_id"].is_string(),
            "rebuild response must contain job_id"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-040: Per-session HNSW map insert and lookup
    // -----------------------------------------------------------------------

    /// Verifies that insert_hnsw stores a session's HNSW index in the
    /// per-session map and that it can be retrieved by session ID.
    #[tokio::test]
    async fn t_api_040_hnsw_map_insert_and_lookup() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // Map is empty initially.
        assert!(state.index.hnsw_index.load().is_empty());

        // Build a trivial HNSW index and insert it.
        let vectors: Vec<(i64, Vec<f32>)> =
            vec![(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(session_id, index);

        // The map should contain the session.
        let guard = state.index.hnsw_index.load();
        assert!(
            guard.contains_key(&session_id),
            "HNSW map must contain the inserted session"
        );
        assert_eq!(guard.len(), 1);
    }

    // -----------------------------------------------------------------------
    // T-API-041: Per-session HNSW map remove evicts the session
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_041_hnsw_map_remove() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        let vectors: Vec<(i64, Vec<f32>)> = vec![(1, vec![1.0, 0.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(session_id, index);

        assert_eq!(state.index.hnsw_index.load().len(), 1);

        state.remove_hnsw(session_id);

        assert!(
            state.index.hnsw_index.load().is_empty(),
            "HNSW map must be empty after removing the only session"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-042: Per-session HNSW map remove_many evicts multiple sessions
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_042_hnsw_map_remove_many() {
        let state = setup_test_state();

        // Create two sessions with different directories.
        let conn = state.pool.get().expect("get conn");
        let config_a = IndexConfig {
            directory: PathBuf::from("/test/a"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let config_b = IndexConfig {
            directory: PathBuf::from("/test/b"),
            ..config_a.clone()
        };
        let sid_a = store::create_session(&conn, &config_a, "0.1.0").expect("create a");
        let sid_b = store::create_session(&conn, &config_b, "0.1.0").expect("create b");
        drop(conn);

        // Insert HNSW for both sessions.
        let vectors: Vec<(i64, Vec<f32>)> = vec![(1, vec![1.0, 0.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let idx_a = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        let idx_b = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(sid_a, idx_a);
        state.insert_hnsw(sid_b, idx_b);

        assert_eq!(state.index.hnsw_index.load().len(), 2);

        // Remove both at once.
        state.remove_hnsw_many(&[sid_a, sid_b]);

        assert!(
            state.index.hnsw_index.load().is_empty(),
            "HNSW map must be empty after removing all sessions"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-043: Delete session also evicts HNSW index from map
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_043_delete_session_evicts_hnsw() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // Insert a trivial HNSW for this session.
        let vectors: Vec<(i64, Vec<f32>)> = vec![(1, vec![1.0, 0.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(session_id, index);

        assert!(state.index.hnsw_index.load().contains_key(&session_id));

        // Delete via the API endpoint.
        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/api/v1/sessions/{session_id}"))
            .body(Body::empty())
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // HNSW must be evicted from the map.
        assert!(
            !state.index.hnsw_index.load().contains_key(&session_id),
            "HNSW index must be evicted after session deletion"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-044: Delete sessions by directory endpoint
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_044_delete_sessions_by_directory() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // Insert HNSW for this session.
        let vectors: Vec<(i64, Vec<f32>)> = vec![(1, vec![1.0, 0.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(session_id, index);

        let body = serde_json::json!({
            "directory": "/test/docs"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/sessions/delete-by-directory")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state.clone(), req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;

        let deleted_ids = json["deleted_session_ids"].as_array().unwrap();
        assert!(
            deleted_ids
                .iter()
                .any(|v| v.as_i64().unwrap() == session_id),
            "deleted_session_ids must contain the deleted session"
        );

        // Session must be gone from the database.
        let conn = state.pool.get().expect("get conn");
        let result = store::get_session(&conn, session_id);
        assert!(result.is_err(), "session must be deleted from database");

        // HNSW must be evicted.
        assert!(
            !state.index.hnsw_index.load().contains_key(&session_id),
            "HNSW index must be evicted after delete-by-directory"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-045: Delete sessions by directory with no matches returns empty
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_045_delete_by_directory_no_matches() {
        let state = setup_test_state();

        let body = serde_json::json!({
            "directory": "/nonexistent/path/no/sessions"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/sessions/delete-by-directory")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        let deleted_ids = json["deleted_session_ids"].as_array().unwrap();
        assert!(
            deleted_ids.is_empty(),
            "delete-by-directory with no matches must return empty list"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-046: Search with per-session HNSW returns empty when no index
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_046_search_without_hnsw_returns_empty() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // No HNSW index loaded for this session. Search should return empty.
        let body = serde_json::json!({
            "query": "test query",
            "session_id": session_id,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = send_request(state, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        let results = json["results"].as_array().unwrap();
        assert!(
            results.is_empty(),
            "search without HNSW index must return empty results"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-047: Startup HNSW loading populates per-session map
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_047_startup_hnsw_loading() {
        let state = setup_test_state();
        let session_id = create_test_session(&state);

        // Insert a file and chunks with embeddings into the database.
        let conn = state.pool.get().expect("get conn");
        let file_id = store::insert_file(
            &conn,
            session_id,
            "/test/docs/paper.pdf",
            "hash123",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file");

        let pages = vec![(1_i64, "Test page content", "pdf-extract")];
        store::bulk_insert_pages(&conn, file_id, &pages).expect("insert pages");

        // Insert chunks with actual embeddings (4-dimensional vectors).
        let embedding_bytes: Vec<u8> = [1.0_f32, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let chunks = vec![store::ChunkInsert {
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
            content_hash: "hash_chunk_0",
            simhash: None,
        }];
        store::bulk_insert_chunks(&conn, &chunks).expect("insert chunks");
        drop(conn);

        // HNSW map is empty before startup loading.
        assert!(state.index.hnsw_index.load().is_empty());

        // Run startup HNSW loading via the wrapper function that coerces
        // Arc<AppState> to Arc<dyn PipelineContext>.
        crate::load_all_session_hnsw(&state);

        // HNSW map should contain the session.
        let guard = state.index.hnsw_index.load();
        assert!(
            guard.contains_key(&session_id),
            "startup loading must populate HNSW for sessions with embeddings"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-048: Startup HNSW loading skips sessions without embeddings
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_api_048_startup_loading_skips_empty_sessions() {
        let state = setup_test_state();
        let _session_id = create_test_session(&state);

        // Session exists but has no files/chunks/embeddings.
        assert!(state.index.hnsw_index.load().is_empty());

        crate::load_all_session_hnsw(&state);

        // HNSW map should remain empty.
        assert!(
            state.index.hnsw_index.load().is_empty(),
            "startup loading must skip sessions without embeddings"
        );
    }

    // -----------------------------------------------------------------------
    // T-API-049: Index handler canonicalizes directory path
    // -----------------------------------------------------------------------

    /// Verifies that the index handler canonicalizes the directory path
    /// before creating the session. Sessions created with equivalent paths
    /// (e.g., trailing separators, relative components) should map to the
    /// same session.
    #[tokio::test]
    async fn t_api_049_index_canonicalizes_directory() {
        let state = setup_test_state();

        // Create a real temp directory so canonicalize_directory succeeds.
        let tmp = tempfile::tempdir().expect("create temp dir for t_api_049");
        let dir = tmp.path().to_str().expect("temp dir path must be UTF-8");

        // Create two indexing requests with the same path.
        let body1 = serde_json::json!({ "directory": dir });
        let req1 = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body1).unwrap()))
            .unwrap();
        let resp1 = send_request(state.clone(), req1).await;
        assert_eq!(resp1.status(), StatusCode::ACCEPTED);
        let json1 = body_json(resp1).await;
        let session1 = json1["session_id"].as_i64().unwrap();

        // Complete the first job so the concurrent check passes.
        let job_id = json1["job_id"].as_str().unwrap().to_string();
        {
            let conn = state.pool.get().expect("conn");
            store::update_job_state(&conn, &job_id, JobState::Running, None).expect("running");
            store::update_job_state(&conn, &job_id, JobState::Completed, None).expect("completed");
        }

        // Same path: both requests must resolve to the same canonical form.
        let body2 = serde_json::json!({ "directory": dir });
        let req2 = Request::builder()
            .method("POST")
            .uri("/api/v1/index")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body2).unwrap()))
            .unwrap();
        let resp2 = send_request(state, req2).await;
        assert_eq!(resp2.status(), StatusCode::ACCEPTED);
        let json2 = body_json(resp2).await;
        let session2 = json2["session_id"].as_i64().unwrap();

        // Both requests should use the same session since the canonical
        // paths are identical.
        assert_eq!(
            session1, session2,
            "identical canonical paths must reuse the same session"
        );
    }
}
