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

//! neuroncite-web: Browser-based frontend server for NeuronCite.
//!
//! This crate provides:
//!
//! - **Assets** (`assets`) -- Embedded static file serving with SPA fallback
//!   routing. The compiled SolidJS frontend from `frontend/dist/` is baked into
//!   the binary via `rust-embed` and served with correct MIME types.
//!
//! - **SSE** (`sse`) -- Server-Sent Event endpoints for real-time streaming of
//!   log messages, indexing progress, job state transitions, and model operations.
//!   Uses `tokio::sync::broadcast` channels subscribed to by SSE consumers.
//!
//! - **Browser** (`browser`) -- Cross-platform browser launcher that opens the
//!   default browser to the server URL after startup.
//!
//! - **Handlers** (`handlers`) -- REST API handlers for web-specific endpoints
//!   under `/api/v1/web/` that the headless API does not provide: file system
//!   browsing, model catalog/download/activation, dependency probes, MCP
//!   registration, and configuration reading.

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod assets;
pub mod broadcast_layer;
pub mod browser;
pub mod handlers;
pub mod middleware;
#[cfg(feature = "gui")]
pub mod native_window;
#[cfg(feature = "gui")]
pub mod splash;
pub mod sse;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use axum::Router;

use neuroncite_api::AppState;

/// Broadcast channel capacity for indexing progress updates. Matches the
/// log channel capacity because progress events are emitted at a similar
/// rate during batch embedding.
#[allow(dead_code)]
const PROGRESS_CHANNEL_CAPACITY: usize = 1024;

/// Broadcast channel capacity for job state transition events. Jobs
/// transition infrequently (queued -> running -> completed), so a smaller
/// capacity suffices.
const JOB_CHANNEL_CAPACITY: usize = 256;

/// Broadcast channel capacity for model operation events (download progress,
/// model switch completion). Model downloads emit frequent progress updates
/// but fewer than log messages.
const MODEL_CHANNEL_CAPACITY: usize = 64;

/// Broadcast channel capacity for citation verification events. The citation
/// agent emits per-row updates and streaming reasoning tokens at moderate
/// throughput.
#[allow(dead_code)]
const CITATION_CHANNEL_CAPACITY: usize = 256;

/// Broadcast channel capacity for source fetching events. Per-entry status
/// updates during BibTeX source downloading.
#[allow(dead_code)]
const SOURCE_CHANNEL_CAPACITY: usize = 128;

/// Shared state for web-specific functionality, holding broadcast channels for
/// SSE event distribution and a reference to the core API application state.
pub struct WebState {
    /// Broadcast channel sender for log messages. The custom tracing Layer
    /// publishes formatted log events here; SSE consumers subscribe via
    /// `log_tx.subscribe()`.
    pub log_tx: tokio::sync::broadcast::Sender<String>,

    /// Broadcast channel sender for indexing progress updates.
    pub progress_tx: tokio::sync::broadcast::Sender<String>,

    /// Broadcast channel sender for job state transition events.
    pub job_tx: tokio::sync::broadcast::Sender<String>,

    /// Broadcast channel sender for model operation events (download progress,
    /// model switch completion, reranker load completion).
    pub model_tx: tokio::sync::broadcast::Sender<String>,

    /// Broadcast channel sender for citation verification events. Carries
    /// per-row status updates, streaming reasoning tokens, and job progress
    /// events from the autonomous citation agent to SSE consumers.
    pub citation_tx: tokio::sync::broadcast::Sender<String>,

    /// Broadcast channel sender for source fetching events. Carries per-entry
    /// status updates from the fetch_sources handler to SSE consumers so the
    /// frontend can display live download progress.
    pub source_tx: tokio::sync::broadcast::Sender<String>,

    /// Reference to the shared core API application state for database pool
    /// and worker handle access from web-specific handlers.
    pub app_state: Arc<AppState>,

    /// Tracks the number of active SSE connections across all event streams.
    /// Incremented when an SSE handler starts streaming, decremented when the
    /// client disconnects. Used to reject connections above the configured
    /// maximum (MAX_SSE_CONNECTIONS) to prevent resource exhaustion from
    /// excessive open HTTP connections.
    pub sse_connections: AtomicUsize,

    /// Whether native OS file/folder dialogs (rfd) are available in this
    /// runtime configuration. True only in GUI mode where the main thread
    /// runs a platform event loop (tao). False in non-GUI web mode and
    /// headless serve mode, where calling rfd would deadlock on macOS
    /// (NSOpenPanel requires the main thread's run loop, but tokio's
    /// block_on parks it) or fail silently on headless Linux.
    pub native_dialogs: bool,
}

impl WebState {
    /// Constructs a new `WebState` with a pre-created log broadcast channel
    /// (shared with the BroadcastLayer tracing subscriber) and fresh channels
    /// for progress, jobs, and model events.
    ///
    /// The `log_tx` parameter is the same sender that was passed to
    /// `BroadcastLayer::new()` during tracing initialization. This ensures
    /// tracing events flow through to SSE `/events/logs` subscribers.
    #[must_use]
    pub fn new(
        app_state: Arc<AppState>,
        log_tx: tokio::sync::broadcast::Sender<String>,
    ) -> Arc<Self> {
        Self::with_native_dialogs(app_state, log_tx, false)
    }

    /// Constructs a new `WebState` with explicit control over whether native
    /// OS dialogs are available. GUI mode passes `true`; non-GUI and headless
    /// modes pass `false`.
    #[must_use]
    pub fn with_native_dialogs(
        app_state: Arc<AppState>,
        log_tx: tokio::sync::broadcast::Sender<String>,
        native_dialogs: bool,
    ) -> Arc<Self> {
        // Clone the progress broadcast sender from AppState. The sender is
        // initialised unconditionally in AppState::new(), so the clone always
        // succeeds. Progress events sent by the job executor flow through this
        // channel to SSE /events/progress subscribers.
        let progress_tx = app_state.sse.progress_tx.clone();

        let (job_tx, _) = tokio::sync::broadcast::channel(JOB_CHANNEL_CAPACITY);
        let (model_tx, _) = tokio::sync::broadcast::channel(MODEL_CHANNEL_CAPACITY);

        // Clone the citation broadcast sender from AppState. Citation agent
        // events flow through this channel to SSE /events/citation subscribers.
        let citation_tx = app_state.sse.citation_tx.clone();

        // Clone the source fetch broadcast sender from AppState. The
        // fetch_sources handler uses this sender to push per-entry progress
        // events to SSE /events/source subscribers.
        let source_tx = app_state.sse.source_tx.clone();

        Arc::new(Self {
            log_tx,
            progress_tx,
            job_tx,
            model_tx,
            citation_tx,
            source_tx,
            app_state,
            sse_connections: AtomicUsize::new(0),
            native_dialogs,
        })
    }
}

/// Builds the complete web router that merges the existing API routes with
/// web-specific endpoints, SSE streams, and embedded static asset serving.
///
/// Route precedence (first match wins):
/// 1. `/api/v1/*` -- Existing REST API endpoints (from neuroncite-api)
/// 2. `/api/v1/web/*` -- Web-specific endpoints (file browse, models, doctor, MCP)
/// 3. `/api/v1/events/*` -- SSE streaming endpoints (logs, progress, jobs, models)
/// 4. `/*` -- Embedded static files with SPA fallback to index.html
pub fn build_web_router(app_state: Arc<AppState>, web_state: Arc<WebState>) -> Router {
    let api_router = neuroncite_api::build_router(app_state.clone());

    let mut web_routes = handlers::web_routes(web_state.clone());
    let mut sse_routes = sse::sse_routes(web_state);

    // Apply the same bearer token authentication to web-specific and SSE routes
    // that the API router already applies to /api/v1/* routes. Without this,
    // a token-secured server would protect /api/v1/search but leave /api/v1/web/browse
    // and /api/v1/events/* open to unauthenticated access.
    if app_state.auth.bearer_token_hash.is_some() {
        let auth_state = app_state.clone();
        let auth_layer = axum::middleware::from_fn(move |req, next| {
            let state = auth_state.clone();
            async move {
                neuroncite_api::middleware::auth::auth_middleware(
                    axum::extract::State(state),
                    req,
                    next,
                )
                .await
            }
        });
        web_routes = web_routes.layer(auth_layer.clone());
        sse_routes = sse_routes.layer(auth_layer);
    }

    Router::new()
        // Existing API routes preserved unchanged (already has auth + body limit)
        .merge(api_router)
        // Web-specific REST endpoints (auth applied above when configured)
        .nest("/api/v1/web", web_routes)
        // SSE streaming endpoints (auth applied above when configured)
        .nest("/api/v1/events", sse_routes)
        // Static frontend assets (must be last -- fallback for SPA routing).
        // Static files are not auth-gated because the browser needs to load
        // index.html and JS/CSS assets before it can present the login flow.
        .fallback(assets::serve_embedded)
        // Limit request body size to 2 MiB (matches the API router limit).
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        // Apply security response headers (X-Content-Type-Options, X-Frame-Options,
        // Referrer-Policy) to all responses from the web server.
        .layer(axum::middleware::from_fn(middleware::security_headers))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use neuroncite_core::{
        AppConfig, EmbeddingBackend, IndexConfig, ModelInfo, NeuronCiteError, Reranker, StorageMode,
    };
    use neuroncite_store as store;

    use crate::WebState;
    use crate::build_web_router;

    // -----------------------------------------------------------------------
    // Stub implementations for this crate's test module.
    //
    // These are intentionally kept as local copies rather than importing from
    // neuroncite_api::test_support because:
    //
    // 1. The test_support module in neuroncite-api is declared as
    //    `pub(crate) mod test_support` behind `#[cfg(test)]`, making it
    //    invisible to downstream crates at both the module and item level.
    //
    // 2. Even if test_support were made `pub`, Rust's `#[cfg(test)]` gating
    //    means it only compiles when neuroncite-api itself is the test target,
    //    not when neuroncite-web's tests compile neuroncite-api as a dependency.
    //
    // 3. Exposing test stubs as a public non-cfg(test) module or feature-gated
    //    module would pollute the production API surface for test-only types.
    //
    // The canonical shared implementations live in neuroncite-api/src/test_support.rs.
    // If the trait contracts for EmbeddingBackend or Reranker change, both the
    // shared module and these local copies must be updated in lockstep.
    //
    // (Audit finding M-014 -- intra-crate duplicates in neuroncite-api have been
    // migrated; this cross-crate copy is retained for the reasons above.)
    // -----------------------------------------------------------------------

    /// Embedding backend that returns deterministic unit vectors without loading
    /// a real ONNX model. Sufficient for testing handler logic that passes through
    /// the worker handle. Reports "BAAI/bge-small-en-v1.5" as the loaded model
    /// ID to match AppConfig::default_model, preventing the index handler's model
    /// guard from rejecting requests.
    struct StubBackend;

    impl EmbeddingBackend for StubBackend {
        fn name(&self) -> &str {
            "stub"
        }

        fn vector_dimension(&self) -> usize {
            4
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
            Ok(())
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }

        fn supports_gpu(&self) -> bool {
            false
        }

        fn available_models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                id: "stub-model".to_string(),
                display_name: "Stub Model".to_string(),
                vector_dimension: 4,
                backend: "stub".to_string(),
            }]
        }

        fn loaded_model_id(&self) -> String {
            "BAAI/bge-small-en-v1.5".to_string()
        }
    }

    /// Stub reranker that returns the candidate's zero-based index as its
    /// relevance score. Matches the behavior of the canonical StubReranker
    /// in neuroncite-api/src/test_support.rs.
    struct StubReranker;

    impl Reranker for StubReranker {
        fn name(&self) -> &str {
            "stub-reranker"
        }

        fn rerank_batch(
            &self,
            _query: &str,
            candidates: &[&str],
        ) -> Result<Vec<f64>, NeuronCiteError> {
            Ok((0..candidates.len()).map(|i| i as f64).collect())
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Test harness helpers
    // -----------------------------------------------------------------------

    /// Creates an in-memory test application state with stub backend/reranker,
    /// a fresh SQLite database, and default configuration.
    fn setup_test_state() -> Arc<neuroncite_api::AppState> {
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");

        {
            let conn = pool.get().expect("get conn for migration");
            store::migrate(&conn).expect("migrate on pool");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let reranker: Arc<dyn Reranker> = Arc::new(StubReranker);
        let handle = neuroncite_api::spawn_worker(backend, Some(reranker));
        let config = AppConfig::default();

        neuroncite_api::AppState::new(pool, handle, config, true, None, 4)
            .expect("test AppState construction must succeed")
    }

    /// Builds the full web router with an in-memory test state.
    fn build_test_app() -> (axum::Router, Arc<neuroncite_api::AppState>, Arc<WebState>) {
        let app_state = setup_test_state();
        let (log_tx, _) = tokio::sync::broadcast::channel(256);
        let web_state = WebState::new(app_state.clone(), log_tx);
        let router = build_web_router(app_state.clone(), web_state.clone());
        (router, app_state, web_state)
    }

    /// Sends a request to the router and returns the response.
    async fn send(router: axum::Router, req: Request<Body>) -> axum::response::Response {
        router.oneshot(req).await.expect("request failed")
    }

    /// Reads the response body as a serde_json::Value.
    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        // 10 MiB is a practical upper bound for test response bodies. Using
        // usize::MAX would allow an unbounded allocation; the cap prevents
        // a misbehaving test response from exhausting memory.
        let bytes = axum::body::to_bytes(resp.into_body(), 10 * 1024 * 1024)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("parse json")
    }

    /// Creates a test session in the database and returns its ID.
    fn create_test_session(state: &Arc<neuroncite_api::AppState>) -> i64 {
        let conn = state.pool.get().expect("get conn");
        let config = IndexConfig {
            directory: PathBuf::from("/test/docs"),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        store::create_session(&conn, &config, "0.1.0").expect("create session")
    }

    // -----------------------------------------------------------------------
    // T-WEB-001: POST /web/browse with empty path returns entries
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_001_browse_root_returns_entries() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/browse")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"path":""}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // On Windows, drives array is non-empty. On Linux, entries contains "/".
        assert!(json["entries"].is_array());
    }

    // -----------------------------------------------------------------------
    // T-WEB-002: POST /web/browse with nonexistent path returns 404
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_002_browse_invalid_path_404() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/browse")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"path":"/nonexistent_path_abc_xyz_12345"}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // -----------------------------------------------------------------------
    // T-WEB-003: POST /web/browse with temp dir lists created files
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_003_browse_tempdir_lists_files() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        std::fs::write(tmp.path().join("hello.txt"), "content").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();

        let (app, _, _) = build_test_app();
        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/browse")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let entries = json["entries"].as_array().expect("entries is array");
        assert_eq!(entries.len(), 2);
    }

    // -----------------------------------------------------------------------
    // T-WEB-004: Browse directories are sorted before files
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_004_browse_directory_sorts_dirs_first() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        std::fs::write(tmp.path().join("aaa_file.txt"), "x").unwrap();
        std::fs::create_dir(tmp.path().join("zzz_dir")).unwrap();

        let (app, _, _) = build_test_app();
        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/browse")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        let json = body_json(resp).await;
        let entries = json["entries"].as_array().expect("entries is array");
        // Directory "zzz_dir" appears before file "aaa_file.txt" despite alphabetical order
        assert_eq!(entries[0]["kind"], "directory");
        assert_eq!(entries[1]["kind"], "file");
    }

    // -----------------------------------------------------------------------
    // T-WEB-005: POST /web/scan-documents in empty dir returns empty list
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_005_scan_documents_empty_dir() {
        let tmp = tempfile::tempdir().expect("create tempdir");

        let (app, _, _) = build_test_app();
        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["files"].as_array().unwrap().len(), 0);
    }

    // -----------------------------------------------------------------------
    // T-WEB-006: POST /web/scan-documents finds nested documents with subfolder
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_006_scan_documents_finds_nested() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        // Create a fake PDF file (just needs .pdf extension for discovery)
        std::fs::write(tmp.path().join("root.pdf"), "fake pdf").unwrap();
        let sub = tmp.path().join("papers");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested.pdf"), "fake pdf 2").unwrap();
        // Non-indexable file should be ignored
        std::fs::write(tmp.path().join("readme.txt"), "not indexable").unwrap();

        let (app, _, _) = build_test_app();
        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let files = json["files"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        // Each entry has the required fields
        for f in files {
            assert!(f["path"].is_string());
            assert!(f["name"].is_string());
            assert!(f["mtime"].is_number());
            assert!(f["status"].is_string());
        }
        // The nested file has a non-empty subfolder
        let nested = files.iter().find(|f| f["name"] == "nested.pdf").unwrap();
        assert_eq!(nested["subfolder"], "papers");
    }

    // -----------------------------------------------------------------------
    // T-WEB-007: POST /web/scan-documents with nonexistent dir returns 400
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_007_scan_documents_invalid_dir_400() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"path":"/nonexistent_path_scan_test_xyz"}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // -----------------------------------------------------------------------
    // T-WEB-008: scan-documents without session_id returns all files as pending
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_008_scan_documents_status_pending_without_session() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        std::fs::write(tmp.path().join("doc.pdf"), "fake").unwrap();

        let (app, _, _) = build_test_app();
        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        let json = body_json(resp).await;
        let files = json["files"].as_array().unwrap();
        assert_eq!(files[0]["status"], "pending");
    }

    // -----------------------------------------------------------------------
    // T-WEB-009: scan-documents with matching indexed file returns "indexed"
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_009_scan_documents_status_indexed() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let pdf_path = tmp.path().join("indexed.pdf");
        std::fs::write(&pdf_path, "fake pdf content").unwrap();

        // Read actual filesystem metadata to match the DB record
        let meta = std::fs::metadata(&pdf_path).unwrap();
        let mtime = meta
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let size = meta.len() as i64;
        let file_path_str = pdf_path.to_string_lossy().to_string();

        let (app, state, _) = build_test_app();
        let session_id = create_test_session(&state);

        // Insert a file record that matches the filesystem metadata
        {
            let conn = state.pool.get().unwrap();
            store::insert_file(
                &conn,
                session_id,
                &file_path_str,
                "hash_abc",
                mtime,
                size,
                1,
                None,
            )
            .unwrap();
        }

        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path, "session_id": session_id});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        let json = body_json(resp).await;
        let files = json["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0]["status"], "indexed");
    }

    // -----------------------------------------------------------------------
    // T-WEB-010: scan-documents with mismatched mtime returns "outdated"
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_010_scan_documents_status_outdated() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let pdf_path = tmp.path().join("stale.pdf");
        std::fs::write(&pdf_path, "fake pdf").unwrap();

        let file_path_str = pdf_path.to_string_lossy().to_string();

        let (app, state, _) = build_test_app();
        let session_id = create_test_session(&state);

        // Insert a file record with a deliberately wrong mtime (1 second off)
        let meta = std::fs::metadata(&pdf_path).unwrap();
        let real_mtime = meta
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        {
            let conn = state.pool.get().unwrap();
            store::insert_file(
                &conn,
                session_id,
                &file_path_str,
                "hash_xyz",
                real_mtime - 100, // Deliberately stale mtime
                meta.len() as i64,
                1,
                None,
            )
            .unwrap();
        }

        let path = tmp.path().to_string_lossy().to_string();
        let body = serde_json::json!({"path": path, "session_id": session_id});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/scan-documents")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        let json = body_json(resp).await;
        let files = json["files"].as_array().unwrap();
        assert_eq!(files[0]["status"], "outdated");
    }

    // -----------------------------------------------------------------------
    // T-WEB-011: GET /web/config returns default configuration
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_011_config_returns_defaults() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/config")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["default_model"].is_string());
        assert!(json["default_strategy"].is_string());
        assert!(json["default_chunk_size"].is_number());
        assert!(json["default_overlap"].is_number());
        // bind_address is intentionally excluded from the config response: it is
        // server-internal configuration that could expose network topology. port is also
        // omitted because the client already knows it from the connection it used to reach
        // this endpoint. The loaded_model_id and loaded_model_dimension are present instead.
        assert!(
            !json.as_object().unwrap().contains_key("bind_address"),
            "bind_address must not be present in config response (security: network topology)"
        );
        assert!(json["loaded_model_id"].is_string());
        assert!(json["loaded_model_dimension"].is_number());
    }

    // -----------------------------------------------------------------------
    // T-WEB-012: GET /web/models/catalog returns structured response
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_012_model_catalog_structure() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/models/catalog")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["gpu_name"].is_string());
        assert!(json["cuda_available"].is_boolean());
        assert!(json["embedding_models"].is_array());
        assert!(json["reranker_models"].is_array());
    }

    // -----------------------------------------------------------------------
    // T-WEB-013: POST /web/models/activate with unknown model returns 400
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_013_model_activate_not_cached_400() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/models/activate")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model_id":"fake/nonexistent-model"}"#))
            .unwrap();
        let resp = send(app, req).await;
        // The model is not cached, so the handler returns 400
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // -----------------------------------------------------------------------
    // T-WEB-014: GET /web/doctor/probes returns dependency list
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_014_doctor_probes_returns_list() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/doctor/probes")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let probes = json.as_array().expect("probes is array");
        // Five probes: pdfium, tesseract, ONNX Runtime (CPU/GPU variant), Ollama, poppler.
        // The ONNX Runtime entry name includes a variant suffix on Windows (" (CPU)" or " (GPU)"),
        // so the exact name is not checked here -- only count and required members.
        assert!(
            probes.len() >= 5,
            "expected at least 5 probes, got {}",
            probes.len()
        );
        let names: Vec<&str> = probes.iter().map(|p| p["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"pdfium"));
        assert!(names.contains(&"tesseract"));
    }

    // -----------------------------------------------------------------------
    // T-WEB-015: POST /web/doctor/install with unknown dependency returns 400
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_015_doctor_install_unknown_400() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/doctor/install")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"dependency":"unicorn"}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // -----------------------------------------------------------------------
    // T-WEB-016: GET /web/mcp/status returns expected fields
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_016_mcp_status_returns_fields() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/mcp/status")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // McpStatusResponse now exposes only `registered` (bool) and `server_version` (string).
        // File paths such as exe_path and config_path were removed to prevent server-side
        // path disclosure to browser-accessible endpoints.
        assert!(
            json["registered"].is_boolean(),
            "registered must be a boolean in mcp status response"
        );
        assert!(
            json["server_version"].is_string(),
            "server_version must be a string in mcp status response"
        );
        assert!(
            !json.as_object().unwrap().contains_key("config_path"),
            "config_path must not be present in mcp status response (path disclosure)"
        );
    }

    // -----------------------------------------------------------------------
    // T-WEB-017: GET /events/logs connects as SSE stream
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_017_sse_log_stream_connects() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/events/logs")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/event-stream"), "content-type: {ct}");
    }

    // -----------------------------------------------------------------------
    // T-WEB-018: GET /events/progress connects as SSE stream
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_018_sse_progress_stream_connects() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/events/progress")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/event-stream"), "content-type: {ct}");
    }

    // -----------------------------------------------------------------------
    // T-WEB-019: GET /api/v1/health works through the merged web router
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_019_api_health_through_web_router() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/health")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // api_version is excluded from the JSON body via #[serde(skip)];
        // it is conveyed by the X-API-Version response header instead.
        assert!(json["gpu_available"].is_boolean());
    }

    // -----------------------------------------------------------------------
    // T-WEB-020: POST /web/browse with a file path (not dir) returns 400
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_020_browse_file_path_returns_400() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let file = tmp.path().join("somefile.txt");
        std::fs::write(&file, "content").unwrap();

        let (app, _, _) = build_test_app();
        let body = serde_json::json!({"path": file.to_string_lossy()});
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/browse")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // -----------------------------------------------------------------------
    // T-WEB-021: native_window module is accessible when gui feature is on
    // -----------------------------------------------------------------------

    /// Verifies that the native_window module is compiled and its public
    /// function signatures are accessible when the gui feature flag is enabled.
    /// This is a compile-time assertion -- if the module or its public API
    /// changes, this test will fail to compile.
    #[cfg(feature = "gui")]
    #[test]
    #[allow(clippy::type_complexity)]
    fn t_web_021_native_window_module_accessible() {
        // Verify both public functions exist with the expected signatures.
        // These are not called (they require a display server) but confirm
        // the module is correctly exposed and linkable.
        let _fn_ptr: fn(
            &str,
            tokio::sync::oneshot::Sender<()>,
        ) -> Result<(), Box<dyn std::error::Error>> = crate::native_window::open_native_window;

        let _splash_fn_ptr: fn(
            std::sync::mpsc::Receiver<String>,
            tokio::sync::oneshot::Sender<()>,
        ) -> Result<(), Box<dyn std::error::Error>> = crate::native_window::run_gui_with_splash;

        let _preflight_fn_ptr: fn() -> Result<(), String> =
            crate::native_window::preflight_gui_check;
    }

    // -----------------------------------------------------------------------
    // T-WEB-022: shutdown oneshot channel fires when sender is invoked
    // -----------------------------------------------------------------------

    /// Tests the shutdown signaling pattern used between the native window
    /// (main thread) and the Axum server (background thread). The oneshot
    /// channel must deliver the signal exactly once.
    #[tokio::test]
    async fn t_web_022_shutdown_channel_delivers_signal() {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Simulate what happens when the native window's CloseRequested
        // handler fires: the sender is consumed and the signal is sent.
        let _ = shutdown_tx.send(());

        // The server side receives the shutdown signal.
        let result = shutdown_rx.await;
        assert!(result.is_ok(), "shutdown signal should be received");
    }

    // -----------------------------------------------------------------------
    // T-WEB-023: shutdown channel receiver completes when sender is dropped
    // -----------------------------------------------------------------------

    /// Tests that dropping the shutdown sender (e.g. when the window creation
    /// fails and the sender is consumed by the error path) causes the receiver
    /// to complete with an error. This is the expected behavior for the
    /// fallback path in run_web().
    #[tokio::test]
    async fn t_web_023_shutdown_channel_sender_drop_completes_receiver() {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Drop the sender without sending a value (simulates window creation
        // failure where the sender is consumed but no signal is sent).
        drop(shutdown_tx);

        // The receiver completes with RecvError.
        let result = shutdown_rx.await;
        assert!(result.is_err(), "dropped sender should produce RecvError");
    }

    // -----------------------------------------------------------------------
    // T-WEB-024: URL channel delivers server URL to main thread
    // -----------------------------------------------------------------------

    /// Tests the std::sync::mpsc channel pattern used to communicate the
    /// actual server URL (with resolved port) from the background server
    /// thread to the main thread that opens the native window.
    #[test]
    fn t_web_024_url_channel_delivers_url() {
        let (url_tx, url_rx) = std::sync::mpsc::channel::<String>();

        // Simulate the server thread sending the URL after port binding.
        let expected_url = "http://127.0.0.1:3035".to_string();
        url_tx.send(expected_url.clone()).unwrap();

        // The main thread receives the URL.
        let received_url = url_rx.recv().unwrap();
        assert_eq!(received_url, expected_url);
    }

    // -----------------------------------------------------------------------
    // T-WEB-025: URL channel error when server thread fails during setup
    // -----------------------------------------------------------------------

    /// Tests that when the server thread terminates during setup (e.g. DB
    /// init failure, port binding failure), the URL sender is dropped without
    /// sending, causing the main thread's recv() to return RecvError. This
    /// triggers the fallback path in run_web().
    #[test]
    fn t_web_025_url_channel_error_on_server_failure() {
        let (url_tx, url_rx) = std::sync::mpsc::channel::<String>();

        // Simulate the server thread failing during setup: the sender is
        // dropped without sending a URL.
        drop(url_tx);

        // The main thread detects the failure.
        let result = url_rx.recv();
        assert!(result.is_err(), "dropped sender should produce RecvError");
    }

    // -----------------------------------------------------------------------
    // T-WEB-026: tokio::select shutdown resolves on window close signal
    // -----------------------------------------------------------------------

    /// Tests the dual shutdown mechanism used by the GUI path: the server
    /// shuts down on either the native window close signal OR a timeout
    /// (standing in for Ctrl+C which cannot be tested without installing
    /// a global signal handler that interferes with parallel test runs).
    #[tokio::test]
    async fn t_web_026_select_shutdown_resolves_on_window_close() {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Simulate window close signal arriving before the timeout.
        let _ = shutdown_tx.send(());

        // Uses a long sleep as a stand-in for ctrl_c() to avoid installing
        // a global signal handler that blocks in test environments.
        let result = tokio::select! {
            _ = shutdown_rx => "window_closed",
            _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => "timeout",
        };
        assert_eq!(result, "window_closed");
    }

    // -----------------------------------------------------------------------
    // T-WEB-027: thread communication round-trip simulates full GUI lifecycle
    // -----------------------------------------------------------------------

    /// Simulates the full thread communication lifecycle of the GUI path:
    /// 1. Server thread starts, binds port, sends URL via mpsc channel
    /// 2. Main thread receives URL
    /// 3. Main thread "opens window" (simulated) and "closes" it
    /// 4. Shutdown signal propagates to server thread via oneshot channel
    /// 5. Server thread completes graceful shutdown
    #[tokio::test]
    async fn t_web_027_full_gui_lifecycle_communication() {
        let (url_tx, url_rx) = tokio::sync::mpsc::channel::<String>(1);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Simulate server thread: starts, binds, sends URL, then waits
        // for shutdown signal.
        let server_handle = tokio::spawn(async move {
            // Server setup complete, send URL to "main thread"
            url_tx
                .send("http://127.0.0.1:3030".to_string())
                .await
                .unwrap();

            // Wait for shutdown signal (simulates axum::serve with graceful
            // shutdown waiting on the oneshot receiver).
            let _ = shutdown_rx.await;
            0i32 // exit code
        });

        // Simulate main thread: receive URL, "open window", "close window".
        // Uses tokio::sync::mpsc to avoid blocking the async runtime thread.
        let mut url_rx = url_rx;
        let url = url_rx.recv().await.unwrap();
        assert_eq!(url, "http://127.0.0.1:3030");

        // Simulate window close: send shutdown signal
        let _ = shutdown_tx.send(());

        // Server thread completes with exit code 0
        let exit_code = server_handle.await.unwrap();
        assert_eq!(exit_code, 0);
    }

    // -----------------------------------------------------------------------
    // T-WEB-027b: preflight_gui_check function signature accessible
    // -----------------------------------------------------------------------

    /// Verifies that the `preflight_gui_check` function is compiled and its
    /// public signature is accessible when the gui feature flag is enabled.
    #[cfg(feature = "gui")]
    #[test]
    fn t_web_027b_preflight_gui_check_accessible() {
        let _fn_ptr: fn() -> Result<(), String> = crate::native_window::preflight_gui_check;
    }

    // -----------------------------------------------------------------------
    // T-WEB-027c: preflight_gui_check returns Ok on the build host
    // -----------------------------------------------------------------------

    /// On CI and developer machines where gui tests run, the WebKitGTK
    /// runtime (Linux) or platform WebView (macOS/Windows) should be
    /// available. This test ensures the dlopen check succeeds in
    /// environments where the GUI is expected to work.
    #[cfg(feature = "gui")]
    #[test]
    fn t_web_027c_preflight_gui_check_passes_on_build_host() {
        let result = crate::native_window::preflight_gui_check();
        assert!(
            result.is_ok(),
            "preflight_gui_check failed on build host: {:?}",
            result.err()
        );
    }

    // ================================================================
    // Ollama proxy and management endpoint tests (T-WEB-028 to T-WEB-036)
    //
    // Since Ollama is not running in CI, these tests verify endpoint
    // behavior when the Ollama server is unreachable (expected 502 for
    // proxy endpoints) and validate the catalog endpoint which is
    // served statically by NeuronCite.
    // ================================================================

    /// T-WEB-028: The status endpoint rejects URLs with non-Ollama ports.
    /// After SSRF hardening (M-004), the Ollama proxy only allows port 11434.
    /// A request with port 19999 is rejected at validation with HTTP 400.
    #[tokio::test]
    async fn t_web_028_ollama_status_non_ollama_port_rejected() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/ollama/status?url=http://127.0.0.1:19999")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-029: The models endpoint rejects URLs with non-Ollama ports.
    /// After SSRF hardening (M-004), port 19999 is blocked at validation.
    #[tokio::test]
    async fn t_web_029_ollama_models_non_ollama_port_rejected() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/ollama/models?url=http://127.0.0.1:19999")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-030: The running endpoint rejects URLs with non-Ollama ports.
    /// After SSRF hardening (M-004), port 19999 is blocked at validation.
    #[tokio::test]
    async fn t_web_030_ollama_running_non_ollama_port_rejected() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/ollama/running?url=http://127.0.0.1:19999")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-031: The catalog endpoint returns a non-empty list of curated
    /// Ollama models with all required fields populated.
    #[tokio::test]
    async fn t_web_031_ollama_catalog_returns_entries() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/ollama/catalog")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value = body_json(resp).await;
        let models = body["models"].as_array().unwrap();
        assert!(!models.is_empty(), "catalog should contain entries");
        // Verify first entry has all required fields
        let first = &models[0];
        assert!(first["name"].is_string());
        assert!(first["display_name"].is_string());
        assert!(first["family"].is_string());
        assert!(first["parameter_size"].is_string());
        assert!(first["size_mb"].is_number());
        assert!(first["description"].is_string());
    }

    /// T-WEB-032: The pull endpoint rejects URLs with non-Ollama ports.
    /// After SSRF hardening (M-004), port 19999 is blocked at validation.
    #[tokio::test]
    async fn t_web_032_ollama_pull_non_ollama_port_rejected() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/ollama/pull")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model": "test:latest", "url": "http://127.0.0.1:19999"}"#,
            ))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-033: The delete endpoint rejects URLs with non-Ollama ports.
    /// After SSRF hardening (M-004), port 19999 is blocked at validation.
    #[tokio::test]
    async fn t_web_033_ollama_delete_non_ollama_port_rejected() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/ollama/delete")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model": "test:latest", "url": "http://127.0.0.1:19999"}"#,
            ))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-034: The pull endpoint returns 400 when the model name is empty.
    #[tokio::test]
    async fn t_web_034_ollama_pull_empty_model_400() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/ollama/pull")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": ""}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-035: The delete endpoint returns 400 when the model name is empty.
    #[tokio::test]
    async fn t_web_035_ollama_delete_empty_model_400() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/web/ollama/delete")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": ""}"#))
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T-WEB-036: All catalog entries have non-empty required fields and
    /// positive size values.
    #[tokio::test]
    async fn t_web_036_ollama_catalog_entries_have_required_fields() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/ollama/catalog")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value = body_json(resp).await;
        let models = body["models"].as_array().unwrap();
        for entry in models {
            let name = entry["name"].as_str().unwrap();
            assert!(!name.is_empty(), "name must not be empty");
            assert!(
                !entry["display_name"].as_str().unwrap().is_empty(),
                "display_name must not be empty for {name}"
            );
            assert!(
                !entry["family"].as_str().unwrap().is_empty(),
                "family must not be empty for {name}"
            );
            assert!(
                !entry["parameter_size"].as_str().unwrap().is_empty(),
                "parameter_size must not be empty for {name}"
            );
            assert!(
                entry["size_mb"].as_u64().unwrap() > 0,
                "size_mb must be positive for {name}"
            );
            assert!(
                !entry["description"].as_str().unwrap().is_empty(),
                "description must not be empty for {name}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-WEB-037: GET /web/check-update route is registered and responds
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_web_037_check_update_route_exists() {
        let (app, _, _) = build_test_app();
        let req = Request::builder()
            .uri("/api/v1/web/check-update")
            .body(Body::empty())
            .unwrap();
        let resp = send(app, req).await;
        // The handler makes an outbound HTTP call to GitHub, which may fail
        // in CI/test environments. We accept either 200 (GitHub reachable)
        // or 502 (outbound call failed). The key assertion is that the route
        // exists and is not 404 or 405.
        let status = resp.status().as_u16();
        assert!(
            status == 200 || status == 502,
            "expected 200 or 502, got {status}"
        );
    }
}
