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

//! Handler for the `neuroncite_index` MCP tool.
//!
//! Supports three mutually exclusive input modes:
//!
//! 1. `directory` -- Scans a directory for supported files and creates a
//!    background indexing job. The actual indexing work happens asynchronously
//!    through the existing job processing pipeline.
//! 2. `urls` -- Indexes previously fetched HTML pages from the cache. This
//!    path runs synchronously within the MCP request because web pages are
//!    typically smaller and fewer than PDF corpora. (Absorbs the former
//!    `neuroncite_html_index` tool.)
//! 3. `files` -- Indexes specific file paths (reserved for future file-type
//!    crates; currently rejected with a descriptive error).
//!
//! The vector_dimension for the session is resolved from the static model
//! catalog based on the requested model ID, not from state.vector_dimension.
//! This prevents dimension mismatches when the MCP server was started with a
//! different model than what the user requests for indexing.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_api::service::index::resolve_vector_dimension;
use neuroncite_core::{IndexConfig, StorageMode};

/// Starts an indexing operation. Three mutually exclusive input modes:
///
/// 1. `directory`: Scans a directory for supported files, creates a background job.
/// 2. `urls`: Indexes previously fetched HTML pages from cache (synchronous).
/// 3. `files`: Indexes specific file paths (reserved for future file-type crates).
///
/// The `chunk_size` parameter is interpreted according to the selected strategy:
/// - `"token"` / `"word"`: chunk_size = tokens/words per chunk, chunk_overlap = overlap
/// - `"sentence"`: chunk_size = maximum words per sentence-based chunk (no overlap)
/// - `"page"`: chunk_size and chunk_overlap are ignored (one chunk per page)
///
/// # Parameters (from MCP tool call)
///
/// - `directory` (optional): Absolute path to the directory. Mutually exclusive
///   with `urls` and `files`.
/// - `urls` (optional): Array of URL strings whose cached HTML to index.
///   Mutually exclusive with `directory` and `files`.
/// - `files` (optional): Array of absolute file paths to index. Reserved for
///   future file-type crates.
/// - `model` (optional): Embedding model ID (default: config's default_model).
/// - `chunk_strategy` (optional): Chunking strategy name (default: "token").
/// - `chunk_size` (optional): Tokens/words per chunk, or max words for sentence
///   strategy (default: 256).
/// - `chunk_overlap` (optional): Overlap between chunks, used by "token" and
///   "word" strategies only (default: 32).
/// - `strip_boilerplate` (optional): For HTML sources, apply readability-based
///   boilerplate removal before section splitting. Defaults to true.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    // Determine input mode from provided parameters.
    let has_directory = params["directory"].as_str().is_some();
    let has_urls = params["urls"].as_array().is_some();
    let has_files = params["files"].as_array().is_some();

    match (has_directory, has_urls, has_files) {
        (true, false, false) => handle_directory(state, params).await,
        (false, true, false) => handle_urls(state, params).await,
        (false, false, true) => Err(
            "the 'files' input mode is reserved for future file-type crates \
                 and not yet implemented; use 'directory' for PDF files or 'urls' \
                 for previously fetched HTML pages"
                .to_string(),
        ),
        (false, false, false) => Err(
            "provide one of: 'directory' (scan for files), 'urls' (index cached HTML), \
                 or 'files' (index specific file paths)"
                .to_string(),
        ),
        _ => Err("'directory', 'urls', and 'files' are mutually exclusive; \
                 provide exactly one"
            .to_string()),
    }
}

/// Handles directory-based indexing: scans for files and creates a background job.
async fn handle_directory(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let directory = params["directory"]
        .as_str()
        .ok_or("missing required parameter: directory")?;

    // Canonicalize the directory path to its absolute form. The directory must
    // exist on disk; a missing path returns Err so the MCP caller receives a
    // clear error message rather than the indexer operating on an unverified path.
    let dir_path = neuroncite_core::paths::canonicalize_directory(std::path::Path::new(directory))
        .map_err(|e| format!("invalid directory: {e}"))?;
    if !dir_path.is_dir() {
        return Err(format!("directory does not exist: {directory}"));
    }

    // Use the config's default_model (which matches the loaded model) when no
    // model is specified. This ensures the default path always uses the
    // server's loaded model rather than a hardcoded string.
    let model = params["model"]
        .as_str()
        .unwrap_or_else(|| state.config.default_model.as_str());
    let strategy = params["chunk_strategy"].as_str().unwrap_or("token");
    let chunk_size = params["chunk_size"].as_u64().unwrap_or(256) as usize;
    let chunk_overlap = params["chunk_overlap"].as_u64().unwrap_or(32) as usize;

    // Resolve the vector dimension from the model catalog. This is the
    // authoritative source for the model's output dimensionality, preventing
    // the bug where state.vector_dimension (set at startup from the initially
    // loaded model) would be stored even when a different model is requested.
    let vector_dimension = resolve_vector_dimension(model)?;

    // Map the unified chunk_size parameter to strategy-specific IndexConfig
    // fields. The "sentence" strategy uses max_words instead of chunk_size.
    // The "page" strategy ignores all size parameters.
    let (cfg_chunk_size, cfg_chunk_overlap, cfg_max_words) = match strategy {
        "token" | "word" => (Some(chunk_size), Some(chunk_overlap), None),
        "sentence" => (None, None, Some(chunk_size)),
        _ => (None, None, None), // "page" has no parameters
    };

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Check for concurrent running index jobs. Only one indexing job is
    // allowed at a time to prevent GPU contention and database write conflicts.
    let jobs = neuroncite_store::list_jobs(&conn).map_err(|e| format!("listing jobs: {e}"))?;
    let running_job = jobs
        .iter()
        .find(|j| j.kind == "index" && j.state == neuroncite_store::JobState::Running);
    if let Some(rj) = running_job {
        let session_info = rj
            .session_id
            .map(|sid| format!(", session_id={sid}"))
            .unwrap_or_default();
        return Err(format!(
            "an indexing job is already running (job_id={}{}); \
             wait for it to complete or cancel it",
            rj.id, session_info
        ));
    }

    let index_config = IndexConfig {
        directory: dir_path,
        model_name: model.to_string(),
        chunk_strategy: strategy.to_string(),
        chunk_size: cfg_chunk_size,
        chunk_overlap: cfg_chunk_overlap,
        max_words: cfg_max_words,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension,
    };

    // Find an existing session with matching config or create a new one.
    let session_id = match neuroncite_store::find_session(&conn, &index_config) {
        Ok(Some(id)) => id,
        _ => neuroncite_store::create_session(&conn, &index_config, env!("CARGO_PKG_VERSION"))
            .map_err(|e| format!("session creation: {e}"))?,
    };

    // Create the job record.
    let job_id = uuid::Uuid::new_v4().to_string();
    neuroncite_store::create_job(&conn, &job_id, "index", Some(session_id))
        .map_err(|e| format!("job creation: {e}"))?;

    // Wake the executor immediately so it picks up the new job without
    // waiting for the next poll interval.
    state.job_notify.notify_one();

    // Return the strategy-specific parameters in the response. Strategies
    // that ignore certain parameters (e.g., "page" ignores chunk_size and
    // chunk_overlap) return null for those fields instead of misleading
    // default values.
    Ok(serde_json::json!({
        "job_id": job_id,
        "session_id": session_id,
        "directory": directory,
        "model": model,
        "chunk_strategy": strategy,
        "chunk_size": cfg_chunk_size,
        "chunk_overlap": cfg_chunk_overlap,
        "max_words": cfg_max_words,
        "vector_dimension": vector_dimension,
        "status": "accepted",
    }))
}

/// Handles URL-based indexing: reads cached HTML, extracts sections, chunks,
/// embeds via the GPU worker, and stores in the database. Runs synchronously
/// within the MCP request because web pages are typically smaller and fewer
/// than PDF corpora. After all files are indexed, the HNSW index is rebuilt.
///
/// This is the logic formerly in `neuroncite_html_index`, moved here so a
/// single `neuroncite_index` tool serves all input modes.
async fn handle_urls(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    use neuroncite_api::html_indexer::{embed_and_store_html_async, extract_and_chunk_html};
    use neuroncite_store::build_hnsw;

    let urls = params["urls"]
        .as_array()
        .ok_or("missing required parameter: urls (must be an array of URL strings)")?;

    if urls.is_empty() {
        return Err("urls array must not be empty".to_string());
    }

    let url_strings: Vec<String> = urls
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_str()
                .map(String::from)
                .ok_or_else(|| format!("urls[{i}] is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Resolve embedding model: use the server's loaded model when none is specified.
    let loaded_model = state.worker_handle.loaded_model_id();
    let model = params["model"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| (*loaded_model).clone());
    let strategy_name = params["chunk_strategy"].as_str().unwrap_or("token");
    let chunk_size = params["chunk_size"].as_u64().unwrap_or(256) as usize;
    let chunk_overlap = params["chunk_overlap"].as_u64().unwrap_or(32) as usize;
    let strip_boilerplate = params["strip_boilerplate"].as_bool().unwrap_or(true);

    let vector_dimension = resolve_vector_dimension(&model)?;

    // Map the unified chunk_size parameter to strategy-specific IndexConfig
    // fields. The "sentence" strategy uses max_words; the "page" strategy
    // ignores all size parameters.
    let (cfg_chunk_size, cfg_chunk_overlap, cfg_max_words) = match strategy_name {
        "token" | "word" => (Some(chunk_size), Some(chunk_overlap), None),
        "sentence" => (None, None, Some(chunk_size)),
        _ => (None, None, None),
    };

    // Build the IndexConfig for session creation/lookup. The directory field
    // stores the HTML cache directory path to distinguish HTML sessions from
    // PDF sessions.
    let cache_dir = neuroncite_html::default_cache_dir();
    let index_config = IndexConfig {
        directory: cache_dir.clone(),
        model_name: model.clone(),
        chunk_strategy: strategy_name.to_string(),
        chunk_size: cfg_chunk_size,
        chunk_overlap: cfg_chunk_overlap,
        max_words: cfg_max_words,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension,
    };

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Find an existing session matching this config or create a session.
    // When the caller provides an explicit session_id, use that instead.
    let session_id = if let Some(sid) = params["session_id"].as_i64() {
        neuroncite_store::get_session(&conn, sid)
            .map_err(|_| format!("session {sid} not found"))?;
        sid
    } else {
        match neuroncite_store::find_session(&conn, &index_config) {
            Ok(Some(id)) => id,
            _ => neuroncite_store::create_session(&conn, &index_config, env!("CARGO_PKG_VERSION"))
                .map_err(|e| format!("session creation: {e}"))?,
        }
    };

    // Release the connection before the embedding loop to avoid holding a
    // connection for the entire duration of GPU-bound embedding work.
    drop(conn);

    // Build the chunking strategy from the resolved parameters.
    let tokenizer_json = state.worker_handle.tokenizer_json();
    let chunk_strategy = neuroncite_chunk::create_strategy(
        strategy_name,
        cfg_chunk_size,
        cfg_chunk_overlap,
        cfg_max_words,
        tokenizer_json.as_deref(),
    )
    .map_err(|e| format!("chunk strategy creation: {e}"))?;

    // Process each URL: read cached HTML, extract sections, chunk, embed,
    // and store in the database. Errors for individual URLs are collected
    // rather than aborting the entire batch.
    let mut total_chunks: usize = 0;
    let mut total_files: usize = 0;
    let mut errors: Vec<serde_json::Value> = Vec::new();

    for url in &url_strings {
        let cache_path = neuroncite_html::cache_path_for_url(&cache_dir, url);

        let raw_html = match std::fs::read(&cache_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                errors.push(serde_json::json!({
                    "url": url,
                    "error": format!(
                        "cached HTML not found at {}: {e}; fetch the URL first via html_fetch or html_crawl",
                        cache_path.display()
                    ),
                }));
                continue;
            }
        };

        let html_str = match std::str::from_utf8(&raw_html) {
            Ok(s) => s,
            Err(e) => {
                errors.push(serde_json::json!({
                    "url": url,
                    "error": format!("cached HTML is not valid UTF-8: {e}"),
                }));
                continue;
            }
        };

        let metadata = neuroncite_html::extract_metadata(html_str, url, 200, None);

        // Phase 1: CPU-bound extraction and chunking.
        let extracted = match extract_and_chunk_html(
            chunk_strategy.as_ref(),
            &cache_path,
            url,
            &metadata,
            &raw_html,
            strip_boilerplate,
        ) {
            Ok(ex) => ex,
            Err(e) => {
                errors.push(serde_json::json!({
                    "url": url,
                    "error": format!("extraction/chunking failed: {e}"),
                }));
                continue;
            }
        };

        if extracted.chunks.is_empty() {
            errors.push(serde_json::json!({
                "url": url,
                "error": "HTML produced zero chunks after extraction and chunking",
            }));
            continue;
        }

        // Phase 2: Embed via the GPU worker and store in the database.
        match embed_and_store_html_async(&state.pool, &state.worker_handle, &extracted, session_id)
            .await
        {
            Ok(result) => {
                total_chunks += result.chunks_created;
                total_files += 1;
            }
            Err(e) => {
                errors.push(serde_json::json!({
                    "url": url,
                    "error": format!("embedding/storage failed: {e}"),
                }));
            }
        }
    }

    // Phase 3: Rebuild the HNSW index for the session so indexed HTML pages
    // become searchable immediately.
    let hnsw_vectors = if total_files > 0 {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let hnsw_chunks = neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
            .map_err(|e| format!("loading embeddings for HNSW: {e}"))?;

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

        let vector_count = vectors.len();
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let index = build_hnsw(&labeled, vector_dimension)
            .map_err(|e| format!("HNSW build failed: {e}"))?;
        state.insert_hnsw(session_id, index);

        Some(vector_count)
    } else {
        None
    };

    Ok(serde_json::json!({
        "session_id": session_id,
        "model": model,
        "chunk_strategy": strategy_name,
        "chunk_size": cfg_chunk_size,
        "chunk_overlap": cfg_chunk_overlap,
        "max_words": cfg_max_words,
        "vector_dimension": vector_dimension,
        "total_files_indexed": total_files,
        "total_chunks_created": total_chunks,
        "total_urls_requested": url_strings.len(),
        "errors": errors,
        "error_count": errors.len(),
        "hnsw_rebuilt": hnsw_vectors.is_some(),
        "hnsw_total_vectors": hnsw_vectors,
        "status": "completed",
    }))
}

#[cfg(test)]
#[cfg(feature = "backend-ort")]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Tests for resolve_vector_dimension (imported from neuroncite_api::service::index).
    // These tests verify that the shared function returns correct catalog
    // values when called from the MCP crate's test context.
    // -----------------------------------------------------------------------

    /// T-MCP-IDX-001: resolve_vector_dimension returns 384 for BGE-small,
    /// matching the model catalog specification.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_001_resolve_bge_small_dimension() {
        let dim = resolve_vector_dimension("BAAI/bge-small-en-v1.5").unwrap();
        assert_eq!(dim, 384);
    }

    /// T-MCP-IDX-002: resolve_vector_dimension returns 1024 for Qwen3-0.6B,
    /// matching the model catalog specification. This is the core regression
    /// test for the bug where Qwen3 sessions were stored with dimension=384.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_002_resolve_qwen3_dimension() {
        let dim = resolve_vector_dimension("Qwen/Qwen3-Embedding-0.6B").unwrap();
        assert_eq!(dim, 1024);
    }

    /// T-MCP-IDX-003: resolve_vector_dimension returns 2560 for Qwen3-4B.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_003_resolve_qwen3_4b_dimension() {
        let dim = resolve_vector_dimension("Qwen/Qwen3-Embedding-4B").unwrap();
        assert_eq!(dim, 2560);
    }

    /// T-MCP-IDX-004: resolve_vector_dimension returns an error for unknown
    /// model IDs. The error message lists available models.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_004_resolve_unknown_model_returns_error() {
        let result = resolve_vector_dimension("nonexistent/model-xyz");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("not in the supported model catalog"),
            "error must mention catalog: {err}"
        );
        assert!(
            err.contains("BAAI/bge-small-en-v1.5"),
            "error must list available models: {err}"
        );
    }

    /// T-MCP-IDX-005: resolve_vector_dimension is independent of any backend
    /// state. It always returns the catalog value regardless of what model
    /// was loaded at server startup.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_005_dimension_independent_of_backend_state() {
        // Simulate the scenario: server loaded BGE-small (384d), but user
        // requests Qwen3 (1024d). resolve_vector_dimension must return 1024.
        let bge_dim = resolve_vector_dimension("BAAI/bge-small-en-v1.5").unwrap();
        let qwen_dim = resolve_vector_dimension("Qwen/Qwen3-Embedding-0.6B").unwrap();
        assert_eq!(bge_dim, 384);
        assert_eq!(qwen_dim, 1024);
        assert_ne!(
            bge_dim, qwen_dim,
            "different models must produce different dimensions"
        );
    }

    /// T-MCP-IDX-006: resolve_vector_dimension returns consistent results
    /// for every model in the catalog. This ensures no model is accidentally
    /// missing from the lookup path.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_idx_006_all_catalog_models_resolvable() {
        for cfg in neuroncite_embed::supported_model_configs() {
            let dim = resolve_vector_dimension(&cfg.model_id);
            assert!(
                dim.is_ok(),
                "model '{}' must be resolvable, got error: {:?}",
                cfg.model_id,
                dim.err()
            );
            assert_eq!(
                dim.unwrap(),
                cfg.vector_dimension,
                "dimension mismatch for model '{}'",
                cfg.model_id
            );
        }
    }

    /// T-MCP-IDX-007: The full index handler creates a session with the
    /// correct vector_dimension from the model catalog, not from
    /// state.vector_dimension. Regression test for the core bug.
    #[cfg(feature = "backend-ort")]
    #[tokio::test]
    async fn t_mcp_idx_007_handler_stores_correct_dimension_for_qwen3() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        // Stub backend simulating a server that was started with BGE-small (384d).
        struct BgeSmallBackend;
        impl EmbeddingBackend for BgeSmallBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384 // Server loaded with BGE-small
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "BAAI/bge-small-en-v1.5".to_string()
            }
        }

        let tmp = tempfile::tempdir().expect("create temp dir");

        // Create a minimal valid PDF so the directory passes validation.
        std::fs::write(
            tmp.path().join("test.pdf"),
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
        )
        .expect("write test pdf");

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

        let backend: std::sync::Arc<dyn EmbeddingBackend> = std::sync::Arc::new(BgeSmallBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let config = AppConfig::default();

        // state.vector_dimension is 384 (from BGE-small loaded at startup).
        // AppState::new returns Arc<AppState>.
        let state = AppState::new(pool, worker_handle, config, true, None, 384)
            .expect("test AppState construction must succeed");
        assert_eq!(
            state
                .index
                .vector_dimension
                .load(std::sync::atomic::Ordering::Relaxed),
            384
        );

        // Simulate an MCP index request that specifies Qwen3-Embedding-0.6B.
        let params = serde_json::json!({
            "directory": tmp.path().to_string_lossy(),
            "model": "Qwen/Qwen3-Embedding-0.6B",
        });

        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handler must succeed: {:?}", result.err());

        let response = result.unwrap();

        // The response must report vector_dimension=1024 (Qwen3), NOT 384 (BGE-small).
        assert_eq!(
            response["vector_dimension"], 1024,
            "session must use Qwen3 dimension (1024), not startup dimension (384)"
        );

        // Verify the session in the database has the correct dimension.
        let session_id = response["session_id"].as_i64().unwrap();
        let conn = state.pool.get().expect("get conn");
        let session = neuroncite_store::get_session(&conn, session_id).expect("get session");
        assert_eq!(
            session.vector_dimension, 1024,
            "database session must store dimension 1024 for Qwen3"
        );
        assert_eq!(
            session.model_name, "Qwen/Qwen3-Embedding-0.6B",
            "database session must store the requested model name"
        );
    }

    /// T-MCP-IDX-009: The "page" strategy returns null for chunk_size,
    /// chunk_overlap, and max_words in the response. Regression test for BUG-3
    /// where the response showed irrelevant default values (256, 32) for
    /// parameters that the page strategy ignores entirely.
    #[cfg(feature = "backend-ort")]
    #[tokio::test]
    async fn t_mcp_idx_009_page_strategy_returns_null_params() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "BAAI/bge-small-en-v1.5".to_string()
            }
        }

        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(
            tmp.path().join("test.pdf"),
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
        )
        .expect("write test pdf");

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

        let backend: std::sync::Arc<dyn EmbeddingBackend> = std::sync::Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let params = serde_json::json!({
            "directory": tmp.path().to_string_lossy(),
            "chunk_strategy": "page",
        });

        let result = handle(&state, &params).await.expect("handler must succeed");

        // BUG-3 regression: page strategy must report null for chunk_size,
        // chunk_overlap, and max_words, not misleading defaults like 256/32.
        assert!(
            result["chunk_size"].is_null(),
            "page strategy must report null for chunk_size, got: {}",
            result["chunk_size"]
        );
        assert!(
            result["chunk_overlap"].is_null(),
            "page strategy must report null for chunk_overlap, got: {}",
            result["chunk_overlap"]
        );
        assert!(
            result["max_words"].is_null(),
            "page strategy must report null for max_words, got: {}",
            result["max_words"]
        );
        assert_eq!(result["chunk_strategy"], "page");
    }

    /// T-MCP-IDX-010: The "token" strategy returns the correct chunk_size and
    /// chunk_overlap values (non-null) and null for max_words.
    #[cfg(feature = "backend-ort")]
    #[tokio::test]
    async fn t_mcp_idx_010_token_strategy_returns_size_and_overlap() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "BAAI/bge-small-en-v1.5".to_string()
            }
        }

        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(
            tmp.path().join("test.pdf"),
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
        )
        .expect("write test pdf");

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

        let backend: std::sync::Arc<dyn EmbeddingBackend> = std::sync::Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let params = serde_json::json!({
            "directory": tmp.path().to_string_lossy(),
            "chunk_strategy": "token",
            "chunk_size": 512,
            "chunk_overlap": 64,
        });

        let result = handle(&state, &params).await.expect("handler must succeed");

        assert_eq!(result["chunk_size"], 512);
        assert_eq!(result["chunk_overlap"], 64);
        assert!(result["max_words"].is_null());
        assert_eq!(result["chunk_strategy"], "token");
    }

    /// T-MCP-IDX-011: The "sentence" strategy returns max_words and null for
    /// chunk_size and chunk_overlap.
    #[cfg(feature = "backend-ort")]
    #[tokio::test]
    async fn t_mcp_idx_011_sentence_strategy_returns_max_words() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "BAAI/bge-small-en-v1.5".to_string()
            }
        }

        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(
            tmp.path().join("test.pdf"),
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
        )
        .expect("write test pdf");

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

        let backend: std::sync::Arc<dyn EmbeddingBackend> = std::sync::Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let params = serde_json::json!({
            "directory": tmp.path().to_string_lossy(),
            "chunk_strategy": "sentence",
            "chunk_size": 300,
        });

        let result = handle(&state, &params).await.expect("handler must succeed");

        // Sentence strategy maps chunk_size to max_words.
        assert!(result["chunk_size"].is_null());
        assert!(result["chunk_overlap"].is_null());
        assert_eq!(result["max_words"], 300);
        assert_eq!(result["chunk_strategy"], "sentence");
    }

    /// T-MCP-IDX-008: The handler rejects unknown model IDs with a
    /// descriptive error instead of silently using the default dimension.
    #[cfg(feature = "backend-ort")]
    #[tokio::test]
    async fn t_mcp_idx_008_handler_rejects_unknown_model() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(
            tmp.path().join("test.pdf"),
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF",
        )
        .expect("write test pdf");

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

        let backend: std::sync::Arc<dyn EmbeddingBackend> = std::sync::Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let params = serde_json::json!({
            "directory": tmp.path().to_string_lossy(),
            "model": "nonexistent/fake-model",
        });

        let result = handle(&state, &params).await;
        assert!(result.is_err(), "unknown model must be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("not in the supported model catalog"),
            "error must mention model catalog: {err}"
        );
    }
}
