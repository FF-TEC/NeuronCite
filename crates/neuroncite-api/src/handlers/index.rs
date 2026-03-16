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

//! Indexing endpoint handler.
//!
//! Accepts a request to start indexing a directory of PDF files. Creates an
//! indexing job, checks idempotency keys, enforces the concurrent job policy
//! (at most one running indexing job at a time), and returns the job and session
//! identifiers for progress polling.
//!
//! The vector_dimension for the session is resolved from the static model
//! catalog based on the requested model name when the `backend-ort` feature is
//! enabled. Without `backend-ort`, the dimension is read from `AppState`
//! (which reflects the currently loaded backend's output dimensionality).
//! The model identity guard always runs before dimension resolution, so in the
//! fallback path the loaded model matches the requested model and the dimension
//! from `AppState` is authoritative.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use neuroncite_core::{IndexConfig, StorageMode};
use neuroncite_store::{self as store, CreateJobResult};

use crate::dto::{API_VERSION, IndexRequest, IndexResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// Resolves the vector dimensionality for the given model ID, with a fallback
/// path for builds without `backend-ort`.
///
/// With `backend-ort`: delegates to the shared service function
/// `crate::service::index::resolve_vector_dimension`, which looks up the model
/// in the static catalog. Returns an error if the model is not found.
///
/// Without `backend-ort`: the static catalog is unavailable, so the dimension
/// is read from `state.index.vector_dimension` (an `AtomicUsize` set at
/// startup from the loaded backend). This is safe because the caller verifies
/// model identity before calling this function, guaranteeing that the requested
/// model is the one whose dimension is stored in `state.vector_dimension`.
fn resolve_vector_dimension(model_id: &str, state: &AppState) -> Result<usize, ApiError> {
    #[cfg(feature = "backend-ort")]
    {
        // With `backend-ort`, the shared service function queries the static
        // model catalog. The `state` parameter is not needed because the
        // catalog is compiled into the binary.
        let _ = state;
        crate::service::index::resolve_vector_dimension(model_id)
            .map_err(|reason| ApiError::BadRequest { reason })
    }
    #[cfg(not(feature = "backend-ort"))]
    {
        // Without the embedding backend crate, the static model catalog is not
        // compiled. Fall back to the dimension tracked in AppState, which was
        // set from the loaded backend at startup. The caller's model guard has
        // already confirmed that the requested model matches the loaded model,
        // so the dimension from AppState is correct for this request.
        let _ = model_id;
        Ok(state
            .index
            .vector_dimension
            .load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// POST /api/v1/index
///
/// Validates the request parameters, checks the idempotency key (if provided),
/// enforces the single-concurrent-job policy, creates a session and job record,
/// and returns 202 Accepted with the job_id and session_id.
#[utoipa::path(
    post,
    path = "/api/v1/index",
    request_body = IndexRequest,
    responses(
        (status = 202, description = "Indexing job created", body = IndexResponse),
        (status = 400, description = "Invalid request parameters"),
        (status = 409, description = "Concurrent indexing job in progress"),
    )
)]
pub async fn start_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexRequest>,
) -> Result<(StatusCode, HeaderMap, Json<IndexResponse>), ApiError> {
    // Validate request fields against the configured input limits. Covers
    // directory emptiness, chunk_size zero and upper bounds, and the
    // overlap-vs-size invariant. Must run before any database or filesystem
    // access so that invalid requests are rejected immediately.
    req.validate(&state.config.limits)?;

    // Path containment: verify the index directory is within the allowed
    // roots when the server is configured with an allowlist.
    crate::util::validate_path_access(std::path::Path::new(&req.directory), &state.config)?;

    // r2d2 requires a mutable borrow of the pooled connection when starting a
    // SQLite transaction. DerefMut on PooledConnection provides the &mut Connection.
    let mut conn = state.pool.get().map_err(ApiError::from)?;

    // Fast-path idempotency check outside the transaction: if the key already
    // exists, return the stored job/session immediately without acquiring any
    // write lock. This avoids write contention for the common replay case.
    if let Some(ref key) = req.idempotency_key
        && let Some(existing) = store::lookup_key(&conn, key).map_err(ApiError::from)?
    {
        let mut headers = HeaderMap::new();
        let location = format!("/api/v1/jobs/{}", existing.job_id);
        if let Ok(v) = HeaderValue::from_str(&location) {
            headers.insert(axum::http::header::LOCATION, v);
        }
        return Ok((
            StatusCode::ACCEPTED,
            headers,
            Json(IndexResponse {
                api_version: API_VERSION.to_string(),
                job_id: existing.job_id,
                session_id: existing.session_id,
            }),
        ));
    }

    // Enforce concurrent job policy: only one running indexing job at a time.
    // Uses has_active_job() with the partial index idx_job_state_active instead
    // of fetching all jobs and filtering client-side.
    let has_running = store::has_active_job(&conn, "index").map_err(ApiError::from)?;
    if has_running {
        return Err(ApiError::Conflict {
            reason: "an indexing job is already in progress".to_string(),
        });
    }

    // Build the IndexConfig from request parameters merged with defaults.
    let model_name = req
        .model_name
        .unwrap_or_else(|| state.config.default_model.clone());

    // Verify that the requested model is the one currently loaded by the worker.
    // The server loads exactly one embedding model at startup and cannot switch
    // models per-request. A mismatch between the requested model and the loaded
    // model would produce embeddings with the wrong dimensionality, causing a
    // panic in the HNSW builder via assert_eq!. Failing here gives the caller
    // a clear error instead of a panic deep in the executor pipeline.
    //
    // This check runs before resolve_vector_dimension so that the non-backend-ort
    // fallback (which reads state.vector_dimension) is guaranteed to return the
    // dimension for the correct model.
    let loaded_model = state.worker_handle.loaded_model_id();
    if model_name != *loaded_model {
        return Err(ApiError::Conflict {
            reason: format!(
                "requested model '{}' is not the model loaded by this server ('{}'); \
                 restart the server with the correct model to index with a different model",
                model_name, loaded_model
            ),
        });
    }

    // Resolve the vector dimension from the model catalog (with backend-ort) or
    // from state.vector_dimension (without backend-ort). The model guard above
    // has already verified that the requested model is the loaded model, so in
    // the fallback path the dimension from state is authoritative.
    let vector_dimension = resolve_vector_dimension(&model_name, &state)?;

    // Falls back to the config's ChunkStrategy enum converted to its lowercase
    // string representation when the request does not specify a strategy.
    let strategy = req
        .chunk_strategy
        .unwrap_or_else(|| state.config.default_strategy.to_string());

    // Validate the chunk_strategy string against the known strategy names.
    // This catches typos (e.g., "words" instead of "word") at the API
    // boundary with a clear error message instead of silently falling through
    // to the default match arm deep in the chunking pipeline.
    if strategy
        .parse::<neuroncite_core::config::ChunkStrategy>()
        .is_err()
    {
        return Err(ApiError::BadRequest {
            reason: format!(
                "unknown chunk_strategy: '{}' (valid: token, word, sentence, page)",
                strategy
            ),
        });
    }

    let chunk_size_val = req.chunk_size.unwrap_or(state.config.default_chunk_size);
    let chunk_overlap_val = req.chunk_overlap.unwrap_or(state.config.default_overlap);

    // Map the unified chunk_size parameter to strategy-specific IndexConfig
    // fields. The "sentence" strategy uses max_words instead of chunk_size.
    // The "page" strategy ignores all size parameters.
    let (cfg_chunk_size, cfg_chunk_overlap, cfg_max_words) = match strategy.as_str() {
        "token" | "word" => (Some(chunk_size_val), Some(chunk_overlap_val), None),
        "sentence" => (None, None, Some(chunk_size_val)),
        _ => (None, None, None), // "page" has no parameters
    };

    // Canonicalize the requested directory path. The path must exist on disk
    // so that the canonical form is authoritative. A non-existent path yields
    // BadRequest so the client receives a clear error rather than the indexer
    // silently operating on an unverified path.
    let canonical_dir =
        neuroncite_core::paths::canonicalize_directory(std::path::Path::new(&req.directory))
            .map_err(|e| ApiError::BadRequest {
                reason: format!("invalid directory: {e}"),
            })?;

    let config = IndexConfig {
        directory: canonical_dir,
        model_name,
        chunk_strategy: strategy,
        chunk_size: cfg_chunk_size,
        chunk_overlap: cfg_chunk_overlap,
        max_words: cfg_max_words,
        ocr_language: state.config.ocr_language.clone(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension,
    };

    // Create or find the session for this configuration.
    let session_id = match store::find_session(&conn, &config).map_err(ApiError::from)? {
        Some(id) => id,
        None => store::create_session(&conn, &config, "0.1.0").map_err(ApiError::from)?,
    };

    let job_id = uuid::Uuid::new_v4().to_string();

    // When an idempotency key is provided, use the atomic function that wraps
    // job creation and key storage in a single BEGIN IMMEDIATE transaction.
    // This prevents the race where two concurrent requests with the same key
    // both pass the fast-path lookup above, then both create jobs, and the
    // second store_key call fails with a PK collision -- leaving a job row with
    // no idempotency record. With create_job_with_key, only one request can
    // win the write lock; the other sees the existing entry and returns early.
    if let Some(ref key) = req.idempotency_key {
        let result = store::create_job_with_key(&mut conn, key, &job_id, "index", session_id)
            .map_err(ApiError::from)?;

        match result {
            CreateJobResult::Existing(entry) => {
                // Another request beat us to the write lock and created the job.
                // Return the existing job/session identifiers idempotently.
                let mut headers = HeaderMap::new();
                let location = format!("/api/v1/jobs/{}", entry.job_id);
                if let Ok(v) = HeaderValue::from_str(&location) {
                    headers.insert(axum::http::header::LOCATION, v);
                }
                return Ok((
                    StatusCode::ACCEPTED,
                    headers,
                    Json(IndexResponse {
                        api_version: API_VERSION.to_string(),
                        job_id: entry.job_id,
                        session_id: entry.session_id,
                    }),
                ));
            }
            CreateJobResult::Created { .. } => {
                // Job and idempotency key stored atomically. Fall through to
                // notify the executor and return 202 Accepted.
            }
        }
    } else {
        // No idempotency key: create the job row directly. This path has no
        // idempotency guarantee but is also used by fire-and-forget clients
        // that do not supply keys.
        store::create_job(&conn, &job_id, "index", Some(session_id)).map_err(ApiError::from)?;
    }

    // Wake the executor immediately so it picks up the new job without
    // waiting for the next poll interval.
    state.job_notify.notify_one();

    let mut headers = HeaderMap::new();
    let location = format!("/api/v1/jobs/{job_id}");
    if let Ok(v) = HeaderValue::from_str(&location) {
        headers.insert(axum::http::header::LOCATION, v);
    }
    Ok((
        StatusCode::ACCEPTED,
        headers,
        Json(IndexResponse {
            api_version: API_VERSION.to_string(),
            job_id,
            session_id,
        }),
    ))
}
