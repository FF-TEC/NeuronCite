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

//! Session management endpoint handlers.
//!
//! Provides handlers for listing sessions, deleting a session (cascading to all
//! dependent records), triggering FTS5 optimization, and triggering HNSW index
//! rebuild.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use neuroncite_store::{self as store};

use crate::dto::{
    API_VERSION, OptimizeResponse, RebuildResponse, SessionDeleteByDirectoryRequest,
    SessionDeleteByDirectoryResponse, SessionDeleteResponse, SessionDto, SessionListResponse,
};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/sessions
///
/// Lists all indexing sessions ordered by creation time descending.
/// Each session includes aggregate statistics (file count, pages, chunks, bytes)
/// fetched via a single aggregate query and joined in-memory by session ID.
#[utoipa::path(
    get,
    path = "/api/v1/sessions",
    responses(
        (status = 200, description = "Session list", body = SessionListResponse),
    )
)]
pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SessionListResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    let rows = store::list_sessions(&conn).map_err(ApiError::from)?;

    // Fetch aggregate statistics for all sessions in a single query,
    // then index by session_id for O(1) lookup during DTO construction.
    let aggregates = store::all_session_aggregates(&conn).map_err(ApiError::from)?;
    let agg_map: HashMap<i64, store::SessionAggregates> =
        aggregates.into_iter().map(|a| (a.session_id, a)).collect();

    let sessions = rows
        .iter()
        .map(|r| {
            let agg = agg_map.get(&r.id);

            SessionDto {
                id: r.id,
                directory_path: r.directory_path.clone(),
                model_name: r.model_name.clone(),
                chunk_strategy: r.chunk_strategy.clone(),
                chunk_size: r.chunk_size,
                chunk_overlap: r.chunk_overlap,
                max_words: r.max_words,
                vector_dimension: r.vector_dimension,
                created_at: r.created_at,
                file_count: agg.map_or(0, |a| a.file_count),
                total_pages: agg.map_or(0, |a| a.total_pages),
                total_chunks: agg.map_or(0, |a| a.total_chunks),
                total_content_bytes: agg.map_or(0, |a| a.total_content_bytes),
            }
        })
        .collect();

    Ok(Json(SessionListResponse {
        api_version: API_VERSION.to_string(),
        sessions,
    }))
}

/// DELETE /api/v1/sessions/{id}
///
/// Deletes a session and all dependent records (files, pages, chunks) via
/// ON DELETE CASCADE.
#[utoipa::path(
    delete,
    path = "/api/v1/sessions/{id}",
    params(("id" = i64, Path, description = "Session ID")),
    responses(
        (status = 200, description = "Session deleted", body = SessionDeleteResponse),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Json<SessionDeleteResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the session exists before attempting deletion.
    store::get_session(&conn, id).map_err(ApiError::from)?;

    let deleted_count = store::delete_session(&conn, id).map_err(ApiError::from)?;

    // Evict the session's HNSW index from the in-memory per-session map.
    state.remove_hnsw(id);

    Ok(Json(SessionDeleteResponse {
        api_version: API_VERSION.to_string(),
        deleted: deleted_count > 0,
    }))
}

/// POST /api/v1/sessions/delete-by-directory
///
/// Deletes all sessions whose directory_path matches the specified directory.
/// The directory path is canonicalized before comparison. All dependent records
/// (files, pages, chunks, embeddings) are removed via ON DELETE CASCADE.
/// HNSW indices for deleted sessions are evicted from the in-memory map.
#[utoipa::path(
    post,
    path = "/api/v1/sessions/delete-by-directory",
    request_body = SessionDeleteByDirectoryRequest,
    responses(
        (status = 200, description = "Sessions deleted", body = SessionDeleteByDirectoryResponse),
    )
)]
pub async fn delete_sessions_by_directory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionDeleteByDirectoryRequest>,
) -> Result<Json<SessionDeleteByDirectoryResponse>, ApiError> {
    if req.directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "directory must not be empty".to_string(),
        });
    }

    // Normalize the requested directory path. Unlike start_index, this endpoint
    // must handle the case where the directory no longer exists on disk (e.g.,
    // the user deleted the folder and now wants to remove its index records).
    // std::fs::canonicalize would fail on a non-existent path, so we strip
    // the Windows extended-length prefix and use the resulting normalized string
    // for the database comparison instead. The stored paths in the database were
    // written by start_index, which canonicalized via fs::canonicalize and then
    // stripped the prefix, so this produces the same string format.
    let path_buf = std::path::PathBuf::from(&req.directory);
    let normalized_str =
        neuroncite_core::paths::strip_extended_length_prefix(&path_buf.to_string_lossy())
            .to_string();

    // Path containment: verify the target directory is within the allowed roots
    // when the server is configured with an allowlist. validate_path_access
    // also attempts canonicalization for the containment check; if the path
    // does not exist it skips the containment check and returns the raw path
    // on loopback bindings, or rejects it on non-loopback bindings without an
    // allowlist. The delete operation is safe in the loopback-no-allowlist case
    // because it only removes records for the exact stored path string.
    let validated = crate::util::validate_path_access(&path_buf, &state.config)?;
    let canonical_str =
        neuroncite_core::paths::strip_extended_length_prefix(&validated.to_string_lossy())
            .to_string();

    // Use the validated canonical string when validate_path_access produced a
    // canonical path (i.e., the directory exists); otherwise fall back to the
    // normalized raw path so that deletion still works for removed directories.
    let lookup_str = if canonical_str.is_empty() {
        normalized_str
    } else {
        canonical_str
    };

    let conn = state.pool.get().map_err(ApiError::from)?;

    let deleted_ids =
        store::delete_sessions_by_directory(&conn, &lookup_str).map_err(ApiError::from)?;

    // Evict all deleted sessions' HNSW indices from the in-memory map.
    state.remove_hnsw_many(&deleted_ids);

    let matched_directory = !deleted_ids.is_empty();
    Ok(Json(SessionDeleteByDirectoryResponse {
        api_version: API_VERSION.to_string(),
        matched_directory,
        deleted_session_ids: deleted_ids,
        directory: lookup_str,
    }))
}

/// POST /api/v1/sessions/{id}/optimize
///
/// Triggers FTS5 optimization for the given session's chunks. The optimize
/// operation merges FTS5 internal b-tree segments for faster query performance.
/// Returns 202 Accepted to indicate the operation was initiated.
#[utoipa::path(
    post,
    path = "/api/v1/sessions/{id}/optimize",
    params(("id" = i64, Path, description = "Session ID")),
    responses(
        (status = 202, description = "Optimization initiated", body = OptimizeResponse),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn optimize_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<(StatusCode, Json<OptimizeResponse>), ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the session exists.
    store::get_session(&conn, id).map_err(ApiError::from)?;

    // Run the FTS5 optimize command.
    store::optimize_fts(&conn).map_err(|e| ApiError::Internal {
        reason: format!("FTS5 optimize failed: {e}"),
    })?;

    Ok((
        StatusCode::ACCEPTED,
        Json(OptimizeResponse {
            api_version: API_VERSION.to_string(),
            status: "fts5 optimization completed".to_string(),
        }),
    ))
}

/// POST /api/v1/sessions/{id}/rebuild
///
/// Creates a background job to rebuild the HNSW index for the given session.
/// Returns 202 Accepted with the job_id.
#[utoipa::path(
    post,
    path = "/api/v1/sessions/{id}/rebuild",
    params(("id" = i64, Path, description = "Session ID")),
    responses(
        (status = 202, description = "Rebuild job created", body = RebuildResponse),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn rebuild_index(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<(StatusCode, HeaderMap, Json<RebuildResponse>), ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the session exists.
    store::get_session(&conn, id).map_err(ApiError::from)?;

    // Create a rebuild job record.
    let job_id = uuid::Uuid::new_v4().to_string();
    store::create_job(&conn, &job_id, "rebuild", Some(id)).map_err(ApiError::from)?;

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
        Json(RebuildResponse {
            api_version: API_VERSION.to_string(),
            job_id,
        }),
    ))
}
