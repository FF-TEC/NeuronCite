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

//! Chunk browsing endpoint handler.
//!
//! Provides paginated access to chunks for a specific file within a session,
//! with optional page-number filtering.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};

use neuroncite_store::{self as store};

use crate::dto::{API_VERSION, ChunkDto, ChunksQuery, ChunksResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/sessions/{session_id}/files/{file_id}/chunks
///
/// Returns paginated chunks for a file, with optional page filtering.
#[utoipa::path(
    get,
    path = "/api/v1/sessions/{session_id}/files/{file_id}/chunks",
    params(
        ("session_id" = i64, Path, description = "Session ID"),
        ("file_id" = i64, Path, description = "File ID"),
        ChunksQuery,
    ),
    responses(
        (status = 200, description = "Chunk list", body = ChunksResponse),
        (status = 404, description = "Session or file not found"),
    )
)]
pub async fn list_chunks(
    State(state): State<Arc<AppState>>,
    Path((session_id, file_id)): Path<(i64, i64)>,
    Query(query): Query<ChunksQuery>,
) -> Result<Json<ChunksResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the session exists (returns 404 if missing).
    store::get_session(&conn, session_id).map_err(ApiError::from)?;

    // Verify the file exists and belongs to the specified session.
    let file = store::get_file(&conn, file_id).map_err(ApiError::from)?;
    if file.session_id != session_id {
        return Err(ApiError::BadRequest {
            reason: format!("file {file_id} does not belong to session {session_id}"),
        });
    }

    let offset = query.offset.unwrap_or(0).max(0);
    let limit = query.limit.unwrap_or(20).clamp(1, 100);

    let (chunks, total) = store::browse_chunks(&conn, file_id, query.page_number, offset, limit)
        .map_err(ApiError::from)?;

    let chunk_dtos: Vec<ChunkDto> = chunks
        .iter()
        .map(|c| ChunkDto {
            chunk_id: c.id,
            chunk_index: c.chunk_index,
            page_start: c.page_start,
            page_end: c.page_end,
            word_count: c.content.split_whitespace().count(),
            byte_count: c.content.len(),
            content: c.content.clone(),
        })
        .collect();

    Ok(Json(ChunksResponse {
        api_version: API_VERSION.to_string(),
        session_id,
        file_id,
        total_chunks: total,
        offset,
        limit,
        returned: chunk_dtos.len(),
        chunks: chunk_dtos,
    }))
}
