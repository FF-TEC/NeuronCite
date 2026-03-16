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

//! Document page retrieval endpoint handler.
//!
//! Serves the text content of a single page from an indexed PDF document.
//! The document is identified by its file ID and the page by its 1-indexed
//! page number.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};

use neuroncite_store::{self as store};

use crate::dto::{API_VERSION, PageResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/documents/{id}/pages/{n}
///
/// Returns the text content, page number, and extraction backend for a
/// specific page of an indexed document.
#[utoipa::path(
    get,
    path = "/api/v1/documents/{id}/pages/{n}",
    params(
        ("id" = i64, Path, description = "File ID"),
        ("n" = i64, Path, description = "Page number (1-indexed)"),
    ),
    responses(
        (status = 200, description = "Page content", body = PageResponse),
        (status = 404, description = "Document or page not found"),
    )
)]
pub async fn get_page(
    State(state): State<Arc<AppState>>,
    Path((file_id, page_number)): Path<(i64, i64)>,
) -> Result<Json<PageResponse>, ApiError> {
    // Page numbers are 1-indexed. Values below 1 indicate a client error
    // (e.g., 0-based indexing or negative sentinel values) and must be
    // rejected with a descriptive error instead of silently returning
    // "page not found" from the database query.
    if page_number < 1 {
        return Err(ApiError::BadRequest {
            reason: format!("page_number must be >= 1 (1-indexed), got {page_number}"),
        });
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    let page = store::get_page(&conn, file_id, page_number).map_err(ApiError::from)?;

    Ok(Json(PageResponse {
        api_version: API_VERSION.to_string(),
        page_number: page.page_number,
        content: page.content,
        backend: page.backend,
    }))
}
