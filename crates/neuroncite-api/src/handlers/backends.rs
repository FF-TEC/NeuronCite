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

//! Embedding backend information endpoint handler.
//!
//! Returns information about all compiled-in embedding backends, including
//! the backend name, GPU support status, and number of available models.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::dto::{API_VERSION, BackendDto, BackendListResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/backends
///
/// Lists all compiled-in embedding backends with their GPU support status
/// and model count.
#[utoipa::path(
    get,
    path = "/api/v1/backends",
    responses(
        (status = 200, description = "Backend list", body = BackendListResponse),
    )
)]
pub async fn list_backends(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<BackendListResponse>, ApiError> {
    let infos = neuroncite_embed::list_available_backends();

    let backends = infos
        .iter()
        .map(|info| BackendDto {
            name: info.name.clone(),
            gpu_supported: info.gpu_supported,
            model_count: info.models.len(),
        })
        .collect();

    Ok(Json(BackendListResponse {
        api_version: API_VERSION.to_string(),
        backends,
    }))
}
