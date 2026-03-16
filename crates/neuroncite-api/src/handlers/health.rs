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

//! Health check endpoint handler.
//!
//! Returns a 200 OK response with basic system status information: application
//! version, active features, GPU availability, backend name, and optional
//! extraction backends (pdfium, tesseract).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::dto::{API_VERSION, HealthResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/health
///
/// Returns the server's health status, including the version string,
/// compiled feature flags, GPU availability, active embedding backend,
/// and optional extraction backend availability.
#[utoipa::path(
    get,
    path = "/api/v1/health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
pub async fn health(State(state): State<Arc<AppState>>) -> Result<Json<HealthResponse>, ApiError> {
    let backends = neuroncite_embed::list_available_backends();

    let (active_backend, gpu_available) = backends
        .first()
        .map(|b| (b.name.clone(), b.gpu_supported))
        .unwrap_or_else(|| ("none".to_string(), false));

    let build_features = backends
        .iter()
        .map(|b| format!("backend-{}", b.name))
        .collect();

    // Pdfium and tesseract availability are determined at compile time via
    // the feature flags forwarded through the crate dependency chain. The
    // binary crate's `pdfium` and `ocr` features propagate to neuroncite-api
    // via Cargo's feature forwarding, so `cfg!(feature = "...")` reflects
    // the actual build configuration rather than a hardcoded value.
    let pdfium_available = cfg!(feature = "pdfium");
    let tesseract_available = cfg!(feature = "ocr");

    Ok(Json(HealthResponse {
        api_version: API_VERSION.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_features,
        gpu_available,
        active_backend,
        reranker_available: state.worker_handle.reranker_available(),
        pdfium_available,
        tesseract_available,
    }))
}
