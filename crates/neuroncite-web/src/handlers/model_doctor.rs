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

//! Model health diagnostics and repair handlers for the web frontend.
//!
//! Provides two endpoints:
//! - GET /api/v1/web/doctor/models -- Returns health status for all models
//!   in the static registry (embedding + reranker), with file-level detail
//!   about cache state, integrity, and repairability.
//! - POST /api/v1/web/doctor/repair-model -- Purges a broken model cache
//!   directory and re-downloads all files from HuggingFace.
//!
//! The diagnostic data comes from `neuroncite_embed::diagnose_model()` which
//! checks file presence, size, and checksum integrity for each model.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::WebState;

/// Single model diagnostic result sent to the frontend. Contains file-level
/// detail about cache state and a derived health status string.
#[derive(Serialize)]
pub struct ModelHealthEntry {
    /// HuggingFace model identifier.
    pub model_id: String,
    /// Whether the model directory exists on disk.
    pub directory_exists: bool,
    /// Filenames present in the model directory.
    pub files_present: Vec<String>,
    /// Filenames expected by the manifest but missing from the directory.
    pub files_missing: Vec<String>,
    /// Whether checksums pass verification. Null when verification cannot
    /// be performed (no manifest, I/O error, or files missing).
    pub checksums_valid: Option<bool>,
    /// Total size in bytes of all present files.
    pub total_size_bytes: u64,
    /// Whether the model can be repaired by purging and re-downloading.
    pub repairable: bool,
    /// Derived health status: "healthy", "incomplete", "corrupt", or "missing".
    pub health: String,
}

/// Response body for GET /api/v1/web/doctor/models.
#[derive(Serialize)]
pub struct ModelDoctorResponse {
    /// Diagnostic results for all known models (embedding + reranker).
    pub models: Vec<ModelHealthEntry>,
}

/// Returns health diagnostics for all models in the static registry.
/// Iterates through embedding and reranker model IDs from the download
/// manifest, running `diagnose_model()` for each and mapping the result
/// to a health status string.
pub async fn model_diagnostics(State(_state): State<Arc<WebState>>) -> Json<ModelDoctorResponse> {
    let all_ids = neuroncite_embed::all_model_ids();
    let mut entries = Vec::with_capacity(all_ids.len());

    for model_id in &all_ids {
        let diagnosis = neuroncite_embed::diagnose_model(model_id, "main");

        // Derive health status from the diagnostic data:
        // - "missing": model directory does not exist on disk
        // - "incomplete": directory exists but required files are missing
        // - "corrupt": all files present but checksum verification failed
        // - "healthy": all files present and checksums pass (or no manifest)
        let health = if !diagnosis.directory_exists {
            "missing".to_string()
        } else if !diagnosis.files_missing.is_empty() {
            "incomplete".to_string()
        } else if diagnosis.checksums_valid == Some(false) {
            "corrupt".to_string()
        } else {
            "healthy".to_string()
        };

        entries.push(ModelHealthEntry {
            model_id: diagnosis.model_id,
            directory_exists: diagnosis.directory_exists,
            files_present: diagnosis.files_present,
            files_missing: diagnosis.files_missing,
            checksums_valid: diagnosis.checksums_valid,
            total_size_bytes: diagnosis.total_size_bytes,
            repairable: diagnosis.repairable,
            health,
        });
    }

    Json(ModelDoctorResponse { models: entries })
}

/// Request body for POST /api/v1/web/doctor/repair-model.
#[derive(Deserialize)]
pub struct RepairModelRequest {
    /// HuggingFace model identifier to repair.
    pub model_id: String,
}

/// Repairs a broken model by purging its cache directory and re-downloading
/// all files from HuggingFace. The repair runs on a blocking thread because
/// it performs synchronous HTTP requests and filesystem I/O.
///
/// Returns 202 Accepted on success, 400 for unknown models, 500 on failure.
pub async fn repair_model(
    State(state): State<Arc<WebState>>,
    Json(req): Json<RepairModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate the model_id against the download manifest.
    if neuroncite_embed::model_expected_files(&req.model_id).is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Unknown model: '{}'. Not in the download manifest.", req.model_id)
            })),
        );
    }

    // Notify SSE subscribers that repair is starting
    let _ = state.model_tx.send(
        serde_json::json!({
            "event": "repair_started",
            "model_id": &req.model_id
        })
        .to_string(),
    );

    let model_id = req.model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Purge the existing cache directory (if it exists)
        neuroncite_embed::purge_cached_model(&model_id, "main")
            .map_err(|e| format!("failed to purge model cache: {e}"))?;

        // Re-download all files from HuggingFace
        neuroncite_embed::download_model(&model_id, "main")
            .map_err(|e| format!("model re-download failed: {e}"))?;

        Ok::<(), String>(())
    })
    .await;

    match result {
        Ok(Ok(())) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "repair_complete",
                    "model_id": &req.model_id
                })
                .to_string(),
            );
            (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({
                    "status": "repaired",
                    "model_id": req.model_id
                })),
            )
        }
        Ok(Err(e)) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "repair_failed",
                    "model_id": &req.model_id,
                    "error": &e
                })
                .to_string(),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Model repair failed: {e}")
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Repair task panicked: {e}")
            })),
        ),
    }
}
