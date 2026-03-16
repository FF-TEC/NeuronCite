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

//! First-run setup status handlers for the web frontend.
//!
//! Two endpoints manage the first-run welcome dialog lifecycle:
//!
//! - `GET /api/v1/web/setup/status` — checks whether the setup marker file
//!   exists. Returns `is_first_run: true` when the file is absent, which
//!   causes the frontend to show the welcome dialog.
//!
//! - `POST /api/v1/web/setup/complete` — creates the marker file. Called by
//!   the frontend after the user either finishes the auto-install sequence or
//!   dismisses the dialog with "I will set it up myself". Subsequent launches
//!   will find the marker file and skip the welcome dialog.
//!
//! Both handlers are stateless: they perform only filesystem operations and
//! do not require access to the embedding backend, database, or broadcast
//! channels. The `State` extractor is therefore not used here.

use axum::Json;
use axum::http::StatusCode;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Response body for GET /api/v1/web/setup/status.
#[derive(Serialize)]
pub struct SetupStatusResponse {
    /// True when the `.setup_complete` marker file does not exist, meaning
    /// this is the first time the application has been launched on this machine.
    pub is_first_run: bool,
    /// Absolute path to the NeuronCite data directory shown in the welcome
    /// dialog so the user knows where all downloaded files will be stored.
    pub data_dir: String,
}

/// Response body for POST /api/v1/web/setup/complete.
#[derive(Serialize)]
pub struct SetupCompleteResponse {
    /// Always true on success. On failure the endpoint returns 500.
    pub ok: bool,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Returns whether the first-run setup has been completed on this machine.
///
/// The response includes the absolute path to the NeuronCite data directory
/// so the welcome dialog can show the user where downloaded files will go
/// before they consent to the downloads.
pub async fn get_setup_status() -> Json<SetupStatusResponse> {
    let marker = neuroncite_core::paths::setup_complete_path();
    let is_first_run = !marker.exists();

    let data_dir = neuroncite_core::paths::base_dir()
        .to_string_lossy()
        .into_owned();

    Json(SetupStatusResponse {
        is_first_run,
        data_dir,
    })
}

/// Creates the `.setup_complete` marker file to suppress the welcome dialog
/// on future application launches.
///
/// The parent directory (`<base_dir>/`) is created if it does not exist yet,
/// which can happen on a completely fresh installation before any download
/// has had the chance to create the directory tree.
///
/// Returns 500 if the filesystem write fails.
pub async fn mark_setup_complete() -> (StatusCode, Json<serde_json::Value>) {
    let marker = neuroncite_core::paths::setup_complete_path();

    // Ensure the parent data directory exists before writing the marker file.
    if let Some(parent) = marker.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Could not create data directory: {e}")
            })),
        );
    }

    // Write an empty marker file. The content is not meaningful; only the
    // presence of the file matters for the first-run check.
    if let Err(e) = std::fs::write(&marker, b"") {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Could not write setup marker: {e}")
            })),
        );
    }

    (StatusCode::OK, Json(serde_json::json!({ "ok": true })))
}
