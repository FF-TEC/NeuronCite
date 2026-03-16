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

//! Web-specific REST API handlers under `/api/v1/web/`.
//!
//! These endpoints provide functionality that the browser-based frontend needs
//! but the headless REST API does not expose: file system browsing (the browser
//! cannot access the local filesystem), model catalog management, dependency
//! probes, MCP registration, and configuration reading.

pub mod browse;
pub mod config;
pub mod doctor;
pub mod mcp;
pub mod model_doctor;
pub mod models;
pub mod ollama;
pub mod setup;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::WebState;

/// Builds the router for all web-specific endpoints under /api/v1/web/.
pub fn web_routes(state: Arc<WebState>) -> Router {
    Router::new()
        // File system browsing, native OS folder/file picker dialogs
        .route("/browse", post(browse::browse_directory))
        .route("/browse/native", post(browse::native_folder_dialog))
        .route("/browse/native-file", post(browse::native_file_dialog))
        .route("/scan-documents", post(browse::scan_documents))
        // Configuration reading
        .route("/config", get(config::get_config))
        // Model catalog and management
        .route("/models/catalog", get(models::model_catalog))
        .route("/models/download", post(models::download_model))
        .route("/models/activate", post(models::activate_model))
        // Reranker model loading (download if needed, then hot-swap into worker)
        .route("/models/load-reranker", post(models::load_reranker))
        // Dependency probes
        .route("/doctor/probes", get(doctor::run_probes))
        .route("/doctor/install", post(doctor::install_dependency))
        // Model-specific diagnostics and repair
        .route("/doctor/models", get(model_doctor::model_diagnostics))
        .route("/doctor/repair-model", post(model_doctor::repair_model))
        // MCP server registration
        .route("/mcp/status", get(mcp::mcp_status))
        .route("/mcp/install", post(mcp::mcp_install))
        .route("/mcp/uninstall", post(mcp::mcp_uninstall))
        // Ollama LLM proxy (connection check, model listing, management)
        .route("/ollama/status", get(ollama::ollama_status))
        .route("/ollama/models", get(ollama::list_ollama_models))
        .route("/ollama/running", get(ollama::list_running_models))
        .route("/ollama/catalog", get(ollama::ollama_catalog))
        .route("/ollama/pull", post(ollama::pull_model))
        .route("/ollama/delete", post(ollama::delete_model))
        .route("/ollama/show", post(ollama::show_model))
        // First-run setup dialog state (marker file presence check and write)
        .route("/setup/status", get(setup::get_setup_status))
        .route("/setup/complete", post(setup::mark_setup_complete))
        .with_state(state)
}
