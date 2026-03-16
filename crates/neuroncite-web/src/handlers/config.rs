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

//! Configuration reading handler for the web frontend.
//!
//! Exposes the current `AppConfig` and runtime model information so the
//! frontend can populate settings fields with default values and display
//! the currently loaded embedding model without hardcoding anything.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::WebState;

/// Application configuration and runtime model state for the web frontend.
///
/// bind_address is intentionally excluded: it is server-internal configuration
/// that could expose the network topology to browser-accessible endpoints.
/// The port is included because the frontend uses it to construct API URLs
/// and it is already known from the connection the client used to reach this
/// endpoint.
#[derive(Serialize)]
pub struct ConfigResponse {
    /// Default embedding model identifier from the configuration file.
    pub default_model: String,
    /// Default chunking strategy name.
    pub default_strategy: String,
    /// Default chunk size in tokens/words.
    pub default_chunk_size: usize,
    /// Default chunk overlap in tokens/words.
    pub default_overlap: usize,
    /// Server port. The bind address is intentionally omitted.
    pub port: u16,
    /// HuggingFace model identifier of the embedding model currently loaded
    /// in the GPU worker. This may differ from `default_model` when the
    /// server was started with `--model <id>` or a model was cached from
    /// a previous session.
    pub loaded_model_id: String,
    /// Output vector dimensionality of the currently loaded embedding model.
    pub loaded_model_dimension: usize,
}

/// Returns the current application configuration and runtime model state.
/// The ChunkStrategy enum is converted to its lowercase string representation
/// for JSON serialization to maintain the same response schema that the
/// frontend expects.
pub async fn get_config(State(state): State<Arc<WebState>>) -> Json<ConfigResponse> {
    let config = &state.app_state.config;
    Json(ConfigResponse {
        default_model: config.default_model.clone(),
        default_strategy: config.default_strategy.to_string(),
        default_chunk_size: config.default_chunk_size,
        default_overlap: config.default_overlap,
        port: config.port,
        loaded_model_id: (*state.app_state.worker_handle.loaded_model_id())
            .clone()
            .to_string(),
        loaded_model_dimension: state
            .app_state
            .index
            .vector_dimension
            .load(std::sync::atomic::Ordering::Relaxed),
    })
}
