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

//! Handler for the `neuroncite_models` MCP tool.
//!
//! Lists all available embedding models from the model catalog with their
//! configuration parameters and local cache status. Model catalog access
//! depends on the backend-ort feature being compiled into neuroncite-embed.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Lists all available embedding models with configuration and cache status.
///
/// For each model, returns the model ID, display name, vector dimension,
/// maximum sequence length, pooling strategy, and whether the model files are
/// cached locally. When the backend-ort feature is not compiled, returns
/// the basic backend list from the registry instead.
pub async fn handle(
    _state: &Arc<AppState>,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    // The detailed model catalog (supported_model_configs) is only available
    // when the backend-ort feature is compiled. Fall back to the backend
    // registry list when it is not available.
    #[cfg(feature = "backend-ort")]
    {
        let configs = neuroncite_embed::supported_model_configs();
        let model_array: Vec<serde_json::Value> = configs
            .iter()
            .map(|c| {
                let cached = neuroncite_embed::is_cached(&c.model_id, "main");
                serde_json::json!({
                    "model_id": c.model_id,
                    "display_name": c.display_name,
                    "vector_dimension": c.vector_dimension,
                    "max_seq_len": c.max_seq_len,
                    "pooling": format!("{:?}", c.pooling),
                    "cached": cached,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "model_count": model_array.len(),
            "models": model_array,
        }))
    }

    #[cfg(not(feature = "backend-ort"))]
    {
        let backends = neuroncite_embed::list_available_backends();
        let backend_array: Vec<serde_json::Value> = backends
            .iter()
            .map(|b| serde_json::json!({"backend": b.name}))
            .collect();

        Ok(serde_json::json!({
            "model_count": 0,
            "models": [],
            "backends": backend_array,
            "note": "detailed model catalog requires the backend-ort feature",
        }))
    }
}
