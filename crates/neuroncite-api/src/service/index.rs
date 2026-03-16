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

// Indexing-related business logic shared between REST and MCP entry points.
//
// This module holds functions that perform pure domain logic related to
// index session creation and configuration. They are free of HTTP and MCP
// protocol concerns, so both the Axum handler and the MCP handler can call
// them directly. The functions return `Result<T, String>` because the error
// representation is protocol-agnostic; each caller converts the String into
// its own error type (ApiError for REST, plain String for MCP).

/// Resolves the vector dimensionality for the given embedding model ID by
/// looking it up in the static model catalog compiled into the binary.
///
/// With the `backend-ort` feature enabled, the function queries
/// `neuroncite_embed::find_model_config` for the model's catalog entry and
/// returns the `vector_dimension` field. If the model ID is not present in
/// the catalog (typo, unsupported model), an error is returned listing the
/// available models so the caller can correct the request.
///
/// Without the `backend-ort` feature, the static model catalog is not
/// compiled into the binary. This function returns an error indicating that
/// catalog-based resolution is unavailable. Callers that still need a
/// dimension value (e.g., the REST API handler) fall back to reading the
/// dimension from `AppState`, which was set at startup from the loaded
/// embedding backend.
pub fn resolve_vector_dimension(model_id: &str) -> Result<usize, String> {
    #[cfg(feature = "backend-ort")]
    {
        neuroncite_embed::find_model_config(model_id)
            .map(|cfg| cfg.vector_dimension)
            .ok_or_else(|| {
                format!(
                    "model '{}' is not in the supported model catalog; available models: {}",
                    model_id,
                    neuroncite_embed::supported_model_configs()
                        .iter()
                        .map(|c| c.model_id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
    #[cfg(not(feature = "backend-ort"))]
    {
        let _ = model_id;
        Err("no embedding backend compiled; enable the 'backend-ort' feature".to_string())
    }
}
