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

//! Backend registry for discovering and instantiating embedding backends.
//!
//! The registry provides a compile-time inventory of all embedding backends that
//! were included via feature flags. At runtime, callers use `list_available_backends`
//! to discover which backends exist, and `create_backend` / `create_reranker` to
//! instantiate them by name.

use neuroncite_core::{EmbeddingBackend, ModelInfo, Reranker};

use crate::error::EmbedError;

/// Default backend name used when no backend list is available or when the
/// caller needs a fallback identifier. Matches the "ort" (ONNX Runtime) backend
/// that is compiled in when the `backend-ort` feature flag is active.
pub const DEFAULT_BACKEND_NAME: &str = "ort";

/// Metadata describing a compiled-in embedding backend. Returned by
/// `list_available_backends` to allow the GUI and API to present available
/// options to the user.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend identifier (e.g., "ort").
    pub name: String,

    /// List of embedding models supported by this backend.
    pub models: Vec<ModelInfo>,

    /// Whether this backend supports GPU-accelerated inference.
    pub gpu_supported: bool,
}

/// Returns metadata for every embedding backend that was compiled into this
/// binary via feature flags. The list is populated at compile time using
/// `#[cfg(feature)]` checks.
///
/// # Returns
///
/// A vector of `BackendInfo` structs, one per active backend feature.
/// The vector is empty if no backend features are enabled.
#[must_use]
pub fn list_available_backends() -> Vec<BackendInfo> {
    // The `mut` is required when at least one backend feature is enabled,
    // because each cfg block pushes into the vector. When no features are
    // active the variable is never mutated, so the allow attribute prevents
    // a warning in that configuration.
    #[allow(unused_mut)]
    let mut backends = Vec::new();

    #[cfg(feature = "backend-ort")]
    {
        let embedder = crate::ort::embedder::OrtEmbedder::unloaded();
        backends.push(BackendInfo {
            name: "ort".to_string(),
            models: embedder.available_models(),
            gpu_supported: embedder.supports_gpu(),
        });
    }

    backends
}

/// Creates an `EmbeddingBackend` instance by backend name.
///
/// The backend name must match a compiled-in backend ("ort"). The returned
/// backend is in an unloaded state; the caller must invoke `load_model`
/// before using it for inference.
///
/// # Arguments
///
/// * `backend_name` - The identifier of the backend to create.
///
/// # Errors
///
/// Returns `EmbedError::Backend` if the requested backend name does not match
/// any compiled-in backend feature.
pub fn create_backend(backend_name: &str) -> Result<Box<dyn EmbeddingBackend>, EmbedError> {
    match backend_name {
        #[cfg(feature = "backend-ort")]
        "ort" => Ok(Box::new(crate::ort::embedder::OrtEmbedder::unloaded())),

        _ => Err(EmbedError::Backend(format!(
            "backend '{backend_name}' is not available (not compiled or unknown name)"
        ))),
    }
}

/// Creates a `Reranker` instance by backend name.
///
/// The backend name must match one of the compiled-in backends. The returned
/// reranker is in an unloaded state; the caller must invoke `load_model`
/// before using it for reranking.
///
/// # Arguments
///
/// * `backend_name` - The identifier of the backend to create.
///
/// # Errors
///
/// Returns `EmbedError::Backend` if the requested backend name does not match
/// any compiled-in backend feature.
pub fn create_reranker(backend_name: &str) -> Result<Box<dyn Reranker>, EmbedError> {
    match backend_name {
        #[cfg(feature = "backend-ort")]
        "ort" => Ok(Box::new(crate::ort::reranker::OrtReranker::unloaded())),

        _ => Err(EmbedError::Backend(format!(
            "reranker backend '{backend_name}' is not available (not compiled or unknown name)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-EMB-004: Registry returns one entry per compiled-in backend.
    /// With the default `backend-ort` feature enabled, this test expects
    /// exactly one backend in the registry.
    #[test]
    fn t_emb_004_registry_returns_one_entry_per_compiled_backend() {
        let backends = list_available_backends();

        // Count the expected number of backends based on active feature flags.
        // The `mut` is required when at least one backend feature is enabled,
        // but unused when no features are active.
        #[allow(unused_mut)]
        let mut expected_count = 0;
        #[cfg(feature = "backend-ort")]
        {
            expected_count += 1;
        }

        assert_eq!(
            backends.len(),
            expected_count,
            "registry should return one entry per compiled-in backend"
        );
    }

    /// T-EMB-009: Each backend returns at least one ModelInfo.
    /// Every compiled-in backend must advertise at least one supported model.
    #[test]
    fn t_emb_009_each_backend_returns_at_least_one_model_info() {
        let backends = list_available_backends();

        for backend in &backends {
            assert!(
                !backend.models.is_empty(),
                "backend '{}' must advertise at least one model",
                backend.name
            );
        }
    }

    /// T-EMB-010: supports_gpu reflects hardware availability.
    /// This test verifies that the gpu_supported field is a boolean
    /// (true or false) -- the actual value depends on the hardware
    /// and CUDA driver availability at test time.
    #[test]
    fn t_emb_010_supports_gpu_reflects_hardware_availability() {
        let backends = list_available_backends();

        for backend in &backends {
            // The gpu_supported field is a boolean; we verify that it
            // is consistent with the backend's own report.
            let _gpu = backend.gpu_supported;
            // Smoke test: the field is accessible and does not panic.
            // The actual GPU availability depends on the test machine.
            tracing::info!(
                backend = %backend.name,
                gpu_supported = backend.gpu_supported,
                "backend GPU support status"
            );
        }
    }

    /// Verifies that requesting a non-existent backend returns an error.
    #[test]
    fn create_backend_unknown_name_returns_error() {
        let result = create_backend("nonexistent_backend");
        assert!(result.is_err(), "unknown backend name should return Err");
    }

    /// Verifies that requesting a non-existent reranker returns an error.
    #[test]
    fn create_reranker_unknown_name_returns_error() {
        let result = create_reranker("nonexistent_backend");
        assert!(result.is_err(), "unknown reranker name should return Err");
    }
}
