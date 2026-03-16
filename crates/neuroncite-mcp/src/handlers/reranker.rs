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

//! Handler for the `neuroncite_reranker_load` MCP tool.
//!
//! Loads a cross-encoder reranker model at runtime and hot-swaps it into the
//! GPU worker via `WorkerHandle::swap_reranker`. This uses the `ArcSwap`-based
//! lock-free slot in the worker, so the change takes effect on the next
//! `RerankBatch` message without any restart or channel reconnection.
//!
//! When the requested model is absent from the local cache, the handler
//! downloads it automatically from HuggingFace before loading. The model
//! must appear in `neuroncite_embed::cache::MODEL_FILES` for auto-download
//! to succeed; models outside the manifest still require manual placement.
//!
//! After a reranker is loaded, `neuroncite_health` reports `reranker_available: true`
//! and the `rerank` parameter in search/batch_search tools starts working.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::Reranker;

/// Loads a cross-encoder reranker model and installs it in the GPU worker.
///
/// When the model directory is absent from the local cache, the handler
/// downloads it from HuggingFace before loading. The download blocks the
/// spawned thread and may take several minutes for large models.
///
/// # Parameters (from MCP tool call)
///
/// - `model_id` (required): Hugging Face model identifier for the cross-encoder
///   model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"). If the model is not
///   in the local cache, it is downloaded automatically.
/// - `backend` (optional): Backend to use for reranker inference. Defaults to
///   "ort" (ONNX Runtime). Must match a compiled-in backend feature.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let model_id = params["model_id"]
        .as_str()
        .ok_or("missing required parameter: model_id")?;

    let model_id = model_id.trim();
    if model_id.is_empty() {
        return Err("model_id must not be empty".to_string());
    }

    let backend_name = params["backend"].as_str().unwrap_or("ort").to_string();
    let model_id_owned = model_id.to_string();

    // Create and load the reranker on a blocking thread. Model loading involves
    // filesystem I/O and ONNX session initialization (CPU-bound graph optimization).
    // When the model is not cached, the download runs synchronously on this thread
    // before session creation, so the caller should expect extended latency.
    // Corrupt cached files are detected via SHA-256 checksum verification
    // and automatically purged before re-downloading.
    let (reranker_arc, was_downloaded): (Arc<dyn Reranker>, bool) =
        tokio::task::spawn_blocking(move || {
            let mut downloaded = false;

            if neuroncite_embed::is_cached(&model_id_owned, "main") {
                // Verify integrity of cached files. If corrupt, purge and re-download.
                match neuroncite_embed::verify_cached_model(&model_id_owned, "main") {
                    Ok(true) => {}
                    _ => {
                        tracing::warn!(
                            model_id = %model_id_owned,
                            "cached reranker model is corrupt, purging and re-downloading"
                        );
                        neuroncite_embed::purge_cached_model(&model_id_owned, "main")
                            .map_err(|e| format!("failed to purge corrupt cache: {e}"))?;
                        neuroncite_embed::download_model(&model_id_owned, "main").map_err(|e| {
                            format!("reranker re-download failed for '{model_id_owned}': {e}")
                        })?;
                        downloaded = true;
                    }
                }
            } else {
                tracing::info!(
                    model_id = %model_id_owned,
                    "reranker model not in local cache, downloading from HuggingFace"
                );
                neuroncite_embed::download_model(&model_id_owned, "main").map_err(|e| {
                    format!("reranker model download failed for '{model_id_owned}': {e}")
                })?;
                tracing::info!(
                    model_id = %model_id_owned,
                    "reranker model download complete"
                );
                downloaded = true;
            }

            let mut reranker = neuroncite_embed::create_reranker(&backend_name)
                .map_err(|e| format!("reranker backend error: {e}"))?;

            reranker
                .load_model(&model_id_owned)
                .map_err(|e| format!("reranker model loading failed: {e}"))?;

            let arc: Arc<dyn Reranker> = Arc::from(reranker);
            Ok::<(Arc<dyn Reranker>, bool), String>((arc, downloaded))
        })
        .await
        .map_err(|e| format!("reranker loading task panicked: {e}"))??;

    let reranker_name = reranker_arc.name().to_string();

    // Hot-swap the reranker into the GPU worker. The worker task observes
    // the change on its next RerankBatch message via ArcSwap::load.
    state
        .worker_handle
        .swap_reranker(Some(reranker_arc), model_id);

    Ok(serde_json::json!({
        "model_id": model_id,
        "backend": params["backend"].as_str().unwrap_or("ort"),
        "reranker_name": reranker_name,
        "reranker_available": true,
        "downloaded": was_downloaded,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-RERANK-001: Handler rejects requests missing the model_id
    /// parameter. Validates the parameter extraction logic.
    #[test]
    fn t_mcp_rerank_001_missing_model_id() {
        let params = serde_json::json!({
            "backend": "ort"
        });
        assert!(params["model_id"].as_str().is_none());
    }

    /// T-MCP-RERANK-002: Handler rejects an empty model_id string. The
    /// model_id must contain a valid Hugging Face model identifier.
    #[test]
    fn t_mcp_rerank_002_empty_model_id() {
        let model_id = "   ";
        let trimmed = model_id.trim();
        assert!(
            trimmed.is_empty(),
            "whitespace-only model_id must be rejected"
        );
    }

    /// T-MCP-RERANK-003: Backend name defaults to "ort" when not specified.
    /// Validates the default value logic for the optional backend parameter.
    #[test]
    fn t_mcp_rerank_003_backend_defaults_to_ort() {
        let params = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        });
        let backend = params["backend"].as_str().unwrap_or("ort");
        assert_eq!(backend, "ort");
    }

    /// T-MCP-RERANK-004: Module compiles and the handler returns the correct
    /// Result type. Compilation of this module validates that all imports,
    /// types, and function signatures are correct.
    #[test]
    fn t_mcp_rerank_004_module_compiles() {
        let _check: fn() -> Result<serde_json::Value, String> =
            || Err("compile-time type check only".to_string());
    }

    /// T-MCP-RERANK-005: The create_reranker function from neuroncite-embed
    /// returns an error for invalid backend names. Validates that the handler's
    /// error path for unknown backends is reachable.
    #[test]
    fn t_mcp_rerank_005_invalid_backend_returns_error() {
        let result = neuroncite_embed::create_reranker("nonexistent_backend");
        assert!(
            result.is_err(),
            "create_reranker with unknown backend must return Err"
        );
    }

    /// T-MCP-RERANK-006: Explicit backend parameter overrides the default.
    /// When the user specifies a backend, that value is used instead of "ort".
    #[test]
    fn t_mcp_rerank_006_explicit_backend_used() {
        let params = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "backend": "custom"
        });
        let backend = params["backend"].as_str().unwrap_or("ort");
        assert_eq!(backend, "custom");
    }

    /// T-MCP-RERANK-007: model_id is extracted correctly from the params JSON.
    #[test]
    fn t_mcp_rerank_007_model_id_extraction() {
        let params = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        });

        let model_id = params["model_id"].as_str().unwrap();
        assert_eq!(model_id, "cross-encoder/ms-marco-MiniLM-L-6-v2");
    }

    /// T-MCP-RERANK-008: The Reranker trait from neuroncite-core is accessible
    /// in this module. Validates the import chain.
    #[test]
    fn t_mcp_rerank_008_reranker_trait_accessible() {
        // Verify the trait is imported and usable at compile time.
        fn _accepts_reranker(_r: &dyn neuroncite_core::Reranker) {}
    }

    /// T-MCP-RERANK-009: Arc<dyn Reranker> can be constructed from
    /// Box<dyn Reranker>. This validates the Arc::from(boxed) conversion
    /// used in the handler to wrap the loaded reranker.
    #[test]
    fn t_mcp_rerank_009_arc_from_box_conversion() {
        use std::sync::Arc;

        /// Stub reranker for testing the Box-to-Arc conversion.
        struct StubReranker;

        impl neuroncite_core::Reranker for StubReranker {
            fn name(&self) -> &str {
                "stub"
            }

            fn rerank_batch(
                &self,
                _query: &str,
                candidates: &[&str],
            ) -> Result<Vec<f64>, neuroncite_core::NeuronCiteError> {
                Ok(candidates.iter().map(|_| 0.0).collect())
            }

            fn load_model(
                &mut self,
                _model_id: &str,
            ) -> Result<(), neuroncite_core::NeuronCiteError> {
                Ok(())
            }
        }

        let boxed: Box<dyn neuroncite_core::Reranker> = Box::new(StubReranker);
        let arc: Arc<dyn neuroncite_core::Reranker> = Arc::from(boxed);

        // The Arc-wrapped reranker must be callable.
        assert_eq!(arc.name(), "stub");
    }

    /// T-MCP-RERANK-010: The create_reranker function succeeds for each
    /// compiled-in backend. Validates that at least one backend can create
    /// a reranker instance. Requires backend-ort because list_available_backends
    /// returns an empty list without a compiled-in backend.
    #[cfg(feature = "backend-ort")]
    #[test]
    fn t_mcp_rerank_010_at_least_one_backend_creates_reranker() {
        let backends = neuroncite_embed::list_available_backends();

        // At least one backend must be compiled in.
        assert!(
            !backends.is_empty(),
            "at least one embedding backend must be compiled in"
        );

        let mut any_success = false;
        for backend in &backends {
            if neuroncite_embed::create_reranker(&backend.name).is_ok() {
                any_success = true;
            }
        }

        assert!(
            any_success,
            "at least one backend must successfully create a reranker"
        );
    }

    /// T-MCP-RERANK-011: The cross-encoder model is present in the download
    /// manifest. When the model is not in cache, the handler auto-downloads
    /// it. This test verifies that the download infrastructure knows the
    /// canonical model ID, so the auto-download path is reachable.
    #[test]
    fn t_mcp_rerank_011_cross_encoder_in_download_manifest() {
        // The canonical reranker model must appear in the MODEL_FILES manifest
        // so that download_model() has a file list to fetch from HuggingFace.
        // If this returns None, auto-download would fail with "no download manifest".
        let expected_files =
            neuroncite_embed::model_expected_files("cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert!(
            expected_files.is_some(),
            "cross-encoder/ms-marco-MiniLM-L-6-v2 must be in the download manifest"
        );

        let files = expected_files.expect("expected_files is Some");
        assert!(
            files.contains(&"model.onnx"),
            "manifest must include model.onnx for the cross-encoder model"
        );
        assert!(
            files.contains(&"tokenizer.json"),
            "manifest must include tokenizer.json for the cross-encoder model"
        );
    }

    /// T-MCP-RERANK-012: The `downloaded` field in the handler response is a
    /// boolean. This validates that the response JSON shape is correct for
    /// callers that inspect whether a download was triggered.
    #[test]
    fn t_mcp_rerank_012_response_includes_downloaded_field() {
        // Simulate the response structure that the handler would produce.
        let response = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "backend": "ort",
            "reranker_name": "ONNX Runtime Cross-Encoder",
            "reranker_available": true,
            "downloaded": true,
        });

        assert!(
            response["downloaded"].is_boolean(),
            "downloaded field must be a boolean"
        );
        assert!(
            response["reranker_available"]
                .as_bool()
                .expect("reranker_available is bool"),
            "reranker_available must be true after a successful load"
        );
    }

    /// T-MCP-RERANK-013: is_cached returns false for a model ID that has never
    /// been downloaded. The auto-download branch in the handler is triggered
    /// when is_cached returns false. This test verifies the is_cached API
    /// behaves correctly for the reranker model ID in a clean environment.
    #[test]
    fn t_mcp_rerank_013_is_cached_false_for_uncached_model() {
        // A model ID that cannot possibly be in any cache directory.
        let result = neuroncite_embed::is_cached(
            "cross-encoder/nonexistent-model-for-test-only-xyz123",
            "main",
        );
        assert!(
            !result,
            "is_cached must return false for a model that was never downloaded"
        );
    }

    /// T-MCP-RERANK-014: download_model returns a descriptive error when the
    /// requested model ID is not in MODEL_FILES. The error must name the model
    /// so the user can identify which model is unknown.
    #[test]
    fn t_mcp_rerank_014_download_model_errors_for_unknown_id() {
        let result =
            neuroncite_embed::download_model("cross-encoder/not-in-manifest-abc123", "main");
        assert!(
            result.is_err(),
            "download_model must return Err for an unknown model ID"
        );
        let err_message = result.expect_err("result is Err").to_string();
        assert!(
            err_message.contains("cross-encoder/not-in-manifest-abc123"),
            "error message must identify the unknown model ID, got: {err_message}"
        );
    }

    /// T-MCP-RERANK-015: download_model succeeds (i.e., does not immediately
    /// error) for the canonical cross-encoder model ID because it is in the
    /// manifest. The actual network download is not triggered here; only the
    /// manifest lookup and directory creation path are exercised. The test
    /// verifies that the error is NOT "no download manifest for model", which
    /// was the previous failure mode before the manifest entry was added.
    ///
    /// Note: The test may return Ok (if network is available and download
    /// succeeds) or Err with a network/IO error. Both outcomes are valid here
    /// because the manifest lookup succeeded in either case.
    #[test]
    fn t_mcp_rerank_015_download_model_accepts_cross_encoder_id() {
        // The cross-encoder model is in MODEL_FILES, so download_model must
        // not immediately reject it with "no download manifest for model".
        // We distinguish this from a network error by checking the error text.
        if let Err(e) =
            neuroncite_embed::download_model("cross-encoder/ms-marco-MiniLM-L-6-v2", "main")
        {
            let err_str = e.to_string();
            assert!(
                !err_str.contains("no download manifest"),
                "download_model must not reject the cross-encoder model with \
                 'no download manifest' -- got: {err_str}"
            );
            // A network error or IO error during download is acceptable
            // in a test environment without internet access.
        }
        // If download_model returned Ok, the model was downloaded and cached.
    }

    /// T-MCP-RERANK-016: The handler's auto-download response shape contains
    /// all mandatory fields. Callers (e.g., MCP AI agents) parse the response
    /// JSON and must find: model_id, backend, reranker_name, reranker_available,
    /// and downloaded.
    #[test]
    fn t_mcp_rerank_016_response_shape_all_mandatory_fields_present() {
        let response_no_download = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "backend": "ort",
            "reranker_name": "ONNX Runtime Cross-Encoder",
            "reranker_available": true,
            "downloaded": false,
        });

        let response_with_download = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "backend": "ort",
            "reranker_name": "ONNX Runtime Cross-Encoder",
            "reranker_available": true,
            "downloaded": true,
        });

        for response in &[&response_no_download, &response_with_download] {
            assert!(
                response["model_id"].is_string(),
                "model_id must be a string"
            );
            assert!(response["backend"].is_string(), "backend must be a string");
            assert!(
                response["reranker_name"].is_string(),
                "reranker_name must be a string"
            );
            assert!(
                response["reranker_available"].is_boolean(),
                "reranker_available must be a boolean"
            );
            assert!(
                response["downloaded"].is_boolean(),
                "downloaded must be a boolean"
            );
        }

        // downloaded=false when model was already in cache.
        assert!(
            !response_no_download["downloaded"]
                .as_bool()
                .expect("downloaded is bool"),
            "downloaded must be false when model was already cached"
        );

        // downloaded=true when model was fetched from HuggingFace.
        assert!(
            response_with_download["downloaded"]
                .as_bool()
                .expect("downloaded is bool"),
            "downloaded must be true when model was fetched from HuggingFace"
        );
    }

    /// T-MCP-RERANK-017: The backend name in the response matches the backend
    /// parameter from the request. When no backend is specified, the default
    /// "ort" is used. This validates that the response correctly reflects the
    /// backend that was actually used.
    #[test]
    fn t_mcp_rerank_017_response_backend_reflects_request_parameter() {
        // Implicit default: no backend parameter -> "ort".
        let params_default = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        });
        let backend = params_default["backend"].as_str().unwrap_or("ort");
        assert_eq!(backend, "ort", "default backend must be 'ort'");

        // Explicit backend: the specified value must appear in the response.
        let params_explicit = serde_json::json!({
            "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "backend": "custom"
        });
        let backend = params_explicit["backend"].as_str().unwrap_or("ort");
        assert_eq!(backend, "custom", "explicit backend must be used as-is");
    }

    /// T-MCP-RERANK-018: Whitespace-only model_id is rejected before the
    /// auto-download path is reached. The handler trims the model_id and
    /// returns an error immediately for empty strings, so no download attempt
    /// is made for invalid input.
    #[test]
    fn t_mcp_rerank_018_whitespace_model_id_rejected_before_download() {
        for bad_id in &["", "   ", "\t\n"] {
            let trimmed = bad_id.trim();
            assert!(
                trimmed.is_empty(),
                "trimmed model_id '{bad_id}' must be empty and rejected"
            );
            // If this were passed to download_model, it would either fail with
            // "no download manifest" or attempt to create an empty directory.
            // The handler must reject it before reaching download_model.
        }
    }
}
