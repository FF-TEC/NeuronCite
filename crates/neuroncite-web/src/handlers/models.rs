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

//! Model catalog and management handlers for the web frontend.
//!
//! Provides the embedding model catalog, reranker catalog, download, and
//! activate endpoints. The catalog data comes from the static model registry
//! in neuroncite-embed, enriched with cached/active status from the filesystem
//! and the running worker handle. Model downloads delegate to the HuggingFace
//! cache layer in neuroncite-embed. Model activation requires a server restart
//! because the WorkerHandle captures the loaded model at construction time.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::WebState;

/// Embedding model entry in the catalog response. Contains display properties
/// from the static model registry plus runtime status (cached, active).
#[derive(Serialize)]
pub struct EmbedModelEntry {
    /// HuggingFace model identifier.
    pub model_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Output embedding vector dimensionality.
    pub vector_dimension: usize,
    /// Maximum input token count.
    pub max_seq_len: usize,
    /// Quality descriptor (e.g., "Good", "Very good").
    pub quality_rating: String,
    /// Language coverage (e.g., "EN", "Multilingual").
    pub language_scope: String,
    /// Cross-lingual retrieval support descriptor.
    pub de_en_retrieval: String,
    /// GPU need descriptor (e.g., "Optional", "Recommended").
    pub gpu_recommendation: String,
    /// Memory requirement descriptor.
    pub ram_requirement: String,
    /// Recommended usage description.
    pub typical_use_case: String,
    /// Approximate ONNX model file size in megabytes.
    pub model_size_mb: u32,
    /// Whether the model files are cached locally in the HuggingFace cache.
    pub cached: bool,
    /// Whether this model is the currently active embedding model.
    pub active: bool,
}

/// Reranker model entry in the catalog response. Carries the same display
/// columns as embedding models (language_scope, gpu_recommendation) so the
/// frontend can render both catalogs with identical table structures.
#[derive(Serialize)]
pub struct RerankModelEntry {
    /// Model identifier.
    pub model_id: String,
    /// Display name.
    pub display_name: String,
    /// Quality descriptor.
    pub quality_rating: String,
    /// Number of transformer layers.
    pub layer_count: u32,
    /// Parameter count in millions.
    pub param_count_m: u32,
    /// Language coverage (e.g., "EN", "Multilingual"). MS MARCO models are
    /// English-only; BGE-reranker-v2-m3 supports multiple languages.
    pub language_scope: String,
    /// Model file size in MB.
    pub model_size_mb: u32,
    /// GPU recommendation level matching the embedding model column.
    pub gpu_recommendation: String,
    /// Approximate runtime RAM in a human-readable range string.
    pub ram_requirement: String,
    /// Whether the model files are cached locally.
    pub cached: bool,
    /// Whether this reranker is currently loaded.
    pub loaded: bool,
}

/// Response body for GET /api/v1/web/models/catalog.
#[derive(Serialize)]
pub struct ModelCatalogResponse {
    /// GPU device name from the embedding backend.
    pub gpu_name: String,
    /// Whether CUDA/GPU acceleration is available.
    pub cuda_available: bool,
    /// Embedding models in the catalog.
    pub embedding_models: Vec<EmbedModelEntry>,
    /// Reranker models in the catalog.
    pub reranker_models: Vec<RerankModelEntry>,
}

/// Returns the full model catalog with cached/active status for each model.
/// The embedding model list is populated from the static registry in
/// neuroncite-embed. Cached status is determined by checking whether the
/// model directory exists in the local HuggingFace cache via
/// `neuroncite_embed::is_cached`.
pub async fn model_catalog(State(state): State<Arc<WebState>>) -> Json<ModelCatalogResponse> {
    let backends = neuroncite_embed::list_available_backends();
    let cuda_available = backends.first().map(|b| b.gpu_supported).unwrap_or(false);
    // Detect the actual GPU device name via nvidia-smi (e.g., "NVIDIA GeForce
    // RTX 4090"). Falls back to "CPU" when no NVIDIA GPU/driver is present.
    let gpu_name = neuroncite_embed::detect_gpu_device_name().unwrap_or_else(|| {
        if cuda_available {
            "GPU (unknown)".to_string()
        } else {
            "CPU".to_string()
        }
    });

    // Build embedding model entries from the static catalog, checking the
    // HuggingFace cache directory for each model to determine cached status.
    // `supported_model_configs()` returns a `&'static [EmbeddingModelConfig]`,
    // so the iterator yields shared references. String fields are cloned into
    // the owned `EmbedModelEntry` struct for the JSON response.
    #[cfg(feature = "backend-ort")]
    let embedding_models: Vec<EmbedModelEntry> = {
        let active_model = state.app_state.worker_handle.loaded_model_id();
        neuroncite_embed::supported_model_configs()
            .iter()
            .map(|cfg| {
                let is_active = cfg.model_id == *active_model;
                let is_cached = neuroncite_embed::is_fully_cached(&cfg.model_id, "main");
                EmbedModelEntry {
                    model_id: cfg.model_id.clone(),
                    display_name: cfg.display_name.clone(),
                    vector_dimension: cfg.vector_dimension,
                    max_seq_len: cfg.max_seq_len,
                    quality_rating: cfg.quality_rating.clone(),
                    language_scope: cfg.language_scope.clone(),
                    de_en_retrieval: cfg.de_en_retrieval.clone(),
                    gpu_recommendation: cfg.gpu_recommendation.clone(),
                    ram_requirement: cfg.ram_requirement.clone(),
                    typical_use_case: cfg.typical_use_case.clone(),
                    model_size_mb: cfg.model_size_mb,
                    cached: is_cached || is_active,
                    active: is_active,
                }
            })
            .collect()
    };

    #[cfg(not(feature = "backend-ort"))]
    let embedding_models: Vec<EmbedModelEntry> = Vec::new();

    // Static reranker catalog listing all cross-encoder models supported by the
    // ONNX reranker backend. The loaded_reranker_id from the worker handle
    // identifies which model is currently active in the reranker slot.
    let loaded_reranker = state.app_state.worker_handle.loaded_reranker_id();

    // Static reranker catalog. Each entry is constructed directly as a
    // RerankModelEntry with placeholder cached/loaded fields. The actual
    // runtime status is patched below from the HuggingFace cache and the
    // active reranker slot in the worker handle.
    let reranker_models: Vec<RerankModelEntry> = [
        (
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "MS MARCO MiniLM L6",
            "Good",
            6,
            22,
            "EN",
            133,
            "Optional",
            "<0.5 GB",
        ),
        (
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "MS MARCO MiniLM L12",
            "Good",
            12,
            33,
            "EN",
            200,
            "Optional",
            "<0.5 GB",
        ),
        (
            "cross-encoder/ms-marco-electra-base",
            "MS MARCO Electra Base",
            "Very good",
            12,
            110,
            "EN",
            440,
            "Recommended",
            "<1 GB",
        ),
        (
            "BAAI/bge-reranker-v2-m3",
            "BGE Reranker v2 M3",
            "Very good",
            12,
            568,
            "Multilingual",
            2300,
            "Recommended",
            "4-6 GB",
        ),
    ]
    .into_iter()
    .map(
        |(id, name, quality, layers, params, lang, size, gpu, ram)| {
            let is_cached = neuroncite_embed::is_fully_cached(id, "main");
            let is_loaded = !loaded_reranker.is_empty() && loaded_reranker.as_str() == id;
            RerankModelEntry {
                model_id: id.into(),
                display_name: name.into(),
                quality_rating: quality.into(),
                layer_count: layers,
                param_count_m: params,
                language_scope: lang.into(),
                model_size_mb: size,
                gpu_recommendation: gpu.into(),
                ram_requirement: ram.into(),
                cached: is_cached || is_loaded,
                loaded: is_loaded,
            }
        },
    )
    .collect();

    Json(ModelCatalogResponse {
        gpu_name,
        cuda_available,
        embedding_models,
        reranker_models,
    })
}

/// Request body for POST /api/v1/web/models/download.
#[derive(Deserialize)]
pub struct ModelDownloadRequest {
    /// HuggingFace model identifier to download.
    pub model_id: String,
}

/// Downloads model files (embedding or reranker) from HuggingFace to the
/// local cache. Validates the model_id against the MODEL_FILES download
/// manifest in neuroncite-embed, which covers both embedding and reranker
/// models. The download runs on a blocking thread via
/// `tokio::task::spawn_blocking` to avoid blocking the async runtime.
/// Returns 202 Accepted on success, 400 for unknown models, 500 on failure.
pub async fn download_model(
    State(state): State<Arc<WebState>>,
    Json(req): Json<ModelDownloadRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate the model_id against the MODEL_FILES download manifest.
    // This covers both embedding models and reranker models. The previous
    // validation only checked supported_model_configs() (embedding models),
    // which caused reranker downloads to fail with "Unknown model".
    let known = neuroncite_embed::model_expected_files(&req.model_id).is_some();
    if !known {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Unknown model: '{}'. Check GET /web/models/catalog for supported models.", req.model_id)
            })),
        );
    }

    // Check if all required files are present to avoid redundant downloads.
    // Uses is_fully_cached() to verify every file in the manifest exists,
    // not just that the directory has at least one file.
    if neuroncite_embed::is_fully_cached(&req.model_id, "main") {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "already_cached",
                "model_id": req.model_id
            })),
        );
    }

    let model_id = req.model_id.clone();
    let model_tx = state.model_tx.clone();

    // Notify SSE subscribers that a download is starting
    let _ = model_tx.send(
        serde_json::json!({
            "event": "download_started",
            "model_id": &model_id
        })
        .to_string(),
    );

    // Run the download on a blocking thread because it performs synchronous
    // HTTP requests and filesystem I/O. The 1-hour timeout prevents a hung
    // HTTP connection from blocking the slot indefinitely. When the timeout
    // fires, the JoinHandle is dropped but the underlying blocking task
    // continues until completion or its own I/O error — this is acceptable
    // because Tokio's blocking threads cannot be cancelled externally.
    let join_result =
        tokio::task::spawn_blocking(move || neuroncite_embed::download_model(&model_id, "main"));
    let result = match tokio::time::timeout(std::time::Duration::from_secs(3600), join_result).await
    {
        Ok(r) => r,
        Err(_elapsed) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "download_error",
                    "model_id": &req.model_id,
                    "error": "download timed out after 1 hour"
                })
                .to_string(),
            );
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": "model download timed out"
                })),
            );
        }
    };

    match result {
        Ok(Ok(_path)) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "model_downloaded",
                    "model_id": &req.model_id
                })
                .to_string(),
            );
            (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({
                    "status": "downloaded",
                    "model_id": req.model_id
                })),
            )
        }
        Ok(Err(e)) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "download_failed",
                    "model_id": &req.model_id,
                    "error": e.to_string()
                })
                .to_string(),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Model download failed: {e}")
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Download task panicked: {e}")
            })),
        ),
    }
}

/// Request body for POST /api/v1/web/models/activate.
#[derive(Deserialize)]
pub struct ModelActivateRequest {
    /// HuggingFace model identifier to activate.
    pub model_id: String,
}

/// Activates a different embedding model at runtime by hot-swapping the
/// backend in the GPU worker. Creates a new backend instance, loads the
/// requested model on a blocking thread, then atomically replaces the
/// active backend via `WorkerHandle::swap_backend()`. Also updates the
/// `AppState.vector_dimension` atomic so that subsequent session creation
/// and search validation use the correct dimensionality.
///
/// Returns 200 with status "activated" on success, 200 with status
/// "already_active" when the model is already loaded, 400 when the model
/// is not cached locally, and 500 on load failure.
pub async fn activate_model(
    State(state): State<Arc<WebState>>,
    Json(req): Json<ModelActivateRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let current_model = state.app_state.worker_handle.loaded_model_id();

    // Already active: no-op
    if req.model_id == *current_model {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "already_active",
                "model_id": req.model_id
            })),
        );
    }

    // Verify that all required model files exist in the local cache before
    // attempting load. Uses diagnose_model() for file-level validation,
    // producing a specific error listing missing files instead of the
    // cryptic "tokenizer file not found" from deep inside the ORT backend.
    let diagnosis = neuroncite_embed::diagnose_model(&req.model_id, "main");
    if !diagnosis.directory_exists {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Model '{}' is not cached locally. Download it first via POST /web/models/download.",
                    req.model_id
                )
            })),
        );
    }
    if !diagnosis.files_missing.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Model '{}' has missing files: {}. Download or repair the model first.",
                    req.model_id,
                    diagnosis.files_missing.join(", ")
                ),
                "missing_files": diagnosis.files_missing,
            })),
        );
    }

    // Notify SSE subscribers that loading has started
    let _ = state.model_tx.send(
        serde_json::json!({
            "event": "activation_loading",
            "model_id": &req.model_id,
            "current_model": *current_model
        })
        .to_string(),
    );

    // Load the model on a blocking thread because `create_backend` and
    // `load_model` perform synchronous file I/O and GPU initialization.
    let model_id = req.model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let backends = neuroncite_embed::list_available_backends();
        let backend_name = backends.first().map(|b| b.name.as_str()).unwrap_or("ort");
        let mut backend = neuroncite_embed::create_backend(backend_name)
            .map_err(|e| format!("backend creation failed: {e}"))?;
        backend
            .load_model(&model_id)
            .map_err(|e| format!("model load failed: {e}"))?;
        Ok::<Box<dyn neuroncite_core::EmbeddingBackend>, String>(backend)
    })
    .await;

    match result {
        Ok(Ok(backend)) => {
            let new_backend: Arc<dyn neuroncite_core::EmbeddingBackend> = Arc::from(backend);
            let new_dim = new_backend.vector_dimension();
            let new_model_id = new_backend.loaded_model_id();

            // Hot-swap the backend in the worker. In-flight requests complete
            // with the previous backend; subsequent requests use the new one.
            state.app_state.worker_handle.swap_backend(new_backend);
            state
                .app_state
                .index
                .vector_dimension
                .store(new_dim, std::sync::atomic::Ordering::Release);

            // Notify SSE subscribers about the completed activation
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "model_switched",
                    "model_id": &new_model_id,
                    "vector_dimension": new_dim
                })
                .to_string(),
            );

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "activated",
                    "model_id": new_model_id,
                    "vector_dimension": new_dim
                })),
            )
        }
        Ok(Err(e)) => {
            // Log the full error server-side; the client receives only a generic
            // message to prevent internal file paths and model details from leaking.
            tracing::error!(model_id = %req.model_id, error = %e, "model activation failed");
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "activation_failed",
                    "model_id": &req.model_id
                })
                .to_string(),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "model activation failed — check server logs for details"
                })),
            )
        }
        Err(e) => {
            // spawn_blocking panic: same sanitization applies.
            tracing::error!(model_id = %req.model_id, error = %e, "activation task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "model activation failed — check server logs for details"
                })),
            )
        }
    }
}

/// Request body for POST /api/v1/web/models/load-reranker.
#[derive(Deserialize)]
pub struct LoadRerankerRequest {
    /// HuggingFace model identifier for the cross-encoder reranker to load.
    pub model_id: String,
}

/// Loads a cross-encoder reranker model and hot-swaps it into the GPU worker.
/// Downloads the model from HuggingFace if not already cached. The model must
/// be present in the MODEL_FILES download manifest. After loading, the reranker
/// becomes available for search reranking via the `rerank` parameter.
///
/// Returns 200 with status "loaded" on success, 200 with status "already_loaded"
/// if the model is already the active reranker, 400 for unknown models, and
/// 500 on download or load failure.
pub async fn load_reranker(
    State(state): State<Arc<WebState>>,
    Json(req): Json<LoadRerankerRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let model_id = req.model_id.trim().to_string();
    if model_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "model_id must not be empty"
            })),
        );
    }

    // Reject unknown model IDs before attempting any work
    if neuroncite_embed::model_expected_files(&model_id).is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Unknown reranker model: '{}'. Not in the download manifest.", model_id)
            })),
        );
    }

    // Skip reload if the same reranker is already active
    let current_reranker = state.app_state.worker_handle.loaded_reranker_id();
    if *current_reranker == *model_id {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "already_loaded",
                "model_id": model_id
            })),
        );
    }

    // Notify SSE subscribers that loading has started
    let _ = state.model_tx.send(
        serde_json::json!({
            "event": "reranker_loading",
            "model_id": &model_id
        })
        .to_string(),
    );

    // Download (if not cached) and load the reranker on a blocking thread.
    // Model loading involves filesystem I/O and ONNX session creation.
    // Corrupt cached files are detected via SHA-256 checksum verification.
    // When corruption is found, the cache directory is purged and a fresh
    // download is triggered before loading.
    let model_id_clone = model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        if neuroncite_embed::is_fully_cached(&model_id_clone, "main") {
            // Verify integrity of cached files against the checksums manifest.
            // If any file is corrupt (incomplete download, disk error, etc.),
            // delete the entire cache directory and re-download.
            match neuroncite_embed::verify_cached_model(&model_id_clone, "main") {
                Ok(true) => {}
                Ok(false) => {
                    tracing::warn!(
                        model_id = %model_id_clone,
                        "cached reranker model is corrupt, purging and re-downloading"
                    );
                    neuroncite_embed::purge_cached_model(&model_id_clone, "main")
                        .map_err(|e| format!("failed to purge corrupt cache: {e}"))?;
                    neuroncite_embed::download_model(&model_id_clone, "main")
                        .map_err(|e| format!("reranker re-download failed: {e}"))?;
                }
                Err(e) => {
                    tracing::warn!(
                        model_id = %model_id_clone,
                        "checksum verification error, purging and re-downloading: {e}"
                    );
                    neuroncite_embed::purge_cached_model(&model_id_clone, "main")
                        .map_err(|e| format!("failed to purge corrupt cache: {e}"))?;
                    neuroncite_embed::download_model(&model_id_clone, "main")
                        .map_err(|e| format!("reranker re-download failed: {e}"))?;
                }
            }
        } else {
            neuroncite_embed::download_model(&model_id_clone, "main")
                .map_err(|e| format!("reranker download failed: {e}"))?;
        }

        let mut reranker = neuroncite_embed::create_reranker("ort")
            .map_err(|e| format!("reranker backend creation failed: {e}"))?;
        reranker
            .load_model(&model_id_clone)
            .map_err(|e| format!("reranker model load failed: {e}"))?;

        let arc: Arc<dyn neuroncite_core::Reranker> = Arc::from(reranker);
        Ok::<Arc<dyn neuroncite_core::Reranker>, String>(arc)
    })
    .await;

    match result {
        Ok(Ok(reranker_arc)) => {
            // Hot-swap the reranker into the GPU worker
            state
                .app_state
                .worker_handle
                .swap_reranker(Some(reranker_arc), &model_id);

            // Notify SSE subscribers about the completed load
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "reranker_loaded",
                    "model_id": &model_id
                })
                .to_string(),
            );

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "loaded",
                    "model_id": model_id,
                    "reranker_available": true
                })),
            )
        }
        Ok(Err(e)) => {
            let _ = state.model_tx.send(
                serde_json::json!({
                    "event": "reranker_load_failed",
                    "model_id": &model_id,
                    "error": &e
                })
                .to_string(),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Reranker load failed: {e}")
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Reranker load task panicked: {e}")
            })),
        ),
    }
}
