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

//! Ollama proxy handlers for the web frontend.
//!
//! These endpoints allow the browser-based frontend to interact with a local
//! Ollama instance without CORS issues. The frontend cannot call Ollama
//! directly because Ollama typically listens on localhost:11434 which is a
//! different origin from the NeuronCite web server.
//!
//! Endpoints provided:
//!
//! - `GET  /api/v1/web/ollama/status`  -- Checks Ollama reachability
//! - `GET  /api/v1/web/ollama/models`  -- Lists all installed models (GET /api/tags)
//! - `GET  /api/v1/web/ollama/running` -- Lists models loaded in GPU/CPU RAM (GET /api/ps)
//! - `GET  /api/v1/web/ollama/catalog` -- Returns a curated list of popular models
//! - `POST /api/v1/web/ollama/pull`    -- Downloads a model (async, progress via SSE)
//! - `POST /api/v1/web/ollama/delete`  -- Removes an installed model
//! - `POST /api/v1/web/ollama/show`    -- Returns detailed metadata for one model

use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};
use url::Url;

use neuroncite_llm::ollama::{OllamaBackend, OllamaModel, OllamaRunningModel};
use neuroncite_llm::types::LlmConfig;

use crate::WebState;

// ---------------------------------------------------------------------------
// Curated model catalog
// ---------------------------------------------------------------------------

/// A single entry in the curated Ollama model catalog. These entries represent
/// popular models from the Ollama registry that users can download directly
/// from the Models panel. The catalog is static because Ollama does not provide
/// a registry search API.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaCatalogEntry {
    /// Ollama model tag for pulling (e.g., "llama3.2:3b").
    pub name: &'static str,
    /// Human-readable display name shown in the UI table.
    pub display_name: &'static str,
    /// Model architecture family (e.g., "llama", "qwen2").
    pub family: &'static str,
    /// Human-readable parameter count (e.g., "3B", "7B").
    pub parameter_size: &'static str,
    /// Approximate disk size after download in megabytes.
    pub size_mb: u32,
    /// Short description of the model's strengths and use case.
    pub description: &'static str,
}

/// Curated list of popular Ollama models covering the most commonly used
/// families and sizes. Ordered by family, then by parameter count ascending.
/// Sizes are approximate disk footprints based on default quantization.
static OLLAMA_CATALOG: &[OllamaCatalogEntry] = &[
    OllamaCatalogEntry {
        name: "llama3.2:1b",
        display_name: "Llama 3.2 1B",
        family: "llama",
        parameter_size: "1B",
        size_mb: 1300,
        description: "Lightweight, fast inference, limited reasoning",
    },
    OllamaCatalogEntry {
        name: "llama3.2:3b",
        display_name: "Llama 3.2 3B",
        family: "llama",
        parameter_size: "3B",
        size_mb: 2000,
        description: "General-purpose, good speed-quality balance",
    },
    OllamaCatalogEntry {
        name: "llama3.1:8b",
        display_name: "Llama 3.1 8B",
        family: "llama",
        parameter_size: "8B",
        size_mb: 4700,
        description: "Strong reasoning, multilingual support",
    },
    OllamaCatalogEntry {
        name: "qwen2.5:3b",
        display_name: "Qwen 2.5 3B",
        family: "qwen2",
        parameter_size: "3B",
        size_mb: 1900,
        description: "Multilingual, strong on structured output",
    },
    OllamaCatalogEntry {
        name: "qwen2.5:7b",
        display_name: "Qwen 2.5 7B",
        family: "qwen2",
        parameter_size: "7B",
        size_mb: 4400,
        description: "Multilingual, strong JSON mode, citation-ready",
    },
    OllamaCatalogEntry {
        name: "qwen2.5:14b",
        display_name: "Qwen 2.5 14B",
        family: "qwen2",
        parameter_size: "14B",
        size_mb: 9000,
        description: "High quality, requires 12+ GB VRAM",
    },
    OllamaCatalogEntry {
        name: "qwen3:8b",
        display_name: "Qwen 3 8B",
        family: "qwen3",
        parameter_size: "8B",
        size_mb: 5200,
        description: "Latest Qwen generation, strong reasoning",
    },
    OllamaCatalogEntry {
        name: "mistral:7b",
        display_name: "Mistral 7B",
        family: "mistral",
        parameter_size: "7B",
        size_mb: 4100,
        description: "Fast, efficient, strong English performance",
    },
    OllamaCatalogEntry {
        name: "mistral-small:latest",
        display_name: "Mistral Small",
        family: "mistral",
        parameter_size: "24B",
        size_mb: 14000,
        description: "High quality, requires 16+ GB VRAM",
    },
    OllamaCatalogEntry {
        name: "gemma2:2b",
        display_name: "Gemma 2 2B",
        family: "gemma2",
        parameter_size: "2B",
        size_mb: 1600,
        description: "Google, compact and efficient",
    },
    OllamaCatalogEntry {
        name: "gemma2:9b",
        display_name: "Gemma 2 9B",
        family: "gemma2",
        parameter_size: "9B",
        size_mb: 5400,
        description: "Google, strong general-purpose performance",
    },
    OllamaCatalogEntry {
        name: "phi4:14b",
        display_name: "Phi 4 14B",
        family: "phi4",
        parameter_size: "14B",
        size_mb: 9100,
        description: "Microsoft, strong reasoning and coding",
    },
    OllamaCatalogEntry {
        name: "deepseek-r1:7b",
        display_name: "DeepSeek R1 7B",
        family: "deepseek",
        parameter_size: "7B",
        size_mb: 4700,
        description: "Chain-of-thought reasoning, math-focused",
    },
    OllamaCatalogEntry {
        name: "deepseek-r1:14b",
        display_name: "DeepSeek R1 14B",
        family: "deepseek",
        parameter_size: "14B",
        size_mb: 9000,
        description: "Strong reasoning, requires 12+ GB VRAM",
    },
    OllamaCatalogEntry {
        name: "codellama:7b",
        display_name: "Code Llama 7B",
        family: "llama",
        parameter_size: "7B",
        size_mb: 3800,
        description: "Code generation and understanding",
    },
];

// ---------------------------------------------------------------------------
// Query and response types
// ---------------------------------------------------------------------------

/// Query parameter for Ollama endpoints. The `url` field specifies the base
/// URL of the Ollama server (e.g., "http://localhost:11434").
#[derive(Debug, Deserialize)]
pub struct OllamaQuery {
    /// Base URL of the Ollama server. Defaults to "http://localhost:11434"
    /// when not provided.
    pub url: Option<String>,
}

/// Request body for POST endpoints that target a specific model.
#[derive(Debug, Deserialize)]
pub struct OllamaModelRequest {
    /// Model tag to operate on (e.g., "llama3.2:3b", "qwen2.5:7b").
    pub model: String,
    /// Base URL of the Ollama server. Defaults to "http://localhost:11434"
    /// when not provided.
    #[serde(default)]
    pub url: Option<String>,
}

/// Response from the status endpoint indicating whether Ollama is reachable.
#[derive(Debug, Serialize)]
pub struct OllamaStatusResponse {
    /// Whether the Ollama server responded to a health check.
    pub connected: bool,
    /// Base URL that was checked.
    pub url: String,
}

/// Response from the models endpoint listing all installed Ollama models.
#[derive(Debug, Serialize)]
pub struct OllamaModelsResponse {
    /// List of models installed on the Ollama server.
    pub models: Vec<OllamaModel>,
    /// Base URL of the Ollama server that was queried.
    pub url: String,
}

/// Response from the running endpoint listing models loaded in RAM.
#[derive(Debug, Serialize)]
pub struct OllamaRunningResponse {
    /// Models currently loaded in GPU/CPU RAM.
    pub models: Vec<OllamaRunningModel>,
    /// Base URL of the Ollama server that was queried.
    pub url: String,
}

/// Response from the catalog endpoint listing available models for download.
#[derive(Debug, Serialize)]
pub struct OllamaCatalogResponse {
    /// Curated list of popular Ollama models.
    pub models: Vec<OllamaCatalogEntry>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Default Ollama server address used when the frontend does not provide one.
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default Ollama API port. Ollama listens on this port unless configured
/// otherwise.
const DEFAULT_OLLAMA_PORT: u16 = 11434;

/// Validates and restricts the Ollama proxy URL to prevent SSRF attacks.
///
/// This proxy exists solely to bridge the browser frontend to a local Ollama
/// instance. Connections are restricted to:
///
/// 1. **Localhost only**: 127.0.0.1, ::1, or the literal hostname "localhost".
///    Ollama runs on the same machine as NeuronCite; there is no legitimate
///    reason to proxy to remote hosts.
/// 2. **Ollama port only**: TCP port 11434 (the Ollama default). This prevents
///    the proxy from being used to scan or access other local services.
/// 3. **Ollama API path prefix only**: The URL path must start with `/api/`
///    (Ollama's API endpoint prefix). The base URL itself (no path or root `/`)
///    is also accepted because `OllamaBackend` appends the API path internally.
/// 4. **http/https schemes only**: Rejects `file://`, `javascript:`, `data:`,
///    and other non-HTTP schemes.
///
/// Returns the validated URL string on success, or an HTTP 400 error tuple
/// on validation failure.
fn validate_url(raw: &Option<String>) -> Result<String, (StatusCode, Json<serde_json::Value>)> {
    let url_str = raw.as_deref().unwrap_or(DEFAULT_OLLAMA_URL);

    let parsed = Url::parse(url_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("invalid URL: {e}") })),
        )
    })?;

    // Only http and https schemes are allowed. Reject javascript:, file:,
    // data:, ftp:, and any other scheme.
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("URL scheme '{scheme}' is not allowed; use http or https"),
                })),
            ));
        }
    }

    // Restrict to localhost addresses only. Ollama runs on the same machine
    // as NeuronCite; proxying to arbitrary remote hosts is an SSRF vector.
    let host = parsed.host_str().unwrap_or("");
    let is_localhost =
        host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]";

    if !is_localhost {
        warn!(
            url = %url_str,
            host = %host,
            "SSRF: blocked Ollama proxy request to non-localhost host"
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Ollama proxy only allows connections to localhost (127.0.0.1, ::1)"
            })),
        ));
    }

    // Restrict to the Ollama port (default 11434). This prevents the proxy
    // from being used to probe or access other services on localhost.
    let port = parsed
        .port()
        .unwrap_or(if parsed.scheme() == "https" { 443 } else { 80 });
    if port != DEFAULT_OLLAMA_PORT {
        warn!(
            url = %url_str,
            port = port,
            "SSRF: blocked Ollama proxy request to non-Ollama port"
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Ollama proxy only allows port {DEFAULT_OLLAMA_PORT}; got port {port}"
                )
            })),
        ));
    }

    // Restrict the URL path to Ollama API endpoints. The base URL (empty path
    // or just "/") is accepted because OllamaBackend appends the API path
    // (e.g., "/api/tags", "/api/ps") when making requests. Explicit paths
    // must start with "/api/" to prevent proxying to non-Ollama endpoints.
    let path = parsed.path();
    if !path.is_empty() && path != "/" && !path.starts_with("/api/") {
        warn!(
            url = %url_str,
            path = %path,
            "SSRF: blocked Ollama proxy request to non-API path"
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Ollama proxy only allows the /api/ path prefix"
            })),
        ));
    }

    Ok(url_str.to_string())
}

/// Constructs an `OllamaBackend` from a pre-validated URL string. The model
/// and generation parameters are irrelevant for management operations
/// (list, pull, delete, show) so they are set to placeholder values.
///
/// Returns an HTTP 500 error tuple if the reqwest HTTP client cannot be
/// constructed (e.g., TLS backend initialization failure).
fn backend_from_validated_url(
    url: &str,
) -> Result<OllamaBackend, (StatusCode, Json<serde_json::Value>)> {
    let config = LlmConfig {
        base_url: url.to_string(),
        model: String::new(),
        temperature: 0.0,
        max_tokens: 0,
        json_mode: false,
    };
    OllamaBackend::new(config).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to create Ollama HTTP client: {e}"),
            })),
        )
    })
}

// ---------------------------------------------------------------------------
// Handler functions
// ---------------------------------------------------------------------------

/// GET /api/v1/web/ollama/status?url=http://localhost:11434
///
/// Checks if the Ollama server is reachable by sending a lightweight request
/// to the /api/tags endpoint. Returns 200 with `connected: true` if the server
/// responds, or 200 with `connected: false` if the connection fails.
pub async fn ollama_status(
    Query(query): Query<OllamaQuery>,
) -> Result<Json<OllamaStatusResponse>, (StatusCode, Json<serde_json::Value>)> {
    let url = validate_url(&query.url)?;
    let backend = backend_from_validated_url(&url)?;
    let connected = backend.is_reachable().await;
    Ok(Json(OllamaStatusResponse { connected, url }))
}

/// GET /api/v1/web/ollama/models?url=http://localhost:11434
///
/// Lists all models installed on the Ollama server by proxying the
/// GET /api/tags endpoint. Returns the model name, disk size, parameter
/// count, family, and quantization level for each installed model.
///
/// Returns 502 Bad Gateway if the Ollama server is unreachable or returns
/// an error response.
pub async fn list_ollama_models(
    Query(query): Query<OllamaQuery>,
) -> Result<Json<OllamaModelsResponse>, impl IntoResponse> {
    let url = validate_url(&query.url)?;
    let backend = backend_from_validated_url(&url)?;

    match backend.list_models().await {
        Ok(models) => Ok(Json(OllamaModelsResponse { models, url })),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "error": format!("failed to reach Ollama at {url}: {e}"),
            })),
        )),
    }
}

/// GET /api/v1/web/ollama/running?url=http://localhost:11434
///
/// Lists models currently loaded into GPU/CPU RAM on the Ollama server
/// by proxying GET /api/ps. Returns memory footprint, VRAM consumption,
/// and auto-unload expiration time for each loaded model.
///
/// Returns 502 Bad Gateway if the Ollama server is unreachable.
pub async fn list_running_models(
    Query(query): Query<OllamaQuery>,
) -> Result<Json<OllamaRunningResponse>, impl IntoResponse> {
    let url = validate_url(&query.url)?;
    let backend = backend_from_validated_url(&url)?;

    match backend.running_models().await {
        Ok(models) => Ok(Json(OllamaRunningResponse { models, url })),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "error": format!("failed to reach Ollama at {url}: {e}"),
            })),
        )),
    }
}

/// GET /api/v1/web/ollama/catalog
///
/// Returns the curated list of popular Ollama models. This static catalog
/// provides download suggestions since Ollama does not offer a registry
/// search API. The frontend merges this with the live installed models
/// list to show status badges.
pub async fn ollama_catalog() -> Json<OllamaCatalogResponse> {
    Json(OllamaCatalogResponse {
        models: OLLAMA_CATALOG.to_vec(),
    })
}

/// POST /api/v1/web/ollama/pull
///
/// Initiates downloading a model from the Ollama registry. The pull operation
/// runs asynchronously in a background tokio task. Progress updates are
/// broadcast via the `model_tx` SSE channel as `ollama_pull_progress` events.
/// On completion, an `ollama_pull_complete` event is sent.
///
/// Returns 202 Accepted immediately after starting the background task.
/// Returns 400 Bad Request if the model name is empty.
/// Returns 502 Bad Gateway if Ollama is unreachable.
pub async fn pull_model(
    State(web_state): State<Arc<WebState>>,
    Json(req): Json<OllamaModelRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), impl IntoResponse> {
    let model = req.model.trim().to_string();
    if model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "model name must not be empty" })),
        ));
    }

    let url = validate_url(&req.url)?;

    // Verify Ollama is reachable before spawning the background task.
    let backend = backend_from_validated_url(&url)?;
    if !backend.is_reachable().await {
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "error": format!("Ollama is not reachable at {url}"),
            })),
        ));
    }

    // Spawn the pull operation on a background task. Progress updates are
    // forwarded to the SSE model channel so the frontend receives them
    // in real-time via the EventSource connection.
    let model_for_task = model.clone();
    let model_tx = web_state.model_tx.clone();
    let url_for_task = url.clone();

    tokio::spawn(async move {
        let backend = match backend_from_validated_url(&url_for_task) {
            Ok(b) => b,
            Err((_status, json)) => {
                error!(
                    model = %model_for_task,
                    "failed to create Ollama HTTP client in pull background task"
                );
                let event = serde_json::json!({
                    "event": "ollama_pull_error",
                    "model": &model_for_task,
                    "error": json.0["error"],
                });
                let _ = model_tx.send(event.to_string());
                return;
            }
        };
        let model_name = model_for_task.clone();
        let tx = model_tx.clone();

        let result = backend
            .pull_model(&model_for_task, |progress| {
                // Broadcast each progress update as an SSE event. Receiver
                // lag is acceptable -- the frontend shows the latest state.
                let event = serde_json::json!({
                    "event": "ollama_pull_progress",
                    "model": &model_name,
                    "status": &progress.status,
                    "total": progress.total,
                    "completed": progress.completed,
                });
                let _ = tx.send(event.to_string());
            })
            .await;

        match result {
            Ok(()) => {
                info!(model = %model_name, "Ollama model pull completed");
                let event = serde_json::json!({
                    "event": "ollama_pull_complete",
                    "model": &model_name,
                });
                let _ = model_tx.send(event.to_string());
            }
            Err(e) => {
                error!(model = %model_name, error = %e, "Ollama model pull failed");
                let event = serde_json::json!({
                    "event": "ollama_pull_error",
                    "model": &model_name,
                    "error": e.to_string(),
                });
                let _ = model_tx.send(event.to_string());
            }
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "status": "pulling",
            "model": model,
        })),
    ))
}

/// POST /api/v1/web/ollama/delete
///
/// Removes an installed model from the Ollama server. The model files are
/// deleted from local storage and the model can no longer be used until
/// re-pulled.
///
/// Returns 400 Bad Request if the model name is empty.
/// Returns 502 Bad Gateway if Ollama is unreachable.
pub async fn delete_model(
    Json(req): Json<OllamaModelRequest>,
) -> Result<Json<serde_json::Value>, impl IntoResponse> {
    let model = req.model.trim().to_string();
    if model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "model name must not be empty" })),
        ));
    }

    let url = validate_url(&req.url)?;
    let backend = backend_from_validated_url(&url)?;

    match backend.delete_model(&model).await {
        Ok(()) => {
            info!(model = %model, "Ollama model deleted");
            Ok(Json(serde_json::json!({
                "status": "deleted",
                "model": model,
            })))
        }
        Err(e) => {
            let status = if matches!(e, neuroncite_llm::LlmError::ModelNotFound { .. }) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            Err((
                status,
                Json(serde_json::json!({
                    "error": format!("failed to delete model '{model}' at {url}: {e}"),
                })),
            ))
        }
    }
}

/// POST /api/v1/web/ollama/show
///
/// Retrieves detailed metadata for a single model including its family,
/// parameter size, quantization level, prompt template, and license text.
///
/// Returns 400 Bad Request if the model name is empty.
/// Returns 502 Bad Gateway if Ollama is unreachable.
/// Returns 404 if the model is not installed.
pub async fn show_model(
    Json(req): Json<OllamaModelRequest>,
) -> Result<Json<serde_json::Value>, impl IntoResponse> {
    let model = req.model.trim().to_string();
    if model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "model name must not be empty" })),
        ));
    }

    let url = validate_url(&req.url)?;
    let backend = backend_from_validated_url(&url)?;

    match backend.show_model(&model).await {
        Ok(detail) => Ok(Json(serde_json::json!({
            "model": detail.model,
            "family": detail.family,
            "parameter_size": detail.parameter_size,
            "quantization_level": detail.quantization_level,
            "template": detail.template,
            "license": detail.license,
            "url": url,
        }))),
        Err(e) => {
            let status = if matches!(e, neuroncite_llm::LlmError::ModelNotFound { .. }) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            Err((
                status,
                Json(serde_json::json!({
                    "error": format!("failed to show model '{model}' at {url}: {e}"),
                })),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-SSRF-001: Default URL (None) resolves to localhost and passes validation.
    #[test]
    fn t_ssrf_001_default_url_passes() {
        let result = validate_url(&None);
        assert!(
            result.is_ok(),
            "default URL (localhost) must pass validation"
        );
        assert_eq!(result.unwrap(), DEFAULT_OLLAMA_URL);
    }

    /// T-SSRF-002: Explicit localhost URL with the Ollama port passes validation.
    #[test]
    fn t_ssrf_002_localhost_ollama_port() {
        let url = Some("http://localhost:11434".to_string());
        let result = validate_url(&url);
        assert!(
            result.is_ok(),
            "localhost with Ollama port must pass validation"
        );
    }

    /// T-SSRF-003: 127.0.0.1 URL with Ollama port passes validation.
    #[test]
    fn t_ssrf_003_ipv4_loopback() {
        let url = Some("http://127.0.0.1:11434".to_string());
        let result = validate_url(&url);
        assert!(result.is_ok(), "127.0.0.1:11434 must pass validation");
    }

    /// T-SSRF-004: javascript: scheme is rejected.
    #[test]
    fn t_ssrf_004_javascript_scheme_rejected() {
        let url = Some("javascript:alert(1)".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "javascript: scheme must be rejected");
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// T-SSRF-005: file:// scheme is rejected.
    #[test]
    fn t_ssrf_005_file_scheme_rejected() {
        let url = Some("file:///etc/passwd".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "file:// scheme must be rejected");
    }

    /// T-SSRF-006: data: scheme is rejected.
    #[test]
    fn t_ssrf_006_data_scheme_rejected() {
        let url = Some("data:text/html,<script>alert(1)</script>".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "data: scheme must be rejected");
    }

    /// T-SSRF-007: ftp: scheme is rejected.
    #[test]
    fn t_ssrf_007_ftp_scheme_rejected() {
        let url = Some("ftp://evil.example.com/model".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "ftp: scheme must be rejected");
    }

    /// T-SSRF-008: Remote host is rejected. The proxy only allows localhost
    /// connections to prevent SSRF to arbitrary external or internal services.
    #[test]
    fn t_ssrf_008_remote_host_rejected() {
        let url = Some("http://169.254.169.254/latest/meta-data/".to_string());
        let result = validate_url(&url);
        assert!(
            result.is_err(),
            "remote host (cloud metadata IP) must be rejected"
        );
    }

    /// T-SSRF-009: https scheme with localhost and Ollama port passes validation.
    #[test]
    fn t_ssrf_009_https_localhost_allowed() {
        let url = Some("https://localhost:11434".to_string());
        let result = validate_url(&url);
        assert!(result.is_ok(), "https://localhost:11434 must be allowed");
    }

    /// T-SSRF-010: Completely invalid URL is rejected.
    #[test]
    fn t_ssrf_010_invalid_url_rejected() {
        let url = Some("not a url at all".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "unparseable URL must be rejected");
    }

    /// T-SSRF-011: Empty string URL is rejected (empty is not a valid URL).
    #[test]
    fn t_ssrf_011_empty_url_rejected() {
        let url = Some(String::new());
        let result = validate_url(&url);
        assert!(result.is_err(), "empty URL string must be rejected");
    }

    /// T-SSRF-012: LAN address (192.168.x.x) is rejected. The Ollama proxy
    /// only allows localhost connections.
    #[test]
    fn t_ssrf_012_lan_address_rejected() {
        let url = Some("http://192.168.1.100:11434".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "LAN address must be rejected");
    }

    /// T-SSRF-013: Localhost with a non-Ollama port is rejected to prevent
    /// scanning or accessing other local services.
    #[test]
    fn t_ssrf_013_non_ollama_port_rejected() {
        let url = Some("http://localhost:8080".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "non-Ollama port must be rejected");
    }

    /// T-SSRF-014: Localhost with a non-API path is rejected to prevent
    /// proxying to non-Ollama endpoints.
    #[test]
    fn t_ssrf_014_non_api_path_rejected() {
        let url = Some("http://localhost:11434/some/other/path".to_string());
        let result = validate_url(&url);
        assert!(result.is_err(), "non-API path must be rejected");
    }

    /// T-SSRF-015: Localhost with /api/ path prefix passes validation.
    #[test]
    fn t_ssrf_015_api_path_accepted() {
        let url = Some("http://localhost:11434/api/tags".to_string());
        let result = validate_url(&url);
        assert!(result.is_ok(), "/api/tags path must be accepted");
    }

    /// T-SSRF-016: IPv6 loopback [::1] with Ollama port passes validation.
    #[test]
    fn t_ssrf_016_ipv6_loopback_accepted() {
        let url = Some("http://[::1]:11434".to_string());
        let result = validate_url(&url);
        assert!(result.is_ok(), "IPv6 loopback [::1]:11434 must be accepted");
    }
}
