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

// Ollama HTTP backend for the LlmBackend trait.
//
// Communicates with a running Ollama instance via its REST API at
// localhost:11434 (configurable). Supports both blocking chat completion
// (stream: false) and streaming completion (stream: true, NDJSON response).
//
// The Ollama API endpoints used:
// - POST /api/chat:   Chat completion with message history
// - GET  /api/tags:   List installed models with their sizes
// - GET  /api/ps:     List models currently loaded in GPU/CPU RAM
// - POST /api/pull:   Download a model from the Ollama registry (NDJSON streaming)
// - POST /api/show:   Retrieve detailed metadata for a single model
// - DELETE /api/delete: Remove a model from local storage
//
// The `OllamaBackend` struct holds a reusable reqwest::Client with a 300s
// timeout. It is Send + Sync and intended to be wrapped in Arc for sharing
// across tokio tasks.

use std::future::Future;
use std::pin::Pin;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::LlmBackend;
use crate::error::LlmError;
use crate::types::{ChatMessage, LlmConfig, LlmResponse};

/// Ollama REST API client implementing the `LlmBackend` trait.
/// Connects to a running Ollama server and forwards chat completion
/// requests via HTTP. The client is stateless -- model loading and
/// GPU memory management are handled by the Ollama server.
pub struct OllamaBackend {
    /// Reusable HTTP client with connection pooling and timeout.
    client: Client,
    /// Configuration specifying the server URL, model, and generation params.
    config: LlmConfig,
}

impl OllamaBackend {
    /// Constructs a new Ollama backend with the given configuration.
    /// The reqwest client is created with a 300-second timeout to accommodate
    /// large models on slow hardware.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::ConnectionFailed` if the reqwest HTTP client cannot
    /// be built (e.g., TLS backend initialization failure on the platform).
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| LlmError::ConnectionFailed {
                reason: format!("failed to build HTTP client: {e}"),
            })?;
        Ok(Self { client, config })
    }

    /// Lists all models installed on the Ollama server.
    /// Calls GET /api/tags and returns the model name, size in bytes,
    /// and parameter count string for each installed model.
    pub async fn list_models(&self) -> Result<Vec<OllamaModel>, LlmError> {
        let url = format!("{}/api/tags", self.config.base_url);
        debug!(url = %url, "Fetching Ollama model list");

        let response =
            self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| LlmError::ConnectionFailed {
                    reason: e.to_string(),
                })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed { status, body });
        }

        let tags_response: OllamaTagsResponse =
            response.json().await.map_err(|e| LlmError::ParseError {
                reason: e.to_string(),
            })?;

        // Convert from the raw nested-details format to the flat public type.
        Ok(tags_response
            .models
            .into_iter()
            .map(OllamaModel::from)
            .collect())
    }

    /// Checks if the Ollama server is reachable by requesting the model list.
    /// Returns true if the server responds with HTTP 200, false otherwise.
    pub async fn is_reachable(&self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url);
        matches!(self.client.get(&url).send().await, Ok(resp) if resp.status().is_success())
    }

    /// Lists models currently loaded into GPU/CPU RAM on the Ollama server.
    /// Calls GET /api/ps and returns each model's name, memory footprint,
    /// VRAM consumption, and auto-unload expiration time.
    pub async fn running_models(&self) -> Result<Vec<OllamaRunningModel>, LlmError> {
        let url = format!("{}/api/ps", self.config.base_url);
        debug!(url = %url, "Fetching Ollama running models");

        let response =
            self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| LlmError::ConnectionFailed {
                    reason: e.to_string(),
                })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed { status, body });
        }

        let ps_response: OllamaPsResponse =
            response.json().await.map_err(|e| LlmError::ParseError {
                reason: e.to_string(),
            })?;

        Ok(ps_response.models)
    }

    /// Downloads a model from the Ollama registry via POST /api/pull.
    /// The pull request is streamed as NDJSON: each line contains a progress
    /// update with status text, layer digest, total bytes, and completed bytes.
    /// The `on_progress` callback is invoked for each progress line, allowing
    /// the caller to forward updates via SSE to the frontend.
    ///
    /// The method blocks until the pull completes or fails. For large models
    /// (several GB), this can take minutes. The 300s reqwest timeout does not
    /// apply to streaming responses where bytes are actively arriving.
    pub async fn pull_model(
        &self,
        model_name: &str,
        on_progress: impl Fn(&OllamaPullProgress) + Send,
    ) -> Result<(), LlmError> {
        let url = format!("{}/api/pull", self.config.base_url);
        debug!(url = %url, model = %model_name, "Starting Ollama model pull");

        let body = serde_json::json!({
            "model": model_name,
            "stream": true,
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed {
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed {
                status,
                body: body_text,
            });
        }

        // Stream NDJSON progress lines using the same line-buffer pattern as
        // chat_completion_streaming. Each complete line is parsed as an
        // OllamaPullProgress and forwarded to the callback.
        let mut byte_stream = response.bytes_stream();
        let mut line_buffer = String::new();

        use futures_util::StreamExt;

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk_bytes = chunk_result.map_err(|e| LlmError::ParseError {
                reason: format!("pull streaming read error: {e}"),
            })?;

            line_buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

            while let Some(newline_pos) = line_buffer.find('\n') {
                let line: String = line_buffer.drain(..=newline_pos).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<OllamaPullProgress>(line) {
                    Ok(progress) => on_progress(&progress),
                    Err(e) => {
                        warn!(line = %line, error = %e, "Skipping unparseable pull progress line");
                    }
                }
            }
        }

        // Process any trailing data without a final newline.
        let remaining = line_buffer.trim();
        if !remaining.is_empty()
            && let Ok(progress) = serde_json::from_str::<OllamaPullProgress>(remaining)
        {
            on_progress(&progress);
        }

        Ok(())
    }

    /// Deletes a model from the Ollama server via DELETE /api/delete.
    /// The model is removed from local storage and can no longer be used
    /// until re-pulled. Returns Ok(()) on success.
    pub async fn delete_model(&self, model_name: &str) -> Result<(), LlmError> {
        let url = format!("{}/api/delete", self.config.base_url);
        debug!(url = %url, model = %model_name, "Deleting Ollama model");

        let body = serde_json::json!({ "model": model_name });

        let response = self
            .client
            .delete(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed {
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            if status == 404 {
                return Err(LlmError::ModelNotFound {
                    model: model_name.to_string(),
                });
            }
            return Err(LlmError::RequestFailed {
                status,
                body: body_text,
            });
        }

        Ok(())
    }

    /// Retrieves detailed metadata for a single model via POST /api/show.
    /// Returns the model's family, parameter size, quantization level,
    /// template string, and license text.
    pub async fn show_model(&self, model_name: &str) -> Result<OllamaModelDetail, LlmError> {
        let url = format!("{}/api/show", self.config.base_url);
        debug!(url = %url, model = %model_name, "Fetching Ollama model details");

        let body = serde_json::json!({ "model": model_name });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ConnectionFailed {
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            if status == 404 {
                return Err(LlmError::ModelNotFound {
                    model: model_name.to_string(),
                });
            }
            return Err(LlmError::RequestFailed {
                status,
                body: body_text,
            });
        }

        let show: OllamaShowResponse = response.json().await.map_err(|e| LlmError::ParseError {
            reason: format!("failed to parse Ollama show response: {e}"),
        })?;

        Ok(OllamaModelDetail {
            model: model_name.to_string(),
            family: show.details.family,
            parameter_size: show.details.parameter_size,
            quantization_level: show.details.quantization_level,
            template: show.template,
            license: show.license,
        })
    }

    /// Builds the JSON request body for the /api/chat endpoint.
    /// When `json_mode` is enabled in the config, includes `"format": "json"`
    /// which constrains Ollama to produce syntactically valid JSON output.
    /// This prevents markdown code fences, prose preambles, and truncated
    /// JSON structures that cause parse failures with smaller models.
    fn build_chat_request(&self, messages: &[ChatMessage], stream: bool) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages.iter().map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            }).collect::<Vec<_>>(),
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        });

        // Ollama's JSON format mode forces the model to produce valid JSON.
        // Only enabled when the caller explicitly requests structured output.
        if self.config.json_mode {
            body["format"] = serde_json::json!("json");
        }

        body
    }
}

impl LlmBackend for OllamaBackend {
    fn name(&self) -> &str {
        "ollama"
    }

    fn chat_completion<'a>(
        &'a self,
        messages: &'a [ChatMessage],
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/api/chat", self.config.base_url);
            let body = self.build_chat_request(messages, false);

            debug!(
                url = %url,
                model = %self.config.model,
                message_count = messages.len(),
                "Sending blocking chat completion request to Ollama"
            );

            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        LlmError::Timeout { seconds: 300 }
                    } else {
                        LlmError::ConnectionFailed {
                            reason: e.to_string(),
                        }
                    }
                })?;

            let status = response.status();
            if !status.is_success() {
                let status_code = status.as_u16();
                let body_text = response.text().await.unwrap_or_default();

                // Ollama returns 404 when the requested model is not installed
                if status_code == 404 {
                    return Err(LlmError::ModelNotFound {
                        model: self.config.model.clone(),
                    });
                }
                return Err(LlmError::RequestFailed {
                    status: status_code,
                    body: body_text,
                });
            }

            let ollama_resp: OllamaChatResponse =
                response.json().await.map_err(|e| LlmError::ParseError {
                    reason: format!("failed to parse Ollama chat response: {e}"),
                })?;

            Ok(LlmResponse {
                content: ollama_resp.message.content,
                prompt_tokens: ollama_resp.prompt_eval_count,
                completion_tokens: ollama_resp.eval_count,
            })
        })
    }

    fn chat_completion_streaming<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        on_token: Box<dyn Fn(&str) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/api/chat", self.config.base_url);
            let body = self.build_chat_request(messages, true);

            debug!(
                url = %url,
                model = %self.config.model,
                message_count = messages.len(),
                "Sending streaming chat completion request to Ollama"
            );

            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        LlmError::Timeout { seconds: 300 }
                    } else {
                        LlmError::ConnectionFailed {
                            reason: e.to_string(),
                        }
                    }
                })?;

            let status = response.status();
            if !status.is_success() {
                let status_code = status.as_u16();
                let body_text = response.text().await.unwrap_or_default();
                if status_code == 404 {
                    return Err(LlmError::ModelNotFound {
                        model: self.config.model.clone(),
                    });
                }
                return Err(LlmError::RequestFailed {
                    status: status_code,
                    body: body_text,
                });
            }

            // Ollama streams NDJSON: each line is a JSON object with
            // { "message": { "content": "token" }, "done": false }
            // The final line has "done": true and includes token counts.
            let mut full_content = String::new();
            let mut prompt_tokens: Option<u64> = None;
            let mut completion_tokens: Option<u64> = None;

            // Read the response body chunk-by-chunk via reqwest's streaming
            // byte reader. Each TCP chunk may contain one or more NDJSON
            // lines, or a partial line that spans multiple chunks. The line
            // buffer accumulates bytes until a newline delimiter is found,
            // then parses each complete line as an OllamaStreamChunk.
            // This delivers tokens to the SSE callback as they arrive from
            // Ollama, rather than buffering the entire response first.
            let mut byte_stream = response.bytes_stream();
            let mut line_buffer = String::new();

            use futures_util::StreamExt;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = chunk_result.map_err(|e| LlmError::ParseError {
                    reason: format!("streaming read error: {e}"),
                })?;

                line_buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

                // Process all complete lines in the buffer. A partial line
                // (no trailing newline) stays in the buffer until the next
                // chunk arrives and completes it.
                while let Some(newline_pos) = line_buffer.find('\n') {
                    let line: String = line_buffer.drain(..=newline_pos).collect();
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    let parsed: OllamaStreamChunk = match serde_json::from_str(line) {
                        Ok(c) => c,
                        Err(e) => {
                            warn!(line = %line, error = %e, "Skipping unparseable Ollama stream chunk");
                            continue;
                        }
                    };

                    if let Some(ref msg) = parsed.message
                        && !msg.content.is_empty()
                    {
                        on_token(&msg.content);
                        full_content.push_str(&msg.content);
                    }

                    if parsed.done {
                        prompt_tokens = parsed.prompt_eval_count;
                        completion_tokens = parsed.eval_count;
                    }
                }
            }

            // Process any remaining data in the buffer after the stream ends.
            // This handles the case where Ollama's final chunk does not end
            // with a newline character.
            let remaining = line_buffer.trim();
            if !remaining.is_empty()
                && let Ok(parsed) = serde_json::from_str::<OllamaStreamChunk>(remaining)
            {
                if let Some(ref msg) = parsed.message
                    && !msg.content.is_empty()
                {
                    on_token(&msg.content);
                    full_content.push_str(&msg.content);
                }
                if parsed.done {
                    prompt_tokens = parsed.prompt_eval_count;
                    completion_tokens = parsed.eval_count;
                }
            }

            Ok(LlmResponse {
                content: full_content,
                prompt_tokens,
                completion_tokens,
            })
        })
    }

    fn config(&self) -> &LlmConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Ollama API response types (private, not exposed outside this module)
// ---------------------------------------------------------------------------

/// Response from GET /api/tags listing all installed models.
/// The raw Ollama JSON nests `parameter_size`, `family`, and
/// `quantization_level` inside a `details` object. This struct
/// deserializes the raw format, then `From` converts it into the
/// flat public `OllamaModel` type.
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaTagsModelRaw>,
}

/// Raw model entry from the Ollama /api/tags response with nested details.
#[derive(Debug, Deserialize)]
struct OllamaTagsModelRaw {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    details: OllamaModelDetails,
}

/// Nested details object within each model entry from /api/tags and /api/ps.
#[derive(Debug, Default, Deserialize)]
struct OllamaModelDetails {
    #[serde(default)]
    parameter_size: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
}

impl From<OllamaTagsModelRaw> for OllamaModel {
    fn from(raw: OllamaTagsModelRaw) -> Self {
        Self {
            name: raw.name,
            size: raw.size,
            parameter_size: raw.details.parameter_size,
            family: raw.details.family,
            quantization_level: raw.details.quantization_level,
        }
    }
}

/// Response from GET /api/ps listing all models loaded in GPU/CPU RAM.
/// The raw Ollama JSON nests model metadata inside a `details` object,
/// same as /api/tags, and adds `size_vram` and `expires_at` fields.
#[derive(Debug, Deserialize)]
struct OllamaPsResponse {
    #[serde(default)]
    models: Vec<OllamaRunningModel>,
}

/// Response from POST /api/show containing detailed model metadata.
#[derive(Debug, Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    details: OllamaModelDetails,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    license: Option<String>,
}

/// A single model entry from the Ollama tags response.
/// The `parameter_size`, `family`, and `quantization_level` fields are
/// extracted from the nested `details` object in the Ollama JSON response
/// via the `OllamaTagsModelRaw` intermediate type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    /// Model tag identifier (e.g., "qwen2.5:14b", "llama3.1:8b").
    pub name: String,
    /// Total size of the model files on disk in bytes.
    #[serde(default)]
    pub size: u64,
    /// Human-readable parameter count string (e.g., "14B", "7B").
    /// Extracted from model metadata; None if the model does not report it.
    #[serde(default)]
    pub parameter_size: Option<String>,
    /// Model family name (e.g., "qwen2.5", "llama3.1").
    #[serde(default)]
    pub family: Option<String>,
    /// Quantization format identifier (e.g., "Q4_K_M", "Q8_0").
    /// Indicates how model weights are compressed on disk.
    #[serde(default)]
    pub quantization_level: Option<String>,
}

/// A model currently loaded in GPU/CPU RAM on the Ollama server.
/// Returned by GET /api/ps. Contains memory footprint details and
/// the auto-unload expiration time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRunningModel {
    /// Model tag identifier matching the name from /api/tags.
    pub name: String,
    /// Memory footprint in bytes (RAM + VRAM combined).
    #[serde(default)]
    pub size: u64,
    /// VRAM consumption in bytes. None when the model runs entirely on CPU.
    /// When equal to `size`, the entire model is loaded on GPU.
    #[serde(default)]
    pub size_vram: Option<u64>,
    /// ISO 8601 timestamp when the model will be automatically unloaded
    /// from memory after idle timeout (default 5 minutes in Ollama).
    #[serde(default)]
    pub expires_at: Option<String>,
    /// Human-readable parameter count string (e.g., "7B").
    #[serde(default)]
    pub parameter_size: Option<String>,
    /// Model family name (e.g., "llama", "qwen2.5").
    #[serde(default)]
    pub family: Option<String>,
}

/// Progress update from POST /api/pull streaming response (NDJSON).
/// Each line in the streaming response is one progress update. During
/// the layer download phase, `total` and `completed` contain byte counts.
/// The final line has `status: "success"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPullProgress {
    /// Human-readable status string describing the current phase.
    /// Values include "pulling manifest", "pulling `<digest>`",
    /// "verifying sha256 digest", "writing manifest", "success".
    pub status: String,
    /// SHA256 digest of the layer currently being downloaded.
    /// Present only during the layer download phase.
    #[serde(default)]
    pub digest: Option<String>,
    /// Total bytes expected for the current layer.
    /// Present only during the layer download phase.
    #[serde(default)]
    pub total: Option<u64>,
    /// Bytes downloaded so far for the current layer.
    /// Present only during the layer download phase.
    #[serde(default)]
    pub completed: Option<u64>,
}

/// Detailed model metadata returned by POST /api/show.
/// Contains information not available in the /api/tags listing such as
/// the Jinja template, license text, and quantization format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetail {
    /// Model tag identifier that was queried.
    pub model: String,
    /// Model architecture family (e.g., "llama", "qwen2.5").
    pub family: Option<String>,
    /// Human-readable parameter count string (e.g., "7B", "14B").
    pub parameter_size: Option<String>,
    /// Quantization format identifier (e.g., "Q4_K_M").
    pub quantization_level: Option<String>,
    /// Jinja prompt template string used by the model.
    pub template: Option<String>,
    /// License text embedded in the model metadata.
    pub license: Option<String>,
}

/// Response from POST /api/chat with stream: false (blocking mode).
#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    /// The generated assistant message.
    message: OllamaChatMessage,
    /// Number of tokens in the prompt evaluation.
    prompt_eval_count: Option<u64>,
    /// Number of tokens generated.
    eval_count: Option<u64>,
}

/// A single message within an Ollama chat response.
#[derive(Debug, Deserialize)]
struct OllamaChatMessage {
    /// The text content of the message.
    content: String,
}

/// A single chunk from the POST /api/chat streaming response (NDJSON).
/// Each line in the response is one chunk. The final chunk has `done: true`
/// and includes aggregate token counts.
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    /// The partial message containing one or more generated tokens.
    message: Option<OllamaChatMessage>,
    /// Whether this is the final chunk in the stream.
    #[serde(default)]
    done: bool,
    /// Prompt token count (present only in the final chunk).
    prompt_eval_count: Option<u64>,
    /// Completion token count (present only in the final chunk).
    eval_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Unit tests for deserialization of Ollama API responses
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the /api/tags response with nested `details` object
    /// is correctly deserialized into flat OllamaModel structs.
    #[test]
    fn test_ollama_model_deserialization() {
        let json = r#"{
            "models": [
                {
                    "name": "llama3.2:3b",
                    "model": "llama3.2:3b",
                    "modified_at": "2025-05-04T17:37:44.706Z",
                    "size": 2019393189,
                    "digest": "a80c4f17acd5",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M"
                    }
                },
                {
                    "name": "qwen2.5:14b",
                    "size": 9000000000,
                    "details": {
                        "family": "qwen2",
                        "parameter_size": "14B",
                        "quantization_level": "Q4_0"
                    }
                }
            ]
        }"#;

        let tags: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        let models: Vec<OllamaModel> = tags.models.into_iter().map(OllamaModel::from).collect();

        assert_eq!(models.len(), 2);

        assert_eq!(models[0].name, "llama3.2:3b");
        assert_eq!(models[0].size, 2019393189);
        assert_eq!(models[0].parameter_size.as_deref(), Some("3.2B"));
        assert_eq!(models[0].family.as_deref(), Some("llama"));
        assert_eq!(models[0].quantization_level.as_deref(), Some("Q4_K_M"));

        assert_eq!(models[1].name, "qwen2.5:14b");
        assert_eq!(models[1].size, 9000000000);
        assert_eq!(models[1].parameter_size.as_deref(), Some("14B"));
        assert_eq!(models[1].family.as_deref(), Some("qwen2"));
        assert_eq!(models[1].quantization_level.as_deref(), Some("Q4_0"));
    }

    /// Verifies that the /api/ps response is correctly deserialized into
    /// OllamaRunningModel structs with VRAM and expiration fields.
    #[test]
    fn test_running_model_deserialization() {
        let json = r#"{
            "models": [
                {
                    "name": "mistral:latest",
                    "model": "mistral:latest",
                    "size": 5137025024,
                    "digest": "2ae6f6dd7a3d",
                    "details": {
                        "family": "llama",
                        "parameter_size": "7.2B",
                        "quantization_level": "Q4_0"
                    },
                    "expires_at": "2024-06-04T14:38:31.837Z",
                    "size_vram": 5137025024
                }
            ]
        }"#;

        let ps: OllamaPsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(ps.models.len(), 1);

        let model = &ps.models[0];
        assert_eq!(model.name, "mistral:latest");
        assert_eq!(model.size, 5137025024);
        assert_eq!(model.size_vram, Some(5137025024));
        assert_eq!(
            model.expires_at.as_deref(),
            Some("2024-06-04T14:38:31.837Z")
        );
    }

    /// Verifies that /api/ps response without size_vram field (CPU-only
    /// inference) deserializes correctly with None for that field.
    #[test]
    fn test_running_model_without_vram() {
        let json = r#"{
            "models": [
                {
                    "name": "phi3:mini",
                    "size": 2300000000
                }
            ]
        }"#;

        let ps: OllamaPsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(ps.models.len(), 1);
        assert_eq!(ps.models[0].name, "phi3:mini");
        assert_eq!(ps.models[0].size_vram, None);
        assert_eq!(ps.models[0].expires_at, None);
    }

    /// Verifies that pull progress lines are correctly deserialized from
    /// the NDJSON streaming response.
    #[test]
    fn test_pull_progress_deserialization() {
        // Manifest phase
        let manifest = r#"{"status": "pulling manifest"}"#;
        let p: OllamaPullProgress = serde_json::from_str(manifest).unwrap();
        assert_eq!(p.status, "pulling manifest");
        assert_eq!(p.digest, None);
        assert_eq!(p.total, None);
        assert_eq!(p.completed, None);

        // Layer download phase
        let download = r#"{
            "status": "pulling e0a42594d802",
            "digest": "sha256:e0a42594d802f1aad36e098aee104e78519e7b872a828f899b77e3d3e6b6e31e",
            "total": 2019393189,
            "completed": 241970
        }"#;
        let p: OllamaPullProgress = serde_json::from_str(download).unwrap();
        assert_eq!(p.status, "pulling e0a42594d802");
        assert!(p.digest.is_some());
        assert_eq!(p.total, Some(2019393189));
        assert_eq!(p.completed, Some(241970));

        // Success phase
        let success = r#"{"status": "success"}"#;
        let p: OllamaPullProgress = serde_json::from_str(success).unwrap();
        assert_eq!(p.status, "success");
    }

    /// Verifies that the /api/show response is correctly deserialized into
    /// an OllamaShowResponse with template and license fields.
    #[test]
    fn test_show_response_deserialization() {
        let json = r#"{
            "license": "MIT License",
            "modelfile": "FROM /path/to/model",
            "parameters": "stop \"<|end|>\"",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8.0B",
                "quantization_level": "Q4_0"
            }
        }"#;

        let show: OllamaShowResponse = serde_json::from_str(json).unwrap();
        assert_eq!(show.details.family.as_deref(), Some("llama"));
        assert_eq!(show.details.parameter_size.as_deref(), Some("8.0B"));
        assert_eq!(show.details.quantization_level.as_deref(), Some("Q4_0"));
        assert_eq!(
            show.template.as_deref(),
            Some("{{ .System }}\n{{ .Prompt }}")
        );
        assert_eq!(show.license.as_deref(), Some("MIT License"));
    }

    /// Verifies that the /api/tags response handles empty model list gracefully.
    #[test]
    fn test_empty_tags_response() {
        let json = r#"{"models": []}"#;
        let tags: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert!(tags.models.is_empty());
    }

    /// Verifies that the /api/tags response with minimal fields (no details)
    /// produces OllamaModel with None for all optional fields.
    #[test]
    fn test_tags_model_minimal_fields() {
        let json = r#"{
            "models": [
                {
                    "name": "custom:latest",
                    "size": 1000000
                }
            ]
        }"#;

        let tags: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        let models: Vec<OllamaModel> = tags.models.into_iter().map(OllamaModel::from).collect();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "custom:latest");
        assert_eq!(models[0].size, 1000000);
        assert_eq!(models[0].parameter_size, None);
        assert_eq!(models[0].family, None);
        assert_eq!(models[0].quantization_level, None);
    }

    /// Verifies that the build_chat_request method includes "format": "json"
    /// when json_mode is enabled in the config. The `new()` call is unwrapped
    /// because the reqwest client builder is expected to succeed in test
    /// environments with a working TLS backend.
    #[test]
    fn test_build_chat_request_json_mode() {
        let config = LlmConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "test".to_string(),
            temperature: 0.1,
            max_tokens: 100,
            json_mode: true,
        };
        let backend = OllamaBackend::new(config).expect("reqwest client creation in test");
        let messages = vec![crate::types::ChatMessage {
            role: crate::types::ChatRole::User,
            content: "hello".to_string(),
        }];

        let body = backend.build_chat_request(&messages, false);
        assert_eq!(body["format"], serde_json::json!("json"));
        assert_eq!(body["stream"], serde_json::json!(false));
    }

    /// Verifies that the build_chat_request method omits the format field
    /// when json_mode is disabled in the config. The `new()` call is unwrapped
    /// because the reqwest client builder is expected to succeed in test
    /// environments with a working TLS backend.
    #[test]
    fn test_build_chat_request_no_json_mode() {
        let config = LlmConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "test".to_string(),
            temperature: 0.5,
            max_tokens: 200,
            json_mode: false,
        };
        let backend = OllamaBackend::new(config).expect("reqwest client creation in test");
        let messages = vec![crate::types::ChatMessage {
            role: crate::types::ChatRole::System,
            content: "you are a helper".to_string(),
        }];

        let body = backend.build_chat_request(&messages, true);
        assert!(body.get("format").is_none());
        assert_eq!(body["stream"], serde_json::json!(true));
    }
}
