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

// Data types for the LLM abstraction layer.
//
// These types are backend-agnostic: they represent the chat completion
// protocol (messages, roles, configuration) without referencing any
// specific LLM provider. Both the Ollama HTTP backend and a future
// native Rust backend use the same types.

use serde::{Deserialize, Serialize};

/// Role of a participant in a multi-turn chat conversation.
/// Maps to the standard system/user/assistant triad used by all major LLM APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// System prompt that sets the LLM's persona and behavioral constraints.
    System,
    /// User-provided input (queries, claims, passages to evaluate).
    User,
    /// LLM-generated response from a previous turn (for multi-turn contexts).
    Assistant,
}

/// A single message in a chat conversation history.
/// The conversation is represented as a chronologically ordered slice of
/// ChatMessage values, starting with an optional system message followed
/// by alternating user/assistant turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Which participant produced this message.
    pub role: ChatRole,
    /// The textual content of the message. For system messages this contains
    /// behavioral instructions; for user messages it contains the task prompt;
    /// for assistant messages it contains prior LLM output.
    pub content: String,
}

/// Response from a chat completion request. Contains the full generated text
/// and optional token usage statistics (when the backend reports them).
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// The complete assistant response text after generation finishes.
    pub content: String,
    /// Number of tokens consumed by the prompt (input). None if the backend
    /// does not report token counts.
    pub prompt_tokens: Option<u64>,
    /// Number of tokens generated in the completion (output). None if the
    /// backend does not report token counts.
    pub completion_tokens: Option<u64>,
}

/// Configuration for connecting to an LLM backend. Passed to the backend
/// constructor and stored for the lifetime of the backend instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Base URL for the LLM API endpoint.
    /// For Ollama: "http://localhost:11434"
    /// For a future native backend: unused (set to empty string).
    pub base_url: String,
    /// Model identifier as recognized by the backend.
    /// For Ollama: the model tag (e.g., "qwen2.5:14b", "llama3.1:8b").
    /// For a future native backend: a HuggingFace model ID or local path.
    pub model: String,
    /// Sampling temperature controlling randomness. 0.0 = deterministic
    /// (greedy decoding), 1.0 = maximum randomness. For citation verification,
    /// values between 0.0 and 0.2 are recommended to minimize hallucination.
    pub temperature: f32,
    /// Maximum number of tokens the LLM may generate in a single response.
    /// Prevents runaway generation. 8192 tokens is the default for citation
    /// verification to accommodate detailed reasoning within a JSON structure.
    pub max_tokens: u32,
    /// When true, the backend constrains output to syntactically valid JSON.
    /// Ollama supports this via the `"format": "json"` request parameter.
    /// Prevents markdown code fences, prose preambles, and incomplete JSON
    /// that cause parse failures with smaller models.
    #[serde(default)]
    pub json_mode: bool,
}
