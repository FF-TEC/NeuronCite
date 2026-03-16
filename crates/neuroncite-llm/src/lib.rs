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

//! neuroncite-llm: Trait-based abstraction for Large Language Model backends.
//!
//! This crate decouples the citation verification agent from any specific LLM
//! implementation. The `LlmBackend` trait defines the contract for chat
//! completion (blocking and streaming). Concrete implementations:
//!
//! - `OllamaBackend`: HTTP client for the Ollama REST API (localhost:11434).
//!   Supports both blocking and streaming chat completion via /api/chat.
//!
//! The crate has zero workspace-internal dependencies. It relies only on
//! tokio (async runtime), reqwest (HTTP client), serde (serialization),
//! thiserror (error types), and tracing (logging). This isolation guarantees
//! that swapping the LLM backend requires changes only within this crate.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod error;
pub mod ollama;
pub mod types;

pub use error::LlmError;
pub use ollama::{
    OllamaBackend, OllamaModel, OllamaModelDetail, OllamaPullProgress, OllamaRunningModel,
};
pub use types::{ChatMessage, ChatRole, LlmConfig, LlmResponse};

use std::future::Future;
use std::pin::Pin;

/// Abstraction over LLM inference backends.
///
/// Implementations handle transport and protocol differences between backends
/// (HTTP for Ollama). The trait is object-safe and
/// requires Send + Sync for sharing across tokio tasks via `Arc<dyn LlmBackend>`.
///
/// The citation agent loop holds `Arc<dyn LlmBackend>` and calls these methods
/// without knowing which backend is active. Swapping backends requires only
/// constructing a different implementation and passing it to the agent.
pub trait LlmBackend: Send + Sync {
    /// Returns a human-readable name identifying this backend (e.g., "ollama").
    /// Used in log messages and SSE events to inform the user
    /// which backend is processing their request.
    fn name(&self) -> &str;

    /// Sends a chat completion request and waits for the full response.
    ///
    /// The `messages` slice contains the conversation history: an optional
    /// system message at index 0, followed by alternating user/assistant
    /// turns. The backend sends all messages to the LLM and returns the
    /// complete assistant response after generation finishes.
    ///
    /// Returns `Pin<Box<dyn Future>>` for object safety with `dyn LlmBackend`.
    /// The lifetime `'a` ties `&self` and `messages` together so the future
    /// can borrow both.
    fn chat_completion<'a>(
        &'a self,
        messages: &'a [ChatMessage],
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + 'a>>;

    /// Sends a chat completion request with token-by-token streaming.
    ///
    /// Each generated token (or token group) is delivered to the `on_token`
    /// callback as it arrives. The callback receives a string slice containing
    /// one or more characters of the ongoing response. After generation
    /// completes, the full response is returned.
    ///
    /// Backends that support native streaming (Ollama) override this
    /// for real-time delivery. The default implementation calls
    /// `chat_completion` and delivers the entire response as a single chunk.
    fn chat_completion_streaming<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        on_token: Box<dyn Fn(&str) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + 'a>>;

    /// Returns the configuration this backend was constructed with.
    /// Used by the agent loop to include model/backend info in SSE events
    /// and log messages.
    fn config(&self) -> &LlmConfig;
}
