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

// Error types for the LLM abstraction layer.
//
// `LlmError` covers all failure modes that can occur when communicating
// with an LLM backend: network failures, HTTP errors, response parsing
// issues, model availability, and timeouts. Each variant carries enough
// context for the caller to decide whether to retry, fail the row, or
// abort the entire agent loop.

use thiserror::Error;

/// Errors that can occur during LLM backend operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The LLM backend is not reachable. This covers DNS resolution failures,
    /// TCP connection refused (Ollama not running), and TLS handshake errors.
    #[error("LLM connection failed: {reason}")]
    ConnectionFailed { reason: String },

    /// The HTTP request completed but the server returned a non-2xx status
    /// code. The response body is included for diagnostic purposes.
    #[error("LLM request failed with status {status}: {body}")]
    RequestFailed { status: u16, body: String },

    /// The response body could not be parsed as valid JSON. This indicates
    /// a protocol mismatch between the client and server (e.g., Ollama
    /// returned HTML instead of JSON due to a misconfigured proxy).
    #[error("LLM response parse error: {reason}")]
    ParseError { reason: String },

    /// The LLM produced valid JSON, but the structure does not match the
    /// expected schema (e.g., missing "verdict" field in the verification
    /// response). The caller should retry with a corrective prompt suffix.
    #[error("LLM output did not match expected schema: {reason}")]
    SchemaViolation { reason: String },

    /// The requested model is not installed on the Ollama server. The user
    /// must run `ollama pull {model}` before starting verification.
    #[error("model '{model}' not found on the LLM server")]
    ModelNotFound { model: String },

    /// The LLM did not produce a response within the configured timeout.
    /// For large models on slow hardware, increasing the timeout or using
    /// a smaller model may resolve this.
    #[error("LLM request timed out after {seconds}s")]
    Timeout { seconds: u64 },
}
