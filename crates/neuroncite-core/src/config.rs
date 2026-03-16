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

// Application-wide configuration deserialized from a TOML file.
// The configuration file is loaded from one of several locations in order
// of precedence (CLI flag, environment variable, working directory, platform
// config directory). If no file is found, built-in defaults are used.
//
// This module defines the `AppConfig` struct and its Default implementation.
// Field names and default values match the specification in the architecture
// document (Section 18.1).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::NeuronCiteError;

/// Tracing log level for the application runtime. Controls the verbosity
/// of log output emitted via the tracing subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "trace"),
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Text chunking strategy selector that determines how extracted document text
/// is split into overlapping segments for embedding. Each variant maps to a
/// different boundary detection algorithm in the neuroncite-chunk crate.
///
/// Serializes to/from lowercase strings ("token", "word", "sentence", "page")
/// via serde rename_all. Also implements `FromStr` for parsing from database
/// TEXT columns and CLI arguments, and `Display` for writing back to the
/// database and log output.
///
/// Note: This enum is distinct from the `ChunkStrategy` trait in
/// `neuroncite_core::traits`, which is the runtime interface that chunking
/// implementations conform to. This enum selects *which* implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkStrategy {
    Token,
    Word,
    Sentence,
    Page,
}

impl std::fmt::Display for ChunkStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token => write!(f, "token"),
            Self::Word => write!(f, "word"),
            Self::Sentence => write!(f, "sentence"),
            Self::Page => write!(f, "page"),
        }
    }
}

impl std::str::FromStr for ChunkStrategy {
    type Err = String;

    /// Parses a chunking strategy from its lowercase string representation.
    /// Accepted values: "token", "word", "sentence", "page".
    /// Returns an error with a descriptive message for unrecognized strings.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "token" => Ok(Self::Token),
            "word" => Ok(Self::Word),
            "sentence" => Ok(Self::Sentence),
            "page" => Ok(Self::Page),
            other => Err(format!(
                "unknown chunk strategy: '{other}' (valid: token, word, sentence, page)"
            )),
        }
    }
}

/// Configurable limits that constrain request parameters at the API boundary.
///
/// Each limit enforces a maximum size or count on a specific request field to
/// prevent resource exhaustion from pathological inputs. All limits have
/// built-in defaults that are conservative enough for single-user use and can
/// be relaxed via the configuration file for deployments with higher load.
///
/// These limits are checked by the `validate()` methods on request DTOs before
/// any database access or embedding computation is performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputLimits {
    /// Maximum number of UTF-8 characters in a search query string. Queries
    /// longer than this limit are rejected with HTTP 400 before the query is
    /// embedded or passed to the search pipeline. Prevents memory exhaustion
    /// from the tokenizer processing extremely long strings.
    #[serde(default = "default_max_query_chars")]
    pub max_query_chars: usize,

    /// Maximum number of chunk IDs in a single verify request. Enforced on the
    /// `chunk_ids` array to prevent the bulk SQL query from producing
    /// excessively large IN-clause parameter lists that can degrade SQLite
    /// performance or cause query plan failures.
    #[serde(default = "default_max_chunk_ids")]
    pub max_chunk_ids: usize,

    /// Maximum number of session IDs in a single multi-search request.
    /// Enforced on the `session_ids` array. Each session requires a separate
    /// HNSW search pass; unbounded session counts create a linear response-
    /// time attack vector.
    #[serde(default = "default_max_session_ids")]
    pub max_session_ids: usize,

    /// Maximum value for the `chunk_size` parameter in an index request.
    /// Enforced to prevent the chunking pipeline from producing single chunks
    /// that are larger than the model's context window (typically 512 tokens
    /// for most embedding models). Values above this limit produce embeddings
    /// that silently truncate at the model boundary, degrading retrieval quality.
    #[serde(default = "default_max_chunk_size_tokens")]
    pub max_chunk_size_tokens: usize,

    /// Maximum number of comma-separated divisor values in a search request's
    /// `refine_divisors` parameter. Each divisor triggers a separate embedding
    /// pass during refinement; unbounded divisor counts create a CPU
    /// exhaustion attack vector.
    #[serde(default = "default_max_refine_divisors")]
    pub max_refine_divisors: usize,
}

fn default_max_query_chars() -> usize {
    10_000
}
fn default_max_chunk_ids() -> usize {
    200
}
fn default_max_session_ids() -> usize {
    10
}
fn default_max_chunk_size_tokens() -> usize {
    16_384
}
fn default_max_refine_divisors() -> usize {
    10
}

impl Default for InputLimits {
    fn default() -> Self {
        Self {
            max_query_chars: default_max_query_chars(),
            max_chunk_ids: default_max_chunk_ids(),
            max_session_ids: default_max_session_ids(),
            max_chunk_size_tokens: default_max_chunk_size_tokens(),
            max_refine_divisors: default_max_refine_divisors(),
        }
    }
}

/// Default values applied to optional request fields when the caller omits
/// them. Centralizes fallback values that were previously hardcoded in
/// individual handlers (e.g. `req.top_k.unwrap_or(10)`) so they can be
/// configured via the TOML config file. Handlers call
/// `req.with_defaults(&state.config.defaults)` before processing to fill
/// in any absent optional fields from this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestDefaults {
    /// Default number of results returned by search endpoints when the
    /// caller does not specify `top_k`. Applies to both single-session and
    /// multi-session search.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    10
}

impl Default for RequestDefaults {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
        }
    }
}

/// Application-wide configuration with fields covering the API server,
/// logging, default indexing parameters, search tuning, CORS policy,
/// and deduplication thresholds.
///
/// # Examples
///
/// ```
/// use neuroncite_core::config::AppConfig;
///
/// let config = AppConfig::default();
/// assert_eq!(config.port, 3030);
/// assert_eq!(config.bind_address, "127.0.0.1");
/// assert_eq!(config.default_chunk_size, 256);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    /// TCP port the API server listens on.
    pub port: u16,

    /// IP address the API server binds to.
    pub bind_address: String,

    /// Minimum log level for stderr output. Valid values: trace, debug,
    /// info, warn, error. The mode-specific default (info for headless,
    /// warn for GUI) is applied when this field is not explicitly set.
    pub log_level: LogLevel,

    /// Path to an append-mode log file. If None, no file logging is performed.
    pub log_file: Option<String>,

    /// Identifier of the default embedding model used when no model is
    /// specified in an indexing request.
    pub default_model: String,

    /// Default chunking strategy. The "token" strategy produces chunks aligned
    /// to the embedding model's subword tokenizer, giving precise control over
    /// the number of tokens per chunk. Other strategies split on word, sentence,
    /// or page boundaries respectively.
    pub default_strategy: ChunkStrategy,

    /// Default chunk size. The unit depends on the active strategy: tokens for
    /// "token", words for "word", words for "sentence" (as max_words). Ignored
    /// by the "page" strategy.
    pub default_chunk_size: usize,

    /// Default overlap between consecutive chunks. The unit matches the active
    /// strategy (tokens for "token", words for "word"). Ignored by "sentence"
    /// and "page" strategies.
    pub default_overlap: usize,

    /// Default Tesseract language code for OCR fallback.
    pub ocr_language: String,

    /// HNSW `ef_search` parameter controlling the trade-off between search
    /// accuracy and speed. Higher values increase recall at the cost of latency.
    pub ef_search: usize,

    /// Reciprocal Rank Fusion parameter k. Controls the weight decay in the
    /// RRF formula: score = 1 / (k + rank).
    pub rrf_k: usize,

    /// List of allowed CORS origins for cross-origin API access. If empty,
    /// all origins are allowed via wildcard. Ignored when bound to localhost.
    pub cors_origins: Vec<String>,

    /// Maximum Hamming distance (out of 64 bits) below which two `SimHash`
    /// values are considered near-duplicates during chunk deduplication.
    pub simhash_threshold: u32,

    /// HNSW orphan ratio above which the system recommends an index rebuild.
    /// Orphans are vector slots that were deleted but not reclaimed.
    pub orphan_ratio_threshold: f64,

    /// Email address for Unpaywall API access. Unpaywall requires an email
    /// as a polite-access identifier in API requests (no account needed).
    /// When this field is None, the DOI resolution chain skips Unpaywall and
    /// starts with Semantic Scholar.
    #[serde(default)]
    pub unpaywall_email: Option<String>,

    /// Filesystem path allowlist for the browse and scan-pdfs web endpoints.
    /// When non-empty, only paths that are descendants of one of these roots
    /// are accessible through the web browsing API. When empty (default),
    /// all paths are accessible -- backwards-compatible for localhost use.
    /// Operators exposing the server to a LAN should populate this field
    /// to prevent the browse endpoint from enumerating arbitrary directories.
    #[serde(default)]
    pub allowed_browse_roots: Vec<String>,

    /// Maximum duration in minutes that a single indexing job execution is
    /// allowed to run before the executor times it out and transitions it
    /// to Failed. This prevents a stuck pipeline from blocking the entire
    /// job queue indefinitely. Valid range: 1..=1440 (1 minute to 24 hours).
    #[serde(default = "default_job_timeout_minutes")]
    pub job_timeout_minutes: u32,

    /// Number of connections in the SQLite r2d2 connection pool. More
    /// connections allow higher concurrent read throughput, but each
    /// connection holds memory for its page cache and mmap region.
    /// Valid range: 1..=64. Default: 8 (1 writer + 7 readers).
    #[serde(default = "default_db_pool_size")]
    pub db_pool_size: u32,

    /// Request parameter limits enforced at the API boundary. Each limit
    /// caps a specific field to prevent resource exhaustion. Defaults are
    /// conservative values that are appropriate for single-user deployments.
    /// All fields within this struct use serde defaults so existing
    /// configuration files without a `[limits]` section continue to work.
    #[serde(default)]
    pub limits: InputLimits,

    /// Default values for optional request fields. When a caller omits an
    /// optional parameter (e.g. `top_k` in a search request), the handler
    /// fills it from this struct instead of using a hardcoded fallback.
    /// Existing configuration files without a `[defaults]` section receive
    /// the built-in defaults via serde.
    #[serde(default)]
    pub defaults: RequestDefaults,
}

/// Returns the default value for `job_timeout_minutes` (30 minutes).
/// Used by serde's `default` attribute during deserialization of configs
/// that do not specify this field.
fn default_job_timeout_minutes() -> u32 {
    30
}

/// Returns the default value for `db_pool_size` (8 connections).
fn default_db_pool_size() -> u32 {
    8
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            port: 3030,
            bind_address: "127.0.0.1".into(),
            log_level: LogLevel::Info,
            log_file: None,
            default_model: "BAAI/bge-small-en-v1.5".into(),
            default_strategy: ChunkStrategy::Token,
            default_chunk_size: 256,
            default_overlap: 32,
            ocr_language: "eng".into(),
            ef_search: 100,
            rrf_k: 60,
            cors_origins: Vec::new(),
            simhash_threshold: 3,
            orphan_ratio_threshold: 0.2,
            unpaywall_email: None,
            allowed_browse_roots: Vec::new(),
            job_timeout_minutes: 30,
            db_pool_size: 8,
            limits: InputLimits::default(),
            defaults: RequestDefaults::default(),
        }
    }
}

/// Returns true when the given address string refers to a loopback interface.
/// Recognized loopback values: "127.0.0.1", "::1", and "localhost".
/// All other values (including "0.0.0.0" which binds all interfaces) are
/// treated as non-loopback, meaning the server is reachable from the network.
/// Returns true if the bind address refers to the loopback interface.
/// Used by server initialization and browse path validation to determine
/// whether the server is only reachable from the same machine (loopback)
/// or from the network.
pub fn is_loopback(addr: &str) -> bool {
    addr == "127.0.0.1" || addr == "::1" || addr == "localhost"
}

impl AppConfig {
    /// Loads configuration from a TOML file at the given path. The file must
    /// contain a complete TOML representation of `AppConfig` (all fields
    /// present).
    ///
    /// After successful deserialization, `validate()` is called to verify
    /// that all field values are within their permitted ranges.
    ///
    /// # Arguments
    ///
    /// * `path` - Filesystem path to the TOML configuration file.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Config` if the file cannot be read, if
    /// the TOML content fails to deserialize into `AppConfig`, or if any
    /// field value is outside its valid range.
    pub fn load_from_file(path: &Path) -> Result<Self, NeuronCiteError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            NeuronCiteError::Config(format!(
                "failed to read config file {}: {e}",
                path.display()
            ))
        })?;

        Self::load_from_str(&contents)
    }

    /// Parses configuration from a TOML string. This is the underlying
    /// parsing function used by `load_from_file` and is also useful for
    /// testing without filesystem access.
    ///
    /// After successful deserialization, `validate()` is called to verify
    /// that all field values are within their permitted ranges.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Config` if the TOML content fails to
    /// deserialize into `AppConfig`, or if any field value is outside
    /// its valid range.
    pub fn load_from_str(toml_content: &str) -> Result<Self, NeuronCiteError> {
        let config: Self = toml::from_str(toml_content)
            .map_err(|e| NeuronCiteError::Config(format!("failed to parse config TOML: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Validates that all configuration field values are within their
    /// permitted ranges. This method is called automatically by
    /// `load_from_str` and `load_from_file` after deserialization.
    ///
    /// # Checked constraints
    ///
    /// - `port` must be in the range 1..=65535 (u16 already guarantees <= 65535,
    ///   but port 0 is rejected because it means "OS-assigned" and breaks
    ///   predictable binding).
    /// - `ef_search` must be greater than zero. A zero ef_search produces
    ///   no results from the HNSW graph traversal.
    /// - `rrf_k` must be greater than zero. A zero rrf_k causes division by
    ///   zero in the RRF formula: score = 1 / (k + rank).
    /// - `simhash_threshold` must be at most 64 because SimHash operates on
    ///   64-bit fingerprints. A threshold above 64 would match every pair.
    /// - `orphan_ratio_threshold` must be in the range 0.0..=1.0 because it
    ///   represents a ratio of deleted slots to total slots.
    /// - `default_chunk_size` must be greater than zero. A zero chunk size
    ///   produces no text in each chunk.
    /// - `default_overlap` must be strictly less than `default_chunk_size`.
    ///   An overlap equal to or larger than the chunk size produces infinite
    ///   loops or empty chunks in the chunking pipeline.
    /// - `job_timeout_minutes` must be in the range 1..=1440. A zero timeout
    ///   causes immediate failure of every job. The maximum of 1440 minutes
    ///   corresponds to 24 hours.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Config` with a message describing the
    /// first invalid field encountered.
    pub fn validate(&self) -> Result<(), NeuronCiteError> {
        if self.port == 0 {
            return Err(NeuronCiteError::Config(
                "port must be in range 1..=65535, got 0".into(),
            ));
        }

        if self.ef_search == 0 {
            return Err(NeuronCiteError::Config(
                "ef_search must be greater than 0".into(),
            ));
        }

        if self.rrf_k == 0 {
            return Err(NeuronCiteError::Config(
                "rrf_k must be greater than 0".into(),
            ));
        }

        if self.simhash_threshold > 64 {
            return Err(NeuronCiteError::Config(format!(
                "simhash_threshold must be at most 64, got {}",
                self.simhash_threshold
            )));
        }

        if !(0.0..=1.0).contains(&self.orphan_ratio_threshold) {
            return Err(NeuronCiteError::Config(format!(
                "orphan_ratio_threshold must be in 0.0..=1.0, got {}",
                self.orphan_ratio_threshold
            )));
        }

        if self.default_chunk_size == 0 {
            return Err(NeuronCiteError::Config(
                "default_chunk_size must be greater than 0".into(),
            ));
        }

        if self.default_overlap >= self.default_chunk_size {
            return Err(NeuronCiteError::Config(format!(
                "default_overlap ({}) must be less than default_chunk_size ({})",
                self.default_overlap, self.default_chunk_size
            )));
        }

        if self.job_timeout_minutes == 0 || self.job_timeout_minutes > 1440 {
            return Err(NeuronCiteError::Config(format!(
                "job_timeout_minutes must be in 1..=1440, got {}",
                self.job_timeout_minutes
            )));
        }

        if self.db_pool_size == 0 || self.db_pool_size > 64 {
            return Err(NeuronCiteError::Config(format!(
                "db_pool_size must be in 1..=64, got {}",
                self.db_pool_size
            )));
        }

        Ok(())
    }

    /// Logs warnings for potentially dangerous configuration combinations.
    /// This method is intended to be called once at application startup,
    /// after the final configuration has been assembled (including any CLI
    /// overrides applied to `bind_address`).
    ///
    /// Checks performed:
    ///
    /// - **Unrestricted filesystem browsing on non-loopback bind**: When the
    ///   server binds to a non-loopback address (anything other than 127.0.0.1,
    ///   ::1, or localhost) and `allowed_browse_roots` is empty, the browse
    ///   and scan-pdfs endpoints permit access to the entire filesystem. This
    ///   is a security risk when the server is reachable from a network, because
    ///   any client can enumerate arbitrary directories and discover file names,
    ///   sizes, and modification times. The method emits a `warn!`-level log
    ///   message instructing the operator to populate `allowed_browse_roots`.
    ///
    /// - **Bearer token transmitted in plaintext over HTTP**: When a bearer
    ///   token is configured (the `bearer_token` field is set) and the server
    ///   binds to a non-loopback address, the authentication token is sent in
    ///   the HTTP `Authorization` header without TLS encryption. Any network
    ///   observer between the client and server can intercept the token. The
    ///   method emits a `warn!`-level log message recommending a reverse proxy
    ///   with TLS termination.
    pub fn check_security_warnings(&self) {
        if !is_loopback(&self.bind_address) && self.allowed_browse_roots.is_empty() {
            tracing::warn!(
                bind_address = %self.bind_address,
                "SECURITY: The server binds to a non-loopback address with an empty \
                 `allowed_browse_roots` list. The browse and scan-pdfs endpoints \
                 permit unrestricted filesystem access to any connected client. \
                 Set `allowed_browse_roots` in the configuration file to restrict \
                 browsing to specific directories."
            );
        }

        // The AppConfig struct does not carry the bearer token directly (it is
        // passed via CLI flag or environment variable and hashed at startup).
        // However, the presence of the token can be signaled by the caller
        // setting the bearer_token field. For this warning, we check a
        // combination: if the bind address is non-loopback, the server is
        // network-accessible and any bearer token configured externally would
        // be transmitted in plaintext HTTP headers.
        if !is_loopback(&self.bind_address) {
            tracing::warn!(
                bind_address = %self.bind_address,
                "SECURITY: The server binds to a non-loopback address. If bearer token \
                 authentication is configured, the token is transmitted in plaintext \
                 over HTTP in the Authorization header. Network observers can intercept \
                 this token. Deploy a reverse proxy with TLS termination (e.g., nginx, \
                 caddy) in front of this server to encrypt traffic."
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// Verifies that `AppConfig::default()` produces values matching the
    /// architecture specification.
    #[test]
    fn default_config_matches_specification() {
        let config = AppConfig::default();
        assert_eq!(config.port, 3030);
        assert_eq!(config.bind_address, "127.0.0.1");
        assert_eq!(config.log_level, LogLevel::Info);
        assert!(config.log_file.is_none());
        assert_eq!(config.default_model, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.default_strategy, ChunkStrategy::Token);
        assert_eq!(config.default_chunk_size, 256);
        assert_eq!(config.default_overlap, 32);
        assert_eq!(config.ocr_language, "eng");
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.rrf_k, 60);
        assert!(config.cors_origins.is_empty());
        assert_eq!(config.simhash_threshold, 3);
        assert!((config.orphan_ratio_threshold - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.job_timeout_minutes, 30);
        assert_eq!(config.defaults.top_k, 10);
    }

    /// Verifies that serializing `AppConfig` to TOML and deserializing it
    /// back produces an identical value.
    #[test]
    fn toml_roundtrip() {
        let config = AppConfig::default();
        let toml_string = toml::to_string(&config).expect("AppConfig serialization to TOML failed");
        let roundtripped = AppConfig::load_from_str(&toml_string)
            .expect("AppConfig deserialization from TOML failed");
        assert_eq!(config, roundtripped);
    }

    /// Verifies that `load_from_str` returns an error for invalid TOML content.
    #[test]
    fn invalid_toml_returns_error() {
        let result = AppConfig::load_from_str("this is not valid toml {{{{");
        assert!(result.is_err());
    }

    /// Verifies that the LogLevel enum serializes to lowercase strings and
    /// Display produces the same representation.
    #[test]
    fn log_level_display_matches_serde() {
        assert_eq!(LogLevel::Trace.to_string(), "trace");
        assert_eq!(LogLevel::Debug.to_string(), "debug");
        assert_eq!(LogLevel::Info.to_string(), "info");
        assert_eq!(LogLevel::Warn.to_string(), "warn");
        assert_eq!(LogLevel::Error.to_string(), "error");
    }

    /// Verifies that the ChunkStrategy enum serializes to lowercase strings
    /// and Display produces the same representation.
    #[test]
    fn chunk_strategy_display_matches_serde() {
        assert_eq!(ChunkStrategy::Token.to_string(), "token");
        assert_eq!(ChunkStrategy::Word.to_string(), "word");
        assert_eq!(ChunkStrategy::Sentence.to_string(), "sentence");
        assert_eq!(ChunkStrategy::Page.to_string(), "page");
    }

    /// Verifies that ChunkStrategy::from_str parses all valid lowercase
    /// strategy names and rejects unrecognized strings.
    #[test]
    fn chunk_strategy_from_str_roundtrip() {
        assert_eq!("token".parse::<ChunkStrategy>(), Ok(ChunkStrategy::Token));
        assert_eq!("word".parse::<ChunkStrategy>(), Ok(ChunkStrategy::Word));
        assert_eq!(
            "sentence".parse::<ChunkStrategy>(),
            Ok(ChunkStrategy::Sentence)
        );
        assert_eq!("page".parse::<ChunkStrategy>(), Ok(ChunkStrategy::Page));
        assert!("unknown".parse::<ChunkStrategy>().is_err());
        assert!("TOKEN".parse::<ChunkStrategy>().is_err());
    }

    /// Verifies that `is_loopback` correctly classifies loopback and
    /// non-loopback addresses. This is the function used by
    /// `check_security_warnings()` to detect potentially dangerous
    /// configurations.
    #[test]
    fn is_loopback_classification() {
        // Loopback addresses: the server is only reachable from the same machine.
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("::1"));
        assert!(is_loopback("localhost"));

        // Non-loopback addresses: the server is reachable from the network.
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("192.168.1.100"));
        assert!(!is_loopback("::"));
        assert!(!is_loopback("10.0.0.1"));
    }

    /// Verifies that `check_security_warnings()` does not panic under any
    /// configuration combination. The actual warning output goes to the
    /// tracing subscriber and cannot be captured in a unit test, but the
    /// method must execute without errors for all input combinations.
    #[test]
    fn check_security_warnings_does_not_panic() {
        // Default config: loopback + empty roots.
        let config = AppConfig::default();
        config.check_security_warnings();

        // Non-loopback + empty roots: the dangerous combination that triggers
        // the security warning.
        let mut config_lan = AppConfig::default();
        config_lan.bind_address = "0.0.0.0".into();
        config_lan.check_security_warnings();

        // Non-loopback + populated roots: safe configuration.
        let mut config_safe = AppConfig::default();
        config_safe.bind_address = "0.0.0.0".into();
        config_safe.allowed_browse_roots = vec!["/data/pdfs".into()];
        config_safe.check_security_warnings();

        // Loopback + populated roots: redundant but valid.
        let mut config_local = AppConfig::default();
        config_local.allowed_browse_roots = vec!["/home/user/docs".into()];
        config_local.check_security_warnings();
    }

    // -----------------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------------

    /// The default AppConfig passes validation without errors because all
    /// built-in default values are within their valid ranges.
    #[test]
    fn t_cfg_val_001_default_config_passes_validation() {
        let config = AppConfig::default();
        assert!(
            config.validate().is_ok(),
            "default config must pass validation"
        );
    }

    /// A port value of 0 is rejected because it means "OS-assigned" and
    /// breaks predictable binding. The u16 type already prevents values
    /// above 65535.
    #[test]
    fn t_cfg_val_002_port_zero_rejected() {
        let mut config = AppConfig::default();
        config.port = 0;
        let result = config.validate();
        assert!(result.is_err(), "port 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("port"),
            "error message must mention 'port', got: {msg}"
        );
    }

    /// ef_search = 0 is rejected because it produces no results from the
    /// HNSW graph traversal.
    #[test]
    fn t_cfg_val_003_ef_search_zero_rejected() {
        let mut config = AppConfig::default();
        config.ef_search = 0;
        let result = config.validate();
        assert!(result.is_err(), "ef_search 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ef_search"),
            "error message must mention 'ef_search', got: {msg}"
        );
    }

    /// rrf_k = 0 is rejected because it causes division by zero in the RRF
    /// formula: score = 1 / (k + rank).
    #[test]
    fn t_cfg_val_004_rrf_k_zero_rejected() {
        let mut config = AppConfig::default();
        config.rrf_k = 0;
        let result = config.validate();
        assert!(result.is_err(), "rrf_k 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("rrf_k"),
            "error message must mention 'rrf_k', got: {msg}"
        );
    }

    /// simhash_threshold > 64 is rejected because SimHash operates on 64-bit
    /// fingerprints and a threshold above 64 would match every pair.
    #[test]
    fn t_cfg_val_005_simhash_threshold_above_64_rejected() {
        let mut config = AppConfig::default();
        config.simhash_threshold = 65;
        let result = config.validate();
        assert!(result.is_err(), "simhash_threshold 65 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("simhash_threshold"),
            "error message must mention 'simhash_threshold', got: {msg}"
        );
    }

    /// simhash_threshold = 64 is the maximum valid value (all bits differ).
    #[test]
    fn t_cfg_val_006_simhash_threshold_64_accepted() {
        let mut config = AppConfig::default();
        config.simhash_threshold = 64;
        assert!(
            config.validate().is_ok(),
            "simhash_threshold 64 must be accepted"
        );
    }

    /// orphan_ratio_threshold below 0.0 is rejected because it represents
    /// a ratio that must be non-negative.
    #[test]
    fn t_cfg_val_007_orphan_ratio_negative_rejected() {
        let mut config = AppConfig::default();
        config.orphan_ratio_threshold = -0.1;
        let result = config.validate();
        assert!(
            result.is_err(),
            "negative orphan_ratio_threshold must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("orphan_ratio_threshold"),
            "error message must mention 'orphan_ratio_threshold', got: {msg}"
        );
    }

    /// orphan_ratio_threshold above 1.0 is rejected because it represents
    /// a ratio of deleted slots to total slots.
    #[test]
    fn t_cfg_val_008_orphan_ratio_above_one_rejected() {
        let mut config = AppConfig::default();
        config.orphan_ratio_threshold = 1.01;
        let result = config.validate();
        assert!(
            result.is_err(),
            "orphan_ratio_threshold > 1.0 must be rejected"
        );
    }

    /// orphan_ratio_threshold = 1.0 is the maximum valid value (100% orphan ratio).
    #[test]
    fn t_cfg_val_009_orphan_ratio_one_accepted() {
        let mut config = AppConfig::default();
        config.orphan_ratio_threshold = 1.0;
        assert!(
            config.validate().is_ok(),
            "orphan_ratio_threshold 1.0 must be accepted"
        );
    }

    /// orphan_ratio_threshold = 0.0 is the minimum valid value (0% orphan ratio).
    #[test]
    fn t_cfg_val_010_orphan_ratio_zero_accepted() {
        let mut config = AppConfig::default();
        config.orphan_ratio_threshold = 0.0;
        assert!(
            config.validate().is_ok(),
            "orphan_ratio_threshold 0.0 must be accepted"
        );
    }

    /// default_chunk_size = 0 is rejected because a zero chunk size produces
    /// no text in each chunk.
    #[test]
    fn t_cfg_val_011_chunk_size_zero_rejected() {
        let mut config = AppConfig::default();
        config.default_chunk_size = 0;
        let result = config.validate();
        assert!(result.is_err(), "default_chunk_size 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("default_chunk_size"),
            "error message must mention 'default_chunk_size', got: {msg}"
        );
    }

    /// default_overlap >= default_chunk_size is rejected because an overlap
    /// equal to or larger than the chunk size would produce infinite loops
    /// or empty chunks in the chunking pipeline.
    #[test]
    fn t_cfg_val_012_overlap_equals_chunk_size_rejected() {
        let mut config = AppConfig::default();
        config.default_chunk_size = 100;
        config.default_overlap = 100;
        let result = config.validate();
        assert!(
            result.is_err(),
            "default_overlap == default_chunk_size must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("default_overlap"),
            "error message must mention 'default_overlap', got: {msg}"
        );
    }

    /// default_overlap > default_chunk_size is rejected.
    #[test]
    fn t_cfg_val_013_overlap_exceeds_chunk_size_rejected() {
        let mut config = AppConfig::default();
        config.default_chunk_size = 100;
        config.default_overlap = 150;
        let result = config.validate();
        assert!(
            result.is_err(),
            "default_overlap > default_chunk_size must be rejected"
        );
    }

    /// default_overlap = 0 with a positive chunk size is a valid configuration
    /// (non-overlapping chunks).
    #[test]
    fn t_cfg_val_014_zero_overlap_accepted() {
        let mut config = AppConfig::default();
        config.default_chunk_size = 256;
        config.default_overlap = 0;
        assert!(
            config.validate().is_ok(),
            "zero overlap with positive chunk size must be accepted"
        );
    }

    /// Validation is called automatically by load_from_str. A TOML string
    /// with an invalid field value produces a Config error even though the
    /// TOML is syntactically valid.
    #[test]
    fn t_cfg_val_015_load_from_str_calls_validate() {
        let mut config = AppConfig::default();
        config.ef_search = 0;
        let toml_string =
            toml::to_string(&config).expect("serialization of invalid config must succeed");
        let result = AppConfig::load_from_str(&toml_string);
        assert!(
            result.is_err(),
            "load_from_str must reject config with ef_search = 0"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ef_search"),
            "error from load_from_str must mention 'ef_search', got: {msg}"
        );
    }

    /// Valid port values at the boundaries (1 and 65535) pass validation.
    #[test]
    fn t_cfg_val_016_port_boundary_values_accepted() {
        let mut config = AppConfig::default();
        config.port = 1;
        assert!(config.validate().is_ok(), "port 1 must be accepted");

        config.port = 65535;
        assert!(config.validate().is_ok(), "port 65535 must be accepted");
    }

    /// job_timeout_minutes = 0 is rejected because it causes immediate timeout
    /// of every job.
    #[test]
    fn t_cfg_val_017_job_timeout_zero_rejected() {
        let mut config = AppConfig::default();
        config.job_timeout_minutes = 0;
        let result = config.validate();
        assert!(result.is_err(), "job_timeout_minutes 0 must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("job_timeout_minutes"),
            "error message must mention 'job_timeout_minutes', got: {msg}"
        );
    }

    /// job_timeout_minutes = 1441 (over 24 hours) is rejected.
    #[test]
    fn t_cfg_val_018_job_timeout_exceeds_max_rejected() {
        let mut config = AppConfig::default();
        config.job_timeout_minutes = 1441;
        let result = config.validate();
        assert!(result.is_err(), "job_timeout_minutes 1441 must be rejected");
    }

    /// job_timeout_minutes boundary values (1 and 1440) pass validation.
    #[test]
    fn t_cfg_val_019_job_timeout_boundary_values_accepted() {
        let mut config = AppConfig::default();
        config.job_timeout_minutes = 1;
        assert!(
            config.validate().is_ok(),
            "job_timeout_minutes 1 must be accepted"
        );

        config.job_timeout_minutes = 1440;
        assert!(
            config.validate().is_ok(),
            "job_timeout_minutes 1440 must be accepted"
        );
    }
}
