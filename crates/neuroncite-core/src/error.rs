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

// Error types covering all failure categories across the NeuronCite application.
// Each variant corresponds to a distinct subsystem that can produce errors,
// allowing callers to match on the error source without inspecting message strings.

use thiserror::Error;

/// Unified error type for the `NeuronCite` application.
///
/// Each variant represents a failure originating from a specific subsystem.
/// The `#[from]` attribute on `Io` enables automatic conversion from
/// `std::io::Error` via the `?` operator. All other variants carry a
/// human-readable message describing the failure context.
///
/// # Design: String-based variants instead of `#[source]`
///
/// Most variants wrap `String` rather than a nested `#[source]` error type.
/// This is intentional: `neuroncite-core` is the leaf crate imported by every
/// other crate in the workspace. It cannot depend on downstream crates like
/// `neuroncite-pdf`, `neuroncite-embed`, or `neuroncite-store` without creating
/// circular dependencies. The `From<DownstreamError> for NeuronCiteError`
/// conversions live in the downstream crate that owns the source error type,
/// and they format the error chain into the String before crossing the boundary.
/// This preserves the full error message while keeping the dependency graph
/// acyclic. The `Io` variant is the exception because `std::io::Error` is from
/// the standard library, which has no dependency ordering constraints.
///
/// # Examples
///
/// ```
/// use neuroncite_core::error::NeuronCiteError;
///
/// let err = NeuronCiteError::Pdf("failed to parse page 3".into());
/// assert!(!err.to_string().is_empty());
/// ```
#[derive(Debug, Error)]
pub enum NeuronCiteError {
    /// Failure during PDF text extraction (any backend).
    #[error("pdf extraction error: {0}")]
    Pdf(String),

    /// Failure during text chunking.
    #[error("chunking error: {0}")]
    Chunk(String),

    /// Failure during embedding computation or model loading.
    #[error("embedding error: {0}")]
    Embed(String),

    /// Failure in the storage layer (`SQLite`, HNSW file, or connection pool).
    #[error("storage error: {0}")]
    Store(String),

    /// Failure during search or ranking operations.
    #[error("search error: {0}")]
    Search(String),

    /// Failure in the REST API layer (request handling, serialization).
    #[error("api error: {0}")]
    Api(String),

    /// Failure in the GUI layer (eframe, egui rendering).
    #[error("gui error: {0}")]
    Gui(String),

    /// Failure during configuration parsing or validation.
    #[error("configuration error: {0}")]
    Config(String),

    /// Underlying I/O error from the standard library.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Caller supplied an invalid argument value.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl NeuronCiteError {
    /// Returns the subsystem name that produced this error.
    /// Useful for structured logging and error categorization.
    #[must_use]
    pub fn subsystem(&self) -> &'static str {
        match self {
            Self::Pdf(_) => "pdf",
            Self::Chunk(_) => "chunk",
            Self::Embed(_) => "embed",
            Self::Store(_) => "store",
            Self::Search(_) => "search",
            Self::Api(_) => "api",
            Self::Gui(_) => "gui",
            Self::Config(_) => "config",
            Self::Io(_) => "io",
            Self::InvalidArgument(_) => "invalid_argument",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CORE-007: Every variant of `NeuronCiteError` produces a non-empty
    /// Display string that does not contain the variant name verbatim (the
    /// message is human-readable, not a raw enum variant name).
    #[test]
    fn t_core_007_error_display_coverage() {
        let variants: Vec<NeuronCiteError> = vec![
            NeuronCiteError::Pdf("page parse failed".into()),
            NeuronCiteError::Chunk("window too small".into()),
            NeuronCiteError::Embed("model file missing".into()),
            NeuronCiteError::Store("database locked".into()),
            NeuronCiteError::Search("index not loaded".into()),
            NeuronCiteError::Api("route not found".into()),
            NeuronCiteError::Gui("render failed".into()),
            NeuronCiteError::Config("invalid port".into()),
            NeuronCiteError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file missing",
            )),
            NeuronCiteError::InvalidArgument("negative value".into()),
        ];

        for variant in &variants {
            let display = format!("{variant}");
            // Display string must be non-empty
            assert!(
                !display.is_empty(),
                "Display string is empty for {:?}",
                variant
            );
            // Display string must not contain the raw variant name like "Pdf(...)"
            // or "Chunk(...)"; it should be a human-readable sentence fragment.
            assert!(
                !display.starts_with("Pdf("),
                "Display contains raw variant name: {display}"
            );
            assert!(
                !display.starts_with("Chunk("),
                "Display contains raw variant name: {display}"
            );
            assert!(
                !display.starts_with("Io("),
                "Display contains raw variant name: {display}"
            );
        }
    }
}
