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

//! Error types for the neuroncite-chunk crate.
//!
//! `ChunkError` covers failures that can occur during text normalization,
//! sentence segmentation, or window construction. Most chunking operations are
//! infallible on valid UTF-8 input, so these errors primarily represent
//! configuration issues (e.g., zero-sized windows) or internal invariant
//! violations.

/// Represents all error conditions that can occur during text chunking.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    /// The caller provided an invalid configuration value (e.g., a zero-sized
    /// window or a missing required parameter for the chosen strategy).
    #[error("invalid chunk configuration: {0}")]
    InvalidConfig(String),

    /// The overlap value equals or exceeds the window size, which would cause
    /// the sliding window to never advance, producing infinite chunks.
    #[error("overlap ({overlap}) must be strictly less than window size ({window})")]
    OverlapExceedsWindow {
        /// The configured overlap value.
        overlap: usize,
        /// The configured window size.
        window: usize,
    },

    /// A tokenizer-related error occurred during token-window chunking. This
    /// variant wraps failures from the caller-provided tokenizer instance.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A text normalization error occurred during preprocessing. This covers
    /// failures in whitespace collapsing, hyphenation repair, or Unicode NFC
    /// normalization.
    #[error("normalization error: {0}")]
    Normalization(String),
}
