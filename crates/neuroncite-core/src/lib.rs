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

//! neuroncite-core: Shared domain types, trait definitions, configuration
//! structures, the document-offset-to-page resolution utility, and error types.
//!
//! This crate has no internal workspace dependencies and serves as the shared
//! type vocabulary imported by every other crate in the workspace. All modules
//! are pub mod because downstream crates need direct access to the types,
//! traits, config structs, offset resolution function, and error variants
//! defined here.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod config;
pub mod disk;
pub mod error;
pub mod math;
pub mod offset;
pub mod paths;
pub mod time;
pub mod traits;
pub mod types;

// Re-export all public types, traits, config structs, and error variants
// at the crate root for ergonomic imports (e.g., `use neuroncite_core::Chunk`
// instead of `use neuroncite_core::types::Chunk`).

pub use config::{AppConfig, InputLimits, RequestDefaults};
pub use error::NeuronCiteError;
pub use math::cosine_similarity;
pub use offset::resolve_page;
pub use time::unix_timestamp_secs;
pub use traits::{ChunkStrategy, EmbeddingBackend, Reranker};
pub use types::{
    Chunk, Citation, EmbeddedChunk, EmbeddingModelConfig, ExtractionBackend, IndexConfig,
    IndexProgress, InferenceCapabilities, ModelInfo, ModelManifest, PageText, PoolingStrategy,
    SearchResult, SourceType, StorageMode,
};
