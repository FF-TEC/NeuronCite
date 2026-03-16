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

//! Indexing pipeline re-exported from neuroncite-pipeline.
//!
//! All indexing pipeline logic (extraction, chunking, embedding, batch utilities,
//! stdout suppression) lives in the `neuroncite-pipeline` crate. This module
//! re-exports the complete public API so that intra-crate paths (e.g.
//! `crate::indexer::extract_and_chunk_file`) continue to resolve without
//! modification to callers inside neuroncite-api.
//!
//! Sub-module shims (`batch`, `pipeline`, `stdio`) expose the nested paths
//! such as `crate::indexer::batch::bytes_to_f32_vec` used in test modules.

pub mod batch;
pub mod pipeline;
pub mod stdio;

// Re-export the public pipeline data types defined in neuroncite-pipeline::indexer.
pub use neuroncite_pipeline::indexer::{ExtractedFile, ExtractionResult, FileIndexResult};

// Re-export batch utilities.
pub use neuroncite_pipeline::indexer::{
    bytes_to_f32_vec, compute_batch_size, f32_slice_to_bytes, length_sorted_permutation,
    restore_original_order,
};

// Re-export pipeline functions.
pub use neuroncite_pipeline::indexer::{
    embed_and_store_file, embed_and_store_file_async, extract_and_chunk_file, run_extraction_phase,
};

// Re-export stdio utilities.
pub use neuroncite_pipeline::indexer::{
    STDOUT_MCP_WRITER_INITIALIZED, StdoutSuppressionGuard, restore_stdout, suppress_stdout,
    writer_from_saved_fd,
};
