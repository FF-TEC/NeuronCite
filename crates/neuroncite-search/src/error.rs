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

//! Error types for the neuroncite-search crate.
//!
//! `SearchError` covers failures in the vector search, keyword search, fusion,
//! deduplication, reranking, and citation assembly stages of the search pipeline.

/// Represents all error conditions that can occur during search pipeline execution.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// The query embedding computation failed (e.g., the embedding backend returned
    /// an error when converting the query text to a vector).
    #[error("query embedding failed: {reason}")]
    QueryEmbedding {
        /// A human-readable description of the embedding failure.
        reason: String,
    },

    /// The HNSW vector index query failed (e.g., index is empty or corrupted).
    #[error("vector search failed: {reason}")]
    VectorSearch {
        /// A human-readable description of the vector search failure.
        reason: String,
    },

    /// The FTS5 keyword search query failed (e.g., invalid FTS5 syntax or
    /// database I/O error).
    #[error("keyword search failed: {reason}")]
    KeywordSearch {
        /// A human-readable description of the keyword search failure.
        reason: String,
    },

    /// A chunk metadata lookup failed during citation assembly or deduplication
    /// (e.g., the chunk ID returned by HNSW does not exist in the chunk table).
    #[error("chunk metadata lookup failed: {reason}")]
    ChunkLookup {
        /// A human-readable description of the lookup failure.
        reason: String,
    },

    /// The cross-encoder reranker failed during scoring.
    #[error("reranking failed: {reason}")]
    Reranking {
        /// A human-readable description of the reranking failure.
        reason: String,
    },

    /// A database query during citation assembly failed.
    #[error("citation assembly failed: {reason}")]
    CitationAssembly {
        /// A human-readable description of the citation failure.
        reason: String,
    },

    /// The search query is empty or otherwise invalid.
    #[error("invalid query: {reason}")]
    InvalidQuery {
        /// A human-readable description of the query validation failure.
        reason: String,
    },

    /// The sub-chunk refinement stage failed (e.g., tokenizer deserialization
    /// error or token encoding failure during sub-chunk generation).
    #[error("refinement failed: {reason}")]
    Refinement {
        /// A human-readable description of the refinement failure.
        reason: String,
    },
}
