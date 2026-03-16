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

//! neuroncite-search: Hybrid search pipeline combining vector and keyword retrieval.
//!
//! This crate orchestrates the full search flow:
//!
//! 1. **Vector search** -- Queries the HNSW index for the K nearest neighbors
//!    of the query embedding vector.
//! 2. **Keyword search** -- Queries the FTS5 index for BM25-ranked keyword matches.
//! 3. **Fusion** -- Merges the two result sets using Reciprocal Rank Fusion (RRF)
//!    to produce a single ranked list.
//! 4. **Deduplication** -- Removes near-duplicate chunks (based on content hash
//!    or SimHash Hamming distance threshold) from the fused results.
//! 5. **Reranking** (optional) -- Passes the top-N fused results through a
//!    cross-encoder reranker model for fine-grained relevance scoring.
//! 6. **Citation assembly** -- Resolves each result chunk back to its source PDF
//!    file and page number, producing structured citation objects for the API
//!    response.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

mod citation;
mod dedup;
mod error;
mod fusion;
mod keyword;
mod pipeline;
pub mod refine;
mod vector;

pub use dedup::compute_simhash;
pub use error::SearchError;
pub use keyword::extract_query_terms;
pub use pipeline::{SearchConfig, SearchOutcome, SearchPipeline, relevance_label};
pub use refine::{
    CachedTokenizer, RefinementConfig, apply_refinement, generate_sub_chunks, parse_divisors,
};
