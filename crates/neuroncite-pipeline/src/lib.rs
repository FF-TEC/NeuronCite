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

//! neuroncite-pipeline: Background job executor, GPU worker, and indexing pipeline.
//!
//! This crate contains the compute-intensive pipeline that is decoupled from
//! the REST API surface of neuroncite-api. Separating these modules reduces
//! compilation time for API handler changes and enables independent testing
//! of the pipeline via the `PipelineContext` trait.
//!
//! # Modules
//!
//! - `context` -- `PipelineContext` trait definition. Implementors provide
//!   access to the database pool, GPU worker, HNSW index map, and SSE channels.
//! - `worker` -- `WorkerHandle` and GPU worker thread management.
//! - `indexer` -- PDF extraction, chunking, embedding storage pipeline.
//!
//! - `executor` -- Background job executor. Generic over `PipelineContext`
//!   via the `spawn_job_executor` and `ensure_hnsw_for_session` public functions.

pub mod context;
pub mod executor;
pub mod indexer;
pub mod worker;

// Re-exports for downstream crates that depend on neuroncite-pipeline.
pub use executor::{
    ensure_hnsw_for_session, ensure_hnsw_for_session_async, load_all_session_hnsw,
    spawn_job_executor,
};
pub use worker::{WorkerHandle, spawn_worker};

// Stub implementations used by tests across this crate (worker.rs, executor.rs).
// Compiled only for test builds.
#[cfg(test)]
pub(crate) mod test_support;
