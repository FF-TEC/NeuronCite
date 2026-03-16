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

//! Background job executor re-exported from neuroncite-pipeline.
//!
//! The executor logic lives in `neuroncite-pipeline::executor`. This module
//! provides thin wrapper functions that accept `Arc<AppState>` and coerce it
//! to `Arc<dyn PipelineContext>` before delegating to the pipeline crate.
//! The coercion is valid because `AppState` implements `PipelineContext`.
//!
//! Public functions:
//!
//! - `spawn_job_executor` -- Spawns the background polling task.
//! - `load_all_session_hnsw` -- Loads HNSW indices for all sessions at startup.
//! - `ensure_hnsw_for_session` -- Builds or retrieves the HNSW for one session.
//! - `ensure_hnsw_for_session_async` -- Async wrapper around the above.

use std::sync::Arc;

use neuroncite_pipeline::context::PipelineContext;

use crate::state::AppState;

// Type alias matching the one in neuroncite-pipeline::executor.
// Arc<AppState> coerces to this via the PipelineContext impl on AppState.
type State = Arc<dyn PipelineContext>;

/// Spawns the background job executor task.
///
/// Delegates to `neuroncite_pipeline::executor::spawn_job_executor`. The
/// `Arc<AppState>` is generic over `PipelineContext` so the coercion is
/// handled by the generic bound in the pipeline crate.
pub fn spawn_job_executor(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    neuroncite_pipeline::executor::spawn_job_executor(state)
}

/// Loads HNSW indices for all sessions present in the database.
///
/// Delegates to `neuroncite_pipeline::executor::load_all_session_hnsw`.
/// `Arc<AppState>` is coerced to `Arc<dyn PipelineContext>` because
/// `AppState` implements `PipelineContext`.
pub fn load_all_session_hnsw(state: &Arc<AppState>) {
    let coerced: State = state.clone();
    neuroncite_pipeline::executor::load_all_session_hnsw(&coerced);
}

/// Builds or retrieves the HNSW index for a single session.
///
/// Returns `true` if the index is available (either already loaded or
/// successfully built), `false` if there are no embeddings for the session.
pub fn ensure_hnsw_for_session(state: &Arc<AppState>, session_id: i64) -> bool {
    let coerced: State = state.clone();
    neuroncite_pipeline::executor::ensure_hnsw_for_session(&coerced, session_id)
}

/// Async wrapper around `ensure_hnsw_for_session`.
///
/// Runs the synchronous HNSW build on a blocking thread pool via
/// `tokio::task::spawn_blocking` so that the calling async task is not stalled.
pub async fn ensure_hnsw_for_session_async(state: &Arc<AppState>, session_id: i64) -> bool {
    let coerced: State = state.clone();
    neuroncite_pipeline::executor::ensure_hnsw_for_session_async(&coerced, session_id).await
}
