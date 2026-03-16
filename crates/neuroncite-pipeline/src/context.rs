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

//! PipelineContext and focused sub-traits for pipeline executor access.
//!
//! Functions in the `executor` module are generic over `C: PipelineContext`
//! instead of taking `Arc<AppState>` directly. This decoupling allows tests
//! to provide a stub implementation without constructing the full server
//! state (database pool, GPU worker, HNSW indices, SSE channels).
//!
//! The single 12-method trait is split into four focused sub-traits:
//! - `DatabaseContext` — SQLite connection pool access.
//! - `ComputeContext` — GPU worker handle access.
//! - `IndexContext` — per-session HNSW index map operations.
//! - `EventContext` — SSE channels, cancellation token, semaphores.
//!
//! `PipelineContext` is the composed bound used by executor functions. It
//! requires all four sub-traits plus provides `config()`. Executor functions
//! that need only a subset of capabilities can bound on the narrower sub-trait
//! instead of the full `PipelineContext`.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use neuroncite_core::AppConfig;
use neuroncite_store::HnswIndex;

use crate::worker::WorkerHandle;

// ---------------------------------------------------------------------------
// DatabaseContext
// ---------------------------------------------------------------------------

/// Provides access to the SQLite connection pool.
///
/// Implemented by `AppState` and by test stubs that construct a minimal
/// pool without the full server stack.
pub trait DatabaseContext: Send + Sync + 'static {
    /// Returns the r2d2 SQLite connection pool. Callers obtain a connection
    /// for each request; the pool blocks if all connections are in use.
    fn pool(&self) -> &Pool<SqliteConnectionManager>;
}

// ---------------------------------------------------------------------------
// ComputeContext
// ---------------------------------------------------------------------------

/// Provides access to the GPU worker handle for embedding and reranking.
///
/// Implemented by `AppState` and by test stubs that construct a fake
/// `WorkerHandle` without loading an ONNX model.
pub trait ComputeContext: Send + Sync + 'static {
    /// Returns the handle to the background GPU worker thread. The executor
    /// submits embedding requests through this handle and awaits the results.
    fn worker_handle(&self) -> &WorkerHandle;
}

// ---------------------------------------------------------------------------
// IndexContext
// ---------------------------------------------------------------------------

/// Provides access to the per-session HNSW index map and mutation operations.
///
/// All mutations use `ArcSwap::rcu()` internally so that concurrent readers
/// always see a consistent snapshot. The trait exposes methods rather than
/// direct field access so that `AppState` can add instrumentation (metrics,
/// logging) without changing call sites.
pub trait IndexContext: Send + Sync + 'static {
    /// Returns the lock-free swappable per-session HNSW index map.
    fn hnsw_index(&self) -> &ArcSwap<HashMap<i64, Arc<HnswIndex>>>;

    /// Inserts a single HNSW index for the given session atomically via rcu().
    fn insert_hnsw(&self, session_id: i64, index: HnswIndex);

    /// Inserts multiple HNSW indices in a single atomic rcu() swap.
    fn insert_hnsw_many(&self, entries: Vec<(i64, HnswIndex)>);

    /// Removes the HNSW index for the given session atomically via rcu().
    fn remove_hnsw(&self, session_id: i64);

    /// Removes HNSW indices for multiple sessions in a single atomic rcu() swap.
    fn remove_hnsw_many(&self, session_ids: &[i64]);
}

// ---------------------------------------------------------------------------
// EventContext
// ---------------------------------------------------------------------------

/// Provides access to SSE channels, the cancellation token, the job-wake
/// notification, and the annotation semaphore.
///
/// Implemented by `AppState` and by test stubs that need only a subset of
/// event infrastructure (e.g. a cancellation token with no SSE channels).
pub trait EventContext: Send + Sync + 'static {
    /// Returns the SSE progress broadcast sender. In headless CLI mode there
    /// are no subscribers; sent messages are silently discarded by the
    /// tokio broadcast runtime when the buffer fills.
    fn progress_tx(&self) -> &broadcast::Sender<String>;

    /// Returns the cancellation token for graceful server shutdown.
    fn cancellation_token(&self) -> &CancellationToken;

    /// Returns the notification channel used to wake the job executor
    /// when a new job is submitted or an existing job is cancelled.
    fn job_notify(&self) -> &tokio::sync::Notify;

    /// Returns the semaphore that serializes annotation pipeline execution.
    /// Initialized with one permit so at most one annotation pipeline runs
    /// at any time. Callers acquire an owned permit before spawning the
    /// blocking pipeline and move it into the closure so it is released
    /// when the closure returns, even after a panic or early return.
    fn annotate_permit(&self) -> &Arc<tokio::sync::Semaphore>;
}

// ---------------------------------------------------------------------------
// PipelineContext (composed bound)
// ---------------------------------------------------------------------------

/// Composed trait used as the bound for all executor and worker functions.
///
/// Requires `DatabaseContext + ComputeContext + IndexContext + EventContext`
/// plus the `config()` method. The executor and indexer modules use
/// `Arc<dyn PipelineContext>` as a type-erased state handle. Test stubs
/// implement all four sub-traits individually before implementing this
/// composed trait.
pub trait PipelineContext:
    DatabaseContext + ComputeContext + IndexContext + EventContext + Send + Sync + 'static
{
    /// Returns the application configuration.
    fn config(&self) -> &AppConfig;
}
