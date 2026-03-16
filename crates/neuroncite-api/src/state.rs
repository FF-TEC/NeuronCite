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

//! Shared application state for the axum server.
//!
//! `AppState` holds all shared resources needed by request handlers: the r2d2
//! database connection pool, the `ArcSwap`-wrapped per-session HNSW index map
//! for lock-free hot-reloading, the GPU worker handle, progress counters,
//! cancellation token, and configuration values. `AppState` is wrapped in `Arc`
//! and passed to axum as extension state.
//!
//! The 15 original fields are grouped into three composable sub-state structs
//! to improve unit testability. Each sub-state can be constructed independently
//! without bringing up the full server stack:
//!
//! - `AuthState` -- bearer token hash, per-IP failure counters, shutdown nonce.
//! - `IndexState` -- HNSW index map, embedding dimension, indexing progress.
//! - `SseChannels` -- three broadcast senders for progress, citation, and
//!   source-fetch SSE streams. All three are initialised unconditionally at
//!   construction time; in headless mode there are no subscribers and messages
//!   sent to these channels are silently discarded by the broadcast runtime.
//!
//! `AppState` embeds all three as public named fields (`auth`, `index`, `sse`).
//! The `Arc<AppState>` wrapper pattern used by axum handlers is unchanged.
//!
//! ## HNSW Index Lifecycle
//!
//! Each session has its own HNSW index built after indexing completes. The index
//! lifecycle has four stages:
//! 1. **Built**: the executor calls `AppState::insert_hnsw()` after building.
//! 2. **Active**: search handlers call `state.index.hnsw_index.load()` to read.
//! 3. **Reloaded**: `insert_hnsw()` uses `ArcSwap::rcu()` for atomic hot-swap.
//! 4. **Evicted**: `remove_hnsw()` is called on session delete.
//!
//! `ArcSwap::rcu()` provides atomic read-copy-update: readers always see either
//! the old or the new snapshot, never a torn intermediate state.
//!
//! ## Connection Pool Sizing
//!
//! The r2d2 connection pool is sized at `max(2, num_cpus)` to ensure that the
//! background executor and concurrent request handlers can both hold connections
//! without deadlocking. A minimum of 2 prevents the executor from starving handlers
//! when the pool has only one connection slot.

use std::collections::HashMap;
use std::fmt;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use arc_swap::ArcSwap;
use dashmap::{DashMap, DashSet};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use neuroncite_core::{AppConfig, IndexProgress};
use neuroncite_pipeline::context::{
    ComputeContext, DatabaseContext, EventContext, IndexContext, PipelineContext,
};
use neuroncite_store::HnswIndex;

use crate::middleware::auth::{GlobalFailureCounter, TimeIndex};
use crate::worker::WorkerHandle;

// ---------------------------------------------------------------------------
// ShutdownNonce
// ---------------------------------------------------------------------------

/// A one-time shutdown nonce that redacts its value in `Debug` and `Display`
/// output to prevent accidental exposure in log files or panic backtraces.
///
/// The raw string is only accessible via `as_str()`, which must be called
/// deliberately by code that needs to compare the nonce against a request body.
pub struct ShutdownNonce(String);

impl ShutdownNonce {
    /// Returns the raw nonce string for comparison with the request body.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ShutdownNonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShutdownNonce(<redacted>)")
    }
}

impl fmt::Display for ShutdownNonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<redacted>")
    }
}

// ---------------------------------------------------------------------------
// AuthState
// ---------------------------------------------------------------------------

/// Authentication and rate-limiting state grouped into a standalone struct.
///
/// Constructed by `AuthState::new()`. Independently constructable in tests
/// without a database pool, GPU worker, or HNSW index, which allows auth
/// middleware unit tests to build minimal state with just a token string.
#[derive(Debug)]
pub struct AuthState {
    /// SHA-256 hash of the bearer token for LAN authentication. The plaintext
    /// token is never stored in memory after construction -- only this hash is
    /// retained. The auth middleware hashes incoming tokens and performs a
    /// constant-time comparison against this value to prevent timing attacks.
    pub bearer_token_hash: Option<[u8; 32]>,

    /// Per-IP tracking of failed authentication attempts for rate limiting.
    /// Each entry maps a client IP address to the number of consecutive failures
    /// and the timestamp of the most recent failure. The auth middleware writes
    /// to this map on each failed attempt and the background eviction task
    /// periodically removes stale entries. `Arc` is required so both the
    /// middleware (via `AppState`) and the eviction task can share ownership
    /// across async tasks.
    pub failed_auth_attempts: Arc<DashMap<IpAddr, (u32, Instant)>>,

    /// Random 32-character hex nonce generated at server startup. The shutdown
    /// endpoint requires this nonce in the request body. Because the nonce is
    /// printed to stdout only at startup, an attacker who cannot observe the
    /// server's initial output cannot trigger a remote shutdown even when no
    /// bearer token is configured. Generated from a UUID v4 (which uses a
    /// cryptographically secure random source internally).
    ///
    /// The value is wrapped in `ShutdownNonce` so that debug output and log
    /// formatting never expose the raw nonce string.
    pub shutdown_nonce: ShutdownNonce,

    /// Server-wide counter of failed authentication attempts within the current
    /// tracking window. Shared between the auth middleware (which calls
    /// `record()` on each failure) and the eviction task (which reads the count
    /// to confirm the counter stays active). When the count exceeds
    /// `GLOBAL_RATE_LIMIT`, all failing requests receive an additional baseline
    /// delay regardless of per-IP state.
    pub global_failure_counter: Arc<GlobalFailureCounter>,

    /// Time-ordered index of per-IP failure timestamps. Maintained alongside
    /// `failed_auth_attempts` to allow the eviction task to identify expired
    /// entries in O(k log n) time rather than scanning the entire DashMap.
    pub time_index: Arc<TimeIndex>,
}

impl AuthState {
    /// Minimum number of characters required for a bearer token.
    /// A token shorter than this value is rejected at construction time
    /// rather than accepted with a warning. 32 characters provides at
    /// least 160 bits of entropy for a uniformly random alphanumeric
    /// token, which exceeds the OWASP minimum for bearer tokens.
    pub const MIN_TOKEN_LEN: usize = 32;

    /// Constructs a new `AuthState`.
    ///
    /// `bearer_token` is trimmed, validated for minimum length, and then
    /// hashed with SHA-256 before storage. Returns `Err` when a token
    /// shorter than `MIN_TOKEN_LEN` is provided so that mis-configured
    /// deployments fail at startup rather than silently accepting weak
    /// credentials. When `None`, the server runs without authentication.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` when `bearer_token` is `Some` but shorter
    /// than `MIN_TOKEN_LEN` after trimming.
    pub fn new(bearer_token: Option<String>) -> Result<Self, String> {
        // Hash the bearer token with SHA-256 so the plaintext is never stored
        // in process memory beyond this point. The auth middleware hashes
        // incoming tokens the same way before comparison.
        let bearer_token_hash = bearer_token
            .map(|token| {
                let token = token.trim().to_string();
                // Reject tokens shorter than MIN_TOKEN_LEN to prevent weak
                // authentication secrets from reaching production. Callers
                // should use `openssl rand -hex 32` to generate a token with
                // sufficient entropy.
                if token.len() < Self::MIN_TOKEN_LEN {
                    return Err(format!(
                        "bearer_token is too short ({} characters); minimum is {} characters; \
                         use `openssl rand -hex 32` to generate a secure token",
                        token.len(),
                        Self::MIN_TOKEN_LEN,
                    ));
                }
                let mut hasher = Sha256::new();
                hasher.update(token.as_bytes());
                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                Ok(hash)
            })
            .transpose()?;

        // Generate a one-time shutdown nonce from a UUID v4 source. The nonce
        // is printed to stdout at startup so the operator can issue a graceful
        // shutdown request. Storing only 32 hex chars (128 bits of entropy from
        // UUID v4) is sufficient to prevent brute-force guessing over a network.
        let shutdown_nonce = ShutdownNonce(uuid::Uuid::new_v4().as_simple().to_string());

        Ok(Self {
            bearer_token_hash,
            failed_auth_attempts: Arc::new(DashMap::new()),
            shutdown_nonce,
            global_failure_counter: Arc::new(GlobalFailureCounter::new()),
            time_index: Arc::new(TimeIndex::new()),
        })
    }
}

// ---------------------------------------------------------------------------
// IndexState
// ---------------------------------------------------------------------------

/// HNSW index map, embedding vector dimension, and indexing progress grouped
/// into a standalone struct.
///
/// Constructed by `IndexState::new()`. Can be built in tests with a fixed
/// vector dimension and no HNSW entries to verify index lifecycle methods.
pub struct IndexState {
    /// Lock-free swappable per-session HNSW index map. Each session ID maps to
    /// its own HNSW index wrapped in `Arc` for cheap cloning during atomic swap.
    /// Updated atomically when an index is built after indexing. Readers see
    /// either the old or the new map (never a torn read).
    pub hnsw_index: ArcSwap<HashMap<i64, Arc<HnswIndex>>>,

    /// Embedding vector dimensionality from the active backend. Stored as
    /// AtomicUsize so `activate_model` can update it when hot-swapping the
    /// embedding model at runtime. Used by the index handler to set the
    /// correct dimension on new sessions.
    pub vector_dimension: AtomicUsize,

    /// Tracks the progress of the current indexing operation.
    /// Protected by a tokio Mutex for async-safe interior mutability.
    pub progress: tokio::sync::Mutex<IndexProgress>,
}

impl IndexState {
    /// Constructs a new `IndexState` with an empty HNSW map and the given
    /// initial `vector_dimension`. Progress counters start at zero.
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            hnsw_index: ArcSwap::from_pointee(HashMap::new()),
            vector_dimension: AtomicUsize::new(vector_dimension),
            progress: tokio::sync::Mutex::new(IndexProgress {
                files_total: 0,
                files_done: 0,
                chunks_created: 0,
                complete: false,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// SseChannels
// ---------------------------------------------------------------------------

/// Broadcast channel buffer capacity for indexing progress events.
/// 64 slots accommodate bursty progress updates from the executor without
/// blocking the sender when no SSE clients are connected.
const PROGRESS_CHANNEL_CAPACITY: usize = 64;

/// Broadcast channel buffer capacity for citation verification events.
/// 256 slots handle the higher event rate from the citation agent (one
/// event per LLM reasoning token during streaming).
const CITATION_CHANNEL_CAPACITY: usize = 256;

/// Broadcast channel buffer capacity for source-fetch events.
/// 256 slots hold per-entry status updates for large bibliographies.
const SOURCE_CHANNEL_CAPACITY: usize = 256;

/// Three broadcast channel senders for SSE event distribution.
///
/// All three channels are created unconditionally in `SseChannels::new()`
/// so that `AppState` can be constructed without the web server being
/// started. In headless (CLI) mode there are no broadcast receivers, so
/// messages sent to these channels are silently discarded by the tokio
/// broadcast runtime when the internal buffer fills and there are no
/// subscribers. The `headless` flag on `AppState` is still available for
/// code paths that want to skip SSE work entirely.
pub struct SseChannels {
    /// Broadcast channel sender for indexing progress SSE events. The job
    /// executor sends `index_progress` and `index_complete` events here.
    /// In headless CLI mode there are no receivers; sent messages are dropped.
    pub progress_tx: broadcast::Sender<String>,

    /// Broadcast channel sender for citation verification SSE events. The
    /// citation agent sends `citation_row_update`, `citation_reasoning_token`,
    /// and `citation_job_progress` events here. In headless CLI mode there
    /// are no receivers; sent messages are dropped.
    pub citation_tx: broadcast::Sender<String>,

    /// Broadcast channel sender for source fetching SSE events. The
    /// fetch_sources handler sends per-entry status updates here so the
    /// frontend can display live download progress. In headless CLI mode
    /// there are no receivers; sent messages are dropped.
    pub source_tx: broadcast::Sender<String>,
}

impl SseChannels {
    /// Constructs a new `SseChannels` with all three broadcast channels
    /// pre-initialized. The initial receiver handles returned by
    /// `broadcast::channel()` are dropped immediately -- callers subscribe
    /// via `Sender::subscribe()` on demand (e.g., when an SSE connection
    /// is accepted). In headless mode no subscriber ever calls `subscribe()`,
    /// so sent messages are silently discarded.
    pub fn new() -> Self {
        let (progress_tx, _) = broadcast::channel(PROGRESS_CHANNEL_CAPACITY);
        let (citation_tx, _) = broadcast::channel(CITATION_CHANNEL_CAPACITY);
        let (source_tx, _) = broadcast::channel(SOURCE_CHANNEL_CAPACITY);
        Self {
            progress_tx,
            citation_tx,
            source_tx,
        }
    }
}

impl Default for SseChannels {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

/// Central application state shared across all axum handlers via `Arc<AppState>`.
///
/// Immutable fields (pool, config, worker_handle, headless) are set at startup
/// and never changed. Mutable state uses interior mutability: `ArcSwap` for
/// the HNSW index map, `AtomicUsize` for the vector dimension, `tokio::Mutex`
/// for indexing progress, and `DashMap` for per-IP auth counters. SSE broadcast
/// channels in `SseChannels` are initialised unconditionally at construction;
/// callers can send to them without checking availability.
///
/// The three sub-state structs (`auth`, `index`, `sse`) group logically related
/// fields. Direct field access (`state.auth.bearer_token_hash`,
/// `state.index.hnsw_index`, `state.sse.progress_tx`) is preferred over
/// forwarding methods to keep the call sites explicit about which sub-state
/// they are reading.
pub struct AppState {
    // -- Database access --
    /// Connection pool for the SQLite database. Handlers obtain a connection
    /// from this pool for each request.
    pub pool: Pool<SqliteConnectionManager>,

    // -- GPU compute --
    /// GPU worker handle for submitting embedding and reranking requests.
    pub worker_handle: WorkerHandle,

    // -- Job lifecycle --
    /// Cancellation token for graceful server shutdown. When canceled,
    /// the axum server stops accepting connections and in-flight
    /// requests are drained.
    pub cancellation_token: CancellationToken,

    /// Notification signal for the background job executor. When a job is
    /// created or canceled, callers notify this to wake the executor
    /// immediately instead of waiting for the next poll interval. This
    /// reduces latency between job submission and execution start from
    /// up to 500ms (POLL_INTERVAL) to near-zero.
    pub job_notify: tokio::sync::Notify,

    /// Semaphore that serializes annotation pipeline execution. Initialized
    /// with one permit so that at most one annotation pipeline runs at a
    /// time. Callers acquire an owned permit before spawning the blocking
    /// pipeline and move the permit into the closure so it is released
    /// when the closure returns -- even after a panic or an early return
    /// following the cancel grace period. This prevents a second annotation
    /// pipeline from loading pdfium (a process-global C library) while a
    /// previous pipeline's blocking thread is still running, which would
    /// cause a deadlock inside the C library's global state.
    pub annotate_permit: Arc<tokio::sync::Semaphore>,

    // -- Configuration --
    /// Application configuration loaded at startup.
    pub config: AppConfig,

    /// Whether the server is running in headless mode (CLI-only, no GUI).
    /// The shutdown endpoint is only available in headless mode.
    pub headless: bool,

    // -- Sub-states --
    /// Authentication and rate-limiting state: bearer token hash, per-IP
    /// failure counters, and one-time shutdown nonce.
    pub auth: AuthState,

    /// HNSW index map, embedding vector dimension, and indexing progress
    /// tracking state.
    pub index: IndexState,

    /// Three broadcast channel senders for SSE event streams
    /// (indexing progress, citation agent, source fetching). All channels
    /// are initialized at construction time; in headless mode there are no
    /// subscribers and sent messages are silently discarded.
    pub sse: SseChannels,

    /// Tracks job IDs with an actively running citation agent. The agent
    /// inserts its job_id at the start of run() and removes it when run()
    /// returns (success or error). The auto_verify handler checks this set
    /// before spawning a new agent to prevent duplicate agents competing
    /// for the same batches on the same job.
    pub active_citation_agents: DashSet<String>,
}

impl AppState {
    /// Constructs a new `AppState` with the given components.
    ///
    /// # Arguments
    ///
    /// * `pool` - r2d2 connection pool for SQLite.
    /// * `worker_handle` - Handle to the GPU worker thread.
    /// * `config` - Application configuration.
    /// * `headless` - Whether the server runs without a GUI.
    /// * `bearer_token` - Optional plaintext authentication token for LAN access.
    ///   Hashed with SHA-256 before storage; the plaintext is not retained.
    /// * `vector_dimension` - Initial embedding dimension from the active backend.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` when `bearer_token` is `Some` but shorter than
    /// `AuthState::MIN_TOKEN_LEN` characters after trimming.
    pub fn new(
        pool: Pool<SqliteConnectionManager>,
        worker_handle: WorkerHandle,
        config: AppConfig,
        headless: bool,
        bearer_token: Option<String>,
        vector_dimension: usize,
    ) -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self {
            pool,
            worker_handle,
            cancellation_token: CancellationToken::new(),
            job_notify: tokio::sync::Notify::new(),
            annotate_permit: Arc::new(tokio::sync::Semaphore::new(1)),
            config,
            headless,
            auth: AuthState::new(bearer_token)?,
            index: IndexState::new(vector_dimension),
            sse: SseChannels::new(),
            active_citation_agents: DashSet::new(),
        }))
    }

    /// Inserts an HNSW index for the given session into the per-session map.
    /// Uses `rcu()` (read-copy-update) to atomically read the current map,
    /// clone it with the new entry, and swap it in. If another thread modifies
    /// the map concurrently, `rcu()` retries the operation automatically,
    /// preventing lost updates that the previous load-clone-store pattern
    /// was susceptible to.
    pub fn insert_hnsw(&self, session_id: i64, index: HnswIndex) {
        let index = Arc::new(index);
        self.index.hnsw_index.rcu(|current| {
            let mut map = (**current).clone();
            map.insert(session_id, index.clone());
            Arc::new(map)
        });
    }

    /// Inserts multiple HNSW indices into the per-session map in a single
    /// atomic `rcu()` swap. Used during startup to load all persisted HNSW
    /// indices at once, avoiding N sequential `rcu()` calls (each of which
    /// clones the entire HashMap).
    pub fn insert_hnsw_many(&self, entries: Vec<(i64, HnswIndex)>) {
        if entries.is_empty() {
            return;
        }
        let arced: Vec<(i64, Arc<HnswIndex>)> = entries
            .into_iter()
            .map(|(id, idx)| (id, Arc::new(idx)))
            .collect();
        self.index.hnsw_index.rcu(|current| {
            let mut map = (**current).clone();
            for (session_id, index) in &arced {
                map.insert(*session_id, index.clone());
            }
            Arc::new(map)
        });
    }

    /// Removes the HNSW index for the given session from the per-session map.
    /// Uses `rcu()` for atomic read-copy-update. No-op if the session ID is
    /// not present (the closure still runs but returns an identical map, which
    /// is acceptable since removals are infrequent).
    pub fn remove_hnsw(&self, session_id: i64) {
        self.index.hnsw_index.rcu(|current| {
            let mut map = (**current).clone();
            map.remove(&session_id);
            Arc::new(map)
        });
    }

    /// Removes HNSW indices for multiple sessions from the per-session map.
    /// Uses `rcu()` for a single atomic swap.
    pub fn remove_hnsw_many(&self, session_ids: &[i64]) {
        if session_ids.is_empty() {
            return;
        }
        self.index.hnsw_index.rcu(|current| {
            let mut map = (**current).clone();
            for id in session_ids {
                map.remove(id);
            }
            Arc::new(map)
        });
    }
}

// ---------------------------------------------------------------------------
// PipelineContext sub-trait implementations for AppState
// ---------------------------------------------------------------------------

/// `DatabaseContext` implementation for `AppState`.
impl DatabaseContext for AppState {
    fn pool(&self) -> &Pool<SqliteConnectionManager> {
        &self.pool
    }
}

/// `ComputeContext` implementation for `AppState`.
impl ComputeContext for AppState {
    fn worker_handle(&self) -> &WorkerHandle {
        &self.worker_handle
    }
}

/// `IndexContext` implementation for `AppState`.
///
/// All mutations delegate to the `AppState` methods which use `rcu()` for
/// atomic read-copy-update without exclusive locking.
impl IndexContext for AppState {
    fn hnsw_index(&self) -> &ArcSwap<HashMap<i64, Arc<HnswIndex>>> {
        &self.index.hnsw_index
    }

    fn insert_hnsw(&self, session_id: i64, index: HnswIndex) {
        AppState::insert_hnsw(self, session_id, index);
    }

    fn insert_hnsw_many(&self, entries: Vec<(i64, HnswIndex)>) {
        AppState::insert_hnsw_many(self, entries);
    }

    fn remove_hnsw(&self, session_id: i64) {
        AppState::remove_hnsw(self, session_id);
    }

    fn remove_hnsw_many(&self, session_ids: &[i64]) {
        AppState::remove_hnsw_many(self, session_ids);
    }
}

/// `EventContext` implementation for `AppState`.
impl EventContext for AppState {
    fn progress_tx(&self) -> &broadcast::Sender<String> {
        &self.sse.progress_tx
    }

    fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }

    fn job_notify(&self) -> &tokio::sync::Notify {
        &self.job_notify
    }

    fn annotate_permit(&self) -> &Arc<tokio::sync::Semaphore> {
        &self.annotate_permit
    }
}

/// `PipelineContext` implementation for `AppState`.
///
/// Each method delegates directly to the corresponding field or sub-state.
/// The executor in neuroncite-pipeline calls these methods rather than
/// accessing `AppState` fields directly, which removes the direct dependency
/// of executor code on the concrete `AppState` type.
impl PipelineContext for AppState {
    fn config(&self) -> &AppConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use std::sync::atomic::Ordering;

    use super::{AuthState, IndexState, SseChannels};

    /// Verifies that `AuthState::new` hashes the bearer token with SHA-256.
    /// The hash stored in `bearer_token_hash` must equal SHA-256("testtoken").
    #[test]
    fn t_auth_state_hashes_token_on_construction() {
        let token = "testtoken_long_enough_to_skip_warning_aaa".to_string();
        let state = AuthState::new(Some(token.clone())).expect("construction must succeed");
        assert!(
            state.bearer_token_hash.is_some(),
            "bearer_token_hash must be Some when a token is provided"
        );
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(
            state.bearer_token_hash.unwrap(),
            expected,
            "stored hash must equal SHA-256 of the input token"
        );
    }

    /// Verifies that `AuthState::new(None)` produces `None` for the hash.
    /// The server must accept all requests when no bearer token is configured.
    #[test]
    fn t_auth_state_no_token_produces_none() {
        let state = AuthState::new(None).expect("construction without token must succeed");
        assert!(
            state.bearer_token_hash.is_none(),
            "bearer_token_hash must be None when no token is provided"
        );
    }

    /// Verifies that `AuthState::new` trims whitespace from the token before
    /// hashing. Tokens with surrounding spaces from config files must hash to
    /// the same value as the trimmed string.
    #[test]
    fn t_auth_state_trims_token_before_hashing() {
        let padded = "  mytoken_long_enough_aaaaaaaaaaaa  ".to_string();
        let trimmed = "mytoken_long_enough_aaaaaaaaaaaa".to_string();
        let state_padded = AuthState::new(Some(padded)).expect("padded token construction");
        let state_trimmed = AuthState::new(Some(trimmed)).expect("trimmed token construction");
        assert_eq!(
            state_padded.bearer_token_hash, state_trimmed.bearer_token_hash,
            "padded and trimmed tokens must produce the same hash"
        );
    }

    /// Verifies that `AuthState::new` rejects tokens shorter than MIN_TOKEN_LEN.
    /// Short tokens must not be silently accepted; the caller receives an Err.
    #[test]
    fn t_auth_state_rejects_short_token() {
        let short_token = "short".to_string();
        let result = AuthState::new(Some(short_token));
        assert!(
            result.is_err(),
            "a token shorter than MIN_TOKEN_LEN must be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("too short"),
            "error message must mention that the token is too short, got: {err}"
        );
    }

    /// Verifies that `AuthState::new` generates a shutdown nonce of exactly
    /// 32 hex characters (UUID v4 simple format: 32 lowercase hex digits).
    #[test]
    fn t_auth_state_shutdown_nonce_is_32_chars() {
        let state = AuthState::new(None).expect("construction without token must succeed");
        assert_eq!(
            state.shutdown_nonce.as_str().len(),
            32,
            "shutdown_nonce must be exactly 32 hex characters"
        );
        assert!(
            state
                .shutdown_nonce
                .as_str()
                .chars()
                .all(|c| c.is_ascii_hexdigit()),
            "shutdown_nonce must contain only ASCII hex digits"
        );
    }

    /// Verifies that the shutdown nonce is redacted when formatted with Debug
    /// or Display. The raw value must not appear in log output.
    #[test]
    fn t_auth_state_shutdown_nonce_is_redacted_in_debug() {
        let state = AuthState::new(None).expect("construction without token must succeed");
        let debug_str = format!("{:?}", state.shutdown_nonce);
        let display_str = format!("{}", state.shutdown_nonce);
        assert!(
            !debug_str.contains(state.shutdown_nonce.as_str()),
            "Debug output must not expose the raw nonce"
        );
        assert!(
            !display_str.contains(state.shutdown_nonce.as_str()),
            "Display output must not expose the raw nonce"
        );
        assert!(
            debug_str.contains("redacted"),
            "Debug output must contain the word 'redacted'"
        );
    }

    /// Verifies that `IndexState::new` stores the given vector dimension in
    /// the `AtomicUsize` with `Ordering::Relaxed` read semantics.
    #[test]
    fn t_index_state_initial_dimension() {
        let state = IndexState::new(384);
        assert_eq!(
            state
                .vector_dimension
                .load(std::sync::atomic::Ordering::Relaxed),
            384,
            "vector_dimension must store the value passed to new()"
        );
    }

    /// Verifies that `IndexState::new` starts with an empty HNSW map. No
    /// sessions are present in a freshly constructed state.
    #[test]
    fn t_index_state_hnsw_map_starts_empty() {
        let state = IndexState::new(128);
        let guard = state.hnsw_index.load();
        assert!(guard.is_empty(), "hnsw_index must be empty on construction");
    }

    /// Verifies that `SseChannels::new` initialises all three broadcast channels
    /// immediately. Subscribers must be creatable without any prior `set()` call.
    /// In headless mode the channels exist but hold zero subscribers; messages
    /// sent to them are silently discarded by the broadcast runtime.
    #[test]
    fn t_sse_channels_new_all_initialised() {
        let channels = SseChannels::new();
        // Each channel must accept a subscriber without error, proving the sender
        // was created and is not in a closed state.
        let _rx_progress = channels.progress_tx.subscribe();
        let _rx_citation = channels.citation_tx.subscribe();
        let _rx_source = channels.source_tx.subscribe();
        // The receiver count after subscribe() must be at least 1.
        assert_eq!(
            channels.progress_tx.receiver_count(),
            1,
            "progress_tx must have at least one subscriber after subscribe()"
        );
        assert_eq!(
            channels.citation_tx.receiver_count(),
            1,
            "citation_tx must have at least one subscriber after subscribe()"
        );
        assert_eq!(
            channels.source_tx.receiver_count(),
            1,
            "source_tx must have at least one subscriber after subscribe()"
        );
    }

    /// Verifies that two separate `AuthState` instances generate different
    /// shutdown nonces. Nonces are generated from UUID v4 random values;
    /// two calls must not produce the same string (collision probability ~2^-128).
    #[test]
    fn t_auth_state_shutdown_nonces_are_unique() {
        let a = AuthState::new(None).expect("construction without token must succeed");
        let b = AuthState::new(None).expect("construction without token must succeed");
        assert_ne!(
            a.shutdown_nonce.as_str(),
            b.shutdown_nonce.as_str(),
            "two AuthState instances must have distinct shutdown nonces"
        );
    }

    /// Verifies that `IndexState::new` initializes `IndexProgress` with all
    /// counters at zero and `complete` set to false.
    #[tokio::test]
    async fn t_index_state_progress_starts_zeroed() {
        let state = IndexState::new(0);
        let progress = state.progress.lock().await;
        assert_eq!(progress.files_total, 0);
        assert_eq!(progress.files_done, 0);
        assert_eq!(progress.chunks_created, 0);
        assert!(!progress.complete);
    }

    /// Verifies that `vector_dimension` can be updated atomically after construction.
    /// The `activate_model` handler uses `store(Ordering::Relaxed)` to update
    /// the dimension when hot-swapping the embedding model.
    #[test]
    fn t_index_state_vector_dimension_is_mutable() {
        let state = IndexState::new(384);
        state
            .vector_dimension
            .store(768, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            state.vector_dimension.load(Ordering::Relaxed),
            768,
            "vector_dimension must reflect the stored value after atomic write"
        );
    }
}
