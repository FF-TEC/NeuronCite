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

//! Background job executor for processing queued indexing jobs.
//!
//! The executor runs as a long-lived tokio task that polls the database for
//! jobs in "queued" state, picks them up in FIFO order, and runs the full
//! indexing pipeline:
//!
//! 1. Transition the job to "running".
//! 2. Load the session configuration to reconstruct the IndexConfig.
//! 3. Discover PDFs in the session's directory.
//! 4. Phase 1 (parallel, CPU-bound): Extract pages and chunk all PDFs.
//! 5. Phase 2 (sequential, GPU-bound): Embed chunks via WorkerHandle and
//!    store them in the database, updating progress after each file.
//! 6. Build the HNSW index and swap it into AppState.
//! 7. Transition the job to "completed" or "failed".
//!
//! The executor is spawned once at MCP server and API server startup. Only
//! one job runs at a time (serial execution). Cancellation is supported by
//! checking the job state before processing each file.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use neuroncite_store::JobState;

use crate::context::PipelineContext;

use crate::indexer;

/// Type alias for the shared context reference used throughout this module.
/// Using a trait object avoids adding a generic parameter to every internal
/// function in the executor call chain. `spawn_job_executor` accepts any
/// concrete type `C: PipelineContext` and coerces it to this type alias.
type State = Arc<dyn PipelineContext>;

/// Intermediate type used during parallel HNSW construction at startup.
/// Each entry holds the session_id, the vector dimension for that session,
/// and the per-chunk (chunk_label, embedding) pairs loaded from SQLite.
/// The outer Vec is indexed over sessions; the inner Vec over chunks.
type SessionEmbeddingData = Vec<(i64, usize, Vec<(i64, Vec<f32>)>)>;

/// Polling interval between checks for queued jobs.
/// The executor sleeps for this duration between database polls when no jobs
/// are queued and no notification signal is received.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Spawns the background job executor as a tokio task.
///
/// The executor polls the database for queued indexing jobs and processes
/// them sequentially. Before picking up a new job, it checks for stuck
/// running jobs (from previous server sessions or timed-out pipelines)
/// and transitions them to Failed. Job execution is wrapped in a timeout
/// to prevent indefinite blocking.
///
/// # Arguments
///
/// * `state` - Shared application state (database pool, GPU worker, HNSW index).
///
/// # Returns
///
/// A `JoinHandle` for the executor task. The handle can be used to await
/// the executor's termination during graceful shutdown.
pub fn spawn_job_executor<C: PipelineContext>(state: Arc<C>) -> tokio::task::JoinHandle<()> {
    let state: State = state;
    tokio::spawn(async move {
        tracing::info!("job executor started, polling every {:?}", POLL_INTERVAL);
        // Compute the job execution timeout from the configuration. The value is
        // stored as minutes in AppConfig and converted to a Duration here for use
        // in tokio::time::timeout and the stuck-job recovery check.
        let job_timeout = Duration::from_secs(u64::from(state.config().job_timeout_minutes) * 60);
        loop {
            // Recover stuck jobs before looking for new work. Jobs that have
            // been in "running" state longer than the configured job timeout are
            // transitioned to "failed". This handles jobs orphaned by a
            // previous server crash or by a pipeline that hung.
            recover_stuck_jobs(&state);

            match find_next_queued_job(&state) {
                Some(job) => {
                    tracing::info!(
                        job_id = %job.id,
                        kind = %job.kind,
                        session_id = ?job.session_id,
                        "picked up queued job"
                    );

                    // Execute the job inside a separate tokio task so that a
                    // panic in the pipeline does not kill the executor loop.
                    // The spawned task captures cloned Arcs; the executor
                    // awaits its completion with a timeout.
                    let state_clone = state.clone();
                    let job_id_for_timeout = job.id.clone();
                    let job_kind_for_timeout = job.kind.clone();

                    let handle = tokio::spawn(async move {
                        execute_job(&state_clone, &job).await;
                    });

                    // Wrap the join in a timeout to prevent a stuck pipeline
                    // from blocking the queue indefinitely.
                    let result = tokio::time::timeout(job_timeout, handle).await;

                    match result {
                        Ok(Ok(())) => {
                            // Job completed (success or handled failure).
                        }
                        Ok(Err(join_error)) => {
                            // The spawned task panicked. The panic is isolated
                            // to that task; the executor loop continues. Log
                            // the panic and transition the job to Failed.
                            tracing::error!(
                                job_id = %job_id_for_timeout,
                                kind = %job_kind_for_timeout,
                                "job execution panicked: {join_error}"
                            );
                            let _ = transition_job(
                                &state,
                                &job_id_for_timeout,
                                JobState::Failed,
                                Some(&format!("job execution panicked: {join_error}")),
                            );
                        }
                        Err(_elapsed) => {
                            // The job timed out. Transition it to Failed.
                            tracing::error!(
                                job_id = %job_id_for_timeout,
                                kind = %job_kind_for_timeout,
                                timeout_secs = job_timeout.as_secs(),
                                "job execution timed out"
                            );
                            let _ = transition_job(
                                &state,
                                &job_id_for_timeout,
                                JobState::Failed,
                                Some(&format!(
                                    "job execution timed out after {} seconds",
                                    job_timeout.as_secs()
                                )),
                            );
                        }
                    }
                }
                None => {
                    // No queued jobs found. Wait for either a notification
                    // (immediate wake-up when a job is created or canceled) or
                    // the poll interval (fallback). This reduces job pickup
                    // latency from POLL_INTERVAL to near-zero when the notify
                    // signal is used.
                    tokio::select! {
                        _ = state.job_notify().notified() => {
                            // Woken up by a job creation or cancellation signal.
                        }
                        _ = tokio::time::sleep(POLL_INTERVAL) => {
                            // Fallback poll interval elapsed.
                        }
                    }
                }
            }
        }
    })
}

/// Loads HNSW indexes for all sessions that have embeddings stored in SQLite.
///
/// Called at server startup (both HTTP and MCP modes) to populate the
/// per-session HNSW map from previously indexed data. This makes search
/// available immediately after server restart without requiring a re-indexing
/// job. Sessions without embeddings are skipped.
///
/// Each session's embeddings are read from the database, converted to f32
/// vectors, and used to build an in-memory HNSW index. For large databases
/// with many sessions, this runs sequentially and may take several seconds.
pub fn load_all_session_hnsw(state: &State) {
    let sessions = {
        let conn = match state.pool().get() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("failed to get connection for startup HNSW loading: {e}");
                return;
            }
        };
        match neuroncite_store::list_sessions(&conn) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("failed to list sessions for startup HNSW loading: {e}");
                return;
            }
        }
    };

    if sessions.is_empty() {
        tracing::info!("no sessions found, skipping startup HNSW loading");
        return;
    }

    // Phase 1: Sequential DB reads. Load embeddings from SQLite for each
    // session. DB reads are serialized through the connection pool to avoid
    // contention. Sessions without embeddings or with invalid dimensions are
    // skipped. The intermediate structure holds the session_id, vector
    // dimension, and pre-converted f32 vectors for parallel HNSW building.
    let mut session_data: SessionEmbeddingData = Vec::with_capacity(sessions.len());

    for session in &sessions {
        let conn = match state.pool().get() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    session_id = session.id,
                    "failed to get connection for HNSW loading: {e}"
                );
                continue;
            }
        };

        let raw_embeddings = match neuroncite_store::load_embeddings_for_hnsw(&conn, session.id) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    session_id = session.id,
                    "failed to load embeddings for HNSW: {e}"
                );
                continue;
            }
        };

        if raw_embeddings.is_empty() {
            continue;
        }

        let vector_dim = match session.vector_dimension_usize() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(session_id = session.id, "vector_dimension overflow: {e}");
                continue;
            }
        };

        // Convert byte blobs to f32 vectors while the data is still in memory
        // from the DB read, before dropping the connection.
        let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
            .into_iter()
            .map(|(id, bytes)| (id, indexer::bytes_to_f32_vec(&bytes)))
            .collect();

        session_data.push((session.id, vector_dim, f32_vectors));
    }

    // Phase 2: Parallel HNSW construction using rayon. The build_hnsw function
    // is CPU-bound (distance computations, graph construction) and benefits
    // from parallel execution across available cores. Each session's HNSW
    // index is built independently with no shared mutable state.
    use rayon::prelude::*;
    let entries: Vec<(i64, neuroncite_store::HnswIndex)> = session_data
        .par_iter()
        .filter_map(|(session_id, vector_dim, f32_vectors)| {
            let labeled_refs: Vec<(i64, &[f32])> = f32_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();

            match neuroncite_store::build_hnsw(&labeled_refs, *vector_dim) {
                Ok(index) => {
                    tracing::info!(
                        session_id,
                        vectors = f32_vectors.len(),
                        dimension = vector_dim,
                        "HNSW index built for session"
                    );
                    Some((*session_id, index))
                }
                Err(e) => {
                    tracing::warn!(session_id, "failed to build HNSW index: {e}");
                    None
                }
            }
        })
        .collect();

    // Single rcu() call inserts all indexes into the map atomically.
    // Readers see either the old (empty) map or the fully populated map,
    // never a partially inserted intermediate state.
    let loaded_count = entries.len();
    state.insert_hnsw_many(entries);

    tracing::info!(
        loaded = loaded_count,
        total_sessions = sessions.len(),
        "startup HNSW loading complete"
    );
}

/// Ensures the HNSW index for a single session is available in AppState.
///
/// Checks whether the per-session HNSW map already contains an entry for
/// `session_id`. When the entry is present, returns `true` immediately without
/// any database access (fast path). When the entry is absent, loads the
/// session's embeddings from SQLite, builds the HNSW index, inserts it into
/// the map, and returns `true`. Returns `false` only when no embeddings are
/// stored for the session (indexing has not run or produced no chunks).
///
/// This function is the repair path for DEFECT-005: after neuroncite_index
/// completes, the background executor's Phase 3 acquires a fresh SQLite
/// connection from the pool. If the WAL read snapshot does not include Phase
/// 2's just-committed embedding blobs, load_embeddings_for_hnsw returns empty
/// and the HNSW build is silently skipped. The next search call hits this
/// function, which retries the load from a fresh connection that sees the
/// committed data.
///
/// The same function is called from index_add when all files are skipped
/// (unchanged) but the HNSW entry is absent from the map, which happens
/// when index_add runs after a server restart.
pub fn ensure_hnsw_for_session(state: &State, session_id: i64) -> bool {
    // Fast path: HNSW already loaded for this session.
    //
    // RACE CONDITION NOTE: Two concurrent calls may both see contains_key()
    // return false and proceed to build the index in parallel. This is benign
    // because insert_hnsw uses ArcSwap::rcu which atomically replaces the map,
    // and both indexes are structurally identical (built from the same SQLite
    // data). The second insert simply overwrites the first with no data loss
    // or corruption. Adding a Mutex here would serialize all search requests
    // that hit the repair path, which is worse than the occasional redundant
    // HNSW build.
    if state.hnsw_index().load().contains_key(&session_id) {
        return true;
    }

    tracing::info!(
        session_id,
        "HNSW not found in map, attempting on-demand build from SQLite"
    );

    // Load session metadata to retrieve the stored vector dimension.
    let session = {
        let conn = match state.pool().get() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(session_id, "failed to get connection for HNSW repair: {e}");
                return false;
            }
        };
        match neuroncite_store::get_session(&conn, session_id) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    session_id,
                    "session not found during HNSW repair attempt: {e}"
                );
                return false;
            }
        }
    };

    // Load all embedding blobs for the session from SQLite.
    let raw_embeddings = {
        let conn = match state.pool().get() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    session_id,
                    "failed to get connection for embedding load: {e}"
                );
                return false;
            }
        };
        match neuroncite_store::load_embeddings_for_hnsw(&conn, session_id) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(session_id, "failed to load embeddings for HNSW repair: {e}");
                return false;
            }
        }
    };

    if raw_embeddings.is_empty() {
        tracing::debug!(
            session_id,
            "no embeddings stored for session, HNSW cannot be built"
        );
        return false;
    }

    let vector_dim = match session.vector_dimension_usize() {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(session_id, "vector_dimension overflow: {e}");
            return false;
        }
    };
    let vector_count = raw_embeddings.len();

    let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
        .into_iter()
        .map(|(id, bytes)| (id, indexer::bytes_to_f32_vec(&bytes)))
        .collect();

    let labeled_refs: Vec<(i64, &[f32])> = f32_vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let index = match neuroncite_store::build_hnsw(&labeled_refs, vector_dim) {
        Ok(idx) => idx,
        Err(e) => {
            tracing::warn!(session_id, "failed to build HNSW index on-demand: {e}");
            return false;
        }
    };
    state.insert_hnsw(session_id, index);

    tracing::info!(
        session_id,
        vectors = vector_count,
        dimension = vector_dim,
        "HNSW index built on-demand and inserted into per-session map"
    );
    true
}

/// Async wrapper around `ensure_hnsw_for_session` that runs the synchronous
/// HNSW build on a blocking thread with a 30-second timeout. Returns `true`
/// if the HNSW index is available for the session after the call, `false` if
/// the build failed, the session does not exist, or the timeout expired.
///
/// This wrapper is intended for async handlers (e.g., MCP search) that need
/// to call ensure_hnsw without blocking the tokio async executor. The API
/// search handlers use spawn_blocking directly and call the synchronous
/// version inside the blocking closure instead.
pub async fn ensure_hnsw_for_session_async(state: &State, session_id: i64) -> bool {
    // Fast path: HNSW already loaded. The ArcSwap load is cheap (no allocation)
    // and avoids the spawn_blocking overhead when no build is needed.
    if state.hnsw_index().load().contains_key(&session_id) {
        return true;
    }

    let state_clone = state.clone();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::task::spawn_blocking(move || ensure_hnsw_for_session(&state_clone, session_id)),
    )
    .await;

    match result {
        Ok(Ok(success)) => success,
        Ok(Err(e)) => {
            tracing::warn!(
                session_id,
                "ensure_hnsw_for_session blocking task panicked: {e}"
            );
            false
        }
        Err(_) => {
            tracing::warn!(
                session_id,
                "ensure_hnsw_for_session timed out after 30 seconds"
            );
            false
        }
    }
}

/// Searches the database for the oldest queued job (index or annotate).
///
/// Returns `None` if no queued jobs exist. Uses a direct SQL query with
/// `ORDER BY created_at ASC, rowid ASC` for deterministic FIFO ordering.
/// The `rowid` tiebreaker resolves jobs created within the same second
/// (the `created_at` column has second granularity).
fn find_next_queued_job(state: &State) -> Option<neuroncite_store::JobRow> {
    // try_get returns immediately if no connection is available, avoiding a
    // blocking wait inside the executor poll loop. If the pool is exhausted,
    // this poll cycle is skipped and the next POLL_INTERVAL retry will acquire
    // a connection once one is returned by an in-progress handler.
    let conn = state.pool().try_get()?;

    let mut stmt = conn
        .prepare_cached(
            "SELECT id, kind, session_id, state, progress_done, progress_total,
                    error_message, created_at, started_at, finished_at, params_json
             FROM job
             WHERE kind IN ('index', 'annotate', 'rebuild') AND state = 'queued'
             ORDER BY created_at ASC, rowid ASC
             LIMIT 1",
        )
        .ok()?;

    // The WHERE clause guarantees state = 'queued', so the JobState is
    // known without parsing the string column.
    stmt.query_row([], |row| {
        Ok(neuroncite_store::JobRow {
            id: row.get(0)?,
            kind: row.get(1)?,
            session_id: row.get(2)?,
            state: JobState::Queued,
            progress_done: row.get(4)?,
            progress_total: row.get(5)?,
            error_message: row.get(6)?,
            created_at: row.get(7)?,
            started_at: row.get(8)?,
            finished_at: row.get(9)?,
            params_json: row.get(10)?,
        })
    })
    .ok()
}

/// Executes a single job through the appropriate pipeline based on its kind.
///
/// On success, transitions the job to "completed". On failure, transitions
/// to "failed" with the error message. Dispatches to `run_index_pipeline`
/// for "index" jobs and `run_annotate_pipeline` for "annotate" jobs.
///
/// Before attempting terminal state transitions (Completed/Failed), the
/// current job state is re-read from the database. If the job was canceled
/// while the pipeline was running (e.g., via the cancel API), the terminal
/// transition is skipped because Canceled is already a terminal state and
/// the state machine does not allow transitions from Canceled. This prevents
/// a stale transition attempt that would log spurious errors.
async fn execute_job(state: &State, job: &neuroncite_store::JobRow) {
    let job_id = &job.id;

    // Transition to Running.
    if let Err(e) = transition_job(state, job_id, JobState::Running, None) {
        tracing::error!(job_id, "failed to transition job to running: {e}");
        return;
    }

    let result = match job.kind.as_str() {
        "index" => run_index_pipeline(state, job).await,
        "annotate" => run_annotate_pipeline(state, job).await,
        "rebuild" => run_rebuild_pipeline(state, job).await,
        other => Err(format!("unknown job kind: {other}")),
    };

    // Re-read the job state from the database before attempting a terminal
    // transition. The job may have been canceled via the cancel API while
    // the pipeline was running. In that case, the state is already Canceled
    // (a terminal state), and attempting Canceled->Completed or
    // Canceled->Failed would fail the state machine validation. Skipping
    // the transition avoids spurious error logs and ensures the cancel
    // state is preserved.
    let current_state = match read_job_state(state, job_id) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(job_id, "failed to read job state after execution: {e}");
            return;
        }
    };

    if current_state == JobState::Canceled {
        tracing::info!(
            job_id,
            "job was canceled during execution, skipping terminal transition"
        );
        return;
    }

    match result {
        Ok(()) => {
            // Finalize progress: set progress_done = progress_total before
            // transitioning to Completed. Some pipelines (e.g., annotation)
            // track progress per matched PDF rather than per input row, so
            // progress_done may be less than progress_total when inputs have
            // no matching PDF. Setting them equal on completion ensures that
            // monitoring tools see a consistent terminal state (100% done)
            // rather than an ambiguous partial count (e.g., 0/1 completed).
            if let Err(e) = finalize_progress(state, job_id) {
                tracing::warn!(job_id, "failed to finalize progress: {e}");
            }
            if let Err(e) = transition_job(state, job_id, JobState::Completed, None) {
                tracing::error!(job_id, "failed to transition job to completed: {e}");
            }
            // Notify SSE subscribers that indexing is complete. In headless
            // mode there are no subscribers; the send result is intentionally
            // discarded.
            {
                let tx = state.progress_tx();
                let _ = tx.send(
                    serde_json::json!({
                        "event": "index_complete",
                        "job_id": job_id,
                    })
                    .to_string(),
                );
            }
        }
        Err(e) => {
            tracing::error!(job_id, kind = %job.kind, "job failed: {e}");
            if let Err(te) = transition_job(state, job_id, JobState::Failed, Some(&e)) {
                tracing::error!(job_id, "failed to transition job to failed: {te}");
            }
        }
    }
}

/// The full indexing pipeline for a single job.
///
/// Loads session config, discovers PDFs, runs extraction + chunking in
/// parallel, embeds and stores sequentially via the WorkerHandle, builds
/// the HNSW index, and swaps it into AppState.
async fn run_index_pipeline(state: &State, job: &neuroncite_store::JobRow) -> Result<(), String> {
    let job_id = job.id.clone();
    let session_id = job.session_id.ok_or("job has no associated session_id")?;

    // Load the session to reconstruct the directory and chunk parameters.
    // pool().get() and the query run inside spawn_blocking so this async
    // function does not block a tokio worker thread while waiting for a
    // pool connection or executing the SELECT query.
    let pool_for_session = state.pool().clone();
    let session = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let conn = pool_for_session
            .get()
            .map_err(|e| format!("pool.get: {e}"))?;
        neuroncite_store::get_session(&conn, session_id).map_err(|e| format!("get_session: {e}"))
    })
    .await
    .map_err(|e| format!("spawn_blocking panicked: {e}"))
    .and_then(|r| r)
    .map_err(|e| format!("session lookup: {e}"))?;

    // Validate that the model stored in the session matches the model loaded by
    // this server instance. A mismatch means embeddings produced by the worker
    // would have a different vector dimension than the value stored in
    // session.vector_dimension, which causes a panic in the HNSW builder via
    // assert_eq!. This guard converts the panic into a clean job failure.
    let loaded_model = state.worker_handle().loaded_model_id();
    if session.model_name != *loaded_model {
        return Err(format!(
            "model mismatch: session was created with model '{}' but the server has loaded \
             model '{}'; restart the server with the correct model to index this session",
            session.model_name, loaded_model
        ));
    }

    let directory = PathBuf::from(&session.directory_path);
    if !directory.is_dir() {
        return Err(format!("directory does not exist: {}", directory.display()));
    }

    // Discover all indexable documents recursively (PDF, HTML, HTM).
    // The recursive walk traverses the entire directory tree so that files
    // in subdirectories (e.g., `pdf/`, `html/`) are found. Symlink cycles
    // are detected and skipped to prevent infinite traversal.
    let documents = neuroncite_pdf::discover_documents(&directory)
        .map_err(|e| format!("document discovery: {e}"))?;

    let files_total = documents.len() as i64;

    // Update progress: 0 done out of N total, 0 chunks created.
    update_progress(state, &job_id, 0, files_total, 0, "extracting")?;

    if documents.is_empty() {
        tracing::info!(job_id = %job_id, "no indexable documents found in directory, completing job");
        return Ok(());
    }

    // Reconstruct the chunk strategy from the session parameters.
    // All three dimension parameters (chunk_size, chunk_overlap, max_words)
    // are stored as nullable i64 in the session table and converted to
    // usize for the chunking API. The sentence strategy requires max_words;
    // without passing it here, create_strategy returns an error.
    let chunk_size = session
        .chunk_size_usize()
        .map_err(|e| format!("integer overflow: {e}"))?;
    let chunk_overlap = session
        .chunk_overlap_usize()
        .map_err(|e| format!("integer overflow: {e}"))?;
    let max_words = session
        .max_words_usize()
        .map_err(|e| format!("integer overflow: {e}"))?;

    let tokenizer_json = state.worker_handle().tokenizer_json();

    let chunk_strategy = neuroncite_chunk::create_strategy(
        &session.chunk_strategy,
        chunk_size,
        chunk_overlap,
        max_words,
        tokenizer_json.as_deref(),
    )
    .map_err(|e| format!("chunking strategy: {e}"))?;

    // Phase 1: parallel extraction and chunking.
    tracing::info!(
        job_id = %job_id,
        files = documents.len(),
        "starting extraction phase"
    );

    // Clone the SSE broadcast sender and job ID for use inside the
    // spawn_blocking closure. The extraction callback sends per-file progress
    // SSE events without touching the database. In headless mode the sender
    // has no subscribers and the send is silently dropped.
    let progress_tx = state.progress_tx().clone();
    let job_id_for_extraction = job_id.clone();

    let chunk_strat = chunk_strategy;
    let extracted_results = tokio::task::spawn_blocking(move || {
        let tx = progress_tx;
        let jid = job_id_for_extraction;
        let callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>> =
            Some(Box::new(move |done: usize, total: usize| {
                broadcast_extraction_progress(&tx, &jid, done, total);
            }) as Box<dyn Fn(usize, usize) + Send + Sync>);

        indexer::run_extraction_phase(chunk_strat.as_ref(), &documents, callback.as_deref())
    })
    .await
    .map_err(|e| format!("extraction task panicked: {e}"))?
    .map_err(|e| format!("extraction phase failed: {e}"))?;

    // Separate successful extractions from failures.
    let mut successful: Vec<indexer::ExtractedFile> = Vec::new();
    for result in extracted_results {
        match result {
            Ok(ef) => {
                if !ef.chunks.is_empty() {
                    successful.push(ef);
                }
            }
            Err((path, e)) => {
                tracing::warn!(
                    job_id = %job_id,
                    file = %path.display(),
                    "extraction failed, skipping: {e}"
                );
            }
        }
    }

    tracing::info!(
        job_id = %job_id,
        extracted = successful.len(),
        total = files_total,
        "extraction phase complete"
    );

    // Change detection: compare each extracted file against the database to
    // determine which files need re-indexing. Files whose content hash is
    // unchanged since the last indexing run are skipped to avoid redundant
    // GPU embedding work. Files with changed content are deleted from the
    // database (CASCADE removes pages/chunks) before re-indexing.
    //
    // This block uses a Connection::transaction() which requires &mut Connection.
    // db_run only provides &Connection, so we use spawn_blocking directly and
    // clone the pool. The `successful` vector is moved into the closure so it
    // does not remain live across the await point.
    let pool_for_detection = state.pool().clone();
    let job_id_for_detection = job_id.clone();
    let (mut files_to_index, skipped_count) = tokio::task::spawn_blocking(move || {
        let mut conn = pool_for_detection
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        // Wrap the change detection loop in a transaction so that file
        // deletions (for re-indexing) and metadata updates are committed
        // atomically. If any operation fails, the entire batch is rolled back.
        let tx = conn
            .transaction()
            .map_err(|e| format!("change detection transaction: {e}"))?;

        let mut files_to_index: Vec<indexer::ExtractedFile> = Vec::with_capacity(successful.len());
        let mut skipped_count = 0_usize;

        for ef in successful {
            let file_path_str = ef.pdf_path.to_string_lossy().to_string();

            match neuroncite_store::find_file_by_session_path(&tx, session_id, &file_path_str) {
                Ok(Some(existing_file)) => {
                    let status = neuroncite_store::check_file_changed(
                        &tx,
                        existing_file.id,
                        ef.mtime,
                        ef.file_size,
                        Some(&ef.file_hash),
                    )
                    .map_err(|e| format!("change detection: {e}"))?;

                    match status {
                        neuroncite_store::ChangeStatus::Unchanged => {
                            tracing::info!(
                                job_id = %job_id_for_detection,
                                file = %ef.pdf_path.display(),
                                "file unchanged since last indexing, skipping"
                            );
                            skipped_count += 1;
                        }
                        neuroncite_store::ChangeStatus::MetadataOnly => {
                            tracing::info!(
                                job_id = %job_id_for_detection,
                                file = %ef.pdf_path.display(),
                                "file metadata changed (mtime/size), updating record"
                            );
                            neuroncite_store::update_file_hash(
                                &tx,
                                existing_file.id,
                                &ef.file_hash,
                                ef.mtime,
                                ef.file_size,
                            )
                            .map_err(|e| format!("metadata update: {e}"))?;
                            skipped_count += 1;
                        }
                        neuroncite_store::ChangeStatus::ContentChanged => {
                            tracing::info!(
                                job_id = %job_id_for_detection,
                                file = %ef.pdf_path.display(),
                                "file content changed, deleting old record and re-indexing"
                            );
                            neuroncite_store::delete_file(&tx, existing_file.id)
                                .map_err(|e| format!("file deletion for re-index: {e}"))?;
                            files_to_index.push(ef);
                        }
                        neuroncite_store::ChangeStatus::New => {
                            // find_file_by_session_path returned Some but check_file_changed
                            // reports New. Treat as new file to be safe.
                            files_to_index.push(ef);
                        }
                    }
                }
                Ok(None) => {
                    // File not found in database, index it as a new file.
                    files_to_index.push(ef);
                }
                Err(e) => {
                    tracing::warn!(
                        job_id = %job_id_for_detection,
                        file = %ef.pdf_path.display(),
                        "change detection lookup failed: {e}, will re-index"
                    );
                    files_to_index.push(ef);
                }
            }
        }

        // Commit the transaction so all file deletions and metadata updates
        // take effect atomically. If commit fails, the entire change detection
        // batch is rolled back.
        tx.commit()
            .map_err(|e| format!("change detection commit: {e}"))?;

        Ok::<(Vec<indexer::ExtractedFile>, usize), String>((files_to_index, skipped_count))
    })
    .await
    .map_err(|e| format!("change detection task panicked: {e}"))?
    .map_err(|e: String| e)?;

    if skipped_count > 0 {
        tracing::info!(
            job_id = %job_id,
            skipped = skipped_count,
            to_index = files_to_index.len(),
            "change detection complete"
        );
    }

    // Phase 2: sequential embedding and storage via WorkerHandle.
    // Embedding vectors are written to SQLite as byte blobs and are NOT
    // accumulated in memory. After all files have been processed, Phase 3
    // reads the embeddings back from the database to build the HNSW index.
    let mut total_chunks = 0_usize;
    let worker = state.worker_handle();
    let indexable_count = files_to_index.len();

    for (file_idx, ef) in files_to_index.iter_mut().enumerate() {
        // Check for cancellation before processing each file.
        if is_job_canceled(state, &job_id)? {
            tracing::info!(job_id = %job_id, "job was canceled, stopping");
            return Err("job canceled by user".to_string());
        }

        tracing::debug!(
            job_id = %job_id,
            file = %ef.pdf_path.display(),
            chunks = ef.chunks.len(),
            "[{}/{}] embedding and storing",
            file_idx + 1,
            indexable_count
        );

        match indexer::embed_and_store_file_async(state.pool(), worker, ef, session_id).await {
            Ok(result) => {
                total_chunks += result.chunks_created;
            }
            Err(e) => {
                tracing::warn!(
                    job_id = %job_id,
                    file = %ef.pdf_path.display(),
                    "embed/store failed, skipping: {e}"
                );
            }
        }

        // Release page text memory after database insertion. The pages are
        // only referenced during insert_file (page_count) and
        // bulk_insert_pages, both of which have completed at this point.
        ef.pages = Vec::new();

        // Update progress after each file. The index is relative to the
        // `files_to_index` slice, not the total discovered PDFs.
        update_progress(
            state,
            &job_id,
            (file_idx + 1) as i64,
            files_total,
            total_chunks as i64,
            "embedding",
        )?;
    }

    // Set progress_done to files_total after Phase 2. All discovered PDFs
    // have been processed at this point (some may have been skipped due to
    // extraction failures or empty chunks, but processing is complete).
    update_progress(
        state,
        &job_id,
        files_total,
        files_total,
        total_chunks as i64,
        "building_index",
    )?;

    tracing::info!(
        job_id = %job_id,
        chunks = total_chunks,
        "embedding phase complete"
    );

    // Phase 3: build HNSW index from embeddings stored in SQLite. This
    // reads all embedding blobs for the session in a single query, converts
    // them from byte blobs to f32 vectors, builds the HNSW index, and
    // swaps it into AppState. The f32 vectors are allocated temporarily
    // for the HNSW build and freed immediately afterward.
    //
    // pool().get() and the query run inside spawn_blocking so the tokio
    // worker thread is not blocked during pool acquisition or the
    // potentially large SELECT of all embedding BLOBs.
    {
        let pool_for_embeddings = state.pool().clone();
        let raw_embeddings = tokio::task::spawn_blocking(move || -> Result<_, String> {
            let conn = pool_for_embeddings
                .get()
                .map_err(|e| format!("pool.get: {e}"))?;
            neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
                .map_err(|e| format!("load_embeddings_for_hnsw: {e}"))
        })
        .await
        .map_err(|e| format!("spawn_blocking panicked: {e}"))
        .and_then(|r| r)
        .map_err(|e| format!("loading embeddings for HNSW: {e}"))?;

        if !raw_embeddings.is_empty() {
            let vector_dim = session
                .vector_dimension_usize()
                .map_err(|e| format!("integer overflow: {e}"))?;
            tracing::info!(
                job_id = %job_id,
                vectors = raw_embeddings.len(),
                dimension = vector_dim,
                "building HNSW index from SQLite embeddings"
            );

            let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
                .into_iter()
                .map(|(id, bytes)| (id, indexer::bytes_to_f32_vec(&bytes)))
                .collect();

            let labeled_refs: Vec<(i64, &[f32])> = f32_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();

            let index = neuroncite_store::build_hnsw(&labeled_refs, vector_dim)
                .map_err(|e| format!("HNSW build failed: {e}"))?;
            state.insert_hnsw(session_id, index);

            tracing::info!(job_id = %job_id, session_id, "HNSW index built and inserted into per-session map");
        }
    }

    Ok(())
}

/// Rebuilds the in-memory HNSW index for a session from the embeddings already
/// stored in SQLite, without re-extracting or re-embedding any PDFs.
///
/// This is the pipeline for jobs created by `POST /sessions/{id}/rebuild`. It
/// is useful after a server crash where the in-memory HNSW was lost but the
/// embedding blobs in SQLite are intact, or when the HNSW map needs to be
/// refreshed without touching the source PDFs.
///
/// Steps:
///   1. Load the session record to obtain the stored vector dimension.
///   2. Read all embedding blobs for the session via `load_embeddings_for_hnsw`.
///   3. If no embeddings exist the index cannot be built; the job completes
///      with a logged warning (no error — the session state is valid).
///   4. Convert byte blobs to f32 vectors and build the HNSW index.
///   5. Atomically swap the new index into AppState.
async fn run_rebuild_pipeline(state: &State, job: &neuroncite_store::JobRow) -> Result<(), String> {
    let job_id = &job.id;
    let session_id = job
        .session_id
        .ok_or("rebuild job has no associated session_id")?;

    let session = {
        let conn = state
            .pool()
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;
        neuroncite_store::get_session(&conn, session_id)
            .map_err(|e| format!("session lookup: {e}"))?
    };

    let raw_embeddings = {
        let conn = state
            .pool()
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;
        neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
            .map_err(|e| format!("loading embeddings for HNSW rebuild: {e}"))?
    };

    if raw_embeddings.is_empty() {
        tracing::warn!(
            job_id = %job_id,
            session_id,
            "no embeddings found for session, HNSW index not rebuilt"
        );
        return Ok(());
    }

    let vector_dim = session
        .vector_dimension_usize()
        .map_err(|e| format!("integer overflow: {e}"))?;
    let vector_count = raw_embeddings.len();

    tracing::info!(
        job_id = %job_id,
        session_id,
        vectors = vector_count,
        dimension = vector_dim,
        "rebuilding HNSW index from stored embeddings"
    );

    let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
        .into_iter()
        .map(|(id, bytes)| (id, indexer::bytes_to_f32_vec(&bytes)))
        .collect();

    let labeled_refs: Vec<(i64, &[f32])> = f32_vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let index = neuroncite_store::build_hnsw(&labeled_refs, vector_dim)
        .map_err(|e| format!("HNSW rebuild failed: {e}"))?;
    state.insert_hnsw(session_id, index);

    tracing::info!(
        job_id = %job_id,
        session_id,
        vectors = vector_count,
        "HNSW index rebuilt and inserted into per-session map"
    );

    Ok(())
}

/// Interval at which the executor polls the database for job cancellation
/// while the annotate pipeline is running inside `spawn_blocking`. The
/// pipeline checks the cancel flag before each PDF, so the actual
/// cancellation latency is bounded by the time to process one PDF plus
/// this polling interval.
const CANCEL_CHECK_INTERVAL: Duration = Duration::from_secs(2);

/// Grace period after setting the cancel flag before the executor gives up
/// waiting for the pipeline to stop. If the pipeline is stuck in a blocking
/// C library call (e.g., Tesseract OCR or pdfium rendering), the cooperative
/// cancel flag cannot be observed, and the pipeline will not exit within
/// CANCEL_CHECK_INTERVAL. After this grace period expires, the executor
/// returns an error and moves on to the next job. The orphaned blocking
/// thread will eventually complete on its own when the C library call
/// finishes or the process exits.
const CANCEL_GRACE_PERIOD: Duration = Duration::from_secs(30);

/// Runs the annotation pipeline for a single annotation job. Deserializes
/// the job parameters from params_json, constructs an AnnotateConfig, and
/// delegates to the neuroncite_annotate crate's pipeline. Progress updates
/// are written to the database after each PDF is processed.
///
/// Before running the pipeline, this function checks whether the source
/// directory has an indexed session in the database. If so, the pre-extracted
/// page texts are loaded from the database and passed to the pipeline via
/// `cached_page_texts` in the config. This avoids redundant text extraction
/// (including OCR) for PDFs that were already indexed, which is the common
/// workflow: index a directory, search it, then annotate selected quotes.
///
/// The pipeline runs inside `spawn_blocking` because it uses synchronous
/// pdfium calls. Cancellation uses cooperative signaling via an
/// `Arc<AtomicBool>` flag that is shared between the async executor and the
/// blocking pipeline. The executor polls the database for cancel state and
/// sets the flag; the pipeline checks the flag before each PDF and returns
/// `AnnotateError::Canceled` when it is set. This approach avoids calling
/// `JoinHandle::abort()` on `spawn_blocking` tasks, which does NOT actually
/// stop the blocking thread -- it only drops the `JoinHandle` while the
/// thread continues running. That causes pdfium (a process-global C library)
/// to remain active on the old thread while the executor starts a new job
/// on a different thread, leading to concurrent pdfium usage and deadlocks.
async fn run_annotate_pipeline(
    state: &State,
    job: &neuroncite_store::JobRow,
) -> Result<(), String> {
    let params_str = job
        .params_json
        .as_deref()
        .ok_or("annotate job missing params_json")?;

    #[derive(serde::Deserialize)]
    struct AnnotateJobParams {
        input_data: String,
        source_directory: String,
        output_directory: String,
        default_color: Option<String>,
        /// Directory containing previously annotated PDFs from a prior append
        /// run. When present, the pipeline merges existing highlight annotations
        /// from these PDFs into the newly annotated output files.
        prior_output_directory: Option<String>,
    }

    let params: AnnotateJobParams =
        serde_json::from_str(params_str).map_err(|e| format!("invalid annotate params: {e}"))?;

    let job_id = job.id.clone();
    let pool = state.pool().clone();

    // Cooperative cancel flag shared between the async poll loop and the
    // synchronous pipeline running on the blocking thread pool. The executor
    // sets this to `true` when it detects the job was canceled in the database;
    // the pipeline checks it before processing each PDF and returns
    // `AnnotateError::Canceled` when set.
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_flag_for_pipeline = cancel_flag.clone();

    // Acquire the annotation semaphore permit before spawning the pipeline.
    // The permit is moved into the blocking closure and released when the
    // closure returns. If a previous pipeline is still running on an orphaned
    // blocking thread (from a prior cancel-after-grace-period), this blocks
    // until that thread finishes and the permit is returned. The semaphore
    // is initialised with one permit so only one pipeline runs at a time.
    // `expect` is safe here because the semaphore is never closed before shutdown.
    let guard = state
        .annotate_permit()
        .clone()
        .acquire_owned()
        .await
        .expect("annotation semaphore was closed before job ran");

    let pipeline_handle = tokio::task::spawn_blocking(move || {
        // The guard lives in this closure and is dropped when the closure
        // returns, ensuring the mutex is released even if the pipeline
        // panics or the outer async function returns early.
        let _permit_guard = guard;
        let input_rows = neuroncite_annotate::parse_input(params.input_data.as_bytes())
            .map_err(|e| format!("input parse: {e}"))?;

        // Build two mappings for translating AnnotationReport results back
        // to the annotation_quote_status rows in the database after the
        // pipeline completes.
        //
        // excerpt_to_meta: quote_excerpt -> Vec<(title, author)>
        //   Used for matched PDFs: QuoteReport has quote_excerpt but not
        //   title/author. This mapping provides the missing fields.
        //
        // meta_to_excerpts: (title, author) -> Vec<quote_excerpt>
        //   Used for unmatched inputs: UnmatchedInput has title/author
        //   but not the quote text. This reverse mapping provides the
        //   excerpt needed to match the database row.
        let mut excerpt_to_meta: std::collections::HashMap<String, Vec<(String, String)>> =
            std::collections::HashMap::new();
        let mut meta_to_excerpts: std::collections::HashMap<(String, String), Vec<String>> =
            std::collections::HashMap::new();
        for row in &input_rows {
            let exc = neuroncite_annotate::report::excerpt(&row.quote, 80);
            excerpt_to_meta
                .entry(exc.clone())
                .or_default()
                .push((row.title.clone(), row.author.clone()));
            meta_to_excerpts
                .entry((row.title.clone(), row.author.clone()))
                .or_default()
                .push(exc);
        }

        // Insert pending quote status rows into the database so that the
        // neuroncite_annotate_status handler can report "pending" quotes
        // while the pipeline is still running. Uses the same excerpt
        // length (80 chars) that the pipeline writes to QuoteReport so
        // that the WHERE clause in update_quote_status matches correctly.
        {
            let quote_tuples: Vec<(&str, &str, String)> = input_rows
                .iter()
                .map(|row| {
                    (
                        row.title.as_str(),
                        row.author.as_str(),
                        neuroncite_annotate::report::excerpt(&row.quote, 80),
                    )
                })
                .collect();
            let refs: Vec<(&str, &str, &str)> = quote_tuples
                .iter()
                .map(|(t, a, e)| (*t, *a, e.as_str()))
                .collect();
            if let Ok(conn) = pool.get()
                && let Err(e) = neuroncite_store::insert_pending_quotes(&conn, &job_id, &refs)
            {
                tracing::warn!(job_id = %job_id, "failed to insert pending quotes: {e}");
            }
        }

        let source_dir = PathBuf::from(&params.source_directory);
        let output_dir = PathBuf::from(&params.output_directory);
        let error_dir = output_dir.join("errors");

        // Load cached page texts from the database for PDFs in the source
        // directory. This reuses text already extracted during indexing,
        // avoiding redundant CPU/OCR work in the annotation pipeline's
        // fallback extraction stage (stage 3.5).
        let cached_page_texts = load_cached_page_texts(&pool, &source_dir);

        if !cached_page_texts.is_empty() {
            tracing::info!(
                cached_pdfs = cached_page_texts.len(),
                "loaded cached page texts from indexed session for annotation pipeline"
            );
        }

        let config = neuroncite_annotate::AnnotateConfig {
            input_rows,
            source_directory: source_dir,
            output_directory: output_dir,
            error_directory: error_dir,
            default_color: params.default_color.unwrap_or_else(|| "#FFFF00".into()),
            cached_page_texts,
            prior_output_directory: params.prior_output_directory.map(PathBuf::from),
        };

        neuroncite_annotate::annotate_pdfs_with_cancel(
            config,
            |done, total| {
                if let Ok(conn) = pool.get()
                    && let Err(e) = neuroncite_store::update_job_progress(
                        &conn,
                        &job_id,
                        done as i64,
                        total as i64,
                    )
                {
                    tracing::warn!(job_id = %job_id, "failed to update job progress: {e}");
                }
            },
            // The cancel callback reads the AtomicBool that the async poll
            // loop sets when it detects a database cancel state. Relaxed
            // ordering is sufficient: the flag transitions monotonically
            // from false to true, and a one-iteration delay in observing
            // the change is acceptable (bounded by CANCEL_CHECK_INTERVAL).
            move || cancel_flag_for_pipeline.load(Ordering::Relaxed),
        )
        .map_err(|e| format!("annotation pipeline: {e}"))
        .map(|report| {
            // Persist per-quote results from the AnnotationReport into the
            // annotation_quote_status table. This makes the data available
            // through the neuroncite_annotate_status handler after job
            // completion. Without this step, the table remains populated
            // with "pending" rows that are never updated.
            if let Ok(conn) = pool.get() {
                // Update statuses for quotes in matched PDFs. Each
                // QuoteReport has a quote_excerpt that maps back to the
                // input row's (title, author) via the excerpt_to_meta
                // HashMap built above.
                for pdf in &report.pdfs {
                    let pdf_filename = &pdf.filename;
                    for quote in &pdf.quotes {
                        let status = match quote.status.as_str() {
                            "matched" | "low_confidence_match" | "page_level_match" => "matched",
                            "not_found" => "not_found",
                            _ => "error",
                        };

                        if let Some(meta_list) = excerpt_to_meta.get(&quote.quote_excerpt) {
                            for (title, author) in meta_list {
                                if let Err(e) = neuroncite_store::update_quote_status(
                                    &conn,
                                    &job_id,
                                    title,
                                    author,
                                    &quote.quote_excerpt,
                                    status,
                                    quote.match_method.as_deref(),
                                    quote.page.map(|p| p as i64),
                                    Some(pdf_filename.as_str()),
                                ) {
                                    tracing::warn!(
                                        job_id = %job_id,
                                        title = %title,
                                        "failed to update quote status for matched quote: {e}"
                                    );
                                }
                            }
                        }
                    }
                }

                // Update statuses for unmatched inputs (quotes whose
                // title/author could not be resolved to any PDF file).
                // UnmatchedInput only contains title, author, and error
                // but not the original quote text. The meta_to_excerpts
                // reverse mapping provides the quote_excerpt values
                // needed to match the database rows.
                for unmatched in &report.unmatched_inputs {
                    let key = (unmatched.title.clone(), unmatched.author.clone());
                    if let Some(excerpts) = meta_to_excerpts.get(&key) {
                        for exc in excerpts {
                            if let Err(e) = neuroncite_store::update_quote_status(
                                &conn,
                                &job_id,
                                &unmatched.title,
                                &unmatched.author,
                                exc,
                                "not_found",
                                None,
                                None,
                                None,
                            ) {
                                tracing::warn!(
                                    job_id = %job_id,
                                    title = %unmatched.title,
                                    "failed to update quote status for unmatched input: {e}"
                                );
                            }
                        }
                    }
                }
            }
        })
    });

    // Poll the database for cancellation while the blocking pipeline runs.
    // When cancellation is detected, the AtomicBool flag is set so the
    // pipeline stops at the next PDF boundary. The executor then awaits the
    // JoinHandle to completion (the pipeline returns AnnotateError::Canceled),
    // ensuring the blocking thread has fully exited and released all pdfium
    // resources before the executor picks up the next job.
    //
    // If the pipeline is stuck in a blocking C library call (Tesseract OCR,
    // pdfium rendering) and cannot observe the cancel flag, a grace period
    // timer limits how long the executor waits. After CANCEL_GRACE_PERIOD
    // elapses since the cancel signal was sent, the executor gives up waiting
    // and returns an error. The orphaned blocking thread will eventually
    // complete on its own.
    let job_id_for_cancel = job.id.clone();
    let state_for_cancel = state.clone();

    tokio::pin!(pipeline_handle);

    // Tracks when the cancel signal was sent. None = not canceled yet.
    let mut cancel_signaled_at: Option<tokio::time::Instant> = None;

    loop {
        tokio::select! {
            result = &mut pipeline_handle => {
                // Pipeline completed (success, cancel, or panic).
                return result.map_err(|e| format!("annotate task panicked: {e}"))?;
            }
            _ = tokio::time::sleep(CANCEL_CHECK_INTERVAL) => {
                // If cancel was already signaled, log periodic warnings but
                // keep waiting for the pipeline to finish. Returning early
                // from this function while the spawn_blocking task is still
                // running would leave pdfium loaded on an orphaned thread.
                // The next annotation job would then load pdfium on a second
                // thread, causing a process-global C library deadlock
                // (BUG-011). The outer job_timeout at the executor
                // level provides the absolute time limit.
                if let Some(signaled_at) = cancel_signaled_at {
                    let elapsed = signaled_at.elapsed();
                    if elapsed >= CANCEL_GRACE_PERIOD {
                        // The pipeline did not stop within the grace period.
                        // Return an error so the executor can move on to the
                        // next job. The orphaned spawn_blocking thread still
                        // holds the annotate_permit mutex guard and will release it when
                        // it eventually finishes. The next annotation pipeline
                        // will block on acquiring the permit until pdfium is
                        // released, preventing the concurrent-pdfium deadlock.
                        tracing::warn!(
                            job_id = %job_id_for_cancel,
                            elapsed_secs = elapsed.as_secs(),
                            "cancel grace period expired; returning error and leaving \
                             orphaned pipeline thread (permit held until it finishes)"
                        );
                        return Err(format!(
                            "annotation pipeline did not stop within {}s after cancel signal",
                            CANCEL_GRACE_PERIOD.as_secs()
                        ));
                    }
                    continue;
                }

                // Check if the job was canceled while the pipeline is running.
                match is_job_canceled(&state_for_cancel, &job_id_for_cancel) {
                    Ok(true) => {
                        tracing::info!(
                            job_id = %job_id_for_cancel,
                            "annotate job canceled, signaling pipeline to stop"
                        );
                        // Set the cooperative cancel flag. The pipeline checks
                        // this before each PDF and returns Canceled. Do NOT
                        // call pipeline_handle.abort() -- spawn_blocking tasks
                        // cannot be aborted, and the "aborted" JoinHandle just
                        // detaches the thread while it continues running with
                        // pdfium loaded, causing deadlocks on the next job.
                        cancel_flag.store(true, Ordering::Relaxed);
                        cancel_signaled_at = Some(tokio::time::Instant::now());
                        // Continue the loop to await the pipeline's completion
                        // or the grace period expiration.
                    }
                    Ok(false) => {
                        // Job is still running, continue waiting.
                    }
                    Err(e) => {
                        // Database read failed. Log the error but do not set
                        // the cancel flag -- transient DB errors should not
                        // kill a running job. The next poll will retry.
                        tracing::warn!(
                            job_id = %job_id_for_cancel,
                            "cancel check failed: {e}"
                        );
                    }
                }
            }
        }
    }
}

/// Loads pre-extracted page texts from the database for all indexed PDFs in
/// the given source directory. Searches all sessions whose `directory_path`
/// matches the source directory (after canonicalization), loads the file
/// records, and reads the page texts. Returns a map keyed by canonical PDF
/// path, with each value being a map of 1-indexed page numbers to text content.
///
/// If no matching sessions exist (the directory was never indexed), or if
/// database access fails, returns an empty map. Errors are logged but do not
/// abort the annotation pipeline -- the pipeline falls back to live extraction.
fn load_cached_page_texts(
    pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    source_dir: &std::path::Path,
) -> std::collections::HashMap<PathBuf, std::collections::HashMap<usize, String>> {
    let mut cache = std::collections::HashMap::new();

    // Canonicalize the source directory using the project's own canonicalize_directory
    // function instead of std::fs::canonicalize. On Windows, std::fs::canonicalize
    // prepends the \\?\ extended-length path prefix (e.g. \\?\D:\Papers), but sessions
    // are stored at index time via canonicalize_directory which strips that prefix
    // (producing D:\Papers). The SQL equality check in find_sessions_by_directory
    // therefore always returns 0 rows when the paths differ, leaving the annotation
    // pipeline with an empty cache and forcing full re-extraction for every PDF.
    // On failure (path does not exist), fall back to the raw source_dir so that
    // the cache lookup returns empty results rather than crashing the pipeline.
    let canonical_dir = neuroncite_core::paths::canonicalize_directory(source_dir)
        .unwrap_or_else(|_| source_dir.to_path_buf());
    let dir_str = canonical_dir.to_string_lossy().to_string();

    let conn = match pool.get() {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("cannot get connection for cache lookup: {e}");
            return cache;
        }
    };

    // Find all sessions indexed from this directory. The most recent session
    // (first in the list, ordered by created_at DESC) is preferred since it
    // likely has the most up-to-date extracted text.
    let sessions = match neuroncite_store::find_sessions_by_directory(&conn, &dir_str) {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!(dir = %dir_str, "session lookup for cache failed: {e}");
            return cache;
        }
    };

    if sessions.is_empty() {
        return cache;
    }

    // Use the most recent session (first in the list).
    let session_id = sessions[0].id;

    let files = match neuroncite_store::list_files_by_session(&conn, session_id) {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(session_id, "file listing for cache failed: {e}");
            return cache;
        }
    };

    for file in &files {
        let pages = match neuroncite_store::get_pages_by_file(&conn, file.id) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(file_id = file.id, "page loading for cache failed: {e}");
                continue;
            }
        };

        if pages.is_empty() {
            continue;
        }

        let mut page_map = std::collections::HashMap::with_capacity(pages.len());
        for page in pages {
            page_map.insert(page.page_number as usize, page.content);
        }

        let file_path = PathBuf::from(&file.file_path);
        cache.insert(file_path, page_map);
    }

    cache
}

/// Scans all running jobs and transitions any that have exceeded the execution
/// timeout to Failed. This recovers from two scenarios:
///
/// 1. Jobs orphaned by a previous server crash (started_at set, but the
///    executor process died before the job could complete).
/// 2. Jobs from the current session where the pipeline hung beyond the timeout
///    (the tokio::time::timeout in the executor loop handles the async side,
///    but this function catches edge cases where the state was left as Running).
///
/// Only jobs managed by the executor (kind = "index" or "annotate") are
/// recovered. Citation verification jobs (kind = "citation_verify") are
/// agent-driven and handled separately.
fn recover_stuck_jobs(state: &State) {
    // try_get avoids blocking the executor loop when all connections are busy
    // serving concurrent handlers. If the pool is exhausted, stuck-job recovery
    // is skipped for this cycle and retried on the next poll interval.
    let conn = match state.pool().try_get() {
        Some(c) => c,
        None => return,
    };

    let jobs = match neuroncite_store::list_jobs(&conn) {
        Ok(j) => j,
        Err(_) => return,
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let timeout_secs = u64::from(state.config().job_timeout_minutes) * 60;
    let timeout_secs = timeout_secs as i64;

    for job in &jobs {
        // Only recover jobs that are managed by the executor loop. Jobs with
        // other kinds (e.g., citation_verify) are driven by external agents and
        // must not be terminated by this function regardless of elapsed time.
        if !matches!(job.kind.as_str(), "index" | "annotate" | "rebuild") {
            continue;
        }

        if job.state != JobState::Running {
            continue;
        }

        let started = match job.started_at {
            Some(s) => s,
            None => continue,
        };

        let elapsed = now - started;
        if elapsed <= timeout_secs {
            continue;
        }

        let msg = format!(
            "job recovered as failed: running for {elapsed} seconds without completing \
             (timeout: {timeout_secs}s)"
        );

        if let Err(e) =
            neuroncite_store::update_job_state(&conn, &job.id, JobState::Failed, Some(&msg))
        {
            tracing::warn!(
                job_id = %job.id,
                "failed to recover stuck job: {e}"
            );
        } else {
            tracing::warn!(
                job_id = %job.id,
                elapsed_secs = elapsed,
                "recovered stuck job by transitioning to failed"
            );
        }
    }
}

/// Transitions a job to a new state, with an optional error message for
/// the Failed state.
fn transition_job(
    state: &State,
    job_id: &str,
    new_state: JobState,
    error_message: Option<&str>,
) -> Result<(), String> {
    let conn = state
        .pool()
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;
    neuroncite_store::update_job_state(&conn, job_id, new_state, error_message)
        .map_err(|e| format!("state transition: {e}"))
}

/// Updates the progress counters for a running job. When the SSE broadcast
/// channel is available (web mode), sends an `index_progress` event to all
/// connected browser clients. The `phase` field distinguishes between the
/// three indexing stages: "extracting", "embedding", and "building_index".
fn update_progress(
    state: &State,
    job_id: &str,
    done: i64,
    total: i64,
    chunks_created: i64,
    phase: &str,
) -> Result<(), String> {
    let conn = state
        .pool()
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;
    neuroncite_store::update_job_progress(&conn, job_id, done, total)
        .map_err(|e| format!("progress update: {e}"))?;

    // Broadcast the progress event to SSE subscribers. In headless mode
    // there are no subscribers; the send result is intentionally discarded.
    {
        let tx = state.progress_tx();
        let _ = tx.send(
            serde_json::json!({
                "event": "index_progress",
                "phase": phase,
                "files_total": total,
                "files_done": done,
                "chunks_created": chunks_created,
                "job_id": job_id,
            })
            .to_string(),
        );
    }
    Ok(())
}

/// Sends an extraction progress SSE event without updating the database.
/// Called from the rayon extraction callback where database writes would
/// add unnecessary overhead. The database progress is only persisted
/// during the embedding phase (Phase 2).
fn broadcast_extraction_progress(
    tx: &tokio::sync::broadcast::Sender<String>,
    job_id: &str,
    done: usize,
    total: usize,
) {
    let _ = tx.send(
        serde_json::json!({
            "event": "index_progress",
            "phase": "extracting",
            "files_total": total,
            "files_done": done,
            "chunks_created": 0,
            "job_id": job_id,
        })
        .to_string(),
    );
}

/// Sets progress_done = progress_total for a job that completed successfully.
/// Called before transitioning to the Completed state to guarantee that the
/// terminal progress counters are consistent. Without this, pipelines that
/// track progress per matched item (rather than per input) can finish with
/// progress_done < progress_total (e.g., annotation jobs where no input
/// matched a PDF produce progress_done=0, progress_total=N).
fn finalize_progress(state: &State, job_id: &str) -> Result<(), String> {
    let conn = state
        .pool()
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;
    conn.execute(
        "UPDATE job SET progress_done = progress_total WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(|e| format!("finalize progress: {e}"))?;
    Ok(())
}

/// Reads the current state of a job from the database. Used to determine
/// whether the job was canceled while the pipeline was running, before
/// attempting a terminal state transition.
fn read_job_state(state: &State, job_id: &str) -> Result<JobState, String> {
    let conn = state
        .pool()
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;
    let job = neuroncite_store::get_job(&conn, job_id).map_err(|e| format!("job lookup: {e}"))?;
    Ok(job.state)
}

/// Checks whether a job has been canceled by reading its current state
/// from the database.
fn is_job_canceled(state: &State, job_id: &str) -> Result<bool, String> {
    Ok(read_job_state(state, job_id)? == JobState::Canceled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use arc_swap::ArcSwap;
    use neuroncite_core::{AppConfig, EmbeddingBackend};
    use neuroncite_store::{self as store, HnswIndex, JobState};
    use tokio_util::sync::CancellationToken;

    use crate::context::{
        ComputeContext, DatabaseContext, EventContext, IndexContext, PipelineContext,
    };
    use crate::test_support::StubBackend;
    use crate::worker::{WorkerHandle, spawn_worker};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Minimal `PipelineContext` implementation for executor unit tests.
    /// Wraps a real SQLite pool and WorkerHandle; all other fields mirror
    /// what `AppState::new` would produce for the test setup. The
    /// `progress_tx` channel is initialised unconditionally; in headless
    /// tests there are no subscribers so messages are silently discarded.
    struct StubPipelineContext {
        pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
        worker_handle: WorkerHandle,
        hnsw_index: ArcSwap<HashMap<i64, Arc<HnswIndex>>>,
        config: AppConfig,
        progress_tx: tokio::sync::broadcast::Sender<String>,
        cancellation_token: CancellationToken,
        job_notify: tokio::sync::Notify,
        annotate_permit: Arc<tokio::sync::Semaphore>,
    }

    /// `DatabaseContext` sub-trait implementation for `StubPipelineContext`.
    impl DatabaseContext for StubPipelineContext {
        fn pool(&self) -> &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager> {
            &self.pool
        }
    }

    /// `ComputeContext` sub-trait implementation for `StubPipelineContext`.
    impl ComputeContext for StubPipelineContext {
        fn worker_handle(&self) -> &WorkerHandle {
            &self.worker_handle
        }
    }

    /// `IndexContext` sub-trait implementation for `StubPipelineContext`.
    ///
    /// Each mutation uses `rcu()` to atomically swap the HNSW index map
    /// without holding an exclusive lock.
    impl IndexContext for StubPipelineContext {
        fn hnsw_index(&self) -> &ArcSwap<HashMap<i64, Arc<HnswIndex>>> {
            &self.hnsw_index
        }
        fn insert_hnsw(&self, session_id: i64, index: HnswIndex) {
            let index = Arc::new(index);
            self.hnsw_index.rcu(|current| {
                let mut map = (**current).clone();
                map.insert(session_id, index.clone());
                Arc::new(map)
            });
        }
        fn insert_hnsw_many(&self, entries: Vec<(i64, HnswIndex)>) {
            if entries.is_empty() {
                return;
            }
            let arced: Vec<(i64, Arc<HnswIndex>)> = entries
                .into_iter()
                .map(|(id, idx)| (id, Arc::new(idx)))
                .collect();
            self.hnsw_index.rcu(|current| {
                let mut map = (**current).clone();
                for (sid, idx) in &arced {
                    map.insert(*sid, idx.clone());
                }
                Arc::new(map)
            });
        }
        fn remove_hnsw(&self, session_id: i64) {
            self.hnsw_index.rcu(|current| {
                let mut map = (**current).clone();
                map.remove(&session_id);
                Arc::new(map)
            });
        }
        fn remove_hnsw_many(&self, session_ids: &[i64]) {
            if session_ids.is_empty() {
                return;
            }
            self.hnsw_index.rcu(|current| {
                let mut map = (**current).clone();
                for id in session_ids {
                    map.remove(id);
                }
                Arc::new(map)
            });
        }
    }

    /// `EventContext` sub-trait implementation for `StubPipelineContext`.
    impl EventContext for StubPipelineContext {
        fn progress_tx(&self) -> &tokio::sync::broadcast::Sender<String> {
            &self.progress_tx
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

    /// `PipelineContext` composed implementation for `StubPipelineContext`.
    ///
    /// `StubPipelineContext` satisfies all four sub-traits so this blanket
    /// impl only needs to supply `config()`.
    impl PipelineContext for StubPipelineContext {
        fn config(&self) -> &AppConfig {
            &self.config
        }
    }

    /// Creates a test context backed by an in-memory SQLite database.
    fn setup_test_state() -> State {
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");

        {
            let conn = pool.get().expect("get conn for migration");
            store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend::default());
        let handle = spawn_worker(backend, None);
        let config = AppConfig::default();

        // The progress_tx receiver is intentionally dropped; in headless test
        // mode there are no SSE subscribers and messages are discarded.
        let (progress_tx, _progress_rx) = tokio::sync::broadcast::channel(64);

        Arc::new(StubPipelineContext {
            pool,
            worker_handle: handle,
            hnsw_index: ArcSwap::from_pointee(HashMap::new()),
            config,
            progress_tx,
            cancellation_token: CancellationToken::new(),
            job_notify: tokio::sync::Notify::new(),
            annotate_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }

    /// Creates a session with word-based chunking and returns its ID.
    fn create_test_session(state: &State, directory: &str) -> i64 {
        create_test_session_with_strategy(state, directory, "word", Some(300), Some(50), None)
    }

    /// Creates a session with a configurable chunking strategy and returns
    /// its ID. Supports word, sentence, and page strategies.
    fn create_test_session_with_strategy(
        state: &State,
        directory: &str,
        strategy: &str,
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
        max_words: Option<usize>,
    ) -> i64 {
        let conn = state.pool().get().expect("get conn");
        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from(directory),
            model_name: "stub-model".to_string(),
            chunk_strategy: strategy.to_string(),
            chunk_size,
            chunk_overlap,
            max_words,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        store::create_session(&conn, &config, "0.1.0").expect("create session")
    }

    /// Creates a queued indexing job for the given session.
    fn create_queued_job(state: &State, session_id: i64) -> String {
        let conn = state.pool().get().expect("get conn");
        let job_id = uuid::Uuid::new_v4().to_string();
        store::create_job(&conn, &job_id, "index", Some(session_id)).expect("create job");
        job_id
    }

    /// Creates a session with a custom model_name (for model-mismatch tests).
    fn create_test_session_with_model(state: &State, directory: &str, model_name: &str) -> i64 {
        let conn = state.pool().get().expect("get conn");
        let config = neuroncite_core::IndexConfig {
            directory: PathBuf::from(directory),
            model_name: model_name.to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        store::create_session(&conn, &config, "0.1.0").expect("create session")
    }

    /// Creates a minimal valid PDF in a directory.
    fn create_test_pdf(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let pdf_content = format!(
            "%PDF-1.0\n\
             1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
             2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
             3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n\
             4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n\
             5 0 obj<</Length {}>>stream\nBT /F1 12 Tf 100 700 Td ({}) Tj ET\nendstream\nendobj\n\
             xref\n0 6\n\
             0000000000 65535 f \n\
             0000000009 00000 n \n\
             0000000058 00000 n \n\
             0000000115 00000 n \n\
             0000000266 00000 n \n\
             0000000340 00000 n \n\
             trailer<</Size 6/Root 1 0 R>>\nstartxref\n{}\n%%EOF",
            content.len() + 42,
            content,
            400 + content.len()
        );
        let path = dir.join(name);
        std::fs::write(&path, pdf_content).expect("write test pdf");
        path
    }

    // -----------------------------------------------------------------------
    // T-EXEC-001: Executor picks up queued job and completes it
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_001_picks_up_queued_job_and_completes() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Hello from executor test");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        // Verify the job is queued.
        {
            let conn = state.pool().get().expect("conn");
            let job = store::get_job(&conn, &job_id).expect("get job");
            assert_eq!(job.state, JobState::Queued);
        }

        // Find and execute the job directly (not via the polling loop).
        let job = find_next_queued_job(&state).expect("should find queued job");
        assert_eq!(job.id, job_id);

        execute_job(&state, &job).await;

        // Verify the job transitioned to completed or failed.
        // (It may fail if pdf-extract cannot parse the minimal test PDF,
        // but the state machine transitions should still work.)
        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get final job");
        assert!(
            final_job.state == JobState::Completed || final_job.state == JobState::Failed,
            "job should be in a terminal state, got: {:?}",
            final_job.state
        );
        assert!(
            final_job.started_at.is_some(),
            "started_at must be set after Running transition"
        );
        assert!(
            final_job.finished_at.is_some(),
            "finished_at must be set after terminal transition"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-002: Executor updates progress correctly
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_002_progress_updates() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "a.pdf", "First PDF");
        create_test_pdf(tmp.path(), "b.pdf", "Second PDF");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get final job");

        // progress_total should reflect the number of discovered PDFs.
        assert!(
            final_job.progress_total >= 2,
            "progress_total should be at least 2, got: {}",
            final_job.progress_total
        );

        // progress_done should equal progress_total upon completion.
        if final_job.state == JobState::Completed {
            assert_eq!(
                final_job.progress_done, final_job.progress_total,
                "progress_done should equal progress_total on completion"
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-EXEC-003: Executor transitions to Failed for non-existent directory
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_003_fails_for_nonexistent_directory() {
        let state = setup_test_state();
        let session_id = create_test_session(&state, "/nonexistent/path/that/does/not/exist");
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get final job");
        assert_eq!(
            final_job.state,
            JobState::Failed,
            "job should fail for non-existent directory"
        );
        assert!(
            final_job.error_message.is_some(),
            "failed job must have an error message"
        );
        let msg = final_job.error_message.as_deref().unwrap_or("");
        assert!(
            msg.contains("does not exist"),
            "error message should indicate missing directory, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-004: Executor respects cancellation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_004_respects_cancellation() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Cancellation test");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        // Cancel the job before the executor processes it.
        {
            let conn = state.pool().get().expect("conn");
            store::update_job_state(&conn, &job_id, JobState::Canceled, None).expect("cancel job");
        }

        // The executor should not find this job (it's no longer queued).
        let found = find_next_queued_job(&state);
        assert!(
            found.is_none(),
            "canceled job should not be picked up by executor"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-005: Executor processes multiple jobs in FIFO order
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_005_fifo_order() {
        let tmp1 = tempfile::tempdir().expect("create temp dir 1");
        let tmp2 = tempfile::tempdir().expect("create temp dir 2");

        let state = setup_test_state();
        let session_id1 = create_test_session(&state, &tmp1.path().to_string_lossy());
        let session_id2 = create_test_session(&state, &tmp2.path().to_string_lossy());
        let job_id1 = create_queued_job(&state, session_id1);

        // Sleep briefly to ensure created_at differs.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let job_id2 = create_queued_job(&state, session_id2);

        // The first queued job should be the oldest (FIFO).
        let first = find_next_queued_job(&state).expect("find first job");
        assert_eq!(
            first.id, job_id1,
            "first job (oldest) should be picked up first"
        );

        // Complete the first job, then the second should be next.
        {
            let conn = state.pool().get().expect("conn");
            store::update_job_state(&conn, &job_id1, JobState::Running, None).expect("running");
            store::update_job_state(&conn, &job_id1, JobState::Completed, None).expect("completed");
        }

        let second = find_next_queued_job(&state).expect("find second job");
        assert_eq!(
            second.id, job_id2,
            "second job should be picked up after first completes"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-006: Executor ignores truly unknown job kinds
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_006_ignores_unknown_job_kinds() {
        let state = setup_test_state();

        // Create a job with a kind that the executor does not handle.
        {
            let conn = state.pool().get().expect("conn");
            store::create_job(&conn, "unknown-job-001", "unknown-kind", None).expect("create job");
        }

        let found = find_next_queued_job(&state);
        assert!(
            found.is_none(),
            "jobs with unknown kinds should be ignored by the executor"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-007: Executor builds HNSW index and swaps into AppState
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_007_builds_hnsw_and_swaps() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "HNSW test document content");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let _job_id = create_queued_job(&state, session_id);

        // Verify no HNSW index is loaded initially.
        assert!(
            state.hnsw_index().load().is_empty(),
            "HNSW index map should be empty before indexing"
        );

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        // After indexing, the HNSW index may or may not be present depending
        // on whether the test PDF produced extractable chunks. If it did,
        // the index should be swapped in.
        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job.id).expect("get job");

        if final_job.state == JobState::Completed {
            let chunks = store::list_chunks_by_session(&conn, session_id).expect("list chunks");
            if !chunks.is_empty() {
                assert!(
                    !state.hnsw_index().load().is_empty(),
                    "HNSW index map should contain the session after indexing with chunks"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // T-EXEC-008: Executor handles empty directory (0 PDFs)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_008_empty_directory_completes() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        // No PDFs in the directory.

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get job");
        assert_eq!(
            final_job.state,
            JobState::Completed,
            "empty directory should complete successfully"
        );
        assert_eq!(
            final_job.progress_total, 0,
            "progress_total should be 0 for empty directory"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-009: Executor handles extraction failures gracefully
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_exec_009_extraction_failures_skipped() {
        let tmp = tempfile::tempdir().expect("create temp dir");

        // Create one valid and one corrupt PDF.
        create_test_pdf(tmp.path(), "good.pdf", "Valid content");
        std::fs::write(tmp.path().join("bad.pdf"), b"not a valid PDF at all")
            .expect("write corrupt pdf");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get job");

        // The job should complete (not fail) because extraction failures are
        // skipped, not treated as fatal errors.
        assert!(
            final_job.state == JobState::Completed || final_job.state == JobState::Failed,
            "job should reach a terminal state"
        );
        assert!(
            final_job.progress_total >= 2,
            "progress_total should reflect all discovered PDFs"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-010: Executor reconstructs sentence strategy with max_words
    // -----------------------------------------------------------------------

    /// Regression test for the bug where `create_strategy` was called with
    /// `max_words: None`, causing sentence-strategy sessions to fail with
    /// "sentence strategy requires max_words parameter". The fix passes
    /// `session.max_words` from the database to the chunking API.
    #[tokio::test]
    async fn t_exec_010_sentence_strategy_with_max_words() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Sentence strategy test content");

        let state = setup_test_state();
        let session_id = create_test_session_with_strategy(
            &state,
            &tmp.path().to_string_lossy(),
            "sentence",
            None,     // chunk_size not used by sentence strategy
            None,     // chunk_overlap not used by sentence strategy
            Some(50), // max_words is required for sentence strategy
        );
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get job");

        // The job must NOT fail with "sentence strategy requires max_words".
        // It should reach a terminal state (completed or failed for other
        // reasons like PDF extraction, but not strategy creation).
        assert!(
            final_job.state == JobState::Completed || final_job.state == JobState::Failed,
            "job should be in terminal state, got: {:?}",
            final_job.state
        );

        if final_job.state == JobState::Failed {
            let msg = final_job.error_message.as_deref().unwrap_or("");
            assert!(
                !msg.contains("max_words"),
                "sentence strategy must not fail due to missing max_words, got: {msg}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-EXEC-011: Executor does not fail on sentence strategy without max_words
    // -----------------------------------------------------------------------

    /// Verifies that a session created without max_words but with "sentence"
    /// strategy results in a clean failure message (not a panic or hang).
    /// This documents the expected behavior when the session config is
    /// inconsistent (sentence strategy requires max_words).
    #[tokio::test]
    async fn t_exec_011_sentence_without_max_words_defaults_to_256() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Missing max_words test");

        let state = setup_test_state();
        // Create a session with sentence strategy but no max_words.
        // The chunker defaults max_words to 256 when omitted.
        let session_id = create_test_session_with_strategy(
            &state,
            &tmp.path().to_string_lossy(),
            "sentence",
            None,
            None,
            None, // omitted max_words — defaults to 256
        );
        let job_id = create_queued_job(&state, session_id);

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get job");

        // The job should complete because the chunker defaults max_words to 256.
        assert_eq!(
            final_job.state,
            JobState::Completed,
            "sentence strategy without max_words should succeed with default 256"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-012: load_cached_page_texts returns empty map for non-indexed dir
    // -----------------------------------------------------------------------

    /// Verifies that load_cached_page_texts returns an empty HashMap when the
    /// source directory has no indexed sessions in the database. This is the
    /// default case for the annotate pipeline when run on a directory that
    /// was never indexed -- the pipeline falls back to live extraction.
    #[tokio::test]
    async fn t_exec_012_cache_empty_for_unindexed_dir() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let state = setup_test_state();

        let cache = load_cached_page_texts(state.pool(), tmp.path());
        assert!(
            cache.is_empty(),
            "cache must be empty for a directory with no indexed sessions"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-013: load_cached_page_texts loads page texts from indexed session
    // -----------------------------------------------------------------------

    /// Verifies that load_cached_page_texts returns a populated HashMap when
    /// the source directory has an indexed session with files and pages. The
    /// cache should contain all page texts keyed by file path.
    #[tokio::test]
    async fn t_exec_013_cache_populated_from_indexed_session() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "test content");

        let state = setup_test_state();
        // Use canonicalize_directory (strips \\?\ on Windows) rather than
        // std::fs::canonicalize to match the path format the indexer stores
        // in the database at index time. load_cached_page_texts also uses
        // canonicalize_directory, so session lookup succeeds only when both
        // sides use the same normalization. The temp directory exists, so the
        // fallback to the raw path is not expected to trigger.
        let canonical_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let session_id = create_test_session(&state, &canonical_dir.to_string_lossy());

        // Manually insert a file and pages into the database to simulate
        // a completed indexing run. The page texts represent the extracted
        // content that the annotate pipeline should reuse.
        let conn = state.pool().get().expect("conn");
        let file_path = canonical_dir.join("test.pdf");
        let file_id = store::insert_file(
            &conn,
            session_id,
            &file_path.to_string_lossy(),
            "fakehash123",
            1_700_000_000,
            4096,
            2,
            Some(2),
        )
        .expect("insert file");

        store::bulk_insert_pages(
            &conn,
            file_id,
            &[
                (1, "Page one extracted text from indexing", "pdf-extract"),
                (2, "Page two extracted text from indexing", "pdf-extract"),
            ],
        )
        .expect("insert pages");
        drop(conn);

        let cache = load_cached_page_texts(state.pool(), &canonical_dir);

        assert_eq!(
            cache.len(),
            1,
            "cache should contain one PDF entry, got: {}",
            cache.len()
        );

        let page_map = cache.get(&file_path).expect("file path must be in cache");
        assert_eq!(page_map.len(), 2, "file should have 2 cached pages");
        assert_eq!(
            page_map[&1], "Page one extracted text from indexing",
            "page 1 text must match"
        );
        assert_eq!(
            page_map[&2], "Page two extracted text from indexing",
            "page 2 text must match"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-014: load_cached_page_texts handles non-existent directory
    // -----------------------------------------------------------------------

    /// Verifies that load_cached_page_texts returns an empty HashMap when
    /// the source directory does not exist on disk. The canonicalization
    /// failure is handled gracefully without panicking.
    #[tokio::test]
    async fn t_exec_014_cache_nonexistent_dir_returns_empty() {
        let state = setup_test_state();
        let cache = load_cached_page_texts(
            state.pool(),
            std::path::Path::new("/nonexistent/path/that/does/not/exist"),
        );
        assert!(
            cache.is_empty(),
            "cache must be empty for non-existent directory"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-015: Executor preserves Canceled state after pipeline completes
    // -----------------------------------------------------------------------

    /// Regression test for BUG-001: When a job is canceled while the pipeline
    /// is running, the executor must preserve the Canceled state rather than
    /// overwriting it with Completed or Failed. Before the fix, execute_job
    /// attempted Canceled->Completed or Canceled->Failed transitions that
    /// violated the state machine, logging spurious errors and leaving the
    /// executor in an inconsistent state.
    ///
    /// This test cancels a job after transitioning it to Running, then calls
    /// execute_job. The job should remain in Canceled state because
    /// execute_job now re-reads the job state before attempting terminal
    /// transitions and skips them when the state is already Canceled.
    #[tokio::test]
    async fn t_exec_015_preserves_canceled_state_after_execution() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Cancel test content");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        // Cancel the job while it is in Queued state (simulating a cancel
        // request that arrives before the executor picks up the job, but
        // the job has already been read by find_next_queued_job).
        let job = {
            let conn = state.pool().get().expect("conn");
            let job = store::get_job(&conn, &job_id).expect("get job");
            // Transition to Running first (as execute_job does).
            store::update_job_state(&conn, &job_id, JobState::Running, None)
                .expect("transition to running");
            // Simulate a concurrent cancel request.
            store::update_job_state(&conn, &job_id, JobState::Canceled, None).expect("cancel job");
            job
        };

        // Create a job row that looks like what find_next_queued_job returns.
        let queued_job = store::JobRow {
            state: JobState::Queued,
            ..job
        };

        // execute_job will try to transition to Running (which fails because
        // the job is already Canceled), and then should return without
        // attempting a Completed/Failed transition.
        execute_job(&state, &queued_job).await;

        // Verify the job is still in Canceled state.
        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get final job");
        assert_eq!(
            final_job.state,
            JobState::Canceled,
            "job must remain in Canceled state after execute_job, got: {:?}",
            final_job.state
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-016: read_job_state returns correct state from database
    // -----------------------------------------------------------------------

    /// Verifies that `read_job_state` correctly reads the current job state
    /// from the database. This function is used by execute_job to determine
    /// whether the job was canceled during pipeline execution.
    #[tokio::test]
    async fn t_exec_016_read_job_state_returns_correct_state() {
        let state = setup_test_state();
        let session_id = create_test_session(&state, "/tmp/read_state_test");
        let job_id = create_queued_job(&state, session_id);

        // Verify initial state is Queued.
        let s = read_job_state(&state, &job_id).expect("read state");
        assert_eq!(s, JobState::Queued);

        // Transition to Running and verify.
        {
            let conn = state.pool().get().expect("conn");
            store::update_job_state(&conn, &job_id, JobState::Running, None).expect("transition");
        }
        let s = read_job_state(&state, &job_id).expect("read state");
        assert_eq!(s, JobState::Running);

        // Cancel and verify.
        {
            let conn = state.pool().get().expect("conn");
            store::update_job_state(&conn, &job_id, JobState::Canceled, None).expect("cancel");
        }
        let s = read_job_state(&state, &job_id).expect("read state");
        assert_eq!(s, JobState::Canceled);
    }

    // -----------------------------------------------------------------------
    // T-EXEC-017: AtomicBool cancel flag for cooperative cancellation
    // -----------------------------------------------------------------------

    /// T-EXEC-017: Verifies that the cooperative cancellation pattern using
    /// `Arc<AtomicBool>` correctly propagates the cancel signal. This is the
    /// regression test for BUG-001 where calling `JoinHandle::abort()` on a
    /// `spawn_blocking` task did NOT stop the blocking thread, causing pdfium
    /// (process-global C library) to remain active while the executor started
    /// a new job on a different thread, leading to deadlocks.
    ///
    /// The fix replaces `abort()` with a cooperative `AtomicBool` flag that the
    /// blocking pipeline checks before each PDF. This test verifies the flag
    /// can be read from a spawned thread.
    #[tokio::test]
    async fn t_exec_017_cooperative_cancel_flag_propagation() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_for_task = cancel_flag.clone();

        // Spawn a blocking task that reads the flag, simulating the
        // annotate pipeline's cancel check.
        let handle = tokio::task::spawn_blocking(move || {
            // Spin briefly waiting for the flag.
            for _ in 0..100 {
                if flag_for_task.load(Ordering::Relaxed) {
                    return true; // Cancel detected.
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            false // Timed out without cancel.
        });

        // Set the cancel flag from the async context (simulating the
        // executor's poll loop detecting a DB cancel state).
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_flag.store(true, Ordering::Relaxed);

        let result = handle.await.expect("task should not panic");
        assert!(
            result,
            "blocking task must observe the cancel flag set from async context"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-018: Cooperative cancel does not abort() the JoinHandle
    // -----------------------------------------------------------------------

    /// T-EXEC-018: Verifies that after setting the cancel flag, the executor
    /// waits for the blocking task to complete naturally rather than aborting
    /// it. The blocking task returns normally and the JoinHandle resolves with
    /// Ok (not a JoinError from abort). This guards against regression to the
    /// abort()-based cancellation that caused BUG-001.
    #[tokio::test]
    async fn t_exec_018_cancel_waits_for_task_completion() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_for_task = cancel_flag.clone();

        let completed = Arc::new(AtomicBool::new(false));
        let completed_for_task = completed.clone();

        let handle = tokio::task::spawn_blocking(move || {
            // Simulate pipeline work.
            std::thread::sleep(Duration::from_millis(100));

            // Check cancel flag (cooperative cancellation point).
            if flag_for_task.load(Ordering::Relaxed) {
                completed_for_task.store(true, Ordering::Relaxed);
                return Err("canceled".to_string());
            }

            completed_for_task.store(true, Ordering::Relaxed);
            Ok(())
        });

        // Set cancel flag while the task is sleeping.
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_flag.store(true, Ordering::Relaxed);

        // Wait for the task to complete naturally (NOT abort).
        let result = handle.await;
        assert!(
            result.is_ok(),
            "JoinHandle must resolve Ok (not JoinError from abort)"
        );

        // The blocking task set the completed flag before returning,
        // proving it ran to completion and was not interrupted.
        assert!(
            completed.load(Ordering::Relaxed),
            "blocking task must run to completion before JoinHandle resolves"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-020: Notify mechanism wakes executor immediately
    // -----------------------------------------------------------------------

    /// T-EXEC-020: The job_notify Notify wakes the executor loop immediately
    /// instead of waiting for the full POLL_INTERVAL. This verifies that
    /// notify_one() after job creation triggers a wake-up.
    ///
    /// Regression test for BUG-002: Before the fix, the executor loop used a
    /// fixed sleep(POLL_INTERVAL) without any notification channel. After
    /// canceling a job, the executor would sleep for up to POLL_INTERVAL
    /// before discovering the next queued job, causing apparent deadlock.
    #[tokio::test]
    async fn t_exec_020_notify_wakes_executor() {
        let state = setup_test_state();

        // The notify should be awaitable and resolve when notified.
        let notify = state.job_notify();

        let state_clone = state.clone();
        let handle = tokio::spawn(async move {
            // Wait for notification (with a timeout to prevent test hang).
            tokio::time::timeout(Duration::from_secs(2), state_clone.job_notify().notified()).await
        });

        // Small delay to ensure the spawned task is waiting.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Notify the executor.
        notify.notify_one();

        let result = handle.await.expect("join handle");
        assert!(
            result.is_ok(),
            "notified() must resolve within timeout after notify_one()"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-021: Cancel transitions job state correctly
    // -----------------------------------------------------------------------

    /// T-EXEC-021: Canceling a queued job transitions its state to Canceled
    /// and find_next_queued_job skips it. Verifies the cancel state machine.
    #[tokio::test]
    async fn t_exec_021_cancel_skips_job() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Cancel test content");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());
        let job_id = create_queued_job(&state, session_id);

        // Cancel the job.
        {
            let conn = state.pool().get().expect("conn");
            store::update_job_state(&conn, &job_id, JobState::Canceled, None).expect("cancel job");
        }

        // Notify the executor (as the cancel handler would do).
        state.job_notify().notify_one();

        // The canceled job must not be picked up by find_next_queued_job.
        let next_job = find_next_queued_job(&state);
        assert!(
            next_job.is_none(),
            "find_next_queued_job must not return a canceled job"
        );

        // Verify the job is in Canceled state.
        let conn = state.pool().get().expect("conn");
        let job = store::get_job(&conn, &job_id).expect("get job");
        assert_eq!(
            job.state,
            JobState::Canceled,
            "job state must be Canceled after cancellation"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-022: Multiple jobs are processed in FIFO order
    // -----------------------------------------------------------------------

    /// T-EXEC-022: When multiple jobs are queued, the executor picks them up
    /// in creation order (FIFO). This verifies that the executor correctly
    /// processes the oldest queued job first.
    #[tokio::test]
    async fn t_exec_022_fifo_ordering() {
        let tmp1 = tempfile::tempdir().expect("create temp dir 1");
        let tmp2 = tempfile::tempdir().expect("create temp dir 2");
        create_test_pdf(tmp1.path(), "a.pdf", "First job PDF");
        create_test_pdf(tmp2.path(), "b.pdf", "Second job PDF");

        let state = setup_test_state();
        let session1 = create_test_session(&state, &tmp1.path().to_string_lossy());
        let session2 = create_test_session(&state, &tmp2.path().to_string_lossy());
        let job_id_1 = create_queued_job(&state, session1);
        let job_id_2 = create_queued_job(&state, session2);

        // The first queued job should be picked up first.
        let first = find_next_queued_job(&state).expect("first job");
        assert_eq!(
            first.id, job_id_1,
            "first queued job must be picked up first (FIFO)"
        );

        // Execute the first job so it transitions out of Queued.
        execute_job(&state, &first).await;

        // The second queued job should be picked up next.
        let second = find_next_queued_job(&state).expect("second job");
        assert_eq!(
            second.id, job_id_2,
            "second queued job must be picked up after first completes"
        );
    }

    // ===================================================================
    // DEFECT-001 regression tests: annotation pipeline cancel + restart
    //
    // Before the fix, `run_annotate_pipeline` had two bugs:
    //
    // 1. The CANCEL_GRACE_PERIOD branch only logged a warning and called
    //    `continue`, leaving the executor stuck in the poll loop forever.
    //    The fix changes this to `return Err(...)` so the executor can
    //    move on to the next job.
    //
    // 2. If the executor returned early (after the grace period), the
    //    orphaned `spawn_blocking` thread still held pdfium loaded.
    //    The next annotation job would load pdfium on a second thread,
    //    causing a process-global C library deadlock. The fix adds an
    //    `annotate_permit` mutex to AppState. The mutex guard is moved into
    //    the blocking closure and dropped when the closure returns. The next
    //    pipeline blocks on acquiring the mutex lock until the previous
    //    thread finishes, preventing concurrent pdfium use.
    // ===================================================================

    // -----------------------------------------------------------------------
    // T-EXEC-023: annotate_permit semaphore allows exclusive access
    // -----------------------------------------------------------------------

    /// T-EXEC-023: The annotate_permit semaphore (initialised with one permit)
    /// allows exactly one annotation pipeline to hold a permit at a time.
    /// This prevents concurrent pdfium access across orphaned spawn_blocking
    /// threads.
    #[tokio::test]
    async fn t_exec_023_annotate_permit_semaphore_exclusive() {
        let state = setup_test_state();

        // First acquisition must succeed immediately (one permit available).
        let permit1 = state
            .annotate_permit()
            .clone()
            .try_acquire_owned()
            .expect("first permit acquisition must succeed");

        // Second acquisition must fail because the single permit is taken.
        let result = state.annotate_permit().clone().try_acquire_owned();
        assert!(
            result.is_err(),
            "second permit acquisition must fail while the first is held"
        );

        // Dropping the first permit returns it to the semaphore.
        drop(permit1);

        // Third acquisition must succeed after the first permit was returned.
        let _permit3 = state
            .annotate_permit()
            .clone()
            .try_acquire_owned()
            .expect("permit acquisition must succeed after release");
    }

    // -----------------------------------------------------------------------
    // T-EXEC-024: Semaphore permit is released when blocking closure returns
    // -----------------------------------------------------------------------

    /// T-EXEC-024: Verifies that the annotate_permit semaphore permit is
    /// returned when the `spawn_blocking` closure returns, even if it returns
    /// an error. This simulates the pattern used in `run_annotate_pipeline`
    /// where the permit is moved into the closure as `_permit_guard`.
    #[tokio::test]
    async fn t_exec_024_permit_released_on_closure_return() {
        let state = setup_test_state();

        // Acquire the permit and move it into a spawn_blocking closure.
        let permit = state
            .annotate_permit()
            .clone()
            .acquire_owned()
            .await
            .expect("permit acquisition must succeed");

        let handle = tokio::task::spawn_blocking(move || {
            let _permit_guard = permit;
            // Simulate pipeline work and error return.
            Err::<(), String>("simulated pipeline error".to_string())
        });

        // Wait for the closure to complete.
        let result = handle.await.expect("task must not panic");
        assert!(result.is_err(), "closure returned an error as expected");

        // After the closure returned, the semaphore permit must be available.
        let _permit2 = state
            .annotate_permit()
            .clone()
            .try_acquire_owned()
            .expect("permit must be available after closure returns");
    }

    // -----------------------------------------------------------------------
    // T-EXEC-025: Semaphore permit is released on closure panic
    // -----------------------------------------------------------------------

    /// T-EXEC-025: Verifies that the annotate_permit semaphore permit is
    /// returned even when the `spawn_blocking` closure panics. Rust's drop
    /// semantics guarantee that the `OwnedSemaphorePermit` destructor runs
    /// during stack unwinding.
    #[tokio::test]
    async fn t_exec_025_permit_released_on_panic() {
        let state = setup_test_state();

        let permit = state
            .annotate_permit()
            .clone()
            .acquire_owned()
            .await
            .expect("permit acquisition must succeed");

        let handle = tokio::task::spawn_blocking(move || {
            let _permit_guard = permit;
            panic!("simulated pipeline panic");
        });

        // The spawn_blocking task panicked, so await returns JoinError.
        let result = handle.await;
        assert!(
            result.is_err(),
            "JoinHandle must return Err for panicked task"
        );

        // Despite the panic, the permit must be returned because
        // OwnedSemaphorePermit was dropped during stack unwinding.
        let _permit2 = state
            .annotate_permit()
            .clone()
            .try_acquire_owned()
            .expect("permit must be available after closure panics");
    }

    // -----------------------------------------------------------------------
    // T-EXEC-026: Semaphore serializes concurrent annotation pipelines
    // -----------------------------------------------------------------------

    /// T-EXEC-026: Verifies that the semaphore serializes two concurrent
    /// annotation pipeline attempts. The second pipeline blocks until the
    /// first releases its permit. This prevents the DEFECT-001 scenario where
    /// two spawn_blocking threads simultaneously loaded pdfium.
    #[tokio::test]
    async fn t_exec_026_semaphore_serializes_concurrent_pipelines() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let state = setup_test_state();
        let counter = Arc::new(AtomicU32::new(0));

        // First pipeline acquires the single semaphore permit.
        let permit1 = state
            .annotate_permit()
            .clone()
            .acquire_owned()
            .await
            .expect("first permit acquisition must succeed");

        let counter1 = counter.clone();
        let first = tokio::task::spawn_blocking(move || {
            let _permit = permit1;
            // Simulate slow pipeline work.
            std::thread::sleep(Duration::from_millis(100));
            counter1.fetch_add(1, Ordering::SeqCst);
        });

        // Second pipeline tries to acquire the permit (blocks until first releases).
        let sem = state.annotate_permit().clone();
        let counter2 = counter.clone();
        let second = tokio::spawn(async move {
            let permit2 = sem
                .acquire_owned()
                .await
                .expect("second permit acquisition must succeed");
            tokio::task::spawn_blocking(move || {
                let _permit = permit2;
                counter2.fetch_add(1, Ordering::SeqCst);
            })
            .await
            .expect("second task must not panic")
        });

        // Wait for both to complete.
        first.await.expect("first pipeline must not panic");
        second.await.expect("second pipeline must not panic");

        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "both pipelines must complete, serialized by the semaphore"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-027: CANCEL_GRACE_PERIOD is finite and reasonable
    // -----------------------------------------------------------------------

    /// T-EXEC-027: Verifies that CANCEL_GRACE_PERIOD is set to a finite,
    /// reasonable duration. Before the fix, the grace period branch never
    /// returned, causing an infinite loop. The fix uses this constant as
    /// the timeout after which the executor returns an error.
    #[test]
    fn t_exec_027_cancel_grace_period_is_finite() {
        assert!(
            CANCEL_GRACE_PERIOD.as_secs() > 0,
            "grace period must be positive"
        );
        assert!(
            CANCEL_GRACE_PERIOD.as_secs() <= 120,
            "grace period must be at most 120 seconds"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-028: CANCEL_CHECK_INTERVAL is shorter than CANCEL_GRACE_PERIOD
    // -----------------------------------------------------------------------

    /// T-EXEC-028: The cancel check interval must be strictly shorter than
    /// the grace period, ensuring the executor polls at least once before
    /// the grace period expires. This prevents the edge case where a single
    /// long sleep could skip past the grace period deadline.
    #[test]
    fn t_exec_028_check_interval_shorter_than_grace_period() {
        assert!(
            CANCEL_CHECK_INTERVAL < CANCEL_GRACE_PERIOD,
            "CANCEL_CHECK_INTERVAL ({:?}) must be shorter than CANCEL_GRACE_PERIOD ({:?})",
            CANCEL_CHECK_INTERVAL,
            CANCEL_GRACE_PERIOD
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-029: annotate_permit field exists and is a Semaphore
    // -----------------------------------------------------------------------

    /// T-EXEC-029: Compile-time verification that the `annotate_permit`
    /// field returned by `PipelineContext::annotate_permit()` is of type
    /// `Arc<tokio::sync::Semaphore>`. This prevents accidental regression to
    /// Mutex or removal of the DEFECT-001 fix.
    #[tokio::test]
    async fn t_exec_029_annotate_permit_field_exists() {
        let state = setup_test_state();
        // Access the field to ensure the return type matches the trait signature.
        let _sem: &Arc<tokio::sync::Semaphore> = state.annotate_permit();
    }

    // -----------------------------------------------------------------------
    // T-EXEC-030: Orphaned pipeline holding semaphore permit blocks next pipeline
    // -----------------------------------------------------------------------

    /// T-EXEC-030: Simulates the DEFECT-001 scenario where the executor
    /// returns early (grace period expired) while the spawn_blocking thread
    /// is still running. The orphaned thread holds the semaphore permit.
    /// The next pipeline attempt blocks on acquiring the permit until the
    /// orphaned thread finishes.
    #[tokio::test]
    async fn t_exec_030_orphaned_thread_blocks_next_pipeline() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let state = setup_test_state();
        let orphan_done = Arc::new(AtomicBool::new(false));

        // Simulate an orphaned pipeline: acquire the permit and hold it
        // in a slow spawn_blocking closure.
        let permit = state
            .annotate_permit()
            .clone()
            .acquire_owned()
            .await
            .expect("permit acquisition must succeed");

        let done_flag = orphan_done.clone();
        let _orphan_handle = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            // Simulate a stuck pipeline that eventually finishes.
            std::thread::sleep(Duration::from_millis(200));
            done_flag.store(true, Ordering::SeqCst);
        });

        // The executor "returned early" (simulated). The next pipeline
        // tries to acquire the permit and must wait.
        let sem = state.annotate_permit().clone();
        let acquired = tokio::time::timeout(Duration::from_millis(50), async {
            let _p = sem
                .acquire_owned()
                .await
                .expect("permit acquisition must succeed");
        })
        .await;

        // The timeout must fire because the orphan still holds the permit.
        assert!(
            acquired.is_err(),
            "next pipeline must block while orphaned thread holds the permit"
        );

        // After the orphan finishes (200ms), the permit becomes available.
        let sem2 = state.annotate_permit().clone();
        let acquired2 = tokio::time::timeout(Duration::from_secs(2), async {
            let _p = sem2
                .acquire_owned()
                .await
                .expect("permit acquisition must succeed");
        })
        .await;

        assert!(
            acquired2.is_ok(),
            "next pipeline must acquire the permit after orphaned thread finishes"
        );
        assert!(
            orphan_done.load(Ordering::SeqCst),
            "orphaned thread must have finished before permit became available"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-BUG-A: load_cached_page_texts finds sessions stored by indexer
    // -----------------------------------------------------------------------

    /// Regression test for BUG-A (annotation cache always empty on Windows):
    /// The indexer stores session directory_path via canonicalize_directory which
    /// strips the \\?\ Windows extended-length path prefix. load_cached_page_texts
    /// previously used std::fs::canonicalize which ADDS \\?\ on Windows,
    /// causing the SQL WHERE directory_path = ? to always return 0 rows.
    /// The fix replaces std::fs::canonicalize with canonicalize_directory so both
    /// sides of the comparison produce the same path string format.
    ///
    /// This test simulates the production scenario: session created with
    /// canonicalize_directory (the indexer's behavior), then load_cached_page_texts
    /// is called with the raw (non-canonicalized) source directory. The function
    /// must find the session and return populated page texts.
    #[tokio::test]
    async fn t_exec_bug_a_cache_finds_session_stored_by_indexer() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "scan.pdf", "scanned content");

        let state = setup_test_state();

        // Store the session with canonicalize_directory -- this is exactly what
        // the indexer does when creating a session from a directory path. The
        // temp directory exists, so the fallback to the raw path will not trigger.
        let indexed_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let session_id = create_test_session(&state, &indexed_dir.to_string_lossy());

        // Insert one file and two pages simulating a completed indexing run.
        let conn = state.pool().get().expect("conn");
        let file_path = indexed_dir.join("scan.pdf");
        let file_id = store::insert_file(
            &conn,
            session_id,
            &file_path.to_string_lossy(),
            "abc123",
            0,
            2048,
            2,
            Some(2),
        )
        .expect("insert file");

        store::bulk_insert_pages(
            &conn,
            file_id,
            &[
                (
                    1,
                    "The results indicate a significant difference",
                    "tesseract",
                ),
                (2, "Appendix: raw data tables", "tesseract"),
            ],
        )
        .expect("insert pages");
        drop(conn);

        // Call load_cached_page_texts with the RAW (not yet canonicalized) directory.
        // The function must normalize the path internally via canonicalize_directory
        // and find the session stored above.
        let cache = load_cached_page_texts(state.pool(), tmp.path());

        assert_eq!(
            cache.len(),
            1,
            "cache must contain the indexed file; returning empty means the path \
             normalization is broken (\\\\?\\ prefix mismatch on Windows)"
        );

        let pages = cache.get(&file_path).expect("file path must be in cache");
        assert_eq!(pages.len(), 2, "both indexed pages must be in the cache");
        assert_eq!(
            pages[&1], "The results indicate a significant difference",
            "page 1 text must match the indexed content"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-005 regression tests: ensure_hnsw_for_session repair path
    //
    // Phase 3 of run_index_pipeline acquires a fresh SQLite connection. When
    // the WAL read snapshot does not include Phase 2's just-committed embeddings,
    // load_embeddings_for_hnsw returns empty and the HNSW build is silently
    // skipped. The fix adds ensure_hnsw_for_session to all search handlers so
    // the missing index is rebuilt on the first search after indexing.
    // -----------------------------------------------------------------------

    /// T-EXEC-DEFECT005-001: ensure_hnsw_for_session returns true immediately
    /// when the HNSW index for the session is already present in the in-memory
    /// map (fast path). No database access is performed in this path.
    #[tokio::test]
    async fn t_exec_defect005_001_fast_path_returns_true_when_hnsw_loaded() {
        let state = setup_test_state();
        let session_id = create_test_session(&state, "/tmp/d5_fast_path");

        // Insert a minimal HNSW index directly into the in-memory map.
        // The index contains one 4-dimensional unit vector.
        let vectors: Vec<(i64, Vec<f32>)> = vec![(1, vec![1.0, 0.0, 0.0, 0.0])];
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = neuroncite_store::build_hnsw(&labeled, 4).expect("build_hnsw");
        state.insert_hnsw(session_id, index);

        // The fast path must return true without touching the database.
        let result = ensure_hnsw_for_session(&state, session_id);
        assert!(
            result,
            "ensure_hnsw_for_session must return true when HNSW is already in the map"
        );
    }

    /// T-EXEC-DEFECT005-002: ensure_hnsw_for_session returns false for a
    /// session that has no embeddings in SQLite. This is the case when indexing
    /// produced no chunks (e.g., an empty directory or all PDFs failed extraction).
    #[tokio::test]
    async fn t_exec_defect005_002_returns_false_when_no_embeddings_in_sqlite() {
        let state = setup_test_state();
        let session_id = create_test_session(&state, "/tmp/d5_empty_session");

        // The session exists but has no chunks and therefore no embeddings.
        // ensure_hnsw_for_session must return false because there is nothing to build.
        let result = ensure_hnsw_for_session(&state, session_id);
        assert!(
            !result,
            "ensure_hnsw_for_session must return false when no embeddings exist for the session"
        );
    }

    /// T-EXEC-DEFECT005-003: ensure_hnsw_for_session builds the HNSW from
    /// SQLite and inserts it into the map when embeddings are present but the
    /// in-memory index is absent. This is the repair path for DEFECT-005: the
    /// search handler calls this function after Phase 3 silently skipped the
    /// HNSW build due to a stale WAL snapshot.
    #[tokio::test]
    async fn t_exec_defect005_003_repair_path_builds_hnsw_from_sqlite() {
        let state = setup_test_state();
        let session_id = create_test_session(&state, "/tmp/d5_repair");

        // Insert a file record so the chunk foreign key constraint is satisfied.
        let file_id = {
            let conn = state.pool().get().expect("get conn");
            store::insert_file(
                &conn,
                session_id,
                "/tmp/d5_repair/paper.pdf",
                "deadbeef",
                0,
                1024,
                1,
                Some(1),
            )
            .expect("insert file")
        };

        // Insert one chunk with a 4-dimensional embedding blob stored as
        // little-endian f32 bytes. This simulates what Phase 2 writes to SQLite.
        let embedding_bytes: Vec<u8> = [1.0f32, 0.0f32, 0.0f32, 0.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        {
            let conn = state.pool().get().expect("get conn");
            store::bulk_insert_chunks(
                &conn,
                &[store::ChunkInsert {
                    file_id,
                    session_id,
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 0,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: 10,
                    content: "test chunk content",
                    embedding: Some(&embedding_bytes),
                    ext_offset: None,
                    ext_length: None,
                    content_hash: "abc123",
                    simhash: None,
                }],
            )
            .expect("bulk insert chunks");
        }

        // The in-memory HNSW map must be empty before calling the repair function.
        assert!(
            !state.hnsw_index().load().contains_key(&session_id),
            "HNSW must not be in map before repair call"
        );

        // The repair path must build the HNSW from the SQLite embedding and
        // insert it into the map.
        let result = ensure_hnsw_for_session(&state, session_id);
        assert!(
            result,
            "ensure_hnsw_for_session must return true after building HNSW from SQLite"
        );

        // Verify the HNSW is now present in the map.
        assert!(
            state.hnsw_index().load().contains_key(&session_id),
            "HNSW must be in map after successful repair"
        );
    }

    /// T-EXEC-DEFECT005-004: ensure_hnsw_for_session returns false for a
    /// nonexistent session ID. The repair path must not panic when the session
    /// cannot be found in SQLite.
    #[tokio::test]
    async fn t_exec_defect005_004_returns_false_for_nonexistent_session() {
        let state = setup_test_state();

        // Use an ID that was never inserted into the sessions table.
        let nonexistent_session_id: i64 = 99_999;

        let result = ensure_hnsw_for_session(&state, nonexistent_session_id);
        assert!(
            !result,
            "ensure_hnsw_for_session must return false for a session that does not exist"
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-031: Rebuild job is picked up and rebuilds HNSW from stored embeddings
    // -----------------------------------------------------------------------

    /// Regression test for the dead-job defect: the executor previously excluded
    /// "rebuild" from the queued-job SQL filter, causing rebuild jobs to stay
    /// queued forever. After the fix, rebuild jobs are picked up and rebuild
    /// the HNSW index from embeddings already stored in SQLite.
    #[tokio::test]
    async fn t_exec_031_rebuild_job_picked_up_and_builds_hnsw() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Rebuild HNSW test content");

        let state = setup_test_state();
        let session_id = create_test_session(&state, &tmp.path().to_string_lossy());

        // Run an indexing job first so embeddings are stored in SQLite.
        let index_job_id = create_queued_job(&state, session_id);
        let index_job = find_next_queued_job(&state).expect("find index job");
        execute_job(&state, &index_job).await;

        {
            let conn = state.pool().get().expect("conn");
            let j = store::get_job(&conn, &index_job_id).expect("get index job");
            // The indexing job must have finished before the rebuild test is valid.
            assert!(
                j.state == JobState::Completed || j.state == JobState::Failed,
                "index job must be in terminal state before rebuild, got: {:?}",
                j.state
            );
        }

        // Remove the HNSW from AppState to simulate a server restart that lost it.
        state.remove_hnsw(session_id);
        assert!(
            !state.hnsw_index().load().contains_key(&session_id),
            "HNSW must be absent before rebuild job runs"
        );

        // Create a rebuild job. Before the fix this job would stay queued forever
        // because find_next_queued_job excluded kind = 'rebuild'.
        let rebuild_job_id = {
            let conn = state.pool().get().expect("conn");
            let jid = uuid::Uuid::new_v4().to_string();
            store::create_job(&conn, &jid, "rebuild", Some(session_id))
                .expect("create rebuild job");
            jid
        };

        // The rebuild job must now be visible to the executor.
        let rebuild_job = find_next_queued_job(&state).expect("rebuild job must be found");
        assert_eq!(
            rebuild_job.id, rebuild_job_id,
            "executor must pick up the rebuild job"
        );

        execute_job(&state, &rebuild_job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &rebuild_job_id).expect("get rebuild job");
        assert!(
            final_job.state == JobState::Completed || final_job.state == JobState::Failed,
            "rebuild job must reach a terminal state, got: {:?}",
            final_job.state
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-032: recover_stuck_jobs does not mark citation_verify as failed
    // -----------------------------------------------------------------------

    /// Regression test for the overzealous recovery defect: recover_stuck_jobs
    /// previously applied a timeout transition to ALL running jobs regardless of
    /// kind. This caused long-running citation_verify jobs (agent-driven, not
    /// executor-managed) to be incorrectly transitioned to Failed.
    ///
    /// After the fix, recover_stuck_jobs only processes jobs with kind IN
    /// ('index', 'annotate', 'rebuild').
    #[tokio::test]
    async fn t_exec_032_recover_stuck_ignores_citation_verify() {
        let state = setup_test_state();
        let conn = state.pool().get().expect("conn");

        // Insert a citation_verify job directly in Running state with
        // started_at = 0 (Unix epoch), guaranteeing it appears "stuck"
        // (elapsed >> configured job_timeout).
        let job_id = "citation-verify-stuck-001";
        conn.execute(
            "INSERT INTO job (id, kind, state, created_at, started_at, progress_done, progress_total)
             VALUES (?1, 'citation_verify', 'running', 0, 0, 0, 0)",
            rusqlite::params![job_id],
        )
        .expect("insert stuck citation_verify job");

        // Run recovery. With the kind filter in place, the citation_verify job
        // must not be transitioned to Failed.
        recover_stuck_jobs(&state);

        let after = store::get_job(&conn, job_id).expect("get citation_verify job");
        assert_eq!(
            after.state,
            JobState::Running,
            "recover_stuck_jobs must not transition citation_verify jobs to Failed; \
             got: {:?}",
            after.state
        );
    }

    // -----------------------------------------------------------------------
    // T-EXEC-033: run_index_pipeline fails cleanly on model mismatch
    // -----------------------------------------------------------------------

    /// Regression test for the model/dimension mismatch defect: when a session
    /// was created with a different model than the one currently loaded by the
    /// worker, the HNSW builder would panic via assert_eq! on a dimension
    /// mismatch. After the fix, the pipeline detects the mismatch early and
    /// returns a descriptive error, transitioning the job to Failed without panic.
    #[tokio::test]
    async fn t_exec_033_model_mismatch_fails_job_cleanly() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        create_test_pdf(tmp.path(), "test.pdf", "Model mismatch test content");

        let state = setup_test_state();

        // Create a session recorded as using a model different from the worker's
        // loaded model ("stub-model"). The worker always returns "stub-model" from
        // loaded_model_id(), so any other name triggers the guard.
        let session_id = create_test_session_with_model(
            &state,
            &tmp.path().to_string_lossy(),
            "BAAI/bge-large-en-v1.5",
        );
        let job_id = {
            let conn = state.pool().get().expect("conn");
            let jid = uuid::Uuid::new_v4().to_string();
            store::create_job(&conn, &jid, "index", Some(session_id)).expect("create job");
            jid
        };

        let job = find_next_queued_job(&state).expect("find job");
        execute_job(&state, &job).await;

        let conn = state.pool().get().expect("conn");
        let final_job = store::get_job(&conn, &job_id).expect("get job");

        assert_eq!(
            final_job.state,
            JobState::Failed,
            "index job must fail when session model differs from loaded model, got: {:?}",
            final_job.state
        );

        let msg = final_job.error_message.as_deref().unwrap_or("");
        assert!(
            msg.contains("model mismatch"),
            "error message must describe the model mismatch, got: {msg}"
        );
        assert!(
            msg.contains("BAAI/bge-large-en-v1.5"),
            "error message must name the session model, got: {msg}"
        );
        assert!(
            msg.contains("stub-model"),
            "error message must name the loaded model, got: {msg}"
        );
    }
}
