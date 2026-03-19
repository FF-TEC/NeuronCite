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

// Background job state tracking.
//
// Records the lifecycle of each background job (indexing, annotation): creation
// timestamp, current status (queued, running, completed, failed, canceled),
// progress counters (done / total), error messages for failed jobs, and optional
// params_json for job-specific parameters. The API server reads job records to
// report progress to the GUI and MCP clients.
//
// State machine:
//   Queued -> Running -> Completed
//   Queued -> Running -> Failed
//   Queued -> Canceled
//   Running -> Canceled
//   Running -> Failed
//   Completed -> Running  (citation_retry re-opens a completed job)
//
// Failed and Canceled are terminal states. All other transitions return an error.

use std::fmt;

use rusqlite::{Connection, params};

use crate::error::StoreError;

/// Represents the lifecycle states of a background job. The state machine
/// allows forward transitions plus the Completed -> Running re-entry
/// used by citation_retry to re-open a finished verification job.
/// Failed and Canceled are terminal states that cannot transition further.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// The job has been created but has not started execution.
    Queued,
    /// The job is actively processing items.
    Running,
    /// The job finished all items successfully.
    Completed,
    /// The job encountered an unrecoverable error.
    Failed,
    /// The job was canceled by the user or system before completion.
    Canceled,
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Canceled => write!(f, "canceled"),
        }
    }
}

impl JobState {
    /// Parses a state string from the database back to the enum variant.
    /// Returns None if the string does not match any known state.
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "canceled" => Some(Self::Canceled),
            _ => None,
        }
    }

    /// Validates whether a transition from `self` to `target` is allowed.
    /// Returns true if the transition is valid according to the state machine.
    /// The Completed -> Running transition enables citation_retry to re-open
    /// a completed job so sub-agents can claim reset rows.
    fn can_transition_to(self, target: Self) -> bool {
        matches!(
            (self, target),
            (Self::Queued, Self::Running | Self::Canceled)
                | (
                    Self::Running,
                    Self::Completed | Self::Failed | Self::Canceled
                )
                | (Self::Completed, Self::Running)
        )
    }
}

/// Row representation of a job record. Contains all columns defined in
/// the job table.
#[derive(Debug, Clone)]
pub struct JobRow {
    pub id: String,
    pub kind: String,
    pub session_id: Option<i64>,
    pub state: JobState,
    pub progress_done: i64,
    pub progress_total: i64,
    pub error_message: Option<String>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    /// Serialized JSON containing job-specific parameters. Used by annotation
    /// jobs to store input data, directories, and color configuration.
    pub params_json: Option<String>,
}

/// Creates a job record with the given UUID, kind, and optional session ID.
/// The initial state is Queued with zero progress. Delegates to
/// `create_job_with_params` with `params_json = None`.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert fails (e.g., duplicate ID).
pub fn create_job(
    conn: &Connection,
    id: &str,
    kind: &str,
    session_id: Option<i64>,
) -> Result<(), StoreError> {
    create_job_with_params(conn, id, kind, session_id, None)
}

/// Creates a job record with the given UUID, kind, optional session ID, and
/// optional serialized JSON parameters. The params_json field stores
/// job-specific configuration (e.g., annotation job input data and paths).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert fails (e.g., duplicate ID).
pub fn create_job_with_params(
    conn: &Connection,
    id: &str,
    kind: &str,
    session_id: Option<i64>,
    params_json: Option<&str>,
) -> Result<(), StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();

    conn.execute(
        "INSERT INTO job (id, kind, session_id, state, progress_done, progress_total, created_at, params_json)
         VALUES (?1, ?2, ?3, ?4, 0, 0, ?5, ?6)",
        params![id, kind, session_id, JobState::Queued.to_string(), now, params_json],
    )?;

    Ok(())
}

/// Retrieves a single job by its primary key.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no job with the given ID exists,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_job(conn: &Connection, id: &str) -> Result<JobRow, StoreError> {
    conn.query_row(
        "SELECT id, kind, session_id, state, progress_done, progress_total,
                error_message, created_at, started_at, finished_at, params_json
         FROM job WHERE id = ?1",
        params![id],
        row_to_job,
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => StoreError::not_found("job", id.to_string()),
        other => other.into(),
    })
}

/// Checks whether at least one job of the given `kind` is in an active state
/// (queued or running). Uses an `EXISTS` subquery against the partial index
/// `idx_job_state_active`, which covers only active states, avoiding a full
/// table scan of completed/failed/canceled jobs.
///
/// This function replaces the previous pattern of calling `list_jobs()` and
/// filtering with `.iter().any(...)`, which fetched all columns of all jobs
/// just to check for a single boolean condition.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn has_active_job(conn: &Connection, kind: &str) -> Result<bool, StoreError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM job
            WHERE kind = ?1 AND state IN ('queued', 'running')
            LIMIT 1
        )",
        rusqlite::params![kind],
        |row| row.get(0),
    )?;
    Ok(exists)
}

/// Maximum number of job rows returned by `list_jobs`. Prevents unbounded
/// memory use when the job table grows large. Completed and failed jobs
/// are cleaned up after 24 hours by `cleanup_expired_jobs`, so this limit
/// is only reached when the cleanup task is disabled or lagging.
const LIST_JOBS_MAX: i64 = 10_000;

/// Lists jobs ordered by creation time descending, capped at `LIST_JOBS_MAX`
/// rows to prevent unbounded memory use.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_jobs(conn: &Connection) -> Result<Vec<JobRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, kind, session_id, state, progress_done, progress_total,
                error_message, created_at, started_at, finished_at, params_json
         FROM job ORDER BY created_at DESC LIMIT ?1",
    )?;

    let rows = stmt.query_map(rusqlite::params![LIST_JOBS_MAX], row_to_job)?;
    let mut jobs = Vec::new();
    for row in rows {
        jobs.push(row?);
    }
    Ok(jobs)
}

/// Transitions a job to a new state. Validates the state transition and
/// updates the `started_at` or `finished_at` timestamps as appropriate.
///
/// When transitioning to Running, sets `started_at` to the current time.
/// When transitioning to Completed, Failed, or Canceled, sets `finished_at`.
///
/// For the Failed state, an optional `error_message` can be provided.
///
/// # Errors
///
/// Returns `StoreError::Manifest` (used here as a validation error)
/// if the state transition is invalid, or `StoreError::Sqlite` if the
/// update fails.
/// Transitions a job to a new state using an optimistic-locking UPDATE.
///
/// The WHERE clause includes both the job ID and the expected current state
/// (`AND state = ?from_state`). If 0 rows are affected, the job either does
/// not exist or was already transitioned by a concurrent caller. This
/// eliminates the TOCTOU race between the old SELECT-then-UPDATE pattern.
///
/// Validation uses `can_transition_to` to check the state machine rules
/// before issuing the UPDATE, matching the old behaviour.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no job with the given ID exists,
/// `StoreError::Manifest` if the current stored state does not allow the
/// requested transition or if a concurrent update won the race,
/// or `StoreError::Sqlite` if the query fails.
pub fn update_job_state(
    conn: &Connection,
    id: &str,
    new_state: JobState,
    error_message: Option<&str>,
) -> Result<(), StoreError> {
    let current = get_job(conn, id)?;

    if !current.state.can_transition_to(new_state) {
        return Err(StoreError::Manifest {
            reason: format!(
                "invalid job state transition: {} -> {new_state}",
                current.state
            ),
        });
    }

    let now = neuroncite_core::unix_timestamp_secs();
    let from_state = current.state.to_string();

    let affected = match new_state {
        JobState::Running => conn.execute(
            "UPDATE job SET state = ?1, started_at = ?2 WHERE id = ?3 AND state = ?4",
            params![new_state.to_string(), now, id, from_state],
        )?,
        JobState::Completed | JobState::Failed | JobState::Canceled => conn.execute(
            "UPDATE job SET state = ?1, finished_at = ?2, error_message = ?3 WHERE id = ?4 AND state = ?5",
            params![new_state.to_string(), now, error_message, id, from_state],
        )?,
        JobState::Queued => conn.execute(
            // Transitioning back to Queued is not permitted by the state machine.
            // This branch exists only to make the match exhaustive and will only
            // be reached if can_transition_to() is wrong, which it is not.
            "UPDATE job SET state = ?1 WHERE id = ?2 AND state = ?3",
            params![new_state.to_string(), id, from_state],
        )?,
    };

    if affected == 0 {
        // 0 rows affected means the stored state changed between the SELECT
        // and the UPDATE (concurrent transition). Treat this as a conflict.
        return Err(StoreError::Manifest {
            reason: format!(
                "job {id} state changed concurrently: expected {from_state}, transition to {new_state} lost the race"
            ),
        });
    }

    Ok(())
}

/// Updates the progress counters (done and total) for a running job.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the update fails.
pub fn update_job_progress(
    conn: &Connection,
    id: &str,
    done: i64,
    total: i64,
) -> Result<(), StoreError> {
    conn.execute(
        "UPDATE job SET progress_done = ?1, progress_total = ?2 WHERE id = ?3",
        params![done, total, id],
    )?;
    Ok(())
}

/// Deletes all jobs whose `finished_at` timestamp is older than 24 hours.
/// Returns the number of jobs deleted.
///
/// Jobs that have not finished (`finished_at` IS NULL) are never deleted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn cleanup_expired_jobs(conn: &Connection) -> Result<usize, StoreError> {
    let cutoff = neuroncite_core::unix_timestamp_secs() - 86400; // 24 hours in seconds
    let count = conn.execute(
        "DELETE FROM job WHERE finished_at IS NOT NULL AND finished_at < ?1",
        params![cutoff],
    )?;
    Ok(count)
}

/// Maps a `rusqlite` row to a `JobRow`. Reads 11 columns: the original 10
/// columns plus the params_json column at index 10.
///
/// Returns `rusqlite::Error::InvalidColumnType` when the state string stored
/// in the database does not match any known `JobState` variant. This prevents
/// silent data corruption: an unrecognized state would previously default to
/// `Queued`, causing a finished job to re-appear as pending.
fn row_to_job(row: &rusqlite::Row<'_>) -> rusqlite::Result<JobRow> {
    let state_str: String = row.get(3)?;
    let state = JobState::from_str(&state_str).ok_or_else(|| {
        rusqlite::Error::InvalidColumnType(3, "state".into(), rusqlite::types::Type::Text)
    })?;

    Ok(JobRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        session_id: row.get(2)?,
        state,
        progress_done: row.get(4)?,
        progress_total: row.get(5)?,
        error_message: row.get(6)?,
        created_at: row.get(7)?,
        started_at: row.get(8)?,
        finished_at: row.get(9)?,
        params_json: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::migrate;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");
        conn
    }

    /// T-STO-093: Job state transitions.
    #[test]
    fn t_sto_093_job_state_transitions() {
        let conn = setup_db();

        let job_id = "test-job-001";
        create_job(&conn, job_id, "index", None).expect("create_job failed");

        let job = get_job(&conn, job_id).expect("get_job failed");
        assert_eq!(job.state, JobState::Queued);

        update_job_state(&conn, job_id, JobState::Running, None)
            .expect("transition to Running failed");
        let job = get_job(&conn, job_id).expect("get_job failed");
        assert_eq!(job.state, JobState::Running);
        assert!(job.started_at.is_some());

        update_job_state(&conn, job_id, JobState::Completed, None)
            .expect("transition to Completed failed");
        let job = get_job(&conn, job_id).expect("get_job failed");
        assert_eq!(job.state, JobState::Completed);
        assert!(job.finished_at.is_some());

        // Completed -> Running is allowed for citation_retry to re-open
        // a finished verification job for re-verification of reset rows.
        update_job_state(&conn, job_id, JobState::Running, None)
            .expect("Completed -> Running transition must succeed for citation_retry");
        let job = get_job(&conn, job_id).expect("get_job after re-open");
        assert_eq!(job.state, JobState::Running);
        assert!(
            job.started_at.is_some(),
            "started_at must be set after Completed -> Running"
        );

        let job_id2 = "test-job-002";
        create_job(&conn, job_id2, "index", None).expect("create_job failed");
        update_job_state(&conn, job_id2, JobState::Running, None).expect("transition failed");
        update_job_state(&conn, job_id2, JobState::Failed, Some("disk full"))
            .expect("transition to Failed failed");
        let job = get_job(&conn, job_id2).expect("get_job failed");
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.error_message.as_deref(), Some("disk full"));

        let job_id3 = "test-job-003";
        create_job(&conn, job_id3, "index", None).expect("create_job failed");
        update_job_state(&conn, job_id3, JobState::Canceled, None)
            .expect("transition to Canceled from Queued failed");
        let job = get_job(&conn, job_id3).expect("get_job failed");
        assert_eq!(job.state, JobState::Canceled);
    }

    /// T-STO-094: Job 24-hour retention cleanup.
    #[test]
    fn t_sto_094_job_24h_retention_cleanup() {
        let conn = setup_db();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock error")
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at, finished_at)
             VALUES ('old-job', 'index', 'completed', 10, 10, ?1, ?2)",
            params![now - 90000, now - 90000],
        )
        .expect("insert old job failed");

        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at, finished_at)
             VALUES ('recent-job', 'index', 'completed', 5, 5, ?1, ?2)",
            params![now - 82800, now - 82800],
        )
        .expect("insert recent job failed");

        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('running-job', 'index', 'running', 3, 10, ?1)",
            params![now - 100_000],
        )
        .expect("insert running job failed");

        let deleted = cleanup_expired_jobs(&conn).expect("cleanup failed");
        assert_eq!(deleted, 1, "only the 25-hour-old job should be deleted");

        let result = get_job(&conn, "old-job");
        assert!(result.is_err(), "old job must be deleted");

        get_job(&conn, "recent-job").expect("recent job must be retained");
        get_job(&conn, "running-job").expect("running job must be retained");
    }

    /// T-STO-095: create_job_with_params stores and retrieves params_json.
    /// Verifies that annotation-style jobs with JSON parameters can be created
    /// and the parameters are correctly read back via get_job.
    #[test]
    fn t_sto_095_create_job_with_params() {
        let conn = setup_db();

        let params = r#"{"source_directory":"/pdfs","output_directory":"/out"}"#;
        create_job_with_params(&conn, "annotate-1", "annotate", None, Some(params))
            .expect("create_job_with_params failed");

        let job = get_job(&conn, "annotate-1").expect("get_job failed");
        assert_eq!(job.kind, "annotate");
        assert_eq!(job.params_json.as_deref(), Some(params));
    }

    /// T-STO-074: create_job (without params) stores NULL for params_json.
    /// Verifies that the legacy create_job function produces a job with
    /// params_json = None.
    #[test]
    fn t_sto_074_create_job_without_params() {
        let conn = setup_db();

        create_job(&conn, "index-1", "index", None).expect("create_job failed");

        let job = get_job(&conn, "index-1").expect("get_job failed");
        assert!(
            job.params_json.is_none(),
            "params_json must be None for jobs created without params"
        );
    }

    /// T-STO-076: list_jobs returns all jobs including params_json. This
    /// test reproduces the original "no such column: params_json" bug by
    /// verifying that list_jobs correctly reads the params_json column.
    #[test]
    fn t_sto_076_list_jobs_includes_params_json() {
        let conn = setup_db();

        let params = r#"{"input_data":"title,author,quote\nA,B,C"}"#;
        create_job(&conn, "idx-1", "index", None).expect("create index job");
        create_job_with_params(&conn, "ann-1", "annotate", None, Some(params))
            .expect("create annotate job");

        let jobs = list_jobs(&conn).expect("list_jobs failed");
        assert_eq!(jobs.len(), 2);

        // list_jobs returns newest first (ORDER BY created_at DESC), but
        // both jobs have the same timestamp (created in the same second),
        // so just check both exist with correct params_json.
        let idx_job = jobs.iter().find(|j| j.id == "idx-1").expect("index job");
        assert!(idx_job.params_json.is_none());

        let ann_job = jobs.iter().find(|j| j.id == "ann-1").expect("annotate job");
        assert_eq!(ann_job.params_json.as_deref(), Some(params));
    }
}
