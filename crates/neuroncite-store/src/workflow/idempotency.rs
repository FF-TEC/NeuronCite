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

// Idempotency key management for resumable indexing.
//
// Each client-supplied idempotency key is stored alongside the associated
// job_id and session_id. Before processing an indexing request, the workflow
// checks whether an idempotency key already exists. If it does, the original
// job and session IDs are returned without re-processing. Entries expire
// after 24 hours, matching the job retention window.
//
// The critical path for new requests uses `create_job_with_key`, which wraps
// the lookup, job creation, and key storage in a single SQLite
// `BEGIN IMMEDIATE` transaction. This prevents the race where two concurrent
// requests both observe a missing key, both create separate jobs, and only
// the first to call `store_key` succeeds while leaving the second job
// orphaned without an idempotency record.

use rusqlite::{Connection, TransactionBehavior, params};

use crate::error::StoreError;

/// Row representation of an idempotency entry. Contains all columns defined
/// in the idempotency table.
#[derive(Debug, Clone)]
pub struct IdempotencyRow {
    pub key: String,
    pub job_id: String,
    pub session_id: i64,
    pub created_at: i64,
}

/// Result of an atomic job-creation-with-idempotency-key operation.
///
/// The caller uses this to distinguish between a newly created job (for which
/// the executor must be notified) and a replay of a previously completed
/// request (for which no further action is needed).
#[derive(Debug)]
pub enum CreateJobResult {
    /// A new job was created. Contains the assigned job_id and session_id.
    Created { job_id: String, session_id: i64 },
    /// A job already existed for this idempotency key. Returns the existing
    /// entry so the handler can return the original job_id and session_id.
    Existing(IdempotencyRow),
}

/// Creates a new indexing job and stores its idempotency key atomically.
///
/// All three logical steps — check for an existing key, insert the job row,
/// and insert the idempotency record — execute inside a single
/// `BEGIN IMMEDIATE` transaction. `BEGIN IMMEDIATE` acquires a write lock
/// before the first statement, so no other writer can interleave between the
/// lookup and the insert. This eliminates the race window that would otherwise
/// allow two concurrent requests with the same key to each create a separate
/// job row before either stores the key.
///
/// # Arguments
///
/// * `conn` - A mutable connection reference used for the transaction.
/// * `key` - The client-supplied idempotency key.
/// * `job_id` - The pre-generated UUID string for the new job.
/// * `job_kind` - Job kind string (e.g., `"index"`).
/// * `session_id` - The session the job belongs to.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the transaction fails (e.g., due to a
/// constraint violation on the job or idempotency tables).
pub fn create_job_with_key(
    conn: &mut Connection,
    key: &str,
    job_id: &str,
    job_kind: &str,
    session_id: i64,
) -> Result<CreateJobResult, StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();

    // BEGIN IMMEDIATE acquires the write lock before any reads inside the
    // transaction. This closes the window between the SELECT on the
    // idempotency table and the subsequent INSERT: any concurrent request
    // with the same key will block until this transaction commits or rolls back.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

    // Step 1: check whether the key already exists.
    let existing: Option<IdempotencyRow> = {
        let result = tx.query_row(
            "SELECT key, job_id, session_id, created_at
             FROM idempotency WHERE key = ?1",
            params![key],
            |row| {
                Ok(IdempotencyRow {
                    key: row.get(0)?,
                    job_id: row.get(1)?,
                    session_id: row.get(2)?,
                    created_at: row.get(3)?,
                })
            },
        );
        match result {
            Ok(entry) => Some(entry),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        }
    };

    if let Some(entry) = existing {
        // Key exists: roll back and return the existing entry. No job is
        // created and the caller must not notify the executor.
        tx.rollback()?;
        return Ok(CreateJobResult::Existing(entry));
    }

    // Step 2: create the job row inside the same transaction.
    tx.execute(
        "INSERT INTO job (id, kind, session_id, state, progress_done, progress_total, created_at)
         VALUES (?1, ?2, ?3, 'queued', 0, 0, ?4)",
        params![job_id, job_kind, session_id, now],
    )?;

    // Step 3: store the idempotency key, linking it to the new job.
    tx.execute(
        "INSERT INTO idempotency (key, job_id, session_id, created_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![key, job_id, session_id, now],
    )?;

    tx.commit()?;

    Ok(CreateJobResult::Created {
        job_id: job_id.to_string(),
        session_id,
    })
}

/// Stores an idempotency key with its associated job and session IDs.
///
/// Used only for code paths that manage job creation separately and need to
/// record the key after the fact. When an idempotency key is present in a new
/// index request, callers must use `create_job_with_key` instead to guarantee
/// atomicity between job creation and key storage.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the key already exists (primary key
/// violation) or if a foreign key constraint is violated.
pub fn store_key(
    conn: &Connection,
    key: &str,
    job_id: &str,
    session_id: i64,
) -> Result<(), StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();

    conn.execute(
        "INSERT INTO idempotency (key, job_id, session_id, created_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![key, job_id, session_id, now],
    )?;

    Ok(())
}

/// Looks up an idempotency key. Returns the full entry if found, or `None`
/// if no entry exists for the given key.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn lookup_key(conn: &Connection, key: &str) -> Result<Option<IdempotencyRow>, StoreError> {
    let result = conn.query_row(
        "SELECT key, job_id, session_id, created_at FROM idempotency WHERE key = ?1",
        params![key],
        |row| {
            Ok(IdempotencyRow {
                key: row.get(0)?,
                job_id: row.get(1)?,
                session_id: row.get(2)?,
                created_at: row.get(3)?,
            })
        },
    );

    match result {
        Ok(entry) => Ok(Some(entry)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Deletes all idempotency entries whose `created_at` timestamp is older than
/// 24 hours. Returns the number of entries deleted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete statement fails.
pub fn cleanup_expired_keys(conn: &Connection) -> Result<usize, StoreError> {
    let cutoff = neuroncite_core::unix_timestamp_secs() - 86400; // 24 hours in seconds
    let count = conn.execute(
        "DELETE FROM idempotency WHERE created_at < ?1",
        params![cutoff],
    )?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::session;
    use crate::schema::migrate;
    use crate::workflow::job;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    /// Helper: creates an in-memory database with schema, a session, and a job.
    fn setup_db_with_job() -> (Connection, i64, String) {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");

        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let session_id =
            session::create_session(&conn, &config, "0.1.0").expect("session creation failed");

        let job_id = "test-job-idem";
        job::create_job(&conn, job_id, "index", Some(session_id)).expect("job creation failed");

        (conn, session_id, job_id.to_string())
    }

    /// Helper: creates an in-memory database with only the schema (no pre-existing session/job).
    fn setup_empty_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");
        conn
    }

    /// Helper: creates a session in the given connection. Returns the session_id.
    fn make_session(conn: &Connection) -> i64 {
        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        session::create_session(conn, &config, "0.1.0").expect("session creation failed")
    }

    /// T-STO-096: Idempotency key lookup. Storing an idempotency entry and
    /// querying by key returns the associated job_id and session_id. A
    /// non-existent key returns None.
    #[test]
    fn t_sto_096_idempotency_key_lookup() {
        let (conn, session_id, job_id) = setup_db_with_job();

        // Store an idempotency key
        store_key(&conn, "idem-key-001", &job_id, session_id).expect("store_key failed");

        // Lookup the stored key
        let entry = lookup_key(&conn, "idem-key-001")
            .expect("lookup_key failed")
            .expect("entry must exist");

        assert_eq!(entry.key, "idem-key-001");
        assert_eq!(entry.job_id, job_id);
        assert_eq!(entry.session_id, session_id);

        // Lookup a non-existent key
        let missing = lookup_key(&conn, "nonexistent-key").expect("lookup_key failed");
        assert!(missing.is_none(), "non-existent key must return None");
    }

    /// T-STO-073: Atomic job creation with idempotency key — first call creates
    /// the job; a second call with the same key returns the existing entry
    /// without creating a duplicate job row.
    #[test]
    fn t_sto_073_create_job_with_key_atomic_idempotency() {
        let mut conn = setup_empty_db();
        let session_id = make_session(&conn);

        let job_id_a = "job-atomic-001";

        // First call: job does not exist yet, must be created.
        let result_a = create_job_with_key(&mut conn, "idem-a", job_id_a, "index", session_id)
            .expect("first create_job_with_key must succeed");

        let returned_job_id = match result_a {
            CreateJobResult::Created {
                job_id,
                session_id: sid,
            } => {
                assert_eq!(job_id, job_id_a);
                assert_eq!(sid, session_id);
                job_id
            }
            CreateJobResult::Existing(_) => {
                panic!("first call must return Created, not Existing")
            }
        };

        // The job row must now exist in the database.
        let job_row = job::get_job(&conn, &returned_job_id).expect("job must be present in DB");
        assert_eq!(job_row.id, job_id_a);
        assert_eq!(job_row.state, crate::workflow::job::JobState::Queued);

        // The idempotency key must be stored.
        let idem = lookup_key(&conn, "idem-a")
            .expect("lookup_key must succeed")
            .expect("idempotency entry must exist");
        assert_eq!(idem.job_id, job_id_a);

        // Second call with the same key must return the existing entry and
        // must NOT create a second job row.
        let result_b =
            create_job_with_key(&mut conn, "idem-a", "job-atomic-002", "index", session_id)
                .expect("second create_job_with_key must succeed");

        match result_b {
            CreateJobResult::Existing(entry) => {
                assert_eq!(entry.job_id, job_id_a, "must return the original job_id");
                assert_eq!(entry.session_id, session_id);
            }
            CreateJobResult::Created { .. } => {
                panic!("second call with same key must return Existing, not Created")
            }
        }

        // Verify that only one job row exists for this session (no duplicate).
        let all_jobs = job::list_jobs(&conn).expect("list_jobs must succeed");
        let index_jobs: Vec<_> = all_jobs
            .iter()
            .filter(|j| j.kind == "index" && j.session_id == Some(session_id))
            .collect();
        assert_eq!(
            index_jobs.len(),
            1,
            "exactly one job row must exist; concurrent call must not create a duplicate"
        );
    }

    /// T-STO-075: Atomic job creation with idempotency key — two different keys
    /// on the same session create two independent jobs.
    #[test]
    fn t_sto_075_create_job_with_key_different_keys_create_separate_jobs() {
        let mut conn = setup_empty_db();
        let session_id = make_session(&conn);

        let result_a = create_job_with_key(&mut conn, "key-x", "job-x-001", "index", session_id)
            .expect("create job X must succeed");
        assert!(matches!(result_a, CreateJobResult::Created { .. }));

        let result_b = create_job_with_key(&mut conn, "key-y", "job-y-001", "index", session_id)
            .expect("create job Y must succeed");
        assert!(matches!(result_b, CreateJobResult::Created { .. }));

        // Both job rows must be present.
        job::get_job(&conn, "job-x-001").expect("job X must exist");
        job::get_job(&conn, "job-y-001").expect("job Y must exist");

        // Both idempotency entries must be present.
        assert!(lookup_key(&conn, "key-x").expect("lookup X").is_some());
        assert!(lookup_key(&conn, "key-y").expect("lookup Y").is_some());
    }
}
