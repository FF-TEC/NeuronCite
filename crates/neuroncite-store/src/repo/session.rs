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

// Session repository operations.
//
// Provides functions to create, read, and delete session records in the
// `SQLite` database. A session represents a single indexed document collection
// rooted at a specific directory path with a specific embedding model and
// chunking configuration. The COALESCE-based unique index on `index_session`
// prevents duplicate sessions for the same configuration, including when
// nullable fields (`chunk_size`, `chunk_overlap`, `max_words`) are NULL.

use rusqlite::{Connection, params};

use neuroncite_core::IndexConfig;

use crate::error::StoreError;
use crate::schema;

/// Row representation of an `index_session` record. Contains all columns
/// defined in the `index_session` table.
#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: i64,
    pub directory_path: String,
    pub model_name: String,
    pub chunk_strategy: String,
    pub chunk_size: Option<i64>,
    pub chunk_overlap: Option<i64>,
    pub max_words: Option<i64>,
    pub ocr_language: String,
    pub vector_dimension: i64,
    pub embedding_storage_mode: String,
    pub schema_version: i64,
    pub app_version: String,
    pub hnsw_total: i64,
    pub hnsw_orphans: i64,
    pub last_rebuild_at: Option<i64>,
    pub created_at: i64,
    /// Optional human-readable label for the session, set via
    /// `update_session_label`. NULL by default.
    pub label: Option<String>,
    /// Optional JSON array of user-defined tag strings for categorizing the
    /// session (e.g. `["finance", "statistics"]`). Stored as a serialized JSON
    /// string in the database. NULL by default.
    pub tags: Option<String>,
    /// Optional JSON object of arbitrary key-value metadata pairs
    /// (e.g. `{"source": "manual", "priority": "high"}`). Stored as a serialized
    /// JSON string in the database. NULL by default.
    pub metadata: Option<String>,
}

impl SessionRow {
    /// Converts the `vector_dimension` field from `i64` (as stored in SQLite) to
    /// `usize` using a checked conversion. Returns `StoreError::IntegerOverflow`
    /// if the stored value is negative or exceeds the platform's `usize::MAX`.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::IntegerOverflow` when the i64 value cannot be
    /// represented as usize on the current platform.
    pub fn vector_dimension_usize(&self) -> Result<usize, StoreError> {
        usize::try_from(self.vector_dimension).map_err(|_| {
            StoreError::integer_overflow(format!(
                "session {} vector_dimension {} cannot be converted to usize",
                self.id, self.vector_dimension
            ))
        })
    }

    /// Converts an `Option<i64>` field (chunk_size, chunk_overlap, or max_words)
    /// to `Option<usize>` using a checked conversion. Returns
    /// `StoreError::IntegerOverflow` if the inner value is negative or exceeds
    /// the platform's `usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `val` - The optional i64 value from the database column.
    /// * `field_name` - The name of the field for use in the error message.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::IntegerOverflow` when the i64 value cannot be
    /// represented as usize on the current platform.
    pub fn i64_to_usize_opt(
        &self,
        val: Option<i64>,
        field_name: &str,
    ) -> Result<Option<usize>, StoreError> {
        match val {
            None => Ok(None),
            Some(v) => {
                let u = usize::try_from(v).map_err(|_| {
                    StoreError::integer_overflow(format!(
                        "session {} {field_name} value {v} cannot be converted to usize",
                        self.id
                    ))
                })?;
                Ok(Some(u))
            }
        }
    }

    /// Converts the `chunk_size` field from `Option<i64>` to `Option<usize>`.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::IntegerOverflow` when the value is negative or
    /// exceeds `usize::MAX`.
    pub fn chunk_size_usize(&self) -> Result<Option<usize>, StoreError> {
        self.i64_to_usize_opt(self.chunk_size, "chunk_size")
    }

    /// Converts the `chunk_overlap` field from `Option<i64>` to `Option<usize>`.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::IntegerOverflow` when the value is negative or
    /// exceeds `usize::MAX`.
    pub fn chunk_overlap_usize(&self) -> Result<Option<usize>, StoreError> {
        self.i64_to_usize_opt(self.chunk_overlap, "chunk_overlap")
    }

    /// Converts the `max_words` field from `Option<i64>` to `Option<usize>`.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::IntegerOverflow` when the value is negative or
    /// exceeds `usize::MAX`.
    pub fn max_words_usize(&self) -> Result<Option<usize>, StoreError> {
        self.i64_to_usize_opt(self.max_words, "max_words")
    }
}

/// Inserts a session record derived from the given `IndexConfig` and returns
/// the auto-generated session ID.
///
/// The `app_version` parameter is the application version string recorded for
/// diagnostic purposes. The `schema_version` is taken from the current schema
/// module constant.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert violates the COALESCE-based
/// unique index (i.e., a session with the same configuration exists).
pub fn create_session(
    conn: &Connection,
    config: &IndexConfig,
    app_version: &str,
) -> Result<i64, StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();

    conn.execute(
        "INSERT INTO index_session (
            directory_path, model_name, chunk_strategy,
            chunk_size, chunk_overlap, max_words,
            ocr_language, vector_dimension, embedding_storage_mode,
            schema_version, app_version,
            hnsw_total, hnsw_orphans, last_rebuild_at, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 0, 0, NULL, ?12)",
        params![
            config.directory.to_string_lossy().as_ref(),
            config.model_name,
            config.chunk_strategy,
            config.chunk_size.map(|v| v as i64),
            config.chunk_overlap.map(|v| v as i64),
            config.max_words.map(|v| v as i64),
            config.ocr_language,
            config.vector_dimension as i64,
            config.embedding_storage_mode.to_string(),
            i64::from(schema::schema_version()),
            app_version,
            now,
        ],
    )?;

    Ok(conn.last_insert_rowid())
}

/// Looks up a session whose configuration matches the given `IndexConfig`.
/// Returns the session ID if found, or `None` if no matching session exists.
///
/// The lookup uses the same COALESCE logic as the unique index to match
/// NULL-valued chunking parameters correctly.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn find_session(conn: &Connection, config: &IndexConfig) -> Result<Option<i64>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id FROM index_session
         WHERE directory_path = ?1
           AND model_name = ?2
           AND chunk_strategy = ?3
           AND COALESCE(chunk_size, -1) = ?4
           AND COALESCE(chunk_overlap, -1) = ?5
           AND COALESCE(max_words, -1) = ?6
           AND ocr_language = ?7
           AND embedding_storage_mode = ?8",
    )?;

    let result = stmt.query_row(
        params![
            config.directory.to_string_lossy().as_ref(),
            config.model_name,
            config.chunk_strategy,
            config.chunk_size.map_or(-1, |v| v as i64),
            config.chunk_overlap.map_or(-1, |v| v as i64),
            config.max_words.map_or(-1, |v| v as i64),
            config.ocr_language,
            config.embedding_storage_mode.to_string(),
        ],
        |row| row.get(0),
    );

    match result {
        Ok(id) => Ok(Some(id)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Retrieves a single session by its primary key. Returns the full
/// `SessionRow` or a `NotFound` error if no session exists with that ID.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no session with the given ID exists,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_session(conn: &Connection, id: i64) -> Result<SessionRow, StoreError> {
    conn.query_row(
        "SELECT id, directory_path, model_name, chunk_strategy,
                chunk_size, chunk_overlap, max_words,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version,
                hnsw_total, hnsw_orphans, last_rebuild_at, created_at,
                label, tags, metadata
         FROM index_session WHERE id = ?1",
        params![id],
        |row| {
            Ok(SessionRow {
                id: row.get(0)?,
                directory_path: row.get(1)?,
                model_name: row.get(2)?,
                chunk_strategy: row.get(3)?,
                chunk_size: row.get(4)?,
                chunk_overlap: row.get(5)?,
                max_words: row.get(6)?,
                ocr_language: row.get(7)?,
                vector_dimension: row.get(8)?,
                embedding_storage_mode: row.get(9)?,
                schema_version: row.get(10)?,
                app_version: row.get(11)?,
                hnsw_total: row.get(12)?,
                hnsw_orphans: row.get(13)?,
                last_rebuild_at: row.get(14)?,
                created_at: row.get(15)?,
                label: row.get(16)?,
                tags: row.get(17)?,
                metadata: row.get(18)?,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => StoreError::not_found("session", id.to_string()),
        other => other.into(),
    })
}

/// Maximum number of session rows returned by `list_sessions`. Prevents
/// unbounded memory use when the database accumulates many sessions. For
/// use cases that require more rows, call `list_sessions_paged` instead.
/// 10 000 sessions exceeds any realistic single-machine deployment.
const LIST_SESSIONS_MAX: i64 = 10_000;

/// Lists sessions in the database, ordered by creation time descending,
/// capped at `LIST_SESSIONS_MAX` rows to prevent unbounded memory use.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_sessions(conn: &Connection) -> Result<Vec<SessionRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, directory_path, model_name, chunk_strategy,
                chunk_size, chunk_overlap, max_words,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version,
                hnsw_total, hnsw_orphans, last_rebuild_at, created_at,
                label, tags, metadata
         FROM index_session ORDER BY created_at DESC LIMIT ?1",
    )?;

    let rows = stmt.query_map(rusqlite::params![LIST_SESSIONS_MAX], |row| {
        Ok(SessionRow {
            id: row.get(0)?,
            directory_path: row.get(1)?,
            model_name: row.get(2)?,
            chunk_strategy: row.get(3)?,
            chunk_size: row.get(4)?,
            chunk_overlap: row.get(5)?,
            max_words: row.get(6)?,
            ocr_language: row.get(7)?,
            vector_dimension: row.get(8)?,
            embedding_storage_mode: row.get(9)?,
            schema_version: row.get(10)?,
            app_version: row.get(11)?,
            hnsw_total: row.get(12)?,
            hnsw_orphans: row.get(13)?,
            last_rebuild_at: row.get(14)?,
            created_at: row.get(15)?,
            label: row.get(16)?,
            tags: row.get(17)?,
            metadata: row.get(18)?,
        })
    })?;

    let mut sessions = Vec::new();
    for row in rows {
        sessions.push(row?);
    }
    Ok(sessions)
}

/// Deletes a session and all dependent records (files, pages, chunks) via
/// ON DELETE CASCADE. Before deleting, detaches any job records that reference
/// this session by setting their `session_id` to NULL. This preserves
/// completed/failed/canceled job history in the `neuroncite_jobs` listing
/// even after the session data is removed.
///
/// Both the job detach and the session delete are wrapped in a single
/// IMMEDIATE transaction so a crash between the two statements cannot
/// leave jobs detached while the session still exists.
///
/// Returns the number of session rows deleted (0 or 1).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the detach, delete, or transaction commit fails.
pub fn delete_session(conn: &Connection, id: i64) -> Result<usize, StoreError> {
    // unchecked_transaction starts an IMMEDIATE transaction without a savepoint.
    // The connection already requires exclusive write access at this point.
    let tx = conn.unchecked_transaction()?;

    // Detach jobs from this session so ON DELETE CASCADE does not remove
    // the job history. The job.session_id column is nullable, so NULLing
    // it out breaks the FK link without violating constraints.
    tx.execute(
        "UPDATE job SET session_id = NULL WHERE session_id = ?1",
        params![id],
    )?;
    let count = tx.execute("DELETE FROM index_session WHERE id = ?1", params![id])?;

    tx.commit()?;
    Ok(count)
}

/// Updates the human-readable `label` column on an existing session.
///
/// Pass `Some("My Label")` to set a label, or `None` to clear it. Returns
/// the number of rows affected (0 if the session ID does not exist, 1 on
/// success).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the UPDATE statement fails.
pub fn update_session_label(
    conn: &Connection,
    id: i64,
    label: Option<&str>,
) -> Result<usize, StoreError> {
    let count = conn.execute(
        "UPDATE index_session SET label = ?1 WHERE id = ?2",
        params![label, id],
    )?;
    Ok(count)
}

/// Updates the `tags` column on an existing session. The value is a serialized
/// JSON array of tag strings (e.g. `["finance", "statistics"]`).
///
/// Pass `Some(json_string)` to set tags, or `None` to clear them. Returns the
/// number of rows affected (0 if the session ID does not exist, 1 on success).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the UPDATE statement fails.
pub fn update_session_tags(
    conn: &Connection,
    id: i64,
    tags: Option<&str>,
) -> Result<usize, StoreError> {
    let count = conn.execute(
        "UPDATE index_session SET tags = ?1 WHERE id = ?2",
        params![tags, id],
    )?;
    Ok(count)
}

/// Updates the `metadata` column on an existing session. The value is a
/// serialized JSON object of arbitrary key-value pairs (e.g. `{"source": "manual"}`).
///
/// Pass `Some(json_string)` to set metadata, or `None` to clear it. Returns
/// the number of rows affected (0 if the session ID does not exist, 1 on
/// success).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the UPDATE statement fails.
pub fn update_session_metadata(
    conn: &Connection,
    id: i64,
    metadata: Option<&str>,
) -> Result<usize, StoreError> {
    let count = conn.execute(
        "UPDATE index_session SET metadata = ?1 WHERE id = ?2",
        params![metadata, id],
    )?;
    Ok(count)
}

/// Returns all sessions whose `directory_path` matches the given directory.
/// Used by the delete-by-directory feature to find all sessions associated
/// with a specific PDF directory before deletion.
///
/// The comparison is case-sensitive and must match the stored path exactly.
/// Callers should canonicalize the directory path before calling this function
/// to ensure consistent matching (see `neuroncite_core::paths::canonicalize_directory`).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn find_sessions_by_directory(
    conn: &Connection,
    directory_path: &str,
) -> Result<Vec<SessionRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, directory_path, model_name, chunk_strategy,
                chunk_size, chunk_overlap, max_words,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version,
                hnsw_total, hnsw_orphans, last_rebuild_at, created_at,
                label, tags, metadata
         FROM index_session WHERE directory_path = ?1 ORDER BY created_at DESC",
    )?;

    let rows = stmt.query_map(params![directory_path], |row| {
        Ok(SessionRow {
            id: row.get(0)?,
            directory_path: row.get(1)?,
            model_name: row.get(2)?,
            chunk_strategy: row.get(3)?,
            chunk_size: row.get(4)?,
            chunk_overlap: row.get(5)?,
            max_words: row.get(6)?,
            ocr_language: row.get(7)?,
            vector_dimension: row.get(8)?,
            embedding_storage_mode: row.get(9)?,
            schema_version: row.get(10)?,
            app_version: row.get(11)?,
            hnsw_total: row.get(12)?,
            hnsw_orphans: row.get(13)?,
            last_rebuild_at: row.get(14)?,
            created_at: row.get(15)?,
            label: row.get(16)?,
            tags: row.get(17)?,
            metadata: row.get(18)?,
        })
    })?;

    let mut sessions = Vec::new();
    for row in rows {
        sessions.push(row?);
    }
    Ok(sessions)
}

/// Deletes all sessions whose `directory_path` matches the given directory.
/// Returns the IDs of the deleted sessions. ON DELETE CASCADE removes all
/// dependent records (files, pages, chunks) for each deleted session.
///
/// Before deleting, detaches any job records referencing these sessions by
/// setting their `session_id` to NULL. This preserves completed/failed/canceled
/// job history in the `neuroncite_jobs` listing even after session data is
/// removed.
///
/// The caller is responsible for evicting the deleted sessions' HNSW indices
/// from the in-memory map after this function returns.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query, detach, or deletion fails.
pub fn delete_sessions_by_directory(
    conn: &Connection,
    directory_path: &str,
) -> Result<Vec<i64>, StoreError> {
    let tx = conn.unchecked_transaction()?;

    // Collect the IDs of matching sessions before deletion so the caller
    // knows which HNSW indices to evict from the in-memory map.
    let mut id_stmt =
        tx.prepare_cached("SELECT id FROM index_session WHERE directory_path = ?1")?;

    let ids: Vec<i64> = id_stmt
        .query_map(params![directory_path], |row| row.get(0))?
        .collect::<Result<Vec<i64>, _>>()?;

    if !ids.is_empty() {
        // Detach all jobs referencing any session in this directory via a
        // correlated subquery. This is a single statement rather than a loop,
        // avoiding multiple round-trips and unbounded parameter lists.
        tx.execute(
            "UPDATE job SET session_id = NULL
             WHERE session_id IN (
                 SELECT id FROM index_session WHERE directory_path = ?1
             )",
            params![directory_path],
        )?;

        tx.execute(
            "DELETE FROM index_session WHERE directory_path = ?1",
            params![directory_path],
        )?;
    }

    drop(id_stmt);
    tx.commit()?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    /// Helper: creates an in-memory database with schema applied.
    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");
        conn
    }

    /// Helper: creates a default `IndexConfig` for testing.
    fn test_config() -> IndexConfig {
        IndexConfig {
            directory: PathBuf::from("/docs/papers"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        }
    }

    /// T-STO-003: Session creation returns a positive ID and the session is
    /// queryable by that ID with the same configuration values.
    #[test]
    fn t_sto_003_session_creation() {
        let conn = setup_db();
        let config = test_config();

        let id = create_session(&conn, &config, "0.1.0").expect("create_session failed");
        assert!(id > 0, "session ID must be positive");

        let row = get_session(&conn, id).expect("get_session failed");
        assert_eq!(row.directory_path, "/docs/papers");
        assert_eq!(row.model_name, "BAAI/bge-small-en-v1.5");
        assert_eq!(row.chunk_strategy, "word");
        assert_eq!(row.chunk_size, Some(300));
        assert_eq!(row.chunk_overlap, Some(50));
        assert!(row.max_words.is_none());
        assert_eq!(row.vector_dimension, 384);
    }

    /// T-STO-004: Duplicate session configuration returns a constraint violation
    /// error on the second insert.
    #[test]
    fn t_sto_004_session_unique_constraint() {
        let conn = setup_db();
        let config = test_config();

        create_session(&conn, &config, "0.1.0").expect("first insert failed");

        let result = create_session(&conn, &config, "0.1.0");
        assert!(
            result.is_err(),
            "second insert with identical config must fail"
        );
    }

    /// T-STO-005: Session COALESCE uniqueness for NULL values. Two sessions
    /// that differ only in having NULL vs NULL for `chunk_size` (page-based
    /// strategy) are correctly rejected by the COALESCE-based unique index.
    #[test]
    fn t_sto_005_coalesce_uniqueness_for_nulls() {
        let conn = setup_db();

        // Page-based strategy: chunk_size, chunk_overlap, and max_words are all None
        let config = IndexConfig {
            directory: PathBuf::from("/docs/papers"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "page".into(),
            chunk_size: None,
            chunk_overlap: None,
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        create_session(&conn, &config, "0.1.0").expect("first page-based insert failed");

        let result = create_session(&conn, &config, "0.1.0");
        assert!(
            result.is_err(),
            "second page-based insert with all NULLs must fail due to COALESCE uniqueness"
        );
    }

    /// T-STO-088: A freshly created session has a NULL label by default.
    /// After calling `update_session_label` with a string, `get_session`
    /// returns that label. Calling it again with `None` clears the label
    /// back to NULL.
    #[test]
    fn t_sto_088_session_label_lifecycle() {
        let conn = setup_db();
        let config = test_config();

        let id = create_session(&conn, &config, "0.1.0").expect("create_session failed");

        // Label is NULL immediately after creation.
        let row = get_session(&conn, id).expect("get_session failed");
        assert!(row.label.is_none(), "label must be NULL after creation");

        // Set a label.
        let updated = update_session_label(&conn, id, Some("Financial Papers"))
            .expect("update_session_label failed");
        assert_eq!(updated, 1, "exactly one row must be updated");

        let row = get_session(&conn, id).expect("get_session failed after label set");
        assert_eq!(
            row.label.as_deref(),
            Some("Financial Papers"),
            "label must match the value passed to update_session_label"
        );

        // Clear the label back to NULL.
        let updated =
            update_session_label(&conn, id, None).expect("update_session_label(None) failed");
        assert_eq!(updated, 1, "exactly one row must be updated on clear");

        let row = get_session(&conn, id).expect("get_session failed after label clear");
        assert!(
            row.label.is_none(),
            "label must be NULL after clearing with None"
        );
    }

    /// T-STO-089: `update_session_label` returns 0 rows affected when called
    /// with a non-existent session ID (no error, just zero updates).
    #[test]
    fn t_sto_089_update_label_nonexistent_session() {
        let conn = setup_db();

        let updated = update_session_label(&conn, 99999, Some("Ghost"))
            .expect("update_session_label should not error on missing ID");
        assert_eq!(
            updated, 0,
            "updating a non-existent session must affect zero rows"
        );
    }

    /// T-STO-090: `list_sessions` includes the label field for each session.
    /// Verifies that the label column is correctly projected in the listing
    /// query, both for sessions with and without labels.
    #[test]
    fn t_sto_090_list_sessions_includes_label() {
        let conn = setup_db();

        let config_a = IndexConfig {
            directory: PathBuf::from("/docs/a"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let config_b = IndexConfig {
            directory: PathBuf::from("/docs/b"),
            ..config_a.clone()
        };

        let id_a = create_session(&conn, &config_a, "0.1.0").expect("create a failed");
        let _id_b = create_session(&conn, &config_b, "0.1.0").expect("create b failed");

        update_session_label(&conn, id_a, Some("Labeled Session"))
            .expect("update_session_label failed");

        let sessions = list_sessions(&conn).expect("list_sessions failed");
        assert_eq!(sessions.len(), 2, "two sessions must exist");

        // Find sessions by directory path.
        let labeled = sessions
            .iter()
            .find(|s| s.directory_path == "/docs/a")
            .unwrap();
        let unlabeled = sessions
            .iter()
            .find(|s| s.directory_path == "/docs/b")
            .unwrap();

        assert_eq!(labeled.label.as_deref(), Some("Labeled Session"));
        assert!(unlabeled.label.is_none());
    }

    /// T-STO-091: `find_session` returns the correct ID for a matching config
    /// and `None` for a config that has no matching session.
    #[test]
    fn t_sto_091_find_session() {
        let conn = setup_db();
        let config = test_config();

        // No session exists yet.
        let found = find_session(&conn, &config).expect("find_session failed");
        assert!(found.is_none(), "must return None when no session matches");

        let id = create_session(&conn, &config, "0.1.0").expect("create_session failed");

        // Session exists with matching config.
        let found = find_session(&conn, &config).expect("find_session failed");
        assert_eq!(
            found,
            Some(id),
            "must return the session ID for a matching config"
        );
    }

    /// T-STO-023: `delete_session` removes the session and returns count=1.
    /// Calling `get_session` afterwards returns a NotFound error.
    #[test]
    fn t_sto_023_delete_session() {
        let conn = setup_db();
        let config = test_config();

        let id = create_session(&conn, &config, "0.1.0").expect("create_session failed");

        let deleted = delete_session(&conn, id).expect("delete_session failed");
        assert_eq!(deleted, 1, "one row must be deleted");

        let result = get_session(&conn, id);
        assert!(result.is_err(), "get_session must fail after deletion");
    }

    /// T-STO-024: The `index_session` table includes the `label` column in
    /// the v1 schema. Verifies via `PRAGMA table_info`.
    #[test]
    fn t_sto_024_schema_has_label_column() {
        let conn = setup_db();

        let has_label: bool = conn
            .prepare("PRAGMA table_info(index_session)")
            .expect("PRAGMA table_info failed")
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })
            .expect("query_map failed")
            .any(|col| col.is_ok_and(|name| name == "label"));

        assert!(has_label, "index_session table must have a 'label' column");
    }

    /// T-STO-025: find_sessions_by_directory returns all sessions for a
    /// given directory path, including sessions with different models or
    /// chunking strategies.
    #[test]
    fn t_sto_025_find_sessions_by_directory() {
        let conn = setup_db();

        let config_word = IndexConfig {
            directory: PathBuf::from("/docs/papers"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let config_page = IndexConfig {
            chunk_strategy: "page".into(),
            chunk_size: None,
            chunk_overlap: None,
            ..config_word.clone()
        };

        let config_other_dir = IndexConfig {
            directory: PathBuf::from("/docs/other"),
            ..config_word.clone()
        };

        let id1 = create_session(&conn, &config_word, "0.1.0").expect("create word session");
        let id2 = create_session(&conn, &config_page, "0.1.0").expect("create page session");
        let _id3 =
            create_session(&conn, &config_other_dir, "0.1.0").expect("create other dir session");

        let found = find_sessions_by_directory(&conn, "/docs/papers")
            .expect("find_sessions_by_directory failed");
        assert_eq!(found.len(), 2, "two sessions match /docs/papers");

        let found_ids: Vec<i64> = found.iter().map(|s| s.id).collect();
        assert!(found_ids.contains(&id1));
        assert!(found_ids.contains(&id2));
    }

    /// T-STO-026: find_sessions_by_directory returns an empty vec when no
    /// sessions match the given directory.
    #[test]
    fn t_sto_026_find_sessions_by_directory_no_match() {
        let conn = setup_db();

        let found = find_sessions_by_directory(&conn, "/nonexistent/path")
            .expect("find_sessions_by_directory failed");
        assert!(found.is_empty(), "no sessions should match");
    }

    /// T-STO-027: delete_sessions_by_directory removes all sessions for a
    /// directory and returns the deleted IDs. ON DELETE CASCADE removes
    /// dependent file and chunk records.
    #[test]
    fn t_sto_027_delete_sessions_by_directory() {
        let conn = setup_db();

        let config_a = IndexConfig {
            directory: PathBuf::from("/docs/finance"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let config_b = IndexConfig {
            chunk_strategy: "page".into(),
            chunk_size: None,
            chunk_overlap: None,
            ..config_a.clone()
        };

        let config_keep = IndexConfig {
            directory: PathBuf::from("/docs/biology"),
            ..config_a.clone()
        };

        let id_a = create_session(&conn, &config_a, "0.1.0").expect("create a");
        let id_b = create_session(&conn, &config_b, "0.1.0").expect("create b");
        let id_keep = create_session(&conn, &config_keep, "0.1.0").expect("create keep");

        // Insert a file in session A to verify CASCADE deletion.
        use crate::repo::file;
        file::insert_file(
            &conn,
            id_a,
            "/docs/finance/paper.pdf",
            "hash",
            1000,
            100,
            5,
            None,
        )
        .expect("insert file");

        let deleted = delete_sessions_by_directory(&conn, "/docs/finance")
            .expect("delete_sessions_by_directory failed");

        assert_eq!(deleted.len(), 2, "two sessions deleted");
        assert!(deleted.contains(&id_a));
        assert!(deleted.contains(&id_b));

        // Finance sessions are gone.
        assert!(get_session(&conn, id_a).is_err());
        assert!(get_session(&conn, id_b).is_err());

        // Biology session is untouched.
        assert!(get_session(&conn, id_keep).is_ok());

        // Files from finance sessions are cascade-deleted.
        let files = file::list_files_by_session(&conn, id_a).expect("list files");
        assert!(files.is_empty(), "files must be cascade-deleted");
    }

    /// T-STO-080: delete_sessions_by_directory returns an empty vec when
    /// no sessions match the given directory (no error, just empty).
    #[test]
    fn t_sto_080_delete_sessions_by_directory_no_match() {
        let conn = setup_db();

        let deleted = delete_sessions_by_directory(&conn, "/nonexistent/path")
            .expect("delete_sessions_by_directory failed");
        assert!(deleted.is_empty(), "no sessions to delete");
    }

    /// T-STO-082: BUG-004 regression: `delete_session` detaches job records
    /// instead of cascade-deleting them. A completed job that referenced a
    /// deleted session must remain in the job table with `session_id = NULL`.
    #[test]
    fn t_sto_082_delete_session_preserves_jobs() {
        let conn = setup_db();
        let config = test_config();

        let session_id = create_session(&conn, &config, "0.1.0").expect("create session");

        // Create a job associated with this session.
        use crate::workflow::job;
        job::create_job(&conn, "preserved-job-1", "index", Some(session_id))
            .expect("create job failed");
        job::update_job_state(&conn, "preserved-job-1", job::JobState::Running, None)
            .expect("transition to running");
        job::update_job_state(&conn, "preserved-job-1", job::JobState::Completed, None)
            .expect("transition to completed");

        // Delete the session.
        let deleted = delete_session(&conn, session_id).expect("delete session");
        assert_eq!(deleted, 1, "session must be deleted");

        // Session is gone.
        assert!(get_session(&conn, session_id).is_err());

        // Job must still exist with session_id = NULL.
        let j = job::get_job(&conn, "preserved-job-1").expect("job must survive session deletion");
        assert_eq!(j.state, job::JobState::Completed);
        assert!(
            j.session_id.is_none(),
            "session_id must be NULL after session deletion, got: {:?}",
            j.session_id
        );
    }

    /// T-STO-084: BUG-004 regression: `delete_sessions_by_directory` detaches
    /// all job records for all deleted sessions. Multiple jobs across multiple
    /// sessions in the same directory must all survive with `session_id = NULL`.
    #[test]
    fn t_sto_084_delete_sessions_by_directory_preserves_jobs() {
        let conn = setup_db();

        let config_a = IndexConfig {
            directory: PathBuf::from("/docs/volatile"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let config_b = IndexConfig {
            chunk_strategy: "page".into(),
            chunk_size: None,
            chunk_overlap: None,
            ..config_a.clone()
        };

        let sid_a = create_session(&conn, &config_a, "0.1.0").expect("create session a");
        let sid_b = create_session(&conn, &config_b, "0.1.0").expect("create session b");

        // Create jobs for both sessions.
        use crate::workflow::job;
        job::create_job(&conn, "job-vol-a", "index", Some(sid_a)).expect("create job a");
        job::update_job_state(&conn, "job-vol-a", job::JobState::Running, None).unwrap();
        job::update_job_state(&conn, "job-vol-a", job::JobState::Completed, None).unwrap();

        job::create_job(&conn, "job-vol-b", "index", Some(sid_b)).expect("create job b");
        job::update_job_state(&conn, "job-vol-b", job::JobState::Running, None).unwrap();
        job::update_job_state(
            &conn,
            "job-vol-b",
            job::JobState::Failed,
            Some("test failure"),
        )
        .unwrap();

        // Delete all sessions for /docs/volatile.
        let deleted_ids =
            delete_sessions_by_directory(&conn, "/docs/volatile").expect("delete by directory");
        assert_eq!(deleted_ids.len(), 2);

        // Both sessions are gone.
        assert!(get_session(&conn, sid_a).is_err());
        assert!(get_session(&conn, sid_b).is_err());

        // Both jobs survive with session_id = NULL.
        let ja = job::get_job(&conn, "job-vol-a").expect("job a must survive");
        assert!(ja.session_id.is_none(), "job a session_id must be NULL");
        assert_eq!(ja.state, job::JobState::Completed);

        let jb = job::get_job(&conn, "job-vol-b").expect("job b must survive");
        assert!(jb.session_id.is_none(), "job b session_id must be NULL");
        assert_eq!(jb.state, job::JobState::Failed);
        assert_eq!(jb.error_message.as_deref(), Some("test failure"));
    }

    /// T-STO-031: Jobs not associated with the deleted session remain
    /// untouched. Only jobs whose `session_id` matches the deleted session
    /// should have their `session_id` set to NULL.
    #[test]
    fn t_sto_031_delete_session_leaves_other_jobs_intact() {
        let conn = setup_db();
        let config_a = test_config();
        let config_b = IndexConfig {
            directory: PathBuf::from("/docs/other"),
            ..config_a.clone()
        };

        let sid_a = create_session(&conn, &config_a, "0.1.0").expect("create a");
        let sid_b = create_session(&conn, &config_b, "0.1.0").expect("create b");

        use crate::workflow::job;
        job::create_job(&conn, "job-delete-target", "index", Some(sid_a)).unwrap();
        job::create_job(&conn, "job-keep-intact", "index", Some(sid_b)).unwrap();

        // Delete only session A.
        delete_session(&conn, sid_a).expect("delete session a");

        // Job for session A has session_id = NULL.
        let j_target = job::get_job(&conn, "job-delete-target").expect("target job must exist");
        assert!(j_target.session_id.is_none());

        // Job for session B still references session B.
        let j_keep = job::get_job(&conn, "job-keep-intact").expect("keep job must exist");
        assert_eq!(
            j_keep.session_id,
            Some(sid_b),
            "unrelated job must retain its session_id"
        );
    }

    /// T-STO-092: A freshly created session has NULL tags and metadata by
    /// default. After calling `update_session_tags` and `update_session_metadata`
    /// with JSON strings, `get_session` returns those values. Calling with
    /// `None` clears them back to NULL.
    #[test]
    fn t_sto_092_session_tags_and_metadata_lifecycle() {
        let conn = setup_db();
        let config = test_config();

        let id = create_session(&conn, &config, "0.1.0").expect("create_session failed");

        // Tags and metadata are NULL immediately after creation.
        let row = get_session(&conn, id).expect("get_session failed");
        assert!(row.tags.is_none(), "tags must be NULL after creation");
        assert!(
            row.metadata.is_none(),
            "metadata must be NULL after creation"
        );

        // Set tags.
        let updated = update_session_tags(&conn, id, Some("[\"finance\",\"statistics\"]"))
            .expect("update_session_tags failed");
        assert_eq!(updated, 1, "exactly one row must be updated for tags");

        let row = get_session(&conn, id).expect("get_session failed after tags set");
        assert_eq!(
            row.tags.as_deref(),
            Some("[\"finance\",\"statistics\"]"),
            "tags must match the value passed to update_session_tags"
        );

        // Set metadata.
        let updated = update_session_metadata(
            &conn,
            id,
            Some("{\"source\":\"manual\",\"priority\":\"high\"}"),
        )
        .expect("update_session_metadata failed");
        assert_eq!(updated, 1, "exactly one row must be updated for metadata");

        let row = get_session(&conn, id).expect("get_session failed after metadata set");
        assert_eq!(
            row.metadata.as_deref(),
            Some("{\"source\":\"manual\",\"priority\":\"high\"}"),
            "metadata must match the value passed to update_session_metadata"
        );

        // Clear tags back to NULL.
        let updated =
            update_session_tags(&conn, id, None).expect("update_session_tags(None) failed");
        assert_eq!(updated, 1, "exactly one row must be updated on clear");

        let row = get_session(&conn, id).expect("get_session failed after tags clear");
        assert!(
            row.tags.is_none(),
            "tags must be NULL after clearing with None"
        );

        // Clear metadata back to NULL.
        let updated =
            update_session_metadata(&conn, id, None).expect("update_session_metadata(None) failed");
        assert_eq!(updated, 1);

        let row = get_session(&conn, id).expect("get_session failed after metadata clear");
        assert!(
            row.metadata.is_none(),
            "metadata must be NULL after clearing with None"
        );
    }

    /// T-STO-051: `update_session_tags` and `update_session_metadata` return 0
    /// rows affected when called with a non-existent session ID.
    #[test]
    fn t_sto_051_update_tags_metadata_nonexistent_session() {
        let conn = setup_db();

        let updated = update_session_tags(&conn, 99999, Some("[\"test\"]"))
            .expect("update_session_tags should not error on missing ID");
        assert_eq!(
            updated, 0,
            "updating tags on non-existent session must affect zero rows"
        );

        let updated = update_session_metadata(&conn, 99999, Some("{\"k\":\"v\"}"))
            .expect("update_session_metadata should not error on missing ID");
        assert_eq!(
            updated, 0,
            "updating metadata on non-existent session must affect zero rows"
        );
    }

    /// T-STO-052: `list_sessions` includes tags and metadata fields for each
    /// session. Verifies that both columns are correctly projected in the listing
    /// query, for sessions with and without tags/metadata.
    #[test]
    fn t_sto_052_list_sessions_includes_tags_and_metadata() {
        let conn = setup_db();

        let config_a = IndexConfig {
            directory: PathBuf::from("/docs/a"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let config_b = IndexConfig {
            directory: PathBuf::from("/docs/b"),
            ..config_a.clone()
        };

        let id_a = create_session(&conn, &config_a, "0.1.0").expect("create a failed");
        let _id_b = create_session(&conn, &config_b, "0.1.0").expect("create b failed");

        update_session_tags(&conn, id_a, Some("[\"tagged\"]")).expect("update_session_tags failed");
        update_session_metadata(&conn, id_a, Some("{\"key\":\"value\"}"))
            .expect("update_session_metadata failed");

        let sessions = list_sessions(&conn).expect("list_sessions failed");
        assert_eq!(sessions.len(), 2, "two sessions must exist");

        let tagged = sessions
            .iter()
            .find(|s| s.directory_path == "/docs/a")
            .unwrap();
        let untagged = sessions
            .iter()
            .find(|s| s.directory_path == "/docs/b")
            .unwrap();

        assert_eq!(tagged.tags.as_deref(), Some("[\"tagged\"]"));
        assert_eq!(tagged.metadata.as_deref(), Some("{\"key\":\"value\"}"));
        assert!(untagged.tags.is_none());
        assert!(untagged.metadata.is_none());
    }

    /// T-STO-053: `find_sessions_by_directory` returns tags and metadata
    /// in the session rows.
    #[test]
    fn t_sto_053_find_sessions_by_directory_includes_tags_metadata() {
        let conn = setup_db();

        let config = IndexConfig {
            directory: PathBuf::from("/docs/tagged"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let id = create_session(&conn, &config, "0.1.0").expect("create session");
        update_session_tags(&conn, id, Some("[\"finance\"]")).unwrap();
        update_session_metadata(&conn, id, Some("{\"src\":\"api\"}")).unwrap();

        let found = find_sessions_by_directory(&conn, "/docs/tagged")
            .expect("find_sessions_by_directory failed");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].tags.as_deref(), Some("[\"finance\"]"));
        assert_eq!(found[0].metadata.as_deref(), Some("{\"src\":\"api\"}"));
    }

    /// T-STO-060: `get_session` returns the correct `vector_dimension` for sessions
    /// created with different model dimensions. This is the data path exercised by
    /// the API search handler's dimension guard, which rejects search requests when
    /// the session dimension does not match the currently loaded model's dimension.
    /// Two sessions with distinct dimensions must each return their own value.
    #[test]
    fn t_sto_060_get_session_preserves_vector_dimension() {
        let conn = setup_db();

        // Session indexed with a 384-dimensional model.
        let config_384 = IndexConfig {
            directory: PathBuf::from("/docs/small"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        // Session indexed with a 768-dimensional model.
        let config_768 = IndexConfig {
            directory: PathBuf::from("/docs/large"),
            model_name: "BAAI/bge-base-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 768,
        };

        let id_384 = create_session(&conn, &config_384, "0.1.0").expect("create 384-dim session");
        let id_768 = create_session(&conn, &config_768, "0.1.0").expect("create 768-dim session");

        let row_384 = get_session(&conn, id_384).expect("get 384-dim session");
        let row_768 = get_session(&conn, id_768).expect("get 768-dim session");

        assert_eq!(
            row_384.vector_dimension, 384,
            "session created with 384d must return vector_dimension=384"
        );
        assert_eq!(
            row_768.vector_dimension, 768,
            "session created with 768d must return vector_dimension=768"
        );

        // Simulate the guard comparison: a server loaded with the 768-dim model
        // must detect the mismatch when asked to search session id_384.
        let loaded_model_dimension: i64 = 768;
        assert_ne!(
            row_384.vector_dimension, loaded_model_dimension,
            "guard must detect dimension mismatch between session (384d) and model (768d)"
        );
        assert_eq!(
            row_768.vector_dimension, loaded_model_dimension,
            "guard must accept matching dimensions"
        );
    }
}
