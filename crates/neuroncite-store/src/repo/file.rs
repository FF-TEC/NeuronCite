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

// File repository operations.
//
// Provides functions to insert, query, update, and delete indexed file records.
// Each file record stores the file's absolute path within the session root,
// its SHA-256 content hash, file size in bytes, modification timestamp,
// extracted page count, and structural PDF page count. The content hash and
// metadata enable two-stage incremental change detection: a fast mtime/size
// pre-filter followed by SHA-256 verification only when metadata differs.
// The pdf_page_count column stores the structural page count from the PDF
// page tree, which may differ from page_count when extraction was incomplete.

use rusqlite::{Connection, params};

use crate::error::StoreError;

/// Row representation of an `indexed_file` record. Contains all columns
/// defined in the `indexed_file` table.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub id: i64,
    pub session_id: i64,
    pub file_path: String,
    pub file_hash: String,
    pub mtime: i64,
    pub size: i64,
    pub page_count: i64,
    /// Structural page count from the PDF page tree (lopdf/pdfium), independent
    /// of how many pages had extractable text. NULL when the structural count
    /// is unavailable (e.g., extraction backend does not report it). Always
    /// NULL for HTML-sourced files.
    pub pdf_page_count: Option<i64>,
    /// Content origin: "pdf" for PDF files, "html" for web-sourced HTML pages.
    /// Defaults to "pdf" for backward compatibility with pre-existing records.
    pub source_type: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Result of the two-stage file change detection algorithm.
/// The caller uses this to decide whether to re-extract, skip, or just
/// update metadata for the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeStatus {
    /// The file's mtime and size match the stored values. No further
    /// processing is needed.
    Unchanged,
    /// The file's mtime or size differs, but the SHA-256 content hash
    /// matches the stored value. Only metadata (mtime, size, `updated_at`)
    /// needs updating.
    MetadataOnly,
    /// The file's content hash differs from the stored value. The file
    /// must be re-extracted, re-chunked, and re-embedded.
    ContentChanged,
    /// No record exists for this file in the database. The file must be
    /// fully processed (extract, chunk, embed, store).
    New,
}

/// Inserts a file tracking record and returns the auto-generated file ID.
///
/// The `pdf_page_count` parameter stores the structural page count from the
/// PDF page tree. Pass `None` when this information is unavailable.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert violates the
/// `UNIQUE(session_id, file_path)` constraint.
#[allow(clippy::too_many_arguments)]
pub fn insert_file(
    conn: &Connection,
    session_id: i64,
    file_path: &str,
    file_hash: &str,
    mtime: i64,
    size: i64,
    page_count: i64,
    pdf_page_count: Option<i64>,
) -> Result<i64, StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();
    conn.execute(
        "INSERT INTO indexed_file (session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, now, now],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Inserts a file tracking record with an explicit source type and returns the
/// auto-generated file ID. The `source_type` parameter distinguishes between
/// content origins ("pdf" for PDF files, "html" for web-scraped HTML pages).
///
/// This function is used by the HTML indexer to insert records with
/// source_type="html". For PDF files, prefer `insert_file` which defaults
/// to source_type="pdf" via the SQL column default.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert violates the
/// `UNIQUE(session_id, file_path)` constraint.
#[allow(clippy::too_many_arguments)]
pub fn insert_file_with_source_type(
    conn: &Connection,
    session_id: i64,
    file_path: &str,
    file_hash: &str,
    mtime: i64,
    size: i64,
    page_count: i64,
    pdf_page_count: Option<i64>,
    source_type: &str,
) -> Result<i64, StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();
    conn.execute(
        "INSERT INTO indexed_file (session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, source_type, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, source_type, now, now],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Retrieves a single file record by primary key.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no file with the given ID exists,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_file(conn: &Connection, id: i64) -> Result<FileRow, StoreError> {
    conn.query_row(
        "SELECT id, session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, source_type, created_at, updated_at
         FROM indexed_file WHERE id = ?1",
        params![id],
        row_to_file,
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => StoreError::not_found("file", id.to_string()),
        other => other.into(),
    })
}

/// Lists all file records belonging to a specific session.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_files_by_session(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<FileRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, source_type, created_at, updated_at
         FROM indexed_file WHERE session_id = ?1 ORDER BY file_path",
    )?;

    let rows = stmt.query_map(params![session_id], row_to_file)?;
    let mut files = Vec::new();
    for row in rows {
        files.push(row?);
    }
    Ok(files)
}

/// Deletes a file record by primary key. ON DELETE CASCADE removes all
/// dependent page and chunk rows. Returns the number of rows deleted (0 or 1).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn delete_file(conn: &Connection, id: i64) -> Result<usize, StoreError> {
    let count = conn.execute("DELETE FROM indexed_file WHERE id = ?1", params![id])?;
    Ok(count)
}

/// Deletes a file record identified by session ID and file path. ON DELETE
/// CASCADE removes all dependent page and chunk rows. Returns the number of
/// rows deleted (0 or 1).
///
/// This function is called before inserting a file during re-indexing. When a
/// session is reused (same directory, model, chunk strategy), stale file records
/// from a previous indexing run remain in the database. Without cleanup, the
/// subsequent INSERT hits the UNIQUE(session_id, file_path) constraint and fails.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn delete_file_by_session_path(
    conn: &Connection,
    session_id: i64,
    file_path: &str,
) -> Result<usize, StoreError> {
    let count = conn.execute(
        "DELETE FROM indexed_file WHERE session_id = ?1 AND file_path = ?2",
        params![session_id, file_path],
    )?;
    Ok(count)
}

/// Updates the `file_hash`, mtime, size, and `updated_at` fields of an existing
/// file record. Used when a file's content or metadata has changed during
/// incremental re-indexing.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the update fails.
pub fn update_file_hash(
    conn: &Connection,
    id: i64,
    file_hash: &str,
    mtime: i64,
    size: i64,
) -> Result<(), StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();
    conn.execute(
        "UPDATE indexed_file SET file_hash = ?1, mtime = ?2, size = ?3, updated_at = ?4
         WHERE id = ?5",
        params![file_hash, mtime, size, now, id],
    )?;
    Ok(())
}

/// Two-stage change detection for a file.
///
/// Stage 1 (fast pre-filter): compares the given mtime and size against the
/// stored values. If both match, the file is classified as `Unchanged`
/// without computing a SHA-256 hash.
///
/// Stage 2 (hash verification): if mtime or size differ, the caller-provided
/// `current_hash` is compared against the stored `file_hash`. If the hashes
/// match, the change is `MetadataOnly`. Otherwise, it is `ContentChanged`.
///
/// If no record exists for the given `file_id`, returns `New`.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn check_file_changed(
    conn: &Connection,
    file_id: i64,
    current_mtime: i64,
    current_size: i64,
    current_hash: Option<&str>,
) -> Result<ChangeStatus, StoreError> {
    let result = conn.query_row(
        "SELECT mtime, size, file_hash FROM indexed_file WHERE id = ?1",
        params![file_id],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        },
    );

    match result {
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(ChangeStatus::New),
        Err(e) => Err(e.into()),
        Ok((stored_mtime, stored_size, stored_hash)) => {
            // Stage 1: fast metadata pre-filter
            if current_mtime == stored_mtime && current_size == stored_size {
                return Ok(ChangeStatus::Unchanged);
            }

            // Stage 2: SHA-256 hash verification (only when metadata differs)
            if let Some(hash) = current_hash
                && hash == stored_hash
            {
                return Ok(ChangeStatus::MetadataOnly);
            }

            Ok(ChangeStatus::ContentChanged)
        }
    }
}

/// Looks up a file record by session ID and file path. Returns `None` if no
/// record exists for this combination. Used by the change detection pipeline
/// to check whether a file was previously indexed in a given session before
/// deciding whether to re-extract and re-embed.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn find_file_by_session_path(
    conn: &Connection,
    session_id: i64,
    file_path: &str,
) -> Result<Option<FileRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, session_id, file_path, file_hash, mtime, size, page_count, pdf_page_count, source_type, created_at, updated_at
         FROM indexed_file WHERE session_id = ?1 AND file_path = ?2",
    )?;

    let result = stmt.query_row(params![session_id, file_path], row_to_file);

    match result {
        Ok(row) => Ok(Some(row)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Maps a `rusqlite` row to a `FileRow`. Column order matches the SELECT
/// statements: id, session_id, file_path, file_hash, mtime, size,
/// page_count, pdf_page_count, source_type, created_at, updated_at.
fn row_to_file(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRow> {
    Ok(FileRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        file_path: row.get(2)?,
        file_hash: row.get(3)?,
        mtime: row.get(4)?,
        size: row.get(5)?,
        page_count: row.get(6)?,
        pdf_page_count: row.get(7)?,
        source_type: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::session;
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    fn setup_db_with_session() -> (Connection, i64) {
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
        (conn, session_id)
    }

    /// T-STO-006: File tracking insert and query.
    #[test]
    fn t_sto_006_file_tracking_insert_and_query() {
        let (conn, session_id) = setup_db_with_session();

        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "abcdef1234567890",
            1_700_000_000,
            1024,
            10,
            None,
        )
        .expect("insert_file failed");

        let files = list_files_by_session(&conn, session_id).expect("list_files failed");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].id, file_id);
        assert_eq!(files[0].file_path, "/docs/paper.pdf");
        assert_eq!(files[0].file_hash, "abcdef1234567890");
        assert_eq!(files[0].mtime, 1_700_000_000);
        assert_eq!(files[0].size, 1024);
        assert_eq!(files[0].page_count, 10);
        assert_eq!(files[0].pdf_page_count, None);
    }

    /// T-STO-007: File change detection mtime/size pre-filter.
    #[test]
    fn t_sto_007_file_change_detection_mtime_size() {
        let (conn, session_id) = setup_db_with_session();

        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_abc",
            1_700_000_000,
            2048,
            5,
            None,
        )
        .expect("insert_file failed");

        let status =
            check_file_changed(&conn, file_id, 1_700_000_000, 2048, None).expect("check failed");
        assert_eq!(status, ChangeStatus::Unchanged);
    }

    /// T-STO-008: File change detection SHA-256 verification.
    #[test]
    fn t_sto_008_file_change_detection_sha256() {
        let (conn, session_id) = setup_db_with_session();

        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_abc",
            1_700_000_000,
            2048,
            5,
            None,
        )
        .expect("insert_file failed");

        let status = check_file_changed(&conn, file_id, 1_700_099_999, 2048, Some("hash_abc"))
            .expect("check failed");
        assert_eq!(status, ChangeStatus::MetadataOnly);

        let status =
            check_file_changed(&conn, file_id, 1_700_099_999, 2048, Some("different_hash"))
                .expect("check failed");
        assert_eq!(status, ChangeStatus::ContentChanged);
    }

    /// T-STO-064: find_file_by_session_path returns the file record when it
    /// exists for the given session and path.
    #[test]
    fn t_sto_011_find_file_by_session_path_existing() {
        let (conn, session_id) = setup_db_with_session();

        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_abc",
            1_700_000_000,
            2048,
            5,
            None,
        )
        .expect("insert_file failed");

        let found = find_file_by_session_path(&conn, session_id, "/docs/paper.pdf")
            .expect("find_file_by_session_path failed");
        assert!(found.is_some(), "must find existing file");
        let row = found.unwrap();
        assert_eq!(row.id, file_id);
        assert_eq!(row.file_hash, "hash_abc");
        assert_eq!(row.mtime, 1_700_000_000);
        assert_eq!(row.size, 2048);
    }

    /// T-STO-065: find_file_by_session_path returns None for a non-existent
    /// file path within the session.
    #[test]
    fn t_sto_012_find_file_by_session_path_missing() {
        let (conn, session_id) = setup_db_with_session();

        let found = find_file_by_session_path(&conn, session_id, "/docs/nonexistent.pdf")
            .expect("find_file_by_session_path failed");
        assert!(found.is_none(), "must return None for missing file");
    }

    /// T-STO-066: find_file_by_session_path returns None when the file exists
    /// in a different session but not the queried session.
    #[test]
    fn t_sto_013_find_file_by_session_path_wrong_session() {
        let (conn, session_id) = setup_db_with_session();

        insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_abc",
            1_700_000_000,
            2048,
            5,
            None,
        )
        .expect("insert_file failed");

        // Query with a different session ID.
        let found = find_file_by_session_path(&conn, session_id + 999, "/docs/paper.pdf")
            .expect("find_file_by_session_path failed");
        assert!(
            found.is_none(),
            "must return None for file in a different session"
        );
    }

    /// T-STO-062: delete_file_by_session_path removes a file record by its
    /// composite key (session_id, file_path) and returns the count of deleted
    /// rows. Verifies that deleting a non-existent path returns 0 and that
    /// the correct file is removed when multiple files exist in the session.
    #[test]
    fn t_sto_009_delete_file_by_session_path() {
        let (conn, session_id) = setup_db_with_session();

        // Insert two file records.
        insert_file(&conn, session_id, "/docs/a.pdf", "h1", 1_000, 100, 2, None)
            .expect("insert a.pdf");
        insert_file(&conn, session_id, "/docs/b.pdf", "h2", 2_000, 200, 5, None)
            .expect("insert b.pdf");

        let files = list_files_by_session(&conn, session_id).expect("list");
        assert_eq!(files.len(), 2);

        // Deleting a non-existent path returns 0.
        let deleted = delete_file_by_session_path(&conn, session_id, "/docs/nonexistent.pdf")
            .expect("delete non-existent");
        assert_eq!(deleted, 0);

        // Delete a.pdf by session path.
        let deleted =
            delete_file_by_session_path(&conn, session_id, "/docs/a.pdf").expect("delete a.pdf");
        assert_eq!(deleted, 1);

        // Only b.pdf remains.
        let files = list_files_by_session(&conn, session_id).expect("list after delete");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_path, "/docs/b.pdf");
    }

    /// T-STO-063: delete_file_by_session_path enables re-insert of the same
    /// file path after deletion, verifying that the UNIQUE constraint is
    /// satisfied after cleanup. This simulates the re-indexing scenario where
    /// a stale file record from a previous run blocks a fresh insert.
    #[test]
    fn t_sto_010_delete_then_reinsert_same_path() {
        let (conn, session_id) = setup_db_with_session();

        // First insert.
        let id1 = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_v1",
            1_000,
            100,
            3,
            None,
        )
        .expect("first insert");
        assert!(id1 > 0);

        // Attempting a second insert with the same path fails.
        let dup_result = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_v2",
            2_000,
            200,
            4,
            None,
        );
        assert!(dup_result.is_err(), "duplicate insert must fail");

        // Delete the stale record.
        let deleted = delete_file_by_session_path(&conn, session_id, "/docs/paper.pdf")
            .expect("delete stale");
        assert_eq!(deleted, 1);

        // Re-insert succeeds after the stale record is removed.
        let id2 = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_v2",
            2_000,
            200,
            4,
            None,
        )
        .expect("re-insert after delete");
        assert!(id2 > 0);

        let file = get_file(&conn, id2).expect("get re-inserted file");
        assert_eq!(file.file_hash, "hash_v2");
    }

    /// T-STO-035: insert_file with pdf_page_count stores and retrieves the
    /// structural page count correctly.
    #[test]
    fn t_sto_035_insert_file_with_pdf_page_count() {
        let (conn, session_id) = setup_db_with_session();

        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/paper.pdf",
            "hash_abc",
            1_700_000_000,
            4096,
            1,
            Some(42),
        )
        .expect("insert_file failed");

        let file = get_file(&conn, file_id).expect("get_file failed");
        assert_eq!(file.page_count, 1);
        assert_eq!(file.pdf_page_count, Some(42));
    }
}
