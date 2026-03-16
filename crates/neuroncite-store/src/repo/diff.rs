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

// Session diff computation.
//
// Compares two index sessions by their indexed file records. The comparison
// key is file_path (the absolute path to the PDF). For files present in both
// sessions, the file hash, page count, and size are compared to detect content
// or structural differences.

use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::StoreError;
use crate::repo::file::{FileRow, list_files_by_session};

/// Result of comparing a file that exists in both session A and session B.
/// Reports whether the content hash, extracted page count, and file size
/// differ between the two sessions.
#[derive(Debug, Clone)]
pub struct FileDiff {
    /// Absolute path to the PDF file.
    pub file_path: String,
    /// file_id in session A.
    pub file_id_a: i64,
    /// file_id in session B.
    pub file_id_b: i64,
    /// True when the SHA-256 content hash differs between sessions.
    pub hash_changed: bool,
    /// True when the extracted page count differs between sessions.
    pub page_count_changed: bool,
    /// True when the file size in bytes differs between sessions.
    pub size_changed: bool,
    /// SHA-256 hash from session A.
    pub hash_a: String,
    /// SHA-256 hash from session B.
    pub hash_b: String,
}

/// Full result of comparing two index sessions by their indexed files.
#[derive(Debug, Clone)]
pub struct SessionDiff {
    /// Files present in session A but absent in session B (by file_path).
    pub only_in_a: Vec<FileRow>,
    /// Files present in session B but absent in session A (by file_path).
    pub only_in_b: Vec<FileRow>,
    /// Files present in both sessions with identical hash, page_count, and size.
    pub identical: Vec<FileDiff>,
    /// Files present in both sessions where at least one attribute differs.
    pub changed: Vec<FileDiff>,
}

/// Compares all indexed files in two sessions and classifies them into four
/// categories: only_in_a, only_in_b, identical, and changed.
///
/// Both sessions are queried in a single connection context. The comparison
/// key is file_path: files with the same path are matched across sessions.
/// For matched files, the SHA-256 hash, extracted page count, and file size
/// are compared.
///
/// Returns StoreError::Sqlite if either session's file list query fails.
pub fn diff_sessions(
    conn: &Connection,
    session_a: i64,
    session_b: i64,
) -> Result<SessionDiff, StoreError> {
    let files_a = list_files_by_session(conn, session_a)?;
    let files_b = list_files_by_session(conn, session_b)?;

    // Build path -> FileRow lookup for O(1) membership checks.
    let map_b: HashMap<&str, &FileRow> =
        files_b.iter().map(|f| (f.file_path.as_str(), f)).collect();

    let map_a: HashMap<&str, &FileRow> =
        files_a.iter().map(|f| (f.file_path.as_str(), f)).collect();

    let mut only_in_a = Vec::new();
    let mut only_in_b = Vec::new();
    let mut identical = Vec::new();
    let mut changed = Vec::new();

    // Classify files from session A against session B.
    for file_a in &files_a {
        match map_b.get(file_a.file_path.as_str()) {
            None => only_in_a.push(file_a.clone()),
            Some(file_b) => {
                let hash_changed = file_a.file_hash != file_b.file_hash;
                let page_count_changed = file_a.page_count != file_b.page_count;
                let size_changed = file_a.size != file_b.size;

                let diff = FileDiff {
                    file_path: file_a.file_path.clone(),
                    file_id_a: file_a.id,
                    file_id_b: file_b.id,
                    hash_changed,
                    page_count_changed,
                    size_changed,
                    hash_a: file_a.file_hash.clone(),
                    hash_b: file_b.file_hash.clone(),
                };

                if hash_changed || page_count_changed || size_changed {
                    changed.push(diff);
                } else {
                    identical.push(diff);
                }
            }
        }
    }

    // Find files in session B that are not in session A.
    for file_b in &files_b {
        if !map_a.contains_key(file_b.file_path.as_str()) {
            only_in_b.push(file_b.clone());
        }
    }

    Ok(SessionDiff {
        only_in_a,
        only_in_b,
        identical,
        changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::migrate;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");
        conn
    }

    fn create_session(conn: &Connection, dir: &str) -> i64 {
        conn.execute(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, created_at
            ) VALUES (?1, 'model', 'page', 'eng', 384, 'sqlite-blob', 1, '0.1.0', 1700000000)",
            rusqlite::params![dir],
        )
        .expect("create session");
        conn.last_insert_rowid()
    }

    fn insert_file(
        conn: &Connection,
        session_id: i64,
        path: &str,
        hash: &str,
        pages: i64,
        size: i64,
    ) -> i64 {
        conn.execute(
            "INSERT INTO indexed_file (
                session_id, file_path, file_hash, mtime, size, page_count, created_at, updated_at
            ) VALUES (?1, ?2, ?3, 1700000000, ?4, ?5, 1700000000, 1700000000)",
            rusqlite::params![session_id, path, hash, size, pages],
        )
        .expect("insert file");
        conn.last_insert_rowid()
    }

    /// T-DIFF-001: diff_sessions identifies files that exist only in session A.
    #[test]
    fn t_diff_001_only_in_a() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/doc1.pdf", "hash1", 10, 4096);
        insert_file(&conn, sa, "/doc2.pdf", "hash2", 20, 8192);
        insert_file(&conn, sa, "/doc3.pdf", "hash3", 5, 2048);
        insert_file(&conn, sb, "/doc1.pdf", "hash1", 10, 4096);
        insert_file(&conn, sb, "/doc2.pdf", "hash2", 20, 8192);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.only_in_a.len(), 1);
        assert_eq!(diff.only_in_a[0].file_path, "/doc3.pdf");
        assert_eq!(diff.only_in_b.len(), 0);
        assert_eq!(diff.identical.len(), 2);
    }

    /// T-DIFF-002: diff_sessions identifies files that exist only in session B.
    #[test]
    fn t_diff_002_only_in_b() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/doc1.pdf", "hash1", 10, 4096);
        insert_file(&conn, sb, "/doc1.pdf", "hash1", 10, 4096);
        insert_file(&conn, sb, "/extra.pdf", "hash_extra", 15, 6000);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.only_in_b.len(), 1);
        assert_eq!(diff.only_in_b[0].file_path, "/extra.pdf");
        assert_eq!(diff.only_in_a.len(), 0);
    }

    /// T-DIFF-003: files with same path and same hash appear in the identical
    /// category.
    #[test]
    fn t_diff_003_identical_files() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/same.pdf", "identical_hash", 10, 4096);
        insert_file(&conn, sb, "/same.pdf", "identical_hash", 10, 4096);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.identical.len(), 1);
        assert_eq!(diff.changed.len(), 0);
        assert!(!diff.identical[0].hash_changed);
        assert!(!diff.identical[0].page_count_changed);
        assert!(!diff.identical[0].size_changed);
    }

    /// T-DIFF-004: files with same path but different hash appear in the
    /// changed category with hash_changed=true.
    #[test]
    fn t_diff_004_changed_hash() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/doc.pdf", "hash_v1", 10, 4096);
        insert_file(&conn, sb, "/doc.pdf", "hash_v2", 10, 4096);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.changed.len(), 1);
        assert!(diff.changed[0].hash_changed);
        assert!(!diff.changed[0].page_count_changed);
        assert!(!diff.changed[0].size_changed);
        assert_eq!(diff.changed[0].hash_a, "hash_v1");
        assert_eq!(diff.changed[0].hash_b, "hash_v2");
    }

    /// T-DIFF-005: files with same hash but different page_count appear in
    /// the changed category.
    #[test]
    fn t_diff_005_changed_page_count() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/doc.pdf", "same_hash", 10, 4096);
        insert_file(&conn, sb, "/doc.pdf", "same_hash", 12, 4096);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.changed.len(), 1);
        assert!(!diff.changed[0].hash_changed);
        assert!(diff.changed[0].page_count_changed);
    }

    /// T-DIFF-006: empty session B produces all files in only_in_a.
    #[test]
    fn t_diff_006_empty_session_b() {
        let conn = setup_db();
        let sa = create_session(&conn, "/dir_a");
        let sb = create_session(&conn, "/dir_b");

        insert_file(&conn, sa, "/doc1.pdf", "h1", 5, 1000);
        insert_file(&conn, sa, "/doc2.pdf", "h2", 10, 2000);

        let diff = diff_sessions(&conn, sa, sb).expect("diff failed");
        assert_eq!(diff.only_in_a.len(), 2);
        assert_eq!(diff.only_in_b.len(), 0);
        assert_eq!(diff.identical.len(), 0);
        assert_eq!(diff.changed.len(), 0);
    }
}
