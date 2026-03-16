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

// FTS5 full-text search index management.
//
// Provides maintenance functions for the `chunk_fts` FTS5 virtual table.
// The actual insert/delete synchronization is handled by `SQLite` triggers
// defined in `schema.rs`. This module provides the optimize command for
// merging FTS5 internal b-tree segments, and an integrity check for
// verifying FTS5 index consistency.

use rusqlite::Connection;

use crate::error::StoreError;

/// Runs the FTS5 optimize command on the `chunk_fts` virtual table. This
/// merges all internal b-tree segments into a single segment, improving
/// query performance for workloads that have accumulated many insert/delete
/// operations.
///
/// This operation may take several seconds for large indexes. The caller
/// should ensure exclusive database access (no concurrent indexing) during
/// the optimize.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the FTS5 optimize command fails.
pub fn optimize_fts(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize');")?;
    Ok(())
}

/// Runs the FTS5 integrity-check command on the `chunk_fts` virtual table.
/// Returns "ok" if the index is consistent, or a descriptive error string
/// if corruption is detected.
///
/// The integrity check verifies that the FTS5 index data matches the
/// content of the backing chunk table. Discrepancies indicate that triggers
/// have failed or that the index has been corrupted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the integrity check command itself fails
/// (distinct from the check reporting corruption).
pub fn integrity_check_fts(conn: &Connection) -> Result<String, StoreError> {
    // FTS5 integrity-check raises an error if the index is inconsistent.
    // If no error is raised, the index is consistent.
    match conn.execute_batch("INSERT INTO chunk_fts(chunk_fts) VALUES('integrity-check');") {
        Ok(()) => Ok("ok".to_string()),
        Err(e) => Ok(format!("integrity check failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::chunk::{ChunkInsert, bulk_insert_chunks};
    use crate::repo::{file as file_repo, session};
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    fn setup_db_with_chunks() -> Connection {
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

        let file_id = file_repo::insert_file(
            &conn,
            session_id,
            "/docs/test.pdf",
            "hash",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("file insert failed");

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 30,
            content: "test content for fts optimization",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "opt_hash",
            simhash: None,
        }];
        bulk_insert_chunks(&conn, &chunks).expect("chunk insert failed");

        conn
    }

    #[test]
    fn optimize_succeeds() {
        let conn = setup_db_with_chunks();
        optimize_fts(&conn).expect("optimize_fts failed");
    }

    #[test]
    fn integrity_check_reports_ok() {
        let conn = setup_db_with_chunks();
        let result = integrity_check_fts(&conn).expect("integrity_check failed");
        assert_eq!(result, "ok");
    }
}
