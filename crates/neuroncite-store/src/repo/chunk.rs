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

// Chunk repository operations.
//
// Provides functions to insert and query chunk records. Each chunk references
// its parent `indexed_file` and its session, stores the chunk text content,
// document-level byte offsets, page range, chunk index, content hash, and
// optionally an embedding blob or external embedding file reference. The
// chunk table uses INTEGER PRIMARY KEY AUTOINCREMENT to prevent rowid reuse
// after deletion, which is critical for HNSW label stability.

use rusqlite::{Connection, params};

use crate::error::StoreError;

/// Row representation of a chunk record. Contains all columns defined
/// in the chunk table.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    pub file_id: i64,
    pub session_id: i64,
    pub page_start: i64,
    pub page_end: i64,
    pub chunk_index: i64,
    pub doc_text_offset_start: i64,
    pub doc_text_offset_end: i64,
    pub content: String,
    pub embedding: Option<Vec<u8>>,
    pub ext_offset: Option<i64>,
    pub ext_length: Option<i64>,
    pub content_hash: String,
    pub is_deleted: bool,
    pub created_at: i64,
}

/// Input parameters for inserting a single chunk. The caller constructs
/// these from the chunking pipeline output and optional embedding data.
#[derive(Debug)]
pub struct ChunkInsert<'a> {
    pub file_id: i64,
    pub session_id: i64,
    pub page_start: i64,
    pub page_end: i64,
    pub chunk_index: i64,
    pub doc_text_offset_start: i64,
    pub doc_text_offset_end: i64,
    pub content: &'a str,
    pub embedding: Option<&'a [u8]>,
    pub ext_offset: Option<i64>,
    pub ext_length: Option<i64>,
    pub content_hash: &'a str,
    /// 64-bit SimHash fingerprint of the chunk text, stored as a signed i64
    /// because SQLite INTEGER is always signed. The indexer computes this at
    /// insert time so that the dedup module can read it directly from the
    /// database instead of loading the full chunk content at search time.
    pub simhash: Option<i64>,
}

/// Bulk inserts multiple chunks within a single transaction. All chunks
/// are inserted atomically: if any single insert fails, the entire batch
/// is rolled back.
///
/// Returns the list of auto-generated chunk IDs in insertion order.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if any insert fails (e.g., duplicate
/// `chunk_index` for the same `file_id`, or foreign key violation).
pub fn bulk_insert_chunks(
    conn: &Connection,
    chunks: &[ChunkInsert<'_>],
) -> Result<Vec<i64>, StoreError> {
    let now = neuroncite_core::unix_timestamp_secs();

    // Use rusqlite's Transaction for RAII rollback safety. If any insert
    // fails and the function returns early via `?`, the Transaction's Drop
    // impl issues a ROLLBACK, preventing a dangling open transaction from
    // corrupting subsequent operations on this connection.
    let tx = conn.unchecked_transaction()?;

    let mut ids = Vec::with_capacity(chunks.len());

    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO chunk (
                file_id, session_id, page_start, page_end,
                chunk_index, doc_text_offset_start, doc_text_offset_end,
                content, embedding, ext_offset, ext_length,
                content_hash, simhash, is_deleted, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 0, ?14)",
        )?;

        for chunk in chunks {
            stmt.execute(params![
                chunk.file_id,
                chunk.session_id,
                chunk.page_start,
                chunk.page_end,
                chunk.chunk_index,
                chunk.doc_text_offset_start,
                chunk.doc_text_offset_end,
                chunk.content,
                chunk.embedding,
                chunk.ext_offset,
                chunk.ext_length,
                chunk.content_hash,
                chunk.simhash,
                now,
            ])?;
            ids.push(tx.last_insert_rowid());
        }
        // Statement is dropped here at block end, releasing the borrow on tx.
    }

    tx.commit()?;

    Ok(ids)
}

/// Retrieves a single chunk by primary key.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no chunk with the given ID exists,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_chunk(conn: &Connection, id: i64) -> Result<ChunkRow, StoreError> {
    conn.query_row(
        "SELECT id, file_id, session_id, page_start, page_end,
                chunk_index, doc_text_offset_start, doc_text_offset_end,
                content, embedding, ext_offset, ext_length,
                content_hash, is_deleted, created_at
         FROM chunk WHERE id = ?1",
        params![id],
        row_to_chunk,
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => StoreError::not_found("chunk", id.to_string()),
        other => other.into(),
    })
}

/// Maximum number of IDs per SQLite IN clause. SQLite's default
/// SQLITE_MAX_VARIABLE_NUMBER is 999; using 800 leaves headroom for any
/// additional bound parameters in future query modifications and avoids
/// hitting the limit on platforms compiled with a lower setting.
const MAX_IN_CLAUSE_SIZE: usize = 800;

/// Retrieves multiple chunks by primary key. Issues `SELECT ... WHERE id IN (...)`
/// statements instead of N individual queries, eliminating the N+1 query pattern
/// that occurs when callers loop over `get_chunk` for each ID.
///
/// When the `ids` slice exceeds `MAX_IN_CLAUSE_SIZE` (800), the IDs are split
/// into batches and each batch is issued as a separate query. Results from all
/// batches are merged into a single returned vector. This prevents hitting
/// SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (999 by default) when callers
/// supply large ID sets.
///
/// The returned vector contains only chunks that exist in the database.
/// IDs that do not match any row are silently omitted from the result.
/// The order of returned rows is determined by SQLite's query plan and is
/// not guaranteed to match the input order of `ids`.
///
/// If the `ids` slice is empty, returns an empty vector without issuing
/// any SQL statement.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if any query execution fails.
pub fn get_chunks_batch(conn: &Connection, ids: &[i64]) -> Result<Vec<ChunkRow>, StoreError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    // Split into batches when the ID count exceeds the IN clause limit.
    // Most callers (verify handler) supply 1-20 IDs so this path is rarely
    // taken; the allocation overhead is negligible for those cases.
    if ids.len() <= MAX_IN_CLAUSE_SIZE {
        return get_chunks_batch_single(conn, ids);
    }

    let mut all_results = Vec::with_capacity(ids.len());
    for batch in ids.chunks(MAX_IN_CLAUSE_SIZE) {
        let batch_results = get_chunks_batch_single(conn, batch)?;
        all_results.extend(batch_results);
    }
    Ok(all_results)
}

/// Executes a single `SELECT ... WHERE id IN (...)` query for a slice of IDs.
/// The caller is responsible for ensuring `ids` does not exceed the SQLite
/// variable limit (use `get_chunks_batch` for arbitrary sizes).
fn get_chunks_batch_single(conn: &Connection, ids: &[i64]) -> Result<Vec<ChunkRow>, StoreError> {
    debug_assert!(!ids.is_empty());
    debug_assert!(ids.len() <= MAX_IN_CLAUSE_SIZE);

    // Build a parameterized IN clause with one placeholder per ID.
    let placeholders: String = ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        "SELECT id, file_id, session_id, page_start, page_end,
                chunk_index, doc_text_offset_start, doc_text_offset_end,
                content, embedding, ext_offset, ext_length,
                content_hash, is_deleted, created_at
         FROM chunk WHERE id IN ({placeholders})"
    );

    let mut stmt = conn.prepare(&sql)?;

    // Bind each ID as a positional parameter. i64 implements ToSql, so we
    // create references directly without heap-allocating Box<dyn ToSql>.
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();

    let rows = stmt.query_map(param_refs.as_slice(), row_to_chunk)?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StoreError::from)
}

/// Lists all active (non-deleted) chunks belonging to a session, ordered by
/// `file_id` and `chunk_index`. The embedding column is excluded from the
/// result set (returned as NULL) because this function is used for display
/// and metadata purposes where the large binary embedding blobs are not
/// needed. Excluding the blob avoids transferring potentially megabytes of
/// binary data per session when only chunk text and metadata are required.
/// Callers that need embeddings should use `load_embeddings_for_hnsw` instead.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_chunks_by_session(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<ChunkRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, session_id, page_start, page_end,
                chunk_index, doc_text_offset_start, doc_text_offset_end,
                content, NULL as embedding, ext_offset, ext_length,
                content_hash, is_deleted, created_at
         FROM chunk WHERE session_id = ?1 AND is_deleted = 0
         ORDER BY file_id, chunk_index",
    )?;

    let rows = stmt.query_map(params![session_id], row_to_chunk)?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StoreError::from)
}

/// Lists all chunks belonging to a specific file, ordered by `chunk_index`.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_chunks_by_file(conn: &Connection, file_id: i64) -> Result<Vec<ChunkRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, session_id, page_start, page_end,
                chunk_index, doc_text_offset_start, doc_text_offset_end,
                content, embedding, ext_offset, ext_length,
                content_hash, is_deleted, created_at
         FROM chunk WHERE file_id = ?1 ORDER BY chunk_index",
    )?;

    let rows = stmt.query_map(params![file_id], row_to_chunk)?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StoreError::from)
}

/// Loads only chunk IDs and raw embedding byte blobs for a session. This is
/// a lightweight query for HNSW index construction that avoids reading the
/// full chunk content, page offsets, and other metadata. Only active
/// (non-deleted) chunks with a non-NULL embedding are returned.
///
/// Returns pairs of (chunk_id, embedding_bytes) where embedding_bytes is the
/// raw little-endian f32 blob. The caller is responsible for converting to
/// `Vec<f32>` via `bytes_to_f32_vec`.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn load_embeddings_for_hnsw(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<(i64, Vec<u8>)>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, embedding FROM chunk
         WHERE session_id = ?1 AND is_deleted = 0 AND embedding IS NOT NULL",
    )?;

    let rows = stmt.query_map(params![session_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StoreError::from)
}

/// Deletes all chunks belonging to a specific file. Returns the number
/// of rows deleted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn delete_chunks_by_file(conn: &Connection, file_id: i64) -> Result<usize, StoreError> {
    let count = conn.execute("DELETE FROM chunk WHERE file_id = ?1", params![file_id])?;
    Ok(count)
}

/// Sets or clears the `is_deleted` flag on a single chunk. When set to true,
/// the FTS5 trigger removes the chunk from the full-text search index.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the update fails.
pub fn set_chunk_deleted(conn: &Connection, id: i64, deleted: bool) -> Result<(), StoreError> {
    let flag: i64 = i64::from(deleted);
    conn.execute(
        "UPDATE chunk SET is_deleted = ?1 WHERE id = ?2",
        params![flag, id],
    )?;
    Ok(())
}

/// Maps a `rusqlite` row to a `ChunkRow`.
fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRow> {
    Ok(ChunkRow {
        id: row.get(0)?,
        file_id: row.get(1)?,
        session_id: row.get(2)?,
        page_start: row.get(3)?,
        page_end: row.get(4)?,
        chunk_index: row.get(5)?,
        doc_text_offset_start: row.get(6)?,
        doc_text_offset_end: row.get(7)?,
        content: row.get(8)?,
        embedding: row.get(9)?,
        ext_offset: row.get(10)?,
        ext_length: row.get(11)?,
        content_hash: row.get(12)?,
        is_deleted: row.get::<_, i64>(13)? != 0,
        created_at: row.get(14)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::{file as file_repo, page as page_repo, session};
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    fn setup_db() -> (Connection, i64, i64) {
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
            "hash123",
            1_700_000_000,
            4096,
            3,
            None,
        )
        .expect("file insert failed");

        (conn, session_id, file_id)
    }

    fn test_chunk(file_id: i64, session_id: i64, idx: i64) -> ChunkInsert<'static> {
        ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: idx,
            doc_text_offset_start: idx * 100,
            doc_text_offset_end: (idx + 1) * 100,
            content: "test chunk content for searching and indexing",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "abcdef0123456789",
            simhash: None,
        }
    }

    /// T-STO-009: Bulk insert 1000 chunks in a transaction.
    #[test]
    fn t_sto_009_bulk_insert_1000_chunks() {
        let (conn, session_id, file_id) = setup_db();

        let chunks: Vec<ChunkInsert<'_>> = (0..1000)
            .map(|i| ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: i * 50,
                doc_text_offset_end: (i + 1) * 50,
                content: "bulk test content",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "hash_bulk",
                simhash: None,
            })
            .collect();

        let ids = bulk_insert_chunks(&conn, &chunks).expect("bulk insert failed");
        assert_eq!(ids.len(), 1000);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(count, 1000);
    }

    /// T-STO-010: AUTOINCREMENT prevents rowid reuse after delete.
    #[test]
    fn t_sto_010_autoincrement_prevents_reuse() {
        let (conn, session_id, file_id) = setup_db();

        let chunk1 = vec![test_chunk(file_id, session_id, 0)];
        let ids1 = bulk_insert_chunks(&conn, &chunk1).expect("first insert failed");
        let first_id = ids1[0];

        conn.execute("DELETE FROM chunk WHERE id = ?1", params![first_id])
            .expect("delete failed");

        let chunk2 = vec![test_chunk(file_id, session_id, 1)];
        let ids2 = bulk_insert_chunks(&conn, &chunk2).expect("second insert failed");
        let second_id = ids2[0];

        assert!(
            second_id > first_id,
            "AUTOINCREMENT must prevent reuse: second_id ({second_id}) > first_id ({first_id})"
        );
    }

    /// T-STO-011: ON DELETE CASCADE from file deletes chunks and pages.
    #[test]
    fn t_sto_011_cascade_delete_from_file() {
        let (conn, session_id, file_id) = setup_db();

        let pages = vec![(1_i64, "page content", "pdf-extract")];
        page_repo::bulk_insert_pages(&conn, file_id, &pages).expect("page insert failed");

        let chunk = vec![test_chunk(file_id, session_id, 0)];
        bulk_insert_chunks(&conn, &chunk).expect("chunk insert failed");

        let chunk_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(chunk_count, 1);

        let page_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM page WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(page_count, 1);

        file_repo::delete_file(&conn, file_id).expect("file delete failed");

        let chunk_count_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(chunk_count_after, 0, "chunks must be cascade-deleted");

        let page_count_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM page WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(page_count_after, 0, "pages must be cascade-deleted");
    }

    /// T-STO-012: FTS5 INSERT trigger sync.
    #[test]
    fn t_sto_012_fts5_insert_trigger() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "quantum entanglement in photonic systems",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "fts_test_hash",
            simhash: None,
        }];

        let ids = bulk_insert_chunks(&conn, &chunks).expect("insert failed");
        let chunk_id = ids[0];

        let fts_rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'entanglement'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 query failed");

        assert_eq!(fts_rowid, chunk_id, "FTS5 rowid must match chunk id");
    }

    /// T-STO-013: FTS5 DELETE removal.
    #[test]
    fn t_sto_013_fts5_delete_removal() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "superconductivity in high temperature materials",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "fts_del_hash",
            simhash: None,
        }];

        let ids = bulk_insert_chunks(&conn, &chunks).expect("insert failed");
        let chunk_id = ids[0];

        conn.execute("DELETE FROM chunk WHERE id = ?1", params![chunk_id])
            .expect("delete failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'superconductivity'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 count query failed");

        assert_eq!(count, 0, "FTS5 entry must be removed after chunk deletion");
    }

    /// T-STO-014: FTS5 `is_deleted` flag removes from search.
    #[test]
    fn t_sto_014_fts5_is_deleted_flag() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "bioluminescent organisms in deep ocean trenches",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "fts_flag_hash",
            simhash: None,
        }];

        bulk_insert_chunks(&conn, &chunks).expect("insert failed");

        let count_before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'bioluminescent'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 query failed");
        assert_eq!(count_before, 1);

        let chunk_id: i64 = conn
            .query_row(
                "SELECT id FROM chunk WHERE file_id = ?1 AND chunk_index = 0",
                params![file_id],
                |row| row.get(0),
            )
            .expect("chunk lookup failed");

        set_chunk_deleted(&conn, chunk_id, true).expect("set_chunk_deleted failed");

        let count_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'bioluminescent'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 count query failed");

        assert_eq!(
            count_after, 0,
            "FTS5 entry must be removed when is_deleted is set to 1"
        );
    }

    /// T-STO-067: FTS5 un-delete trigger restores chunk to search index.
    /// When a chunk's is_deleted flag changes from 1 back to 0, the un-delete
    /// trigger re-inserts its content into the FTS5 index so it appears in
    /// keyword search results again. Without the trigger, un-deleted chunks
    /// are permanently lost from FTS5.
    #[test]
    fn t_sto_067_fts5_undelete_trigger_restores_search() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "magnetohydrodynamics in stellar plasma",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "undelete_test_hash",
            simhash: None,
        }];

        bulk_insert_chunks(&conn, &chunks).expect("insert failed");

        let chunk_id: i64 = conn
            .query_row(
                "SELECT id FROM chunk WHERE file_id = ?1 AND chunk_index = 0",
                params![file_id],
                |row| row.get(0),
            )
            .expect("chunk lookup failed");

        // Soft-delete the chunk; the v1 trigger removes it from FTS5.
        set_chunk_deleted(&conn, chunk_id, true).expect("soft delete failed");

        let count_deleted: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'magnetohydrodynamics'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 count after delete failed");
        assert_eq!(
            count_deleted, 0,
            "FTS5 entry must be absent after soft-delete"
        );

        // Un-delete the chunk; the un-delete trigger re-inserts it into FTS5.
        set_chunk_deleted(&conn, chunk_id, false).expect("un-delete failed");

        let count_restored: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'magnetohydrodynamics'",
                [],
                |row| row.get(0),
            )
            .expect("FTS5 count after un-delete failed");
        assert_eq!(
            count_restored, 1,
            "FTS5 entry must be restored after un-delete (is_deleted 1->0)"
        );
    }

    /// T-STO-016: bulk_insert_chunks rolls back on failure.
    /// When a constraint violation occurs mid-batch (duplicate chunk_index for
    /// the same file_id), the RAII Transaction guard rolls back the entire
    /// batch. No partial rows remain in the database.
    #[test]
    fn t_sto_016_bulk_insert_rollback_on_failure() {
        let (conn, session_id, file_id) = setup_db();

        // Insert a chunk at index 0 first.
        let first = vec![test_chunk(file_id, session_id, 0)];
        bulk_insert_chunks(&conn, &first).expect("first insert failed");

        // Attempt a batch containing a duplicate chunk_index (0) for the same
        // file_id. The UNIQUE(file_id, chunk_index) constraint causes the batch
        // to fail. The transaction guard must roll back the entire batch.
        let batch = vec![
            test_chunk(file_id, session_id, 1), // valid
            test_chunk(file_id, session_id, 0), // duplicate -> constraint violation
        ];

        let result = bulk_insert_chunks(&conn, &batch);
        assert!(
            result.is_err(),
            "bulk_insert_chunks must fail on duplicate chunk_index"
        );

        // Verify that chunk_index 1 from the failed batch was rolled back.
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunk WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(
            count, 1,
            "only the first chunk (from the earlier batch) must remain; \
             the failed batch must be fully rolled back"
        );
    }

    /// T-STO-028: load_embeddings_for_hnsw returns (id, embedding) pairs for
    /// active chunks with non-NULL embeddings, excluding deleted chunks and
    /// chunks without embeddings.
    #[test]
    fn t_sto_028_load_embeddings_for_hnsw() {
        let (conn, session_id, file_id) = setup_db();

        // Create a 4-dimensional f32 embedding as bytes (little-endian).
        let embedding_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let embedding_bytes: Vec<u8> = embedding_f32.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Chunk 0: has embedding (should be returned).
        let chunk_with_emb = ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "chunk with embedding",
            embedding: Some(&embedding_bytes),
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_emb",
            simhash: Some(42),
        };

        // Chunk 1: no embedding (should NOT be returned).
        let chunk_no_emb = ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 1,
            doc_text_offset_start: 50,
            doc_text_offset_end: 100,
            content: "chunk without embedding",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_no_emb",
            simhash: None,
        };

        let ids =
            bulk_insert_chunks(&conn, &[chunk_with_emb, chunk_no_emb]).expect("insert chunks");
        assert_eq!(ids.len(), 2);

        // Soft-delete chunk 0 to test the is_deleted filter.
        // First, load to confirm both are present initially.
        let results_before_delete =
            load_embeddings_for_hnsw(&conn, session_id).expect("load embeddings before delete");
        assert_eq!(
            results_before_delete.len(),
            1,
            "only chunk with embedding should be returned"
        );
        assert_eq!(results_before_delete[0].0, ids[0]);
        assert_eq!(results_before_delete[0].1, embedding_bytes);

        // Soft-delete chunk 0 and verify it is excluded.
        set_chunk_deleted(&conn, ids[0], true).expect("soft delete");
        let results_after_delete =
            load_embeddings_for_hnsw(&conn, session_id).expect("load embeddings after delete");
        assert!(
            results_after_delete.is_empty(),
            "deleted chunk must be excluded from HNSW load"
        );
    }

    /// T-STO-029: The simhash column is stored and retrievable. Verifies that
    /// SimHash values inserted via ChunkInsert are persisted in the database
    /// and can be read back through a direct SQL query.
    #[test]
    fn t_sto_029_simhash_column_stored() {
        let (conn, session_id, file_id) = setup_db();

        let simhash_value: i64 = 0x0123_4567_89AB_CDEF_u64 as i64;
        let chunk = ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 30,
            content: "simhash test content",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_simhash_test",
            simhash: Some(simhash_value),
        };

        let ids = bulk_insert_chunks(&conn, &[chunk]).expect("insert chunk");
        assert_eq!(ids.len(), 1);

        // Read the simhash back from the database.
        let stored_simhash: Option<i64> = conn
            .query_row(
                "SELECT simhash FROM chunk WHERE id = ?1",
                params![ids[0]],
                |row| row.get(0),
            )
            .expect("simhash query failed");

        assert_eq!(
            stored_simhash,
            Some(simhash_value),
            "stored simhash must match the inserted value"
        );
    }

    /// T-STO-030: The simhash column accepts NULL for backward compatibility.
    /// Chunks inserted without a simhash value store NULL in the database.
    #[test]
    fn t_sto_030_simhash_nullable() {
        let (conn, session_id, file_id) = setup_db();

        let chunk = ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 30,
            content: "null simhash test",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_null_simhash",
            simhash: None,
        };

        let ids = bulk_insert_chunks(&conn, &[chunk]).expect("insert chunk");
        let stored_simhash: Option<i64> = conn
            .query_row(
                "SELECT simhash FROM chunk WHERE id = ?1",
                params![ids[0]],
                |row| row.get(0),
            )
            .expect("simhash query failed");

        assert!(
            stored_simhash.is_none(),
            "simhash must be NULL when not provided"
        );
    }

    /// T-STO-050: list_chunks_by_session excludes embedding blobs from the
    /// result set. The embedding column is returned as NULL to avoid
    /// transferring large binary data when only metadata is needed.
    #[test]
    fn t_sto_050_list_chunks_by_session_excludes_embedding() {
        let (conn, session_id, file_id) = setup_db();

        // Create a chunk with an embedding blob.
        let embedding_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let chunk = ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 30,
            content: "embedding exclusion test",
            embedding: Some(&embedding_bytes),
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_emb_excl",
            simhash: None,
        };

        bulk_insert_chunks(&conn, &[chunk]).expect("insert chunk");

        // list_chunks_by_session returns NULL for the embedding column.
        let chunks =
            list_chunks_by_session(&conn, session_id).expect("list_chunks_by_session failed");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].embedding.is_none(),
            "list_chunks_by_session must return NULL for embedding column"
        );

        // Verify the embedding is still stored in the database via get_chunk.
        let full_chunk = get_chunk(&conn, chunks[0].id).expect("get_chunk failed");
        assert!(
            full_chunk.embedding.is_some(),
            "get_chunk must return the actual embedding blob"
        );
        assert_eq!(
            full_chunk.embedding.unwrap(),
            embedding_bytes,
            "get_chunk embedding must match the inserted bytes"
        );
    }

    // -------------------------------------------------------------------
    // get_chunks_batch tests
    // -------------------------------------------------------------------

    /// T-STO-068: get_chunks_batch returns an empty vector when called with
    /// an empty ID slice, without issuing any SQL statement.
    #[test]
    fn t_sto_068_get_chunks_batch_empty_ids() {
        let (conn, _session_id, _file_id) = setup_db();

        let result = get_chunks_batch(&conn, &[]).expect("empty batch must succeed");
        assert!(
            result.is_empty(),
            "empty ID slice must return empty result vector"
        );
    }

    /// T-STO-069: get_chunks_batch retrieves a single chunk correctly when
    /// given a one-element ID slice.
    #[test]
    fn t_sto_069_get_chunks_batch_single_id() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![test_chunk(file_id, session_id, 0)];
        let ids = bulk_insert_chunks(&conn, &chunks).expect("insert failed");
        assert_eq!(ids.len(), 1);

        let result = get_chunks_batch(&conn, &ids).expect("single-ID batch must succeed");
        assert_eq!(result.len(), 1, "must return exactly one chunk");
        assert_eq!(
            result[0].id, ids[0],
            "returned chunk ID must match the requested ID"
        );
        assert_eq!(
            result[0].session_id, session_id,
            "returned chunk session_id must match"
        );
        assert_eq!(
            result[0].content, "test chunk content for searching and indexing",
            "returned chunk content must match the inserted content"
        );
    }

    /// T-STO-070: get_chunks_batch retrieves multiple chunks in a single
    /// query. All requested IDs are returned with correct content.
    #[test]
    fn t_sto_070_get_chunks_batch_multiple_ids() {
        let (conn, session_id, file_id) = setup_db();

        let inserts: Vec<ChunkInsert<'_>> = (0..5)
            .map(|i| ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: i * 50,
                doc_text_offset_end: (i + 1) * 50,
                content: "batch retrieval test content",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "hash_batch_get",
                simhash: None,
            })
            .collect();

        let ids = bulk_insert_chunks(&conn, &inserts).expect("insert failed");
        assert_eq!(ids.len(), 5);

        let result = get_chunks_batch(&conn, &ids).expect("batch retrieval must succeed");
        assert_eq!(result.len(), 5, "must return all 5 chunks");

        // Verify all requested IDs are present in the result set.
        let returned_ids: std::collections::HashSet<i64> = result.iter().map(|c| c.id).collect();
        for &id in &ids {
            assert!(
                returned_ids.contains(&id),
                "returned set must contain chunk ID {id}"
            );
        }
    }

    /// T-STO-071: get_chunks_batch silently omits IDs that do not exist in
    /// the database. Requesting a mix of valid and non-existent IDs returns
    /// only the valid ones.
    #[test]
    fn t_sto_071_get_chunks_batch_nonexistent_ids() {
        let (conn, session_id, file_id) = setup_db();

        let chunks = vec![test_chunk(file_id, session_id, 0)];
        let ids = bulk_insert_chunks(&conn, &chunks).expect("insert failed");
        let valid_id = ids[0];

        // Request the valid ID plus two IDs that do not exist in the database.
        let query_ids = vec![valid_id, 999_999, 888_888];
        let result =
            get_chunks_batch(&conn, &query_ids).expect("batch with non-existent must succeed");

        assert_eq!(
            result.len(),
            1,
            "only the existing chunk must be returned; non-existent IDs are omitted"
        );
        assert_eq!(result[0].id, valid_id);
    }

    /// T-STO-072: get_chunks_batch returns an empty vector when all
    /// requested IDs are non-existent.
    #[test]
    fn t_sto_072_get_chunks_batch_all_nonexistent() {
        let (conn, _session_id, _file_id) = setup_db();

        let result =
            get_chunks_batch(&conn, &[999_999, 888_888]).expect("all-nonexistent must succeed");
        assert!(
            result.is_empty(),
            "all non-existent IDs must produce an empty result"
        );
    }
}
