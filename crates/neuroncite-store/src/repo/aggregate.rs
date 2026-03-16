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

// Aggregate query operations.
//
// Provides batch-efficient SQL aggregate functions used by the statistics,
// quality, discovery, and comparison endpoints. All functions accept a
// borrowed `rusqlite::Connection` and return lightweight structs with
// aggregate counts. These functions avoid loading full entity rows
// (no content blobs, no embeddings) to minimize memory usage.

use rusqlite::{Connection, params};

use crate::error::StoreError;
use crate::repo::file::FileRow;

// ---------------------------------------------------------------------------
// Session-level aggregates
// ---------------------------------------------------------------------------

/// Aggregate counts for a single index session: file count, total extracted
/// pages, active (non-deleted) chunk count, and total content bytes across
/// all pages.
#[derive(Debug, Clone)]
pub struct SessionAggregates {
    pub session_id: i64,
    pub file_count: i64,
    pub total_pages: i64,
    pub total_chunks: i64,
    pub total_content_bytes: i64,
}

/// Retrieves aggregate counts for ALL sessions in a single query. Uses LEFT
/// JOIN subqueries with GROUP BY so that sessions with zero files/chunks
/// return zero counts instead of being omitted. Ordered by session
/// `created_at DESC` to match the `list_sessions` ordering.
pub fn all_session_aggregates(conn: &Connection) -> Result<Vec<SessionAggregates>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT
            s.id AS session_id,
            COALESCE(fc.cnt, 0) AS file_count,
            COALESCE(fc.pages, 0) AS total_pages,
            COALESCE(cc.cnt, 0) AS total_chunks,
            COALESCE(bc.bytes, 0) AS total_content_bytes
        FROM index_session s
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS cnt, SUM(page_count) AS pages
            FROM indexed_file
            GROUP BY session_id
        ) fc ON s.id = fc.session_id
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS cnt
            FROM chunk
            WHERE is_deleted = 0
            GROUP BY session_id
        ) cc ON s.id = cc.session_id
        LEFT JOIN (
            SELECT f.session_id, SUM(p.byte_count) AS bytes
            FROM page p
            JOIN indexed_file f ON p.file_id = f.id
            GROUP BY f.session_id
        ) bc ON s.id = bc.session_id
        ORDER BY s.created_at DESC",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(SessionAggregates {
            session_id: row.get(0)?,
            file_count: row.get(1)?,
            total_pages: row.get(2)?,
            total_chunks: row.get(3)?,
            total_content_bytes: row.get(4)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Per-file chunk statistics
// ---------------------------------------------------------------------------

/// Chunk statistics for a single file: count of active chunks and
/// min/max/avg content byte lengths.
#[derive(Debug, Clone)]
pub struct FileChunkStats {
    pub file_id: i64,
    pub chunk_count: i64,
    pub min_content_len: i64,
    pub max_content_len: i64,
    pub avg_content_len: f64,
}

/// Retrieves per-file chunk statistics for all files in a session. Only
/// active (non-deleted) chunks are counted. Files with zero chunks are
/// not included in the result (use a HashMap lookup with a fallback for
/// files that have no chunks).
pub fn file_chunk_stats_by_session(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<FileChunkStats>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT
            file_id,
            COUNT(*) AS chunk_count,
            MIN(LENGTH(content)) AS min_content_len,
            MAX(LENGTH(content)) AS max_content_len,
            AVG(LENGTH(content)) AS avg_content_len
        FROM chunk
        WHERE session_id = ?1 AND is_deleted = 0
        GROUP BY file_id",
    )?;

    let rows = stmt.query_map(params![session_id], |row| {
        Ok(FileChunkStats {
            file_id: row.get(0)?,
            chunk_count: row.get(1)?,
            min_content_len: row.get(2)?,
            max_content_len: row.get(3)?,
            avg_content_len: row.get(4)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Retrieves chunk statistics for a single file. Returns a zero-count struct
/// when the file has no active chunks.
pub fn single_file_chunk_stats(
    conn: &Connection,
    file_id: i64,
) -> Result<FileChunkStats, StoreError> {
    let result = conn.query_row(
        "SELECT
            COUNT(*) AS chunk_count,
            COALESCE(MIN(LENGTH(content)), 0) AS min_content_len,
            COALESCE(MAX(LENGTH(content)), 0) AS max_content_len,
            COALESCE(AVG(LENGTH(content)), 0.0) AS avg_content_len
        FROM chunk
        WHERE file_id = ?1 AND is_deleted = 0",
        params![file_id],
        |row| {
            Ok(FileChunkStats {
                file_id,
                chunk_count: row.get(0)?,
                min_content_len: row.get(1)?,
                max_content_len: row.get(2)?,
                avg_content_len: row.get(3)?,
            })
        },
    )?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Per-file page backend statistics
// ---------------------------------------------------------------------------

/// Page statistics for a single (file_id, backend) combination: count of
/// pages extracted by that backend, total byte count of their content,
/// and count of pages with zero bytes (empty pages).
#[derive(Debug, Clone)]
pub struct FilePageStats {
    pub file_id: i64,
    pub backend: String,
    pub page_count: i64,
    pub total_bytes: i64,
    pub empty_count: i64,
}

/// Retrieves per-file, per-backend page statistics for all files in a
/// session. Groups by (file_id, backend) so each file may have multiple
/// rows when different pages used different extraction backends.
pub fn file_page_stats_by_session(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<FilePageStats>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT
            p.file_id,
            p.backend,
            COUNT(*) AS page_count,
            SUM(p.byte_count) AS total_bytes,
            SUM(CASE WHEN p.byte_count = 0 THEN 1 ELSE 0 END) AS empty_count
        FROM page p
        JOIN indexed_file f ON p.file_id = f.id
        WHERE f.session_id = ?1
        GROUP BY p.file_id, p.backend",
    )?;

    let rows = stmt.query_map(params![session_id], |row| {
        Ok(FilePageStats {
            file_id: row.get(0)?,
            backend: row.get(1)?,
            page_count: row.get(2)?,
            total_bytes: row.get(3)?,
            empty_count: row.get(4)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Chunk browsing (paginated, without search)
// ---------------------------------------------------------------------------

/// A chunk row for browsing purposes. Excludes the embedding blob and
/// external storage references to minimize memory and bandwidth.
#[derive(Debug, Clone)]
pub struct ChunkBrowseRow {
    pub id: i64,
    pub chunk_index: i64,
    pub page_start: i64,
    pub page_end: i64,
    pub content: String,
    pub content_hash: String,
}

/// Retrieves paginated chunks for a file, ordered by chunk_index. When
/// `page_filter` is `Some(n)`, only chunks whose page range spans page `n`
/// are returned. Returns `(chunks, total_count)` where `total_count` is the
/// unfiltered (but page-filtered if applicable) count for pagination headers.
pub fn browse_chunks(
    conn: &Connection,
    file_id: i64,
    page_filter: Option<i64>,
    offset: i64,
    limit: i64,
) -> Result<(Vec<ChunkBrowseRow>, i64), StoreError> {
    let total: i64 = match page_filter {
        None => conn.query_row(
            "SELECT COUNT(*) FROM chunk WHERE file_id = ?1 AND is_deleted = 0",
            params![file_id],
            |row| row.get(0),
        )?,
        Some(page) => conn.query_row(
            "SELECT COUNT(*) FROM chunk
             WHERE file_id = ?1 AND is_deleted = 0
               AND page_start <= ?2 AND page_end >= ?2",
            params![file_id, page],
            |row| row.get(0),
        )?,
    };

    let chunks = match page_filter {
        None => {
            let mut stmt = conn.prepare_cached(
                "SELECT id, chunk_index, page_start, page_end, content, content_hash
                 FROM chunk
                 WHERE file_id = ?1 AND is_deleted = 0
                 ORDER BY chunk_index
                 LIMIT ?2 OFFSET ?3",
            )?;
            let rows = stmt.query_map(params![file_id, limit, offset], row_to_browse)?;
            let mut v = Vec::new();
            for row in rows {
                v.push(row?);
            }
            v
        }
        Some(page) => {
            let mut stmt = conn.prepare_cached(
                "SELECT id, chunk_index, page_start, page_end, content, content_hash
                 FROM chunk
                 WHERE file_id = ?1 AND is_deleted = 0
                   AND page_start <= ?2 AND page_end >= ?2
                 ORDER BY chunk_index
                 LIMIT ?3 OFFSET ?4",
            )?;
            let rows = stmt.query_map(params![file_id, page, limit, offset], row_to_browse)?;
            let mut v = Vec::new();
            for row in rows {
                v.push(row?);
            }
            v
        }
    };

    Ok((chunks, total))
}

/// Maps a row to a `ChunkBrowseRow`. Column order: id, chunk_index,
/// page_start, page_end, content, content_hash.
fn row_to_browse(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkBrowseRow> {
    Ok(ChunkBrowseRow {
        id: row.get(0)?,
        chunk_index: row.get(1)?,
        page_start: row.get(2)?,
        page_end: row.get(3)?,
        content: row.get(4)?,
        content_hash: row.get(5)?,
    })
}

// ---------------------------------------------------------------------------
// Cross-session file search
// ---------------------------------------------------------------------------

/// Finds all indexed_file records across all sessions whose file_path matches
/// the given LIKE pattern. The pattern uses SQL LIKE syntax (% = any,
/// _ = single) with backslash as the ESCAPE character. Callers that
/// construct the pattern from user input must escape literal `%`, `_`, and
/// `\` characters with a leading backslash (see `escape_like_pattern` in the
/// compare handler) so that those characters are matched literally rather
/// than interpreted as LIKE wildcards.
pub fn find_files_by_path_pattern(
    conn: &Connection,
    pattern: &str,
) -> Result<Vec<FileRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, session_id, file_path, file_hash, mtime, size, page_count,
                pdf_page_count, source_type, created_at, updated_at
         FROM indexed_file
         WHERE file_path LIKE ?1 ESCAPE '\\'
         ORDER BY session_id, file_path",
    )?;

    let rows = stmt.query_map(params![pattern], |row| {
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
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Quality report data
// ---------------------------------------------------------------------------

/// Per-file quality data: page backend distribution, empty page count,
/// total content bytes, and structural vs extracted page counts. Used by
/// the quality report endpoint to flag extraction issues.
#[derive(Debug, Clone)]
pub struct FileQualityRow {
    pub file_id: i64,
    pub file_path: String,
    pub page_count: i64,
    pub pdf_page_count: Option<i64>,
    pub native_pages: i64,
    pub ocr_pages: i64,
    pub empty_pages: i64,
    pub total_bytes: i64,
}

/// Retrieves per-file quality data for all files in a session. Each row
/// aggregates page backend distribution and byte totals for one file.
pub fn session_quality_data(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<FileQualityRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT
            f.id AS file_id,
            f.file_path,
            f.page_count,
            f.pdf_page_count,
            COALESCE(SUM(CASE WHEN p.backend IN ('pdf-extract', 'pdfium') THEN 1 ELSE 0 END), 0) AS native_pages,
            COALESCE(SUM(CASE WHEN p.backend = 'ocr' THEN 1 ELSE 0 END), 0) AS ocr_pages,
            COALESCE(SUM(CASE WHEN p.byte_count = 0 THEN 1 ELSE 0 END), 0) AS empty_pages,
            COALESCE(SUM(p.byte_count), 0) AS total_bytes
        FROM indexed_file f
        LEFT JOIN page p ON p.file_id = f.id
        WHERE f.session_id = ?1
        GROUP BY f.id
        ORDER BY f.file_path",
    )?;

    let rows = stmt.query_map(params![session_id], |row| {
        Ok(FileQualityRow {
            file_id: row.get(0)?,
            file_path: row.get(1)?,
            page_count: row.get(2)?,
            pdf_page_count: row.get(3)?,
            native_pages: row.get(4)?,
            ocr_pages: row.get(5)?,
            empty_pages: row.get(6)?,
            total_bytes: row.get(7)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Per-page quality detail for a single page that has a quality issue
/// (empty content or OCR-extracted). Used by the quality report handler
/// to list specific page numbers that need attention.
#[derive(Debug, Clone)]
pub struct PageQualityDetail {
    /// File ID this page belongs to.
    pub file_id: i64,
    /// 1-indexed page number within the file.
    pub page_number: i64,
    /// Extraction backend used for this page ("pdf-extract", "pdfium", "ocr").
    pub backend: String,
    /// Byte count of the extracted text content. 0 indicates an empty page.
    pub byte_count: i64,
}

/// Retrieves per-page quality details for all pages in a session that have
/// quality issues: empty content (byte_count = 0) or OCR-extracted text.
/// Returns rows sorted by file_id and page_number for deterministic grouping.
pub fn page_quality_details(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<PageQualityDetail>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT p.file_id, p.page_number, p.backend, p.byte_count
        FROM page p
        INNER JOIN indexed_file f ON f.id = p.file_id
        WHERE f.session_id = ?1
          AND (p.byte_count = 0 OR p.backend = 'ocr')
        ORDER BY p.file_id, p.page_number",
    )?;

    let rows = stmt.query_map(params![session_id], |row| {
        Ok(PageQualityDetail {
            file_id: row.get(0)?,
            page_number: row.get(1)?,
            backend: row.get(2)?,
            byte_count: row.get(3)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::{
        chunk::{ChunkInsert, bulk_insert_chunks},
        file::insert_file,
        page::bulk_insert_pages,
        session::create_session,
    };
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    /// Creates an in-memory database with schema applied.
    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable FK");
        migrate(&conn).expect("migrate");
        conn
    }

    /// Creates a test IndexConfig for the given directory.
    fn test_config(dir: &str) -> IndexConfig {
        IndexConfig {
            directory: PathBuf::from(dir),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        }
    }

    /// Inserts a session, a file, pages, and chunks for testing.
    fn populate_session(conn: &Connection, dir: &str) -> (i64, i64) {
        let session_id = create_session(conn, &test_config(dir), "0.1.0").expect("create session");
        let file_id = insert_file(
            conn,
            session_id,
            "/docs/paper.pdf",
            "hash1",
            1000,
            4096,
            3,
            Some(3),
        )
        .expect("insert file");

        let pages = vec![
            (
                1_i64,
                "First page content with several words for testing",
                "pdf-extract",
            ),
            (2_i64, "Second page also has content", "pdf-extract"),
            (3_i64, "", "pdf-extract"),
        ];
        bulk_insert_pages(conn, file_id, &pages).expect("insert pages");

        let chunks: Vec<ChunkInsert<'_>> = (0..5)
            .map(|i| ChunkInsert {
                file_id,
                session_id,
                page_start: (i / 2) + 1,
                page_end: (i / 2) + 1,
                chunk_index: i,
                doc_text_offset_start: i * 50,
                doc_text_offset_end: (i + 1) * 50,
                content: "chunk text content for aggregate testing purposes",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "hash_agg",
                simhash: None,
            })
            .collect();
        bulk_insert_chunks(conn, &chunks).expect("insert chunks");

        (session_id, file_id)
    }

    /// T-AGG-001: all_session_aggregates returns correct counts for a session
    /// with files, pages, and chunks.
    #[test]
    fn t_agg_001_session_aggregates_with_data() {
        let conn = setup_db();
        let (session_id, _) = populate_session(&conn, "/docs");

        let aggs = all_session_aggregates(&conn).expect("aggregates");
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].session_id, session_id);
        assert_eq!(aggs[0].file_count, 1);
        assert_eq!(aggs[0].total_pages, 3);
        assert_eq!(aggs[0].total_chunks, 5);
        assert!(aggs[0].total_content_bytes > 0);
    }

    /// T-AGG-002: all_session_aggregates returns zero counts for an empty session.
    #[test]
    fn t_agg_002_session_aggregates_empty() {
        let conn = setup_db();
        create_session(&conn, &test_config("/empty"), "0.1.0").expect("create session");

        let aggs = all_session_aggregates(&conn).expect("aggregates");
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].file_count, 0);
        assert_eq!(aggs[0].total_pages, 0);
        assert_eq!(aggs[0].total_chunks, 0);
        assert_eq!(aggs[0].total_content_bytes, 0);
    }

    /// T-AGG-003: file_chunk_stats_by_session returns per-file chunk counts.
    #[test]
    fn t_agg_003_file_chunk_stats() {
        let conn = setup_db();
        let (session_id, file_id) = populate_session(&conn, "/docs");

        let stats = file_chunk_stats_by_session(&conn, session_id).expect("chunk stats");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].file_id, file_id);
        assert_eq!(stats[0].chunk_count, 5);
        assert!(stats[0].min_content_len > 0);
        assert!(stats[0].max_content_len >= stats[0].min_content_len);
    }

    /// T-AGG-004: single_file_chunk_stats returns zero-count for a file
    /// with no chunks.
    #[test]
    fn t_agg_004_single_file_chunk_stats_no_chunks() {
        let conn = setup_db();
        let session_id =
            create_session(&conn, &test_config("/docs"), "0.1.0").expect("create session");
        let file_id = insert_file(
            &conn,
            session_id,
            "/docs/empty.pdf",
            "h",
            1000,
            100,
            1,
            None,
        )
        .expect("insert file");

        let stats = single_file_chunk_stats(&conn, file_id).expect("stats");
        assert_eq!(stats.chunk_count, 0);
        assert_eq!(stats.min_content_len, 0);
        assert_eq!(stats.max_content_len, 0);
    }

    /// T-AGG-005: file_page_stats_by_session returns per-file backend distribution.
    #[test]
    fn t_agg_005_file_page_stats() {
        let conn = setup_db();
        let (session_id, _) = populate_session(&conn, "/docs");

        let stats = file_page_stats_by_session(&conn, session_id).expect("page stats");
        assert!(!stats.is_empty());
        // All pages used "pdf-extract" backend in the test data.
        assert_eq!(stats[0].backend, "pdf-extract");
        assert_eq!(stats[0].page_count, 3);
        // One empty page (page 3 has empty content).
        assert_eq!(stats[0].empty_count, 1);
    }

    /// T-AGG-006: browse_chunks returns paginated results.
    #[test]
    fn t_agg_006_browse_chunks_pagination() {
        let conn = setup_db();
        let (_, file_id) = populate_session(&conn, "/docs");

        // First page: offset=0, limit=3 -> returns 3 of 5 total.
        let (chunks, total) = browse_chunks(&conn, file_id, None, 0, 3).expect("browse p1");
        assert_eq!(total, 5);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[2].chunk_index, 2);

        // Second page: offset=3, limit=3 -> returns 2 remaining.
        let (chunks, total) = browse_chunks(&conn, file_id, None, 3, 3).expect("browse p2");
        assert_eq!(total, 5);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_index, 3);

        // Beyond total: offset=10 -> returns 0.
        let (chunks, total) = browse_chunks(&conn, file_id, None, 10, 3).expect("browse beyond");
        assert_eq!(total, 5);
        assert!(chunks.is_empty());
    }

    /// T-AGG-007: browse_chunks with page_filter narrows results to chunks
    /// spanning the specified page.
    #[test]
    fn t_agg_007_browse_chunks_page_filter() {
        let conn = setup_db();
        let (_, file_id) = populate_session(&conn, "/docs");

        // Page 1 should have chunks at indices 0 and 1 (page_start = 1).
        let (chunks, total) = browse_chunks(&conn, file_id, Some(1), 0, 100).expect("browse p1");
        assert!(total > 0);
        for c in &chunks {
            assert!(c.page_start <= 1 && c.page_end >= 1);
        }
    }

    /// T-AGG-008: find_files_by_path_pattern finds files across sessions.
    #[test]
    fn t_agg_008_find_files_by_pattern() {
        let conn = setup_db();
        populate_session(&conn, "/docs");

        let files = find_files_by_path_pattern(&conn, "%paper%").expect("find");
        assert_eq!(files.len(), 1);
        assert!(files[0].file_path.contains("paper"));

        // Non-matching pattern.
        let files = find_files_by_path_pattern(&conn, "%nonexistent%").expect("find empty");
        assert!(files.is_empty());
    }

    /// T-AGG-009: session_quality_data returns per-file quality rows.
    #[test]
    fn t_agg_009_session_quality_data() {
        let conn = setup_db();
        let (session_id, _) = populate_session(&conn, "/docs");

        let rows = session_quality_data(&conn, session_id).expect("quality");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].page_count, 3);
        assert_eq!(rows[0].pdf_page_count, Some(3));
        assert_eq!(rows[0].native_pages, 3);
        assert_eq!(rows[0].ocr_pages, 0);
        assert_eq!(rows[0].empty_pages, 1);
        assert!(rows[0].total_bytes > 0);
    }

    /// T-AGG-010: page_quality_details returns empty and OCR pages.
    #[test]
    fn t_agg_010_page_quality_details_basic() {
        let conn = setup_db();
        let config = test_config("/quality_test");
        let session_id = create_session(&conn, &config, "0.1.0").expect("session");

        let file_id = insert_file(
            &conn,
            session_id,
            "/quality_test/test.pdf",
            "hash123",
            1000,
            5000,
            5,
            Some(5),
        )
        .expect("insert file");

        // Pages: 1=native+content, 2=empty, 3=OCR+content, 4=empty, 5=native+content.
        let content = "test content";
        let pages = vec![
            (1, content, "pdf-extract"),
            (2, "", "pdf-extract"),
            (3, content, "ocr"),
            (4, "", "pdf-extract"),
            (5, content, "pdf-extract"),
        ];
        bulk_insert_pages(&conn, file_id, &pages).expect("insert pages");

        let details = page_quality_details(&conn, session_id).expect("details");

        // Should return 3 rows: page 2 (empty), page 3 (OCR), page 4 (empty).
        assert_eq!(details.len(), 3, "3 pages have quality issues");

        // Page 2: empty.
        assert_eq!(details[0].page_number, 2);
        assert_eq!(details[0].byte_count, 0);

        // Page 3: OCR-extracted with content.
        assert_eq!(details[1].page_number, 3);
        assert_eq!(details[1].backend, "ocr");
        assert!(details[1].byte_count > 0);

        // Page 4: empty.
        assert_eq!(details[2].page_number, 4);
        assert_eq!(details[2].byte_count, 0);
    }

    /// T-AGG-011: page_quality_details returns empty Vec for a session
    /// with only clean pages (no empty or OCR pages).
    #[test]
    fn t_agg_011_page_quality_details_clean_session() {
        let conn = setup_db();
        let config = test_config("/clean_test");
        let session_id = create_session(&conn, &config, "0.1.0").expect("session");

        let file_id = insert_file(
            &conn,
            session_id,
            "/clean_test/clean.pdf",
            "cleanhash",
            500,
            3000,
            3,
            Some(3),
        )
        .expect("insert file");

        let content = "full content here";
        let pages = vec![
            (1, content, "pdf-extract"),
            (2, content, "pdf-extract"),
            (3, content, "pdfium"),
        ];
        bulk_insert_pages(&conn, file_id, &pages).expect("insert pages");

        let details = page_quality_details(&conn, session_id).expect("details");
        assert!(
            details.is_empty(),
            "clean session must have no quality detail rows"
        );
    }
}
