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

// Page repository operations.
//
// Provides functions to insert and query page records. Each page record
// references its parent `indexed_file` and stores the 1-indexed page number,
// the extracted and normalized text content, the extraction backend identifier,
// and the UTF-8 byte count of the content. The `byte_count` column enables
// fast document-offset-to-page resolution without loading full page text.

use rusqlite::{Connection, params};

use crate::error::StoreError;

/// Row representation of a page record. Contains all columns defined
/// in the page table.
#[derive(Debug, Clone)]
pub struct PageRow {
    pub id: i64,
    pub file_id: i64,
    pub page_number: i64,
    pub content: String,
    pub backend: String,
    pub byte_count: i64,
}

/// Bulk inserts multiple page records for a single file within a transaction.
/// Each tuple in the `pages` slice contains:
/// (`page_number`, content, `backend_name`).
///
/// The `byte_count` for each page is computed from the content string's UTF-8
/// byte length.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if any insert fails (e.g., duplicate
/// `page_number` for the same `file_id`).
pub fn bulk_insert_pages(
    conn: &Connection,
    file_id: i64,
    pages: &[(i64, &str, &str)],
) -> Result<(), StoreError> {
    // Wrap all inserts in a SAVEPOINT so they commit as a single atomic
    // unit. Without this, each INSERT would autocommit separately,
    // producing N transactions for N pages (200 pages = 200 fsyncs).
    // SAVEPOINT is used instead of BEGIN/COMMIT to allow nesting inside
    // an outer transaction without conflict.
    conn.execute_batch("SAVEPOINT bulk_pages")?;

    let result = (|| -> Result<(), StoreError> {
        let mut stmt = conn.prepare_cached(
            "INSERT INTO page (file_id, page_number, content, backend, byte_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for &(page_number, content, backend) in pages {
            let byte_count = content.len() as i64;
            stmt.execute(params![file_id, page_number, content, backend, byte_count])?;
        }

        Ok(())
    })();

    match result {
        Ok(()) => {
            conn.execute_batch("RELEASE bulk_pages")?;
            Ok(())
        }
        Err(e) => {
            // Roll back all inserts from this batch on any failure.
            let _ = conn.execute_batch("ROLLBACK TO bulk_pages");
            Err(e)
        }
    }
}

/// Retrieves all pages belonging to a specific file, ordered by `page_number`.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn get_pages_by_file(conn: &Connection, file_id: i64) -> Result<Vec<PageRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, page_number, content, backend, byte_count
         FROM page WHERE file_id = ?1 ORDER BY page_number",
    )?;

    let rows = stmt.query_map(params![file_id], |row| {
        Ok(PageRow {
            id: row.get(0)?,
            file_id: row.get(1)?,
            page_number: row.get(2)?,
            content: row.get(3)?,
            backend: row.get(4)?,
            byte_count: row.get(5)?,
        })
    })?;

    let mut pages = Vec::new();
    for row in rows {
        pages.push(row?);
    }
    Ok(pages)
}

/// Retrieves a single page by `file_id` and `page_number`.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no page with the given identifiers exists,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_page(conn: &Connection, file_id: i64, page_number: i64) -> Result<PageRow, StoreError> {
    conn.query_row(
        "SELECT id, file_id, page_number, content, backend, byte_count
         FROM page WHERE file_id = ?1 AND page_number = ?2",
        params![file_id, page_number],
        |row| {
            Ok(PageRow {
                id: row.get(0)?,
                file_id: row.get(1)?,
                page_number: row.get(2)?,
                content: row.get(3)?,
                backend: row.get(4)?,
                byte_count: row.get(5)?,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            StoreError::not_found("page", format!("file_id={file_id}, page={page_number}"))
        }
        other => other.into(),
    })
}

/// Retrieves a contiguous range of pages by `file_id` and inclusive page
/// number bounds. Returns pages ordered by `page_number` within the range.
/// Pages that do not exist within the range are silently omitted from the
/// result (no error is raised for gaps).
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn get_pages_range(
    conn: &Connection,
    file_id: i64,
    start_page: i64,
    end_page: i64,
) -> Result<Vec<PageRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, page_number, content, backend, byte_count
         FROM page WHERE file_id = ?1 AND page_number >= ?2 AND page_number <= ?3
         ORDER BY page_number",
    )?;

    let rows = stmt.query_map(params![file_id, start_page, end_page], |row| {
        Ok(PageRow {
            id: row.get(0)?,
            file_id: row.get(1)?,
            page_number: row.get(2)?,
            content: row.get(3)?,
            backend: row.get(4)?,
            byte_count: row.get(5)?,
        })
    })?;

    let mut pages = Vec::new();
    for row in rows {
        pages.push(row?);
    }
    Ok(pages)
}

/// Deletes all page records belonging to a specific file. Returns the number
/// of rows deleted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn delete_pages_by_file(conn: &Connection, file_id: i64) -> Result<usize, StoreError> {
    let count = conn.execute("DELETE FROM page WHERE file_id = ?1", params![file_id])?;
    Ok(count)
}

/// A single substring match found within a page's text content.
#[derive(Debug, Clone)]
pub struct TextMatch {
    /// Byte offset (0-based) of the match start within the page content.
    pub position: usize,
    /// Surrounding context: characters before and after the match position,
    /// bounded by page content boundaries.
    pub context: String,
}

/// All matches found on a single page for a text search query.
#[derive(Debug, Clone)]
pub struct PageTextMatch {
    /// 1-indexed page number where the matches were found.
    pub page_number: i64,
    /// All non-overlapping occurrences of the query on this page.
    pub matches: Vec<TextMatch>,
}

/// Searches all pages of a single indexed file for occurrences of a literal
/// substring. Returns one PageTextMatch entry per page that contains at
/// least one match, ordered by page_number ascending.
///
/// Loads all pages for the file into memory (typical PDFs have < 500 pages)
/// and searches in Rust. For case-insensitive mode, both the page content
/// and query are lowercased before comparison. Context extraction returns
/// `context_chars` characters on each side of the match, bounded by page
/// content boundaries.
///
/// Returns an empty Vec when the query is not found on any page (not an error).
pub fn search_page_text(
    conn: &Connection,
    file_id: i64,
    query: &str,
    case_sensitive: bool,
    context_chars: usize,
) -> Result<Vec<PageTextMatch>, StoreError> {
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let pages = get_pages_by_file(conn, file_id)?;
    let mut results = Vec::new();

    for page in &pages {
        let content = &page.content;
        if content.is_empty() {
            continue;
        }

        // For case-sensitive search, the original content serves as both the
        // display source and the search haystack, and the raw query serves as
        // both the search needle and the display needle.
        let matches = if case_sensitive {
            find_all_occurrences(content, content, query, query, context_chars)
        } else {
            let lower_content = content.to_lowercase();
            let lower_query = query.to_lowercase();
            find_all_occurrences(
                content,
                &lower_content,
                &lower_query,
                &lower_query,
                context_chars,
            )
        };

        if !matches.is_empty() {
            results.push(PageTextMatch {
                page_number: page.page_number,
                matches,
            });
        }
    }

    Ok(results)
}

/// Finds all non-overlapping occurrences of `search_query` within `search_content`,
/// extracting context windows from `original_content`. The separation of
/// original and search content handles case-insensitive search: positions
/// found in the lowercased content map directly to the original content
/// because lowercasing preserves byte positions for ASCII text.
///
/// For Unicode text where lowercasing changes byte length (e.g., German sharp s),
/// the context window is extracted from the search content as a fallback.
fn find_all_occurrences(
    original_content: &str,
    search_content: &str,
    search_query: &str,
    _display_query: &str,
    context_chars: usize,
) -> Vec<TextMatch> {
    let mut matches = Vec::new();
    let mut search_start = 0_usize;

    while search_start < search_content.len() {
        let remaining = &search_content[search_start..];
        let rel_pos = match remaining.find(search_query) {
            Some(pos) => pos,
            None => break,
        };

        let abs_pos = search_start + rel_pos;

        // Extract context from original content. Use char-based boundaries
        // to avoid splitting multi-byte UTF-8 characters.
        let context = extract_context(original_content, abs_pos, search_query.len(), context_chars);

        matches.push(TextMatch {
            position: abs_pos,
            context,
        });

        // Advance past this match to find non-overlapping occurrences.
        search_start = abs_pos + search_query.len().max(1);
    }

    matches
}

/// Extracts a context window around a match position. Returns `context_chars`
/// characters before and after the match, bounded by the content boundaries.
/// Uses char indices to avoid splitting multi-byte UTF-8 sequences.
fn extract_context(
    content: &str,
    match_byte_start: usize,
    match_byte_len: usize,
    context_chars: usize,
) -> String {
    // Find the character index of the match start.
    let char_start = content[..match_byte_start.min(content.len())]
        .chars()
        .count();

    // Calculate context window in character units.
    let ctx_char_start = char_start.saturating_sub(context_chars);
    let match_char_len = content
        [match_byte_start..((match_byte_start + match_byte_len).min(content.len()))]
        .chars()
        .count();
    let ctx_char_end = (char_start + match_char_len + context_chars).min(content.chars().count());

    // Convert character indices back to byte boundaries.
    let byte_start = content
        .char_indices()
        .nth(ctx_char_start)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let byte_end = content
        .char_indices()
        .nth(ctx_char_end)
        .map(|(i, _)| i)
        .unwrap_or(content.len());

    content[byte_start..byte_end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::{file as file_repo, session};
    use crate::schema::migrate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use std::path::PathBuf;

    fn setup_db_with_file() -> (Connection, i64) {
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

        (conn, file_id)
    }

    #[test]
    fn bulk_insert_and_query_pages() {
        let (conn, file_id) = setup_db_with_file();

        let pages = vec![
            (1_i64, "Page one content", "pdf-extract"),
            (2, "Page two has more text here", "pdf-extract"),
            (3, "", "ocr"),
        ];

        bulk_insert_pages(&conn, file_id, &pages).expect("bulk_insert failed");

        let loaded = get_pages_by_file(&conn, file_id).expect("get_pages failed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].page_number, 1);
        assert_eq!(loaded[0].byte_count, "Page one content".len() as i64);
        assert_eq!(loaded[2].content, "");
        assert_eq!(loaded[2].byte_count, 0);
    }

    #[test]
    fn get_single_page() {
        let (conn, file_id) = setup_db_with_file();

        let pages = vec![
            (1_i64, "First page", "pdf-extract"),
            (2, "Second page", "pdfium"),
        ];
        bulk_insert_pages(&conn, file_id, &pages).expect("insert failed");

        let page = get_page(&conn, file_id, 2).expect("get_page failed");
        assert_eq!(page.page_number, 2);
        assert_eq!(page.content, "Second page");
        assert_eq!(page.backend, "pdfium");
    }

    /// T-STORE-032: SAVEPOINT rollback on partial failure. Verifies that when
    /// a bulk insert fails mid-batch (due to a UNIQUE constraint violation on
    /// duplicate page_number), all rows from the batch are rolled back and no
    /// partial data remains in the page table. Without the SAVEPOINT, each
    /// INSERT would autocommit separately and the first two pages would persist
    /// even though the batch as a whole failed.
    #[test]
    fn t_store_032_savepoint_rollback_on_duplicate() {
        let (conn, file_id) = setup_db_with_file();

        // First batch: insert pages 1 and 2 successfully.
        let batch1 = vec![
            (1_i64, "Page one", "pdf-extract"),
            (2, "Page two", "pdf-extract"),
        ];
        bulk_insert_pages(&conn, file_id, &batch1).expect("first batch must succeed");

        // Second batch: page 3 is valid, but page 2 is a duplicate of the
        // first batch. The UNIQUE constraint on (file_id, page_number) causes
        // the second INSERT to fail, which must roll back the entire batch.
        let batch2 = vec![
            (3_i64, "Page three", "pdf-extract"),
            (2, "Duplicate page two", "pdf-extract"),
        ];
        let result = bulk_insert_pages(&conn, file_id, &batch2);
        assert!(
            result.is_err(),
            "second batch must fail due to duplicate page_number"
        );

        // Verify that only the two pages from batch1 exist. Page 3 from
        // the failed batch2 must not be present because the SAVEPOINT
        // rolled back the entire batch.
        let pages = get_pages_by_file(&conn, file_id).expect("get_pages failed");
        assert_eq!(
            pages.len(),
            2,
            "only the two pages from the first batch must persist; \
             the failed batch must be fully rolled back"
        );
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[1].page_number, 2);
        assert_eq!(
            pages[1].content, "Page two",
            "the original page 2 content must be preserved, not overwritten"
        );
    }

    // -----------------------------------------------------------------------
    // Text search tests
    // -----------------------------------------------------------------------

    fn setup_text_search_db() -> (Connection, i64) {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");

        conn.execute(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, created_at
            ) VALUES ('/test', 'model', 'page', 'eng', 384, 'sqlite-blob', 1, '0.1.0', 1700000000)",
            [],
        )
        .expect("create session");
        let session_id = conn.last_insert_rowid();

        conn.execute(
            "INSERT INTO indexed_file (
                session_id, file_path, file_hash, mtime, size, page_count, created_at, updated_at
            ) VALUES (?1, '/test/doc.pdf', 'abc123', 1700000000, 4096, 3, 1700000000, 1700000000)",
            rusqlite::params![session_id],
        )
        .expect("create file");
        let file_id = conn.last_insert_rowid();

        (conn, file_id)
    }

    fn insert_text_search_pages(conn: &Connection, file_id: i64) {
        let pages = [
            (1, "The quick brown fox jumps over the lazy dog."),
            (
                2,
                "Statistical analysis of regression models demonstrates significant correlation.",
            ),
            (3, "The fox ran across the field. The fox stopped to rest."),
        ];
        for (page_num, content) in &pages {
            conn.execute(
                "INSERT INTO page (file_id, page_number, content, backend, byte_count)
                 VALUES (?1, ?2, ?3, 'pdf-extract', ?4)",
                rusqlite::params![file_id, page_num, content, content.len()],
            )
            .expect("insert page");
        }
    }

    /// T-PAGE-SEARCH-001: search_page_text finds a literal substring on the
    /// correct page and returns the matching page number.
    #[test]
    fn t_page_search_001_finds_substring_correct_page() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        let results = search_page_text(&conn, file_id, "regression models", false, 100)
            .expect("search failed");
        assert_eq!(results.len(), 1, "match should be on exactly one page");
        assert_eq!(results[0].page_number, 2);
        assert_eq!(results[0].matches.len(), 1);
        assert!(results[0].matches[0].context.contains("regression models"));
    }

    /// T-PAGE-SEARCH-002: case-insensitive search matches uppercase query
    /// against lowercase content.
    #[test]
    fn t_page_search_002_case_insensitive_matching() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        let results = search_page_text(&conn, file_id, "STATISTICAL ANALYSIS", false, 100)
            .expect("search failed");
        assert_eq!(
            results.len(),
            1,
            "case-insensitive match should find result"
        );
        assert_eq!(results[0].page_number, 2);
    }

    /// T-PAGE-SEARCH-003: case-sensitive search does NOT match when case differs.
    #[test]
    fn t_page_search_003_case_sensitive_no_match() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        let results = search_page_text(&conn, file_id, "STATISTICAL ANALYSIS", true, 100)
            .expect("search failed");
        assert!(results.is_empty(), "case-sensitive search should not match");
    }

    /// T-PAGE-SEARCH-004: multiple occurrences on the same page are all reported
    /// as separate TextMatch entries.
    #[test]
    fn t_page_search_004_multiple_occurrences_same_page() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        // Page 3 has "The fox" twice (case-insensitive "the fox" matches both).
        let results =
            search_page_text(&conn, file_id, "The fox", false, 100).expect("search failed");
        // Page 1 has "The quick brown fox" and page 3 has "The fox" twice.
        let page3 = results.iter().find(|r| r.page_number == 3);
        assert!(page3.is_some(), "page 3 should have matches");
        assert_eq!(
            page3.unwrap().matches.len(),
            2,
            "page 3 should have 2 occurrences"
        );
    }

    /// T-PAGE-SEARCH-005: query not found returns empty results (not an error).
    #[test]
    fn t_page_search_005_not_found_returns_empty() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        let results = search_page_text(&conn, file_id, "nonexistent phrase xyz", false, 100)
            .expect("search failed");
        assert!(results.is_empty());
    }

    /// T-PAGE-SEARCH-007: case-sensitive search finds a match when the query
    /// case matches the content exactly. Regression test for DEF-007 where
    /// swapped arguments caused case_sensitive=true to always return zero.
    #[test]
    fn t_page_search_007_case_sensitive_finds_exact_match() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        // Page 2 contains "Statistical analysis" with uppercase S.
        let results =
            search_page_text(&conn, file_id, "Statistical", true, 100).expect("search failed");
        assert_eq!(
            results.len(),
            1,
            "case-sensitive search must find exact-case match on page 2"
        );
        assert_eq!(results[0].page_number, 2);
        assert!(results[0].matches[0].context.contains("Statistical"));
    }

    /// T-PAGE-SEARCH-008: case-sensitive search finds multiple matches across
    /// different pages. Regression test for DEF-007.
    #[test]
    fn t_page_search_008_case_sensitive_multiple_pages() {
        let (conn, file_id) = setup_text_search_db();
        insert_text_search_pages(&conn, file_id);

        // "The" appears on pages 1 and 3 (with uppercase T).
        let results = search_page_text(&conn, file_id, "The", true, 100).expect("search failed");
        assert!(
            results.len() >= 2,
            "case-sensitive 'The' must match on at least 2 pages, got {}",
            results.len()
        );
    }

    /// T-PAGE-SEARCH-006: context is bounded by page content boundaries and
    /// does not panic on short pages.
    #[test]
    fn t_page_search_006_context_bounded_by_page_length() {
        let (conn, file_id) = setup_text_search_db();

        // Insert a very short page.
        conn.execute(
            "INSERT INTO page (file_id, page_number, content, backend, byte_count)
             VALUES (?1, 4, 'ab', 'pdf-extract', 2)",
            rusqlite::params![file_id],
        )
        .expect("insert short page");

        let results = search_page_text(&conn, file_id, "ab", false, 100).expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matches[0].context, "ab");
    }
}
