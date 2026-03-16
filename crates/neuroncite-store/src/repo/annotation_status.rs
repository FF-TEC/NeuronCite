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

// Repository functions for the annotation_quote_status table.
//
// Tracks per-quote progress during annotation pipeline execution. Each row
// corresponds to one quote from the annotation input CSV/JSON data and records
// whether the quote was located in a PDF (matched), not found, or encountered
// an error during processing.
//
// The lifecycle of a quote status row:
//   1. insert_pending_quotes() creates rows with status "pending" at job start.
//   2. update_quote_status() transitions each row to "matched", "not_found",
//      or "error" as the pipeline processes each quote.
//   3. list_quote_statuses() and count_quote_statuses_by_status() provide
//      read access for the MCP handler that serves neuroncite_annotate_status.

use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::StoreError;

/// A single row from the annotation_quote_status table. Represents the
/// processing status of one quote from the annotation input data.
#[derive(Debug, Clone)]
pub struct AnnotationQuoteStatusRow {
    /// Auto-incremented primary key.
    pub id: i64,
    /// Job ID referencing the annotation job in the job table.
    pub job_id: String,
    /// Title of the work being annotated (from the input data).
    pub title: String,
    /// Author of the work being annotated (from the input data).
    pub author: String,
    /// Truncated excerpt of the quote text (first 100 chars) for display.
    pub quote_excerpt: String,
    /// Processing status: "pending", "matched", "not_found", or "error".
    pub status: String,
    /// Text location method that successfully found the quote (e.g., "exact",
    /// "normalized", "fuzzy", "ocr"). None when the quote has not been
    /// processed or was not found.
    pub match_method: Option<String>,
    /// 1-indexed page number where the quote was located. None when the
    /// quote has not been processed or was not found.
    pub page: Option<i64>,
    /// Filename of the PDF that was matched to this quote. None when no
    /// PDF was matched.
    pub pdf_filename: Option<String>,
    /// Unix timestamp (seconds) of the last status update.
    pub updated_at: i64,
}

/// Inserts pending quote status rows for all quotes in an annotation job.
/// Called at the start of the annotation pipeline before processing begins.
///
/// Each entry in `quotes` is a tuple of (title, author, quote_excerpt).
/// All rows are inserted with status "pending" and the current timestamp.
///
/// All inserts are wrapped in a single transaction so a crash during the
/// batch leaves the job with either all rows inserted or none, preventing
/// a partial-insert state that would cause the pipeline to skip quotes.
pub fn insert_pending_quotes(
    conn: &Connection,
    job_id: &str,
    quotes: &[(&str, &str, &str)],
) -> Result<usize, StoreError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let tx = conn.unchecked_transaction()?;

    let mut stmt = tx.prepare_cached(
        "INSERT INTO annotation_quote_status (job_id, title, author, quote_excerpt, status, updated_at)
         VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
    )?;

    let mut inserted = 0;
    for (title, author, excerpt) in quotes {
        stmt.execute(rusqlite::params![job_id, title, author, excerpt, now])?;
        inserted += 1;
    }

    drop(stmt);
    tx.commit()?;
    Ok(inserted)
}

/// Updates the status of a specific quote in an annotation job. Called by the
/// pipeline callback as each quote is processed. Matches on the combination
/// of (job_id, title, author, quote_excerpt) to find the target row.
///
/// Updates the status, match_method, page, pdf_filename, and updated_at fields.
/// Returns the number of rows affected (0 if no matching row was found).
///
/// The parameter count exceeds the default clippy threshold because each
/// parameter maps 1:1 to a SQL column in the WHERE/SET clauses. Wrapping
/// them in a struct would add indirection without improving clarity.
#[allow(clippy::too_many_arguments)]
pub fn update_quote_status(
    conn: &Connection,
    job_id: &str,
    title: &str,
    author: &str,
    quote_excerpt: &str,
    status: &str,
    match_method: Option<&str>,
    page: Option<i64>,
    pdf_filename: Option<&str>,
) -> Result<usize, StoreError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let affected = conn.execute(
        "UPDATE annotation_quote_status
         SET status = ?5, match_method = ?6, page = ?7, pdf_filename = ?8, updated_at = ?9
         WHERE job_id = ?1 AND title = ?2 AND author = ?3 AND quote_excerpt = ?4",
        rusqlite::params![
            job_id,
            title,
            author,
            quote_excerpt,
            status,
            match_method,
            page,
            pdf_filename,
            now
        ],
    )?;

    Ok(affected)
}

/// Returns quote status rows for a job with pagination support. Rows are
/// ordered by id (insertion order, which matches the input data order).
pub fn list_quote_statuses(
    conn: &Connection,
    job_id: &str,
    limit: i64,
    offset: i64,
) -> Result<Vec<AnnotationQuoteStatusRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, job_id, title, author, quote_excerpt, status,
                match_method, page, pdf_filename, updated_at
         FROM annotation_quote_status
         WHERE job_id = ?1
         ORDER BY id ASC
         LIMIT ?2 OFFSET ?3",
    )?;

    let rows = stmt
        .query_map(rusqlite::params![job_id, limit, offset], |row| {
            Ok(AnnotationQuoteStatusRow {
                id: row.get(0)?,
                job_id: row.get(1)?,
                title: row.get(2)?,
                author: row.get(3)?,
                quote_excerpt: row.get(4)?,
                status: row.get(5)?,
                match_method: row.get(6)?,
                page: row.get(7)?,
                pdf_filename: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(rows)
}

/// Returns aggregate counts of quote statuses for a job. The returned map
/// keys are status strings ("pending", "matched", "not_found", "error") and
/// values are the count of rows with that status.
pub fn count_quote_statuses_by_status(
    conn: &Connection,
    job_id: &str,
) -> Result<HashMap<String, i64>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT status, COUNT(*) FROM annotation_quote_status
         WHERE job_id = ?1
         GROUP BY status",
    )?;

    let mut counts = HashMap::new();
    let rows = stmt.query_map(rusqlite::params![job_id], |row| {
        let status: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((status, count))
    })?;

    for row in rows {
        let (status, count) = row?;
        counts.insert(status, count);
    }

    Ok(counts)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates an in-memory database with the full schema and a test job.
    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        crate::schema::migrate(&conn).expect("migration failed");

        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('test-ann-job', 'annotate', 'running', 0, 0, 1700000000)",
            [],
        )
        .expect("insert test job");

        conn
    }

    /// T-ANN-STATUS-001: insert_pending_quotes inserts 3 rows with status "pending".
    #[test]
    fn t_ann_status_001_insert_pending() {
        let conn = setup_db();
        let quotes = vec![
            ("Title A", "Author A", "first quote excerpt..."),
            ("Title B", "Author B", "second quote excerpt..."),
            ("Title C", "Author C", "third quote excerpt..."),
        ];

        let inserted =
            insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert pending quotes");
        assert_eq!(inserted, 3);

        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list quotes");
        assert_eq!(rows.len(), 3);

        for row in &rows {
            assert_eq!(row.status, "pending");
            assert!(row.match_method.is_none());
            assert!(row.page.is_none());
            assert!(row.pdf_filename.is_none());
        }
    }

    /// T-ANN-STATUS-002: update_quote_status transitions a row to "matched"
    /// and sets match_method, page, and pdf_filename.
    #[test]
    fn t_ann_status_002_update_to_matched() {
        let conn = setup_db();
        let quotes = vec![("Title A", "Author A", "some text...")];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        let affected = update_quote_status(
            &conn,
            "test-ann-job",
            "Title A",
            "Author A",
            "some text...",
            "matched",
            Some("exact"),
            Some(5),
            Some("Title_A.pdf"),
        )
        .expect("update status");
        assert_eq!(affected, 1);

        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "matched");
        assert_eq!(rows[0].match_method.as_deref(), Some("exact"));
        assert_eq!(rows[0].page, Some(5));
        assert_eq!(rows[0].pdf_filename.as_deref(), Some("Title_A.pdf"));
    }

    /// T-ANN-STATUS-003: CASCADE delete removes quote status rows when the
    /// parent job is deleted.
    #[test]
    fn t_ann_status_003_cascade_delete() {
        let conn = setup_db();
        let quotes = vec![
            ("Title A", "Author A", "text A..."),
            ("Title B", "Author B", "text B..."),
        ];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        // Verify rows exist.
        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list");
        assert_eq!(rows.len(), 2);

        // Delete the parent job.
        conn.execute("DELETE FROM job WHERE id = 'test-ann-job'", [])
            .expect("delete job");

        // Rows must be gone.
        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list after delete");
        assert_eq!(
            rows.len(),
            0,
            "CASCADE delete must remove all quote status rows"
        );
    }

    /// T-ANN-STATUS-004: Pagination with limit and offset. 10 rows inserted,
    /// limit=3, offset=3 returns rows 4-6.
    #[test]
    fn t_ann_status_004_pagination() {
        let conn = setup_db();
        let mut quotes = Vec::new();
        for i in 0..10 {
            quotes.push((
                format!("Title {i}"),
                format!("Author {i}"),
                format!("quote {i}..."),
            ));
        }
        let quote_refs: Vec<(&str, &str, &str)> = quotes
            .iter()
            .map(|(t, a, q)| (t.as_str(), a.as_str(), q.as_str()))
            .collect();
        insert_pending_quotes(&conn, "test-ann-job", &quote_refs).expect("insert");

        let page = list_quote_statuses(&conn, "test-ann-job", 3, 3).expect("list with pagination");
        assert_eq!(page.len(), 3, "limit=3 must return 3 rows");

        // Rows are ordered by id (insertion order), so offset=3 skips rows 0-2.
        assert_eq!(page[0].title, "Title 3");
        assert_eq!(page[1].title, "Title 4");
        assert_eq!(page[2].title, "Title 5");
    }

    /// T-ANN-STATUS-005: count_quote_statuses_by_status returns correct counts
    /// for mixed status values.
    #[test]
    fn t_ann_status_005_count_by_status() {
        let conn = setup_db();
        let quotes = vec![
            ("Title A", "Author A", "text A..."),
            ("Title B", "Author B", "text B..."),
            ("Title C", "Author C", "text C..."),
        ];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        // Update one to "matched" and one to "not_found".
        update_quote_status(
            &conn,
            "test-ann-job",
            "Title A",
            "Author A",
            "text A...",
            "matched",
            Some("exact"),
            Some(1),
            Some("a.pdf"),
        )
        .expect("update A");

        update_quote_status(
            &conn,
            "test-ann-job",
            "Title B",
            "Author B",
            "text B...",
            "not_found",
            None,
            None,
            None,
        )
        .expect("update B");

        let counts =
            count_quote_statuses_by_status(&conn, "test-ann-job").expect("count by status");

        assert_eq!(counts.get("pending").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("matched").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("not_found").copied().unwrap_or(0), 1);
    }

    /// T-ANN-STATUS-006: update_quote_status returns 0 when no matching row
    /// exists (wrong job_id or quote combination).
    #[test]
    fn t_ann_status_006_update_nonexistent_returns_zero() {
        let conn = setup_db();
        let quotes = vec![("Title A", "Author A", "text A...")];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        let affected = update_quote_status(
            &conn,
            "test-ann-job",
            "Wrong Title",
            "Wrong Author",
            "wrong text...",
            "matched",
            Some("exact"),
            Some(1),
            Some("x.pdf"),
        )
        .expect("update nonexistent");
        assert_eq!(
            affected, 0,
            "updating a non-matching quote must affect 0 rows"
        );
    }

    // -----------------------------------------------------------------------
    // DEF-003 regression tests: Verifying the insert + update round-trip
    // that the executor relies on after the fix. The executor inserts
    // pending quotes before the pipeline runs, then updates statuses
    // from the AnnotationReport after the pipeline completes.
    // -----------------------------------------------------------------------

    /// T-ANN-STATUS-007: Full round-trip simulating the executor's lifecycle.
    /// Insert pending quotes, update some to "matched" and some to "not_found",
    /// verify counts reflect the final state.
    #[test]
    fn t_ann_status_007_full_round_trip_insert_then_update() {
        let conn = setup_db();
        let quotes = vec![
            ("Risk Paper", "Bollerslev", "GARCH model estimation..."),
            ("Risk Paper", "Bollerslev", "conditional heterosked..."),
            ("Missing Paper", "Unknown", "this quote is nowhere..."),
        ];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        // Verify all 3 rows are pending.
        let counts = count_quote_statuses_by_status(&conn, "test-ann-job").expect("count");
        assert_eq!(counts.get("pending").copied().unwrap_or(0), 3);

        // Update first quote to "matched" (simulating pipeline success).
        update_quote_status(
            &conn,
            "test-ann-job",
            "Risk Paper",
            "Bollerslev",
            "GARCH model estimation...",
            "matched",
            Some("normalized"),
            Some(7),
            Some("Bollerslev_1986.pdf"),
        )
        .expect("update first");

        // Update second quote to "matched" with a different method.
        update_quote_status(
            &conn,
            "test-ann-job",
            "Risk Paper",
            "Bollerslev",
            "conditional heterosked...",
            "matched",
            Some("fuzzy"),
            Some(12),
            Some("Bollerslev_1986.pdf"),
        )
        .expect("update second");

        // Update third quote to "not_found" (simulating unmatched input).
        update_quote_status(
            &conn,
            "test-ann-job",
            "Missing Paper",
            "Unknown",
            "this quote is nowhere...",
            "not_found",
            None,
            None,
            None,
        )
        .expect("update third");

        // Verify final counts.
        let counts =
            count_quote_statuses_by_status(&conn, "test-ann-job").expect("count after update");
        assert_eq!(
            counts.get("pending").copied().unwrap_or(0),
            0,
            "no quotes must remain pending after full processing"
        );
        assert_eq!(counts.get("matched").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("not_found").copied().unwrap_or(0), 1);

        // Verify individual row details.
        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list");
        assert_eq!(rows.len(), 3);

        assert_eq!(rows[0].status, "matched");
        assert_eq!(rows[0].match_method.as_deref(), Some("normalized"));
        assert_eq!(rows[0].page, Some(7));
        assert_eq!(rows[0].pdf_filename.as_deref(), Some("Bollerslev_1986.pdf"));

        assert_eq!(rows[1].status, "matched");
        assert_eq!(rows[1].match_method.as_deref(), Some("fuzzy"));

        assert_eq!(rows[2].status, "not_found");
        assert!(rows[2].match_method.is_none());
        assert!(rows[2].page.is_none());
    }

    /// T-ANN-STATUS-008: The excerpt_to_meta mapping approach used by the
    /// executor correctly handles multiple quotes from the same work.
    /// Two quotes with the same (title, author) but different excerpts
    /// are updated independently.
    #[test]
    fn t_ann_status_008_same_title_author_different_excerpts() {
        let conn = setup_db();
        let quotes = vec![
            ("Paper X", "Author Y", "first excerpt from Paper X..."),
            ("Paper X", "Author Y", "second excerpt from Paper X..."),
        ];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        // Update only the first quote.
        let affected = update_quote_status(
            &conn,
            "test-ann-job",
            "Paper X",
            "Author Y",
            "first excerpt from Paper X...",
            "matched",
            Some("exact"),
            Some(3),
            Some("paper_x.pdf"),
        )
        .expect("update first");
        assert_eq!(affected, 1, "only the first row must be affected");

        // Second quote must still be pending.
        let rows = list_quote_statuses(&conn, "test-ann-job", 100, 0).expect("list");
        assert_eq!(rows[0].status, "matched");
        assert_eq!(rows[1].status, "pending");
    }

    /// T-ANN-STATUS-009: Status rows survive across separate connection
    /// instances (persisted to SQLite, not held in memory). This validates
    /// that the data is available after the executor's blocking thread
    /// completes and the MCP handler reads from a different connection.
    #[test]
    fn t_ann_status_009_data_persists_across_connections() {
        let tmp_db = tempfile::NamedTempFile::new().expect("create temp file");
        let db_path = tmp_db.path().to_str().expect("path to str");

        // Connection 1: insert and update (simulates the executor thread).
        {
            let conn = rusqlite::Connection::open(db_path).expect("open conn 1");
            conn.execute_batch("PRAGMA foreign_keys = ON;")
                .expect("pragma");
            crate::schema::migrate(&conn).expect("migrate");

            conn.execute(
                "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
                 VALUES ('persist-job', 'annotate', 'completed', 2, 2, 1700000000)",
                [],
            )
            .expect("insert job");

            let quotes = vec![
                ("Title A", "Author A", "excerpt A..."),
                ("Title B", "Author B", "excerpt B..."),
            ];
            insert_pending_quotes(&conn, "persist-job", &quotes).expect("insert quotes");

            update_quote_status(
                &conn,
                "persist-job",
                "Title A",
                "Author A",
                "excerpt A...",
                "matched",
                Some("exact"),
                Some(1),
                Some("title_a.pdf"),
            )
            .expect("update A");
        }

        // Connection 2: read (simulates the MCP handler on a different thread).
        {
            let conn = rusqlite::Connection::open(db_path).expect("open conn 2");
            let rows = list_quote_statuses(&conn, "persist-job", 100, 0).expect("list");
            assert_eq!(
                rows.len(),
                2,
                "both rows must be readable from a new connection"
            );
            assert_eq!(rows[0].status, "matched");
            assert_eq!(rows[1].status, "pending");

            let counts = count_quote_statuses_by_status(&conn, "persist-job").expect("count");
            assert_eq!(counts.get("matched").copied().unwrap_or(0), 1);
            assert_eq!(counts.get("pending").copied().unwrap_or(0), 1);
        }
    }

    /// T-ANN-STATUS-010: Updating all quotes to terminal statuses produces
    /// zero "pending" count. The executor must leave no rows in "pending"
    /// state after the pipeline completes.
    #[test]
    fn t_ann_status_010_no_pending_after_full_update() {
        let conn = setup_db();
        let quotes = vec![
            ("A", "Auth A", "quote A..."),
            ("B", "Auth B", "quote B..."),
            ("C", "Auth C", "quote C..."),
        ];
        insert_pending_quotes(&conn, "test-ann-job", &quotes).expect("insert");

        // Update all quotes to various terminal statuses.
        update_quote_status(
            &conn,
            "test-ann-job",
            "A",
            "Auth A",
            "quote A...",
            "matched",
            Some("exact"),
            Some(1),
            Some("a.pdf"),
        )
        .expect("update A");
        update_quote_status(
            &conn,
            "test-ann-job",
            "B",
            "Auth B",
            "quote B...",
            "not_found",
            None,
            None,
            None,
        )
        .expect("update B");
        update_quote_status(
            &conn,
            "test-ann-job",
            "C",
            "Auth C",
            "quote C...",
            "error",
            None,
            None,
            None,
        )
        .expect("update C");

        let counts = count_quote_statuses_by_status(&conn, "test-ann-job").expect("count");
        assert_eq!(
            counts.get("pending").copied().unwrap_or(0),
            0,
            "no rows must remain in 'pending' status after full processing"
        );
        assert_eq!(counts.get("matched").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("not_found").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("error").copied().unwrap_or(0), 1);
    }
}
