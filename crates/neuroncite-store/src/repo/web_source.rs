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

// Web source repository operations.
//
// Provides CRUD functions for the web_source table, which stores web-specific
// metadata for HTML-sourced indexed files. Each web_source record is linked 1:1
// to an indexed_file record via file_id, storing the fetched URL, HTML head
// metadata (title, Open Graph tags, author, language), domain, HTTP response
// details, and optionally the raw HTML bytes for re-processing.

use rusqlite::{Connection, params};

use crate::error::StoreError;

/// Row representation of a `web_source` record. Contains all columns
/// defined in the `web_source` table.
#[derive(Debug, Clone)]
pub struct WebSourceRow {
    pub id: i64,
    /// Foreign key referencing the indexed_file record this metadata belongs to.
    pub file_id: i64,
    /// The URL that was fetched to obtain this HTML page.
    pub url: String,
    /// Canonical URL from `<link rel="canonical">`, if present in the HTML head.
    pub canonical_url: Option<String>,
    /// Page title from `<title>` tag.
    pub title: Option<String>,
    /// Content of `<meta name="description">`.
    pub meta_description: Option<String>,
    /// Language code from `<html lang="...">` or `<meta http-equiv="Content-Language">`.
    pub language: Option<String>,
    /// Open Graph image URL from `<meta property="og:image">`.
    pub og_image: Option<String>,
    /// Open Graph title from `<meta property="og:title">`.
    pub og_title: Option<String>,
    /// Open Graph description from `<meta property="og:description">`.
    pub og_description: Option<String>,
    /// Author from `<meta name="author">`.
    pub author: Option<String>,
    /// Published date from `<meta name="article:published_time">` or similar.
    pub published_date: Option<String>,
    /// Domain name extracted from the URL (e.g., "example.com").
    pub domain: String,
    /// Unix timestamp (seconds) when the page was fetched.
    pub fetch_timestamp: i64,
    /// HTTP status code of the response (e.g., 200, 301, 404).
    pub http_status: i64,
    /// Content-Type header value from the HTTP response.
    pub content_type: Option<String>,
    /// Raw HTML bytes, stored for re-processing without re-fetching. NULL when
    /// raw storage is disabled to save database space.
    pub raw_html: Option<Vec<u8>>,
}

/// Insert parameters for a web_source record. Uses borrowed references to
/// avoid unnecessary allocations at the call site.
pub struct WebSourceInsert<'a> {
    pub file_id: i64,
    pub url: &'a str,
    pub canonical_url: Option<&'a str>,
    pub title: Option<&'a str>,
    pub meta_description: Option<&'a str>,
    pub language: Option<&'a str>,
    pub og_image: Option<&'a str>,
    pub og_title: Option<&'a str>,
    pub og_description: Option<&'a str>,
    pub author: Option<&'a str>,
    pub published_date: Option<&'a str>,
    pub domain: &'a str,
    pub fetch_timestamp: i64,
    pub http_status: i64,
    pub content_type: Option<&'a str>,
    pub raw_html: Option<&'a [u8]>,
}

/// Inserts a web_source record and returns the auto-generated row ID.
/// The file_id must reference an existing indexed_file record with
/// source_type='html'.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the insert violates the UNIQUE(file_id)
/// constraint or if the foreign key reference to indexed_file is invalid.
pub fn insert_web_source(conn: &Connection, ws: &WebSourceInsert<'_>) -> Result<i64, StoreError> {
    conn.execute(
        "INSERT INTO web_source (
            file_id, url, canonical_url, title, meta_description, language,
            og_image, og_title, og_description, author, published_date,
            domain, fetch_timestamp, http_status, content_type, raw_html
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            ws.file_id,
            ws.url,
            ws.canonical_url,
            ws.title,
            ws.meta_description,
            ws.language,
            ws.og_image,
            ws.og_title,
            ws.og_description,
            ws.author,
            ws.published_date,
            ws.domain,
            ws.fetch_timestamp,
            ws.http_status,
            ws.content_type,
            ws.raw_html,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Retrieves the web_source record associated with the given indexed_file ID.
///
/// # Errors
///
/// Returns `StoreError::NotFound` if no web_source exists for this file_id,
/// or `StoreError::Sqlite` if the query fails.
pub fn get_web_source_by_file(conn: &Connection, file_id: i64) -> Result<WebSourceRow, StoreError> {
    conn.query_row(
        "SELECT id, file_id, url, canonical_url, title, meta_description, language,
                og_image, og_title, og_description, author, published_date,
                domain, fetch_timestamp, http_status, content_type, raw_html
         FROM web_source WHERE file_id = ?1",
        params![file_id],
        row_to_web_source,
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            StoreError::not_found("web_source", format!("file_id={file_id}"))
        }
        other => other.into(),
    })
}

/// Looks up a web_source record by URL. Returns None if no record exists
/// for this URL. Used by the fetch pipeline to check whether a URL has
/// already been cached before re-fetching.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn get_web_source_by_url(
    conn: &Connection,
    url: &str,
) -> Result<Option<WebSourceRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, url, canonical_url, title, meta_description, language,
                og_image, og_title, og_description, author, published_date,
                domain, fetch_timestamp, http_status, content_type, raw_html
         FROM web_source WHERE url = ?1",
    )?;

    let result = stmt.query_row(params![url], row_to_web_source);

    match result {
        Ok(row) => Ok(Some(row)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Lists all web_source records belonging to files within the given session.
/// Joins web_source with indexed_file to filter by session_id.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_web_sources(
    conn: &Connection,
    session_id: i64,
) -> Result<Vec<WebSourceRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT ws.id, ws.file_id, ws.url, ws.canonical_url, ws.title, ws.meta_description,
                ws.language, ws.og_image, ws.og_title, ws.og_description, ws.author,
                ws.published_date, ws.domain, ws.fetch_timestamp, ws.http_status,
                ws.content_type, ws.raw_html
         FROM web_source ws
         JOIN indexed_file f ON f.id = ws.file_id
         WHERE f.session_id = ?1
         ORDER BY ws.url",
    )?;

    let rows = stmt.query_map(params![session_id], row_to_web_source)?;
    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Lists all web_source records across all sessions. Used by the
/// neuroncite_html_list tool when no session filter is specified.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the query fails.
pub fn list_web_sources_all(conn: &Connection) -> Result<Vec<WebSourceRow>, StoreError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, file_id, url, canonical_url, title, meta_description, language,
                og_image, og_title, og_description, author, published_date,
                domain, fetch_timestamp, http_status, content_type, raw_html
         FROM web_source
         ORDER BY url",
    )?;

    let rows = stmt.query_map([], row_to_web_source)?;
    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Deletes the web_source record associated with the given indexed_file ID.
/// Returns the number of rows deleted (0 or 1). This function is typically
/// not called directly because ON DELETE CASCADE on the file_id foreign key
/// automatically removes the web_source record when the indexed_file is deleted.
///
/// # Errors
///
/// Returns `StoreError::Sqlite` if the delete fails.
pub fn delete_web_source(conn: &Connection, file_id: i64) -> Result<usize, StoreError> {
    let count = conn.execute(
        "DELETE FROM web_source WHERE file_id = ?1",
        params![file_id],
    )?;
    Ok(count)
}

/// Maps a `rusqlite` row to a `WebSourceRow`. Column order matches the SELECT
/// statements: id, file_id, url, canonical_url, title, meta_description,
/// language, og_image, og_title, og_description, author, published_date,
/// domain, fetch_timestamp, http_status, content_type, raw_html.
fn row_to_web_source(row: &rusqlite::Row<'_>) -> rusqlite::Result<WebSourceRow> {
    Ok(WebSourceRow {
        id: row.get(0)?,
        file_id: row.get(1)?,
        url: row.get(2)?,
        canonical_url: row.get(3)?,
        title: row.get(4)?,
        meta_description: row.get(5)?,
        language: row.get(6)?,
        og_image: row.get(7)?,
        og_title: row.get(8)?,
        og_description: row.get(9)?,
        author: row.get(10)?,
        published_date: row.get(11)?,
        domain: row.get(12)?,
        fetch_timestamp: row.get(13)?,
        http_status: row.get(14)?,
        content_type: row.get(15)?,
        raw_html: row.get(16)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::migrate;

    /// Helper: creates an in-memory database with foreign keys enabled and
    /// schema applied, then inserts a session and an indexed_file record with
    /// source_type='html'.
    fn setup_db_with_html_file() -> (Connection, i64, i64) {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");

        // Create a session.
        conn.execute(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, hnsw_total, hnsw_orphans, created_at
            ) VALUES ('/html_cache', 'model', 'token', 'eng', 384, 'sqlite-blob', 1, '0.1.0', 0, 0, 1700000000)",
            [],
        )
        .expect("insert session");
        let session_id = conn.last_insert_rowid();

        // Create an indexed_file record with source_type='html'.
        conn.execute(
            "INSERT INTO indexed_file (
                session_id, file_path, file_hash, mtime, size, page_count,
                source_type, created_at, updated_at
            ) VALUES (?1, 'https://example.com/page', 'hash123', 1700000000, 5000, 3, 'html', 1700000000, 1700000000)",
            params![session_id],
        )
        .expect("insert html file");
        let file_id = conn.last_insert_rowid();

        (conn, session_id, file_id)
    }

    /// Returns a WebSourceInsert with all fields populated for testing.
    fn sample_insert(file_id: i64) -> WebSourceInsert<'static> {
        WebSourceInsert {
            file_id,
            url: "https://example.com/page",
            canonical_url: Some("https://example.com/page"),
            title: Some("Example Page"),
            meta_description: Some("A description of the example page."),
            language: Some("en"),
            og_image: Some("https://example.com/image.png"),
            og_title: Some("OG Example Title"),
            og_description: Some("OG description text."),
            author: Some("John Doe"),
            published_date: Some("2025-06-15"),
            domain: "example.com",
            fetch_timestamp: 1700000000,
            http_status: 200,
            content_type: Some("text/html; charset=utf-8"),
            raw_html: Some(b"<html><body>Hello</body></html>"),
        }
    }

    /// T-STORE-WS-001: insert_web_source + get_web_source_by_file roundtrip.
    /// Inserting a web_source record and retrieving it by file_id returns all
    /// stored fields.
    #[test]
    fn t_store_ws_001_insert_and_get_by_file() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        let ws_id = insert_web_source(&conn, &ws).expect("insert_web_source failed");
        assert!(ws_id > 0);

        let row = get_web_source_by_file(&conn, file_id).expect("get_web_source_by_file failed");
        assert_eq!(row.id, ws_id);
        assert_eq!(row.file_id, file_id);
        assert_eq!(row.url, "https://example.com/page");
        assert_eq!(
            row.canonical_url.as_deref(),
            Some("https://example.com/page")
        );
        assert_eq!(row.title.as_deref(), Some("Example Page"));
        assert_eq!(
            row.meta_description.as_deref(),
            Some("A description of the example page.")
        );
        assert_eq!(row.language.as_deref(), Some("en"));
        assert_eq!(
            row.og_image.as_deref(),
            Some("https://example.com/image.png")
        );
        assert_eq!(row.og_title.as_deref(), Some("OG Example Title"));
        assert_eq!(row.og_description.as_deref(), Some("OG description text."));
        assert_eq!(row.author.as_deref(), Some("John Doe"));
        assert_eq!(row.published_date.as_deref(), Some("2025-06-15"));
        assert_eq!(row.domain, "example.com");
        assert_eq!(row.fetch_timestamp, 1700000000);
        assert_eq!(row.http_status, 200);
        assert_eq!(
            row.content_type.as_deref(),
            Some("text/html; charset=utf-8")
        );
        assert_eq!(
            row.raw_html.as_deref(),
            Some(b"<html><body>Hello</body></html>".as_slice())
        );
    }

    /// T-STORE-WS-002: get_web_source_by_url finds a record by URL.
    #[test]
    fn t_store_ws_002_get_by_url() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        let found = get_web_source_by_url(&conn, "https://example.com/page")
            .expect("get_web_source_by_url failed");
        assert!(found.is_some(), "must find web_source by URL");
        assert_eq!(found.unwrap().file_id, file_id);
    }

    /// T-STORE-WS-003: get_web_source_by_url returns None for a URL that does
    /// not exist in the database.
    #[test]
    fn t_store_ws_003_get_by_url_not_found() {
        let (conn, _session_id, _file_id) = setup_db_with_html_file();

        let found = get_web_source_by_url(&conn, "https://nonexistent.example.com/page")
            .expect("get_web_source_by_url failed");
        assert!(found.is_none(), "must return None for non-existent URL");
    }

    /// T-STORE-WS-004: list_web_sources returns only records belonging to files
    /// within the specified session.
    #[test]
    fn t_store_ws_004_list_by_session() {
        let (conn, session_id, file_id) = setup_db_with_html_file();

        // Insert web_source for the file in the test session.
        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        // Create a second session with a separate file and web_source.
        conn.execute(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, hnsw_total, hnsw_orphans, created_at
            ) VALUES ('/other_cache', 'model', 'token', 'eng', 384, 'sqlite-blob', 1, '0.1.0', 0, 0, 1700000000)",
            [],
        )
        .expect("insert second session");
        let session_id_2 = conn.last_insert_rowid();

        conn.execute(
            "INSERT INTO indexed_file (
                session_id, file_path, file_hash, mtime, size, page_count,
                source_type, created_at, updated_at
            ) VALUES (?1, 'https://other.com/page', 'hash456', 1700000000, 3000, 1, 'html', 1700000000, 1700000000)",
            params![session_id_2],
        )
        .expect("insert second html file");
        let file_id_2 = conn.last_insert_rowid();

        let ws2 = WebSourceInsert {
            file_id: file_id_2,
            url: "https://other.com/page",
            canonical_url: None,
            title: Some("Other Page"),
            meta_description: None,
            language: None,
            og_image: None,
            og_title: None,
            og_description: None,
            author: None,
            published_date: None,
            domain: "other.com",
            fetch_timestamp: 1700000001,
            http_status: 200,
            content_type: None,
            raw_html: None,
        };
        insert_web_source(&conn, &ws2).expect("insert second web_source failed");

        // list_web_sources for session 1 returns only the first record.
        let sources = list_web_sources(&conn, session_id).expect("list_web_sources failed");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].url, "https://example.com/page");

        // list_web_sources for session 2 returns only the second record.
        let sources_2 = list_web_sources(&conn, session_id_2).expect("list_web_sources failed");
        assert_eq!(sources_2.len(), 1);
        assert_eq!(sources_2[0].url, "https://other.com/page");
    }

    /// T-STORE-WS-005: delete_web_source removes the record and cascade from
    /// indexed_file deletion also works.
    #[test]
    fn t_store_ws_005_delete() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        // Direct delete by file_id.
        let deleted = delete_web_source(&conn, file_id).expect("delete_web_source failed");
        assert_eq!(deleted, 1);

        // Verify the record is gone.
        let result = get_web_source_by_file(&conn, file_id);
        assert!(result.is_err(), "web_source must be deleted");
    }

    /// T-STORE-WS-006: insert_web_source with all optional fields as None.
    #[test]
    fn t_store_ws_006_insert_minimal() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = WebSourceInsert {
            file_id,
            url: "https://minimal.example.com",
            canonical_url: None,
            title: None,
            meta_description: None,
            language: None,
            og_image: None,
            og_title: None,
            og_description: None,
            author: None,
            published_date: None,
            domain: "minimal.example.com",
            fetch_timestamp: 1700000000,
            http_status: 200,
            content_type: None,
            raw_html: None,
        };

        let ws_id = insert_web_source(&conn, &ws).expect("insert_web_source failed");
        assert!(ws_id > 0);

        let row = get_web_source_by_file(&conn, file_id).expect("get_web_source_by_file failed");
        assert_eq!(row.url, "https://minimal.example.com");
        assert!(row.title.is_none());
        assert!(row.canonical_url.is_none());
        assert!(row.meta_description.is_none());
        assert!(row.raw_html.is_none());
    }

    /// T-STORE-WS-007: insert_web_source with raw_html BLOB stores and retrieves
    /// binary data correctly.
    #[test]
    fn t_store_ws_007_raw_html_blob() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let html_bytes = b"<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Heading</h1><p>Content</p></body></html>";
        let ws = WebSourceInsert {
            file_id,
            url: "https://blob.example.com",
            canonical_url: None,
            title: None,
            meta_description: None,
            language: None,
            og_image: None,
            og_title: None,
            og_description: None,
            author: None,
            published_date: None,
            domain: "blob.example.com",
            fetch_timestamp: 1700000000,
            http_status: 200,
            content_type: Some("text/html"),
            raw_html: Some(html_bytes),
        };

        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        let row = get_web_source_by_file(&conn, file_id).expect("get_web_source_by_file failed");
        assert_eq!(row.raw_html.as_deref(), Some(html_bytes.as_slice()));
    }

    /// T-STORE-WS-008: Schema migration creates the web_source table with the
    /// expected column set.
    #[test]
    fn t_store_ws_008_web_source_column_set() {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");

        let query = "PRAGMA table_info(web_source)";
        let columns: Vec<String> = conn
            .prepare(query)
            .expect("PRAGMA failed")
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })
            .expect("query_map failed")
            .filter_map(|r| r.ok())
            .collect();

        let expected = [
            "id",
            "file_id",
            "url",
            "canonical_url",
            "title",
            "meta_description",
            "language",
            "og_image",
            "og_title",
            "og_description",
            "author",
            "published_date",
            "domain",
            "fetch_timestamp",
            "http_status",
            "content_type",
            "raw_html",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in web_source"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "web_source must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STORE-WS-009: Migration is idempotent -- running migrate() twice does
    /// not produce errors or duplicate the web_source table.
    #[test]
    fn t_store_ws_009_migration_idempotent() {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("first migration failed");
        migrate(&conn).expect("second migration must succeed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'web_source'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(count, 1, "web_source table must not be duplicated");
    }

    /// T-STORE-WS-010: CASCADE delete removes web_source when the parent
    /// indexed_file record is deleted.
    #[test]
    fn t_store_ws_010_cascade_delete_from_indexed_file() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        // Delete the indexed_file record -- CASCADE must remove web_source.
        conn.execute("DELETE FROM indexed_file WHERE id = ?1", params![file_id])
            .expect("delete indexed_file failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM web_source WHERE file_id = ?1",
                params![file_id],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(
            count, 0,
            "CASCADE delete must remove web_source when indexed_file is deleted"
        );
    }

    /// T-STORE-WS-011: UNIQUE constraint on file_id prevents duplicate
    /// web_source records for the same indexed_file.
    #[test]
    fn t_store_ws_011_unique_file_id_constraint() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("first insert must succeed");

        // Second insert with the same file_id must fail.
        let result = insert_web_source(&conn, &ws);
        assert!(
            result.is_err(),
            "duplicate insert for the same file_id must fail due to UNIQUE constraint"
        );
    }

    /// T-STORE-WS-012: list_web_sources_all returns records across all sessions.
    #[test]
    fn t_store_ws_012_list_all() {
        let (conn, _session_id, file_id) = setup_db_with_html_file();

        let ws = sample_insert(file_id);
        insert_web_source(&conn, &ws).expect("insert_web_source failed");

        let all = list_web_sources_all(&conn).expect("list_web_sources_all failed");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].url, "https://example.com/page");
    }
}
