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

//! BM25 keyword search via the FTS5 index.
//!
//! Queries the SQLite FTS5 virtual table with the user's search terms and
//! returns results ranked by BM25 relevance score. FTS5's `bm25()` function
//! returns lower-is-better scores (more negative = more relevant). This module
//! converts those raw scores into 1-indexed rank positions suitable for
//! Reciprocal Rank Fusion.

use rusqlite::Connection;

use crate::error::SearchError;

/// A single result from the FTS5 keyword search. Contains the chunk's database
/// row ID, its raw BM25 score, and its 1-indexed rank position.
#[derive(Debug, Clone)]
pub struct KeywordHit {
    /// Primary key of the chunk in the `chunk` table (matches chunk_fts rowid).
    pub chunk_id: i64,
    /// Raw BM25 score from FTS5. Lower (more negative) values indicate higher
    /// relevance. This field is retained for diagnostic logging and is accessible
    /// to callers who need the original FTS5 score alongside the rank position.
    pub raw_bm25_score: f64,
    /// 1-indexed rank position derived from the raw BM25 score. The chunk with
    /// the lowest (most negative) raw score receives rank 1.
    pub bm25_rank: usize,
}

/// Queries the FTS5 `chunk_fts` virtual table for chunks matching the given
/// query string, ranked by BM25 relevance. Returns results with 1-indexed
/// rank positions.
///
/// FTS5's `bm25()` function returns scores where lower (more negative) values
/// indicate higher relevance. This function sorts by ascending raw score
/// (most relevant first) and assigns 1-indexed ranks.
///
/// Results are always scoped to the given `session_id` via a JOIN with the
/// `chunk` table. The FTS5 virtual table (`chunk_fts`) is global across all
/// sessions; without this session filter, BM25 results from unrelated sessions
/// would leak into the response.
///
/// When `file_ids` is `Some` with a non-empty slice, results are further
/// restricted to chunks belonging to the specified file IDs.
///
/// When `page_range` is `Some((start, end))`, results are restricted to chunks
/// whose page range overlaps [start, end] (inclusive, 1-indexed).
///
/// # Arguments
///
/// * `conn` - Database connection with FTS5 table available.
/// * `query` - The search query string. Passed directly to the FTS5 MATCH operator.
/// * `limit` - Maximum number of results to return.
/// * `session_id` - Only chunks belonging to this session are returned. Required
///   because the FTS5 virtual table is shared across all sessions.
/// * `file_ids` - When `Some`, only chunks belonging to these file IDs are returned.
/// * `page_range` - When `Some((start, end))`, only chunks overlapping the page range
///   [start, end] (inclusive, 1-indexed) are returned.
///
/// # Errors
///
/// Returns `SearchError::KeywordSearch` if the FTS5 query fails (e.g., invalid
/// syntax or database I/O error).
pub fn keyword_search(
    conn: &Connection,
    query: &str,
    limit: usize,
    session_id: i64,
    file_ids: Option<&[i64]>,
    page_range: Option<(i64, i64)>,
) -> Result<Vec<KeywordHit>, SearchError> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Sanitize the query for FTS5 MATCH syntax. Raw user input can contain
    // characters that FTS5 interprets as operators (e.g., hyphens as NOT,
    // colons as column filters, quotes, parentheses). Each whitespace-
    // delimited token is wrapped in double quotes so FTS5 treats it as a
    // literal term. Tokens that already contain double quotes have those
    // quotes stripped to prevent syntax errors.
    let sanitized = sanitize_fts5_query(query);
    if sanitized.is_empty() {
        return Ok(Vec::new());
    }

    // Query the FTS5 virtual table with a mandatory JOIN on the chunk table
    // to scope results to the target session. The FTS5 virtual table
    // (chunk_fts) is global across all sessions, so the JOIN + session_id
    // filter is required to prevent cross-session result leakage.
    //
    // The bm25() function returns a score where lower values indicate higher
    // relevance. ORDER BY score ASC places the most relevant results first.
    // The LIMIT clause caps the number of results returned by the database.
    //
    // When file_ids is provided, results are further restricted to chunks
    // belonging to the specified file IDs.
    let use_file_filter = matches!(file_ids, Some(fids) if !fids.is_empty());
    let use_page_filter = page_range.is_some();

    // Build SQL dynamically. The base query joins chunk_fts with chunk for
    // session scoping. Optional clauses for file_ids and page_range are
    // appended, with parameter indices tracked via a counter.
    let mut sql = String::from(
        "SELECT chunk_fts.rowid, bm25(chunk_fts) AS score \
         FROM chunk_fts \
         JOIN chunk ON chunk_fts.rowid = chunk.id \
         WHERE chunk_fts MATCH ?1 AND chunk.session_id = ?2 AND chunk.is_deleted = 0",
    );

    // Track the next parameter index (1-indexed SQL params).
    let mut param_idx: usize = 3;

    if use_file_filter {
        let fids = file_ids.expect("checked above");
        let file_placeholders: String = (0..fids.len())
            .map(|i| format!("?{}", param_idx + i))
            .collect::<Vec<_>>()
            .join(", ");
        sql.push_str(&format!(" AND chunk.file_id IN ({file_placeholders})"));
        param_idx += fids.len();
    }

    // Page range overlap: chunk.page_start <= range_end AND chunk.page_end >= range_start
    if use_page_filter {
        sql.push_str(&format!(
            " AND chunk.page_start <= ?{} AND chunk.page_end >= ?{}",
            param_idx,
            param_idx + 1
        ));
        param_idx += 2;
    }

    sql.push_str(&format!(" ORDER BY score ASC LIMIT ?{param_idx}"));

    let mut stmt = conn
        .prepare_cached(&sql)
        .map_err(|e| SearchError::KeywordSearch {
            reason: format!("FTS5 query preparation failed: {e}"),
        })?;

    // Build parameter list using trait object references instead of
    // Box<dyn ToSql> to avoid per-parameter heap allocation. All values
    // whose references are pushed into the params vector are stored in
    // local variables that outlive the query execution, following the same
    // pattern as the vector search module's batch_active_chunk_ids().
    let limit_val = limit as i64;

    let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::new();
    params.push(&sanitized as &dyn rusqlite::types::ToSql);
    params.push(&session_id as &dyn rusqlite::types::ToSql);
    if let Some(fids) = file_ids
        && !fids.is_empty()
    {
        for fid in fids {
            params.push(fid as &dyn rusqlite::types::ToSql);
        }
    }

    // Page range bounds must outlive the query execution, so they are stored
    // in local variables before borrowing into the parameter slice.
    let page_end_val: i64;
    let page_start_val: i64;
    if let Some((ps, pe)) = page_range {
        page_end_val = pe;
        page_start_val = ps;
        params.push(&page_end_val as &dyn rusqlite::types::ToSql); // page_start <= range_end
        params.push(&page_start_val as &dyn rusqlite::types::ToSql); // page_end >= range_start
    }
    params.push(&limit_val as &dyn rusqlite::types::ToSql);

    let rows = stmt
        .query_map(params.as_slice(), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })
        .map_err(|e| SearchError::KeywordSearch {
            reason: format!("FTS5 query execution failed: {e}"),
        })?;

    let mut hits = Vec::with_capacity(limit);
    for row_result in rows {
        let (chunk_id, raw_bm25_score) = row_result.map_err(|e| SearchError::KeywordSearch {
            reason: format!("FTS5 row read failed: {e}"),
        })?;
        hits.push((chunk_id, raw_bm25_score));
    }

    // Assign 1-indexed rank positions. The hits are already sorted by ascending
    // BM25 score (most relevant first) from the SQL ORDER BY clause.
    let ranked: Vec<KeywordHit> = hits
        .into_iter()
        .enumerate()
        .map(|(i, (chunk_id, raw_bm25_score))| {
            tracing::trace!(
                chunk_id,
                raw_bm25 = raw_bm25_score,
                rank = i + 1,
                "FTS5 result assigned rank"
            );
            KeywordHit {
                chunk_id,
                raw_bm25_score,
                bm25_rank: i + 1,
            }
        })
        .collect();

    Ok(ranked)
}

/// Extracts the individual search terms from a raw query string using the
/// same sanitization logic as `sanitize_fts5_query`. Each returned string
/// is the cleaned token (without FTS5 quoting) in its original case.
///
/// This function is used by the MCP search handler to report which BM25
/// keywords matched in each result's content. The caller can compare these
/// terms against the result text (case-insensitive) to build a per-result
/// `matched_terms` array.
///
/// Returns an empty Vec if the query contains no alphanumeric tokens.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|token| token.chars().filter(|c| *c != '"').collect::<String>())
        .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
        .collect()
}

/// Converts a raw user query string into a safe FTS5 MATCH expression.
///
/// FTS5 query syntax reserves several characters as operators:
/// - `-` prefix means NOT
/// - `:` means column filter (e.g., `title:word`)
/// - `"` delimits phrase queries
/// - `(`, `)` group sub-expressions
/// - `*` is a prefix wildcard
/// - `+` is sometimes treated as a required-term prefix
///
/// This function splits the query on whitespace, strips internal double
/// quotes from each token, and wraps each token in double quotes. The
/// resulting tokens are joined with spaces, producing an implicit AND
/// query where every token is matched literally.
///
/// Returns an empty string if the input contains no alphanumeric characters
/// after sanitization.
fn sanitize_fts5_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|token| {
            // Strip characters that are FTS5 operators or would break quoting.
            let cleaned: String = token.chars().filter(|c| *c != '"').collect();
            cleaned
        })
        .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
        .map(|t| format!("\"{t}\""))
        .collect();

    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{ChunkInsert, bulk_insert_chunks};
    use rusqlite::Connection;
    use std::path::PathBuf;

    /// Sets up an in-memory database with schema, session, file, and the
    /// specified chunk contents inserted. Returns the connection, chunk IDs,
    /// and the session ID (needed for keyword_search's session scoping).
    fn setup_fts_db(contents: &[&str]) -> (Connection, Vec<i64>, i64) {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };

        let session_id = neuroncite_store::repo::session::create_session(&conn, &config, "0.1.0")
            .expect("create session");

        let file_id = neuroncite_store::repo::file::insert_file(
            &conn,
            session_id,
            "/docs/test.pdf",
            "hash123",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file");

        // Pre-compute hash strings so they outlive the ChunkInsert references.
        let hash_strings: Vec<String> = (0..contents.len()).map(|i| format!("hash_{i}")).collect();

        let chunk_inserts: Vec<ChunkInsert<'_>> = contents
            .iter()
            .enumerate()
            .map(|(i, content)| ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: i as i64,
                doc_text_offset_start: 0,
                doc_text_offset_end: content.len() as i64,
                content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &hash_strings[i],
                simhash: None,
            })
            .collect();

        let ids = bulk_insert_chunks(&conn, &chunk_inserts).expect("bulk insert");
        (conn, ids, session_id)
    }

    /// T-SCH-003: BM25 search returns ranked IDs. An FTS5 query for a term
    /// present in exactly 3 chunks returns those 3 chunk IDs.
    #[test]
    fn t_sch_003_bm25_search_returns_ranked_ids() {
        let contents = [
            "the theory of quantum mechanics explains particle behavior",
            "classical mechanics describes macroscopic motion",
            "quantum computing leverages quantum mechanical phenomena",
            "biology studies living organisms and ecosystems",
            "quantum field theory unifies quantum mechanics and relativity",
        ];

        let (conn, ids, session_id) = setup_fts_db(&contents);

        let results =
            keyword_search(&conn, "quantum", 10, session_id, None, None).expect("keyword search");

        // The term "quantum" appears in chunks 0, 2, and 4
        assert_eq!(
            results.len(),
            3,
            "FTS5 query for 'quantum' must return exactly 3 chunks"
        );

        let result_ids: Vec<i64> = results.iter().map(|h| h.chunk_id).collect();
        assert!(result_ids.contains(&ids[0]), "chunk 0 contains 'quantum'");
        assert!(result_ids.contains(&ids[2]), "chunk 2 contains 'quantum'");
        assert!(result_ids.contains(&ids[4]), "chunk 4 contains 'quantum'");
    }

    /// T-SCH-004: BM25 rank positions are 1-indexed. The highest-ranked result
    /// has bm25_rank = 1.
    #[test]
    fn t_sch_004_bm25_rank_positions_are_1_indexed() {
        let contents = [
            "statistical analysis of variance in experimental data",
            "the variance of the sample mean decreases with sample size",
            "machine learning for pattern recognition",
        ];

        let (conn, _ids, session_id) = setup_fts_db(&contents);

        let results =
            keyword_search(&conn, "variance", 10, session_id, None, None).expect("keyword search");

        assert!(
            !results.is_empty(),
            "must return at least one result for 'variance'"
        );

        // The first result must have rank 1
        assert_eq!(
            results[0].bm25_rank, 1,
            "highest-ranked result must have bm25_rank = 1"
        );

        // Ranks must be consecutive 1-indexed positions
        for (i, hit) in results.iter().enumerate() {
            assert_eq!(
                hit.bm25_rank,
                i + 1,
                "rank positions must be consecutive and 1-indexed"
            );
        }
    }

    /// T-SCH-020: Sanitize wraps each token in double quotes so FTS5 treats
    /// hyphens, colons, and other operator characters as literal text.
    #[test]
    fn t_sch_020_sanitize_wraps_tokens_in_quotes() {
        let result = sanitize_fts5_query("short-term volatility");
        assert_eq!(result, "\"short-term\" \"volatility\"");
    }

    /// T-SCH-021: Sanitize strips embedded double quotes from tokens to
    /// prevent FTS5 syntax errors from unbalanced quote characters.
    #[test]
    fn t_sch_021_sanitize_strips_embedded_quotes() {
        let result = sanitize_fts5_query("the \"Box-Cox\" family");
        assert_eq!(result, "\"the\" \"Box-Cox\" \"family\"");
    }

    /// T-SCH-012: Sanitize returns an empty string for input that contains
    /// no alphanumeric characters (only operators/punctuation).
    #[test]
    fn t_sch_012_sanitize_empty_for_no_alphanumeric() {
        let result = sanitize_fts5_query("--- ::: *** ()");
        assert!(
            result.is_empty(),
            "non-alphanumeric input must produce empty output"
        );
    }

    /// T-SCH-013: Keyword search with hyphenated terms succeeds instead of
    /// failing with "no such column" FTS5 error. The hyphen in "short-term"
    /// was previously interpreted as the FTS5 NOT operator.
    #[test]
    fn t_sch_013_hyphenated_query_does_not_error() {
        let contents = [
            "short-term volatility measures are range-based estimators",
            "long-term investment strategies require patience",
            "the variance of short-term returns exhibits clustering",
        ];

        let (conn, ids, session_id) = setup_fts_db(&contents);

        // This query previously failed with "no such column: term"
        let results = keyword_search(&conn, "short-term volatility", 10, session_id, None, None)
            .expect("hyphenated query must not error");

        assert!(!results.is_empty(), "hyphenated query must return results");

        let result_ids: Vec<i64> = results.iter().map(|h| h.chunk_id).collect();
        assert!(
            result_ids.contains(&ids[0]),
            "chunk 0 contains 'short-term' and 'volatility'"
        );
    }

    /// T-SCH-014: BM25 keyword search with session_id scoping only returns
    /// chunks from the specified session, even though FTS5 is a global virtual
    /// table shared across all sessions.
    ///
    /// Regression test for BUG-001: Before the fix, keyword_search did not
    /// filter by session_id, causing results from unrelated sessions to leak
    /// into BM25 results. The RRF merge then included these cross-session
    /// results in the final output.
    #[test]
    fn t_sch_014_bm25_search_isolates_sessions() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

        let config_a = IndexConfig {
            directory: PathBuf::from("/docs_a"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let config_b = IndexConfig {
            directory: PathBuf::from("/docs_b"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };

        // Create two separate sessions (simulating two different directories).
        let session_a = neuroncite_store::repo::session::create_session(&conn, &config_a, "0.1.0")
            .expect("create session A");
        let session_b = neuroncite_store::repo::session::create_session(&conn, &config_b, "0.1.0")
            .expect("create session B");

        // Insert a file and chunks into session A.
        let file_a = neuroncite_store::repo::file::insert_file(
            &conn,
            session_a,
            "/docs_a/paper.pdf",
            "hash_a",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file A");

        let hash_a = "hash_chunk_a".to_string();
        let chunk_a = ChunkInsert {
            file_id: file_a,
            session_id: session_a,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "heteroskedasticity consistent covariance matrix estimator",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: &hash_a,
            simhash: None,
        };
        let ids_a = bulk_insert_chunks(&conn, &[chunk_a]).expect("insert chunks A");

        // Insert a file and chunks into session B with the same term.
        let file_b = neuroncite_store::repo::file::insert_file(
            &conn,
            session_b,
            "/docs_b/other.pdf",
            "hash_b",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file B");

        let hash_b = "hash_chunk_b".to_string();
        let chunk_b = ChunkInsert {
            file_id: file_b,
            session_id: session_b,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "heteroskedasticity robust inference methods",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: &hash_b,
            simhash: None,
        };
        let ids_b = bulk_insert_chunks(&conn, &[chunk_b]).expect("insert chunks B");

        // Search session A for "heteroskedasticity" -- must only return chunk
        // from session A, not from session B.
        let results_a = keyword_search(&conn, "heteroskedasticity", 10, session_a, None, None)
            .expect("keyword search session A");

        assert_eq!(
            results_a.len(),
            1,
            "session A search must return exactly 1 chunk from session A"
        );
        assert_eq!(
            results_a[0].chunk_id, ids_a[0],
            "result must be the chunk from session A"
        );

        // Search session B for "heteroskedasticity" -- must only return chunk
        // from session B, not from session A.
        let results_b = keyword_search(&conn, "heteroskedasticity", 10, session_b, None, None)
            .expect("keyword search session B");

        assert_eq!(
            results_b.len(),
            1,
            "session B search must return exactly 1 chunk from session B"
        );
        assert_eq!(
            results_b[0].chunk_id, ids_b[0],
            "result must be the chunk from session B"
        );
    }

    /// T-SCH-016: extract_query_terms produces cleaned tokens from a query
    /// string. Tokens are stripped of double quotes but preserve their
    /// original case and hyphenation.
    #[test]
    fn t_sch_016_extract_query_terms_basic() {
        let terms = extract_query_terms("heteroskedasticity covariance");
        assert_eq!(terms, vec!["heteroskedasticity", "covariance"]);
    }

    /// T-SCH-017: extract_query_terms strips embedded double quotes from
    /// tokens (same sanitization as sanitize_fts5_query).
    #[test]
    fn t_sch_017_extract_query_terms_strips_quotes() {
        let terms = extract_query_terms("the \"Box-Cox\" family");
        assert_eq!(terms, vec!["the", "Box-Cox", "family"]);
    }

    /// T-SCH-018: extract_query_terms returns an empty Vec for input
    /// containing only punctuation and operator characters.
    #[test]
    fn t_sch_018_extract_query_terms_empty_for_punctuation() {
        let terms = extract_query_terms("--- ::: *** ()");
        assert!(terms.is_empty());
    }

    /// T-SCH-019: extract_query_terms preserves hyphenated terms as single
    /// tokens, matching the FTS5 sanitization behavior.
    #[test]
    fn t_sch_019_extract_query_terms_preserves_hyphens() {
        let terms = extract_query_terms("short-term volatility range-based");
        assert_eq!(terms, vec!["short-term", "volatility", "range-based"]);
    }

    /// T-SCH-015: BM25 keyword search with file_ids filter combined with
    /// session_id scoping. Ensures both filters work together correctly.
    #[test]
    fn t_sch_015_bm25_session_and_file_filter_combined() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };

        let session_id = neuroncite_store::repo::session::create_session(&conn, &config, "0.1.0")
            .expect("create session");

        // Create two files in the same session.
        let file_1 = neuroncite_store::repo::file::insert_file(
            &conn,
            session_id,
            "/docs/paper1.pdf",
            "h1",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file 1");
        let file_2 = neuroncite_store::repo::file::insert_file(
            &conn,
            session_id,
            "/docs/paper2.pdf",
            "h2",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file 2");

        let h1 = "hash_c1".to_string();
        let h2 = "hash_c2".to_string();
        let chunks = [
            ChunkInsert {
                file_id: file_1,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 40,
                content: "volatility clustering in financial returns",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &h1,
                simhash: None,
            },
            ChunkInsert {
                file_id: file_2,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 40,
                content: "volatility estimation using range-based methods",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &h2,
                simhash: None,
            },
        ];

        let ids = bulk_insert_chunks(&conn, &chunks).expect("insert chunks");

        // Without file filter: both chunks match.
        let all_results = keyword_search(&conn, "volatility", 10, session_id, None, None)
            .expect("keyword search without filter");
        assert_eq!(all_results.len(), 2, "both chunks contain 'volatility'");

        // With file filter: only file_1's chunk matches.
        let filtered_results =
            keyword_search(&conn, "volatility", 10, session_id, Some(&[file_1]), None)
                .expect("keyword search with file filter");
        assert_eq!(
            filtered_results.len(),
            1,
            "file filter must restrict to file_1"
        );
        assert_eq!(filtered_results[0].chunk_id, ids[0]);
    }
}
