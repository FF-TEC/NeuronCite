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

//! Citation assembly from search result chunks.
//!
//! Resolves each result chunk back to its source PDF file path and page number
//! by reading chunk metadata from the database. Produces structured `Citation`
//! objects that include the file display name, page range, and a text excerpt
//! in quotation marks for display in the API response and GUI results panel.
//!
//! Citation format rules:
//! - Single page: `"filename.pdf, p. 5: 'excerpt...'"` when page_start == page_end
//! - Multi-page: `"filename.pdf, pp. 5-7: 'excerpt...'"` when page_start != page_end
//! - Excerpt: first 200 characters of the chunk content, with "..." appended if truncated

use std::path::PathBuf;

use rusqlite::Connection;

use neuroncite_core::Citation;

use crate::error::SearchError;

/// Maximum number of characters from the chunk content to include in the
/// citation excerpt.
const EXCERPT_MAX_CHARS: usize = 200;

/// Raw metadata for a single chunk, loaded by the batch query in
/// `batch_load_chunk_meta`. Contains all fields needed to construct
/// both the `Citation` and the `SearchResult`.
pub(crate) struct ChunkMeta {
    pub chunk_id: i64,
    pub file_id: i64,
    pub page_start: i64,
    pub page_end: i64,
    pub chunk_index: i64,
    pub content: String,
    pub doc_offset_start: i64,
    pub doc_offset_end: i64,
    pub file_path: String,
}

/// Batch-loads chunk metadata and file paths for all given chunk IDs in a
/// single JOIN query. Returns a map of chunk_id -> ChunkMeta.
///
/// Replaces N individual queries (1 per chunk for metadata + 1 per chunk for
/// file path) with a single `WHERE id IN (...)` query joining `chunk` and
/// `indexed_file` tables.
///
/// # Errors
///
/// Returns `SearchError::CitationAssembly` if the query preparation or
/// row reading fails.
pub(crate) fn batch_load_chunk_meta(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<std::collections::HashMap<i64, ChunkMeta>, SearchError> {
    use std::collections::HashMap;

    if chunk_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let placeholders: String = std::iter::repeat_n("?", chunk_ids.len())
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        "SELECT c.id, c.file_id, c.page_start, c.page_end, c.chunk_index,
                c.content, c.doc_text_offset_start, c.doc_text_offset_end,
                f.file_path
         FROM chunk c
         JOIN indexed_file f ON c.file_id = f.id
         WHERE c.id IN ({placeholders})"
    );

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| SearchError::CitationAssembly {
            reason: format!("batch chunk metadata preparation failed: {e}"),
        })?;

    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();

    let rows = stmt
        .query_map(params.as_slice(), |row| {
            Ok(ChunkMeta {
                chunk_id: row.get(0)?,
                file_id: row.get(1)?,
                page_start: row.get(2)?,
                page_end: row.get(3)?,
                chunk_index: row.get(4)?,
                content: row.get(5)?,
                doc_offset_start: row.get(6)?,
                doc_offset_end: row.get(7)?,
                file_path: row.get(8)?,
            })
        })
        .map_err(|e| SearchError::CitationAssembly {
            reason: format!("batch chunk metadata query failed: {e}"),
        })?;

    let mut metas = HashMap::with_capacity(chunk_ids.len());
    for row in rows {
        let meta = row.map_err(|e| SearchError::CitationAssembly {
            reason: format!("batch chunk metadata row read failed: {e}"),
        })?;
        metas.insert(meta.chunk_id, meta);
    }

    Ok(metas)
}

/// Constructs a `Citation` from pre-loaded `ChunkMeta` without any additional
/// database queries. Formats the page reference as "p. X" for single-page
/// chunks or "pp. X-Y" for multi-page chunks, and includes a text excerpt
/// from the chunk content.
pub(crate) fn citation_from_meta(meta: &ChunkMeta) -> Citation {
    let source_file = PathBuf::from(&meta.file_path);

    let file_display_name = source_file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| meta.file_path.clone());

    let excerpt = build_excerpt(&meta.content);

    let page_ref = if meta.page_start == meta.page_end {
        format!("p. {}", meta.page_start)
    } else {
        format!("pp. {}-{}", meta.page_start, meta.page_end)
    };

    let formatted = format!("{file_display_name}, {page_ref}: '{excerpt}'");

    Citation {
        file_id: meta.file_id,
        source_file,
        file_display_name,
        // Use try_from to safely convert i64 page numbers to usize. Negative
        // page numbers (which should not occur in a valid database row) are
        // mapped to 0 rather than wrapping to a large usize via `as` cast.
        page_start: usize::try_from(meta.page_start).unwrap_or(0),
        page_end: usize::try_from(meta.page_end).unwrap_or(0),
        doc_offset_start: usize::try_from(meta.doc_offset_start).unwrap_or(0),
        doc_offset_end: usize::try_from(meta.doc_offset_end).unwrap_or(0),
        formatted,
    }
}

/// Builds a text excerpt from the chunk content. Takes the first
/// `EXCERPT_MAX_CHARS` characters and appends "..." if the content is longer.
fn build_excerpt(content: &str) -> String {
    if content.chars().count() <= EXCERPT_MAX_CHARS {
        content.to_string()
    } else {
        let truncated: String = content.chars().take(EXCERPT_MAX_CHARS).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{ChunkInsert, bulk_insert_chunks};
    use rusqlite::Connection;
    use std::path::PathBuf;

    /// Sets up a database with a single chunk spanning the given page range.
    /// Returns the connection and the chunk ID for use in citation tests.
    fn setup_citation_db(page_start: i64, page_end: i64, content: &str) -> (Connection, i64) {
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
            "/docs/statistics.pdf",
            "hash123",
            1_700_000_000,
            4096,
            10,
            None,
        )
        .expect("insert file");

        let chunks = vec![ChunkInsert {
            file_id,
            session_id,
            page_start,
            page_end,
            chunk_index: 0,
            doc_text_offset_start: 100,
            doc_text_offset_end: 100 + content.len() as i64,
            content,
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "citation_test_hash",
            simhash: None,
        }];

        let ids = bulk_insert_chunks(&conn, &chunks).expect("bulk insert");
        (conn, ids[0])
    }

    /// T-SCH-010: Citation single-page format. Uses "p. 5" when
    /// page_start == page_end == 5. Verifies the batch-loaded citation path.
    #[test]
    fn t_sch_010_citation_single_page_format() {
        let content = "Regression analysis is a statistical method for modeling relationships.";
        let (conn, chunk_id) = setup_citation_db(5, 5, content);

        let metas = batch_load_chunk_meta(&conn, &[chunk_id]).expect("batch_load_chunk_meta");
        let meta = metas.get(&chunk_id).expect("chunk meta must be present");
        let citation = citation_from_meta(meta);

        assert_eq!(citation.page_start, 5);
        assert_eq!(citation.page_end, 5);
        assert!(
            citation.formatted.contains("p. 5"),
            "single-page citation must contain 'p. 5', found: {}",
            citation.formatted,
        );
        assert!(
            !citation.formatted.contains("pp."),
            "single-page citation must not contain 'pp.', found: {}",
            citation.formatted,
        );
        assert!(
            citation.formatted.contains("statistics.pdf"),
            "citation must contain the file display name"
        );
    }

    /// T-SCH-011: Citation multi-page format. Uses "pp. 5-7" when
    /// page_start=5, page_end=7. Verifies the batch-loaded citation path.
    #[test]
    fn t_sch_011_citation_multi_page_format() {
        let content = "Cross-validation is a technique for assessing model generalization.";
        let (conn, chunk_id) = setup_citation_db(5, 7, content);

        let metas = batch_load_chunk_meta(&conn, &[chunk_id]).expect("batch_load_chunk_meta");
        let meta = metas.get(&chunk_id).expect("chunk meta must be present");
        let citation = citation_from_meta(meta);

        assert_eq!(citation.page_start, 5);
        assert_eq!(citation.page_end, 7);
        assert!(
            citation.formatted.contains("pp. 5-7"),
            "multi-page citation must contain 'pp. 5-7', found: {}",
            citation.formatted,
        );
        assert!(
            citation.formatted.contains("statistics.pdf"),
            "citation must contain the file display name"
        );
    }

    /// Verifies that excerpt truncation works for content longer than 200 characters.
    #[test]
    fn excerpt_truncation_appends_ellipsis() {
        let long_content = "a".repeat(300);
        let excerpt = build_excerpt(&long_content);
        assert_eq!(excerpt.len(), 203); // 200 chars + "..."
        assert!(excerpt.ends_with("..."));
    }

    /// Verifies that short content is not truncated.
    #[test]
    fn excerpt_preserves_short_content() {
        let short = "short text";
        let excerpt = build_excerpt(short);
        assert_eq!(excerpt, "short text");
        assert!(!excerpt.ends_with("..."));
    }

    /// Verifies that batch_load_chunk_meta returns an empty map for empty input.
    #[test]
    fn batch_load_empty_ids_returns_empty_map() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

        let metas = batch_load_chunk_meta(&conn, &[]).expect("batch_load_chunk_meta");
        assert!(metas.is_empty(), "empty input must produce empty map");
    }
}
