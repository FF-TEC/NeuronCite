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

//! Handler for the `neuroncite_files` MCP tool.
//!
//! Lists all indexed documents within a session with per-file chunk statistics,
//! page backend distribution, text byte counts, extraction status, and source
//! type. For HTML sources, each file entry includes web_source metadata (URL,
//! title, domain, author, language). Supports optional filtering by source_type
//! and domain, sorting, and single-file detail mode via the `file_id` parameter.
//!
//! This handler absorbs the functionality of the former `neuroncite_html_list`
//! tool: use the `source_type` filter with value "html" and the `domain` filter
//! to replicate html_list queries.

use std::collections::HashMap;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;

/// Lists indexed files in a session with chunk and page statistics.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Session to list files from.
/// - `file_id` (optional): Return details for a single file instead of all files.
/// - `sort_by` (optional): Sort order for the file list. One of "name" (default),
///   "size", "pages", "chunks".
/// - `source_type` (optional): Filter files by source type (e.g., "pdf", "html",
///   "txt", "docx"). Only files matching this type are returned.
/// - `domain` (optional): Filter HTML files by domain substring (case-sensitive).
///   Files whose web_source domain contains this substring are included. Non-HTML
///   files are excluded when this filter is active.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;
    let sort_by = params["sort_by"].as_str().unwrap_or("name");
    let single_file_id = params["file_id"].as_i64();
    let source_type_filter = params["source_type"].as_str();
    let domain_filter = params["domain"].as_str();

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Verify the session exists before listing files.
    neuroncite_store::get_session(&conn, session_id)
        .map_err(|_| format!("session {session_id} not found"))?;

    // Fetch file records: single file or all files in session.
    let files = if let Some(fid) = single_file_id {
        let f =
            neuroncite_store::get_file(&conn, fid).map_err(|_| format!("file {fid} not found"))?;
        if f.session_id != session_id {
            return Err(format!(
                "file {fid} does not belong to session {session_id}"
            ));
        }
        vec![f]
    } else {
        neuroncite_store::list_files_by_session(&conn, session_id)
            .map_err(|e| format!("listing files: {e}"))?
    };

    // Apply source_type filter. When domain filter is active, only HTML files
    // can match (domain is a web_source field that only exists for HTML).
    let files: Vec<_> = files
        .into_iter()
        .filter(|f| {
            if let Some(st) = source_type_filter
                && f.source_type != st
            {
                return false;
            }
            if domain_filter.is_some() && f.source_type != "html" {
                return false;
            }
            true
        })
        .collect();

    // Pre-fetch web_source metadata for all HTML files in one pass. The
    // web_source table has a 1:1 relationship with indexed_file for HTML
    // sources. Caching avoids N+1 queries when the session contains many
    // HTML files.
    let mut web_source_map: HashMap<i64, serde_json::Value> = HashMap::new();
    for f in &files {
        if f.source_type == "html"
            && let Ok(ws) = neuroncite_store::get_web_source_by_file(&conn, f.id)
        {
            // Apply domain filter: skip HTML files whose domain does not
            // contain the filter substring.
            if let Some(domain) = domain_filter
                && !ws.domain.contains(domain)
            {
                continue;
            }
            web_source_map.insert(
                f.id,
                serde_json::json!({
                    "url": ws.url,
                    "title": ws.title,
                    "domain": ws.domain,
                    "author": ws.author,
                    "language": ws.language,
                }),
            );
        }
    }

    // When domain filter is active, only keep HTML files that passed the
    // domain check (present in web_source_map).
    let files: Vec<_> = if domain_filter.is_some() {
        files
            .into_iter()
            .filter(|f| f.source_type != "html" || web_source_map.contains_key(&f.id))
            .collect()
    } else {
        files
    };

    // Batch-fetch per-file chunk statistics: count, min/max/avg content length.
    let chunk_stats = neuroncite_store::file_chunk_stats_by_session(&conn, session_id)
        .map_err(|e| format!("fetching chunk stats: {e}"))?;
    let chunk_map: HashMap<i64, neuroncite_store::FileChunkStats> =
        chunk_stats.into_iter().map(|s| (s.file_id, s)).collect();

    // Batch-fetch per-file page backend distribution.
    let page_stats = neuroncite_store::file_page_stats_by_session(&conn, session_id)
        .map_err(|e| format!("fetching page stats: {e}"))?;

    // Group page stats by file_id for O(1) lookup.
    let mut page_map: HashMap<i64, Vec<neuroncite_store::FilePageStats>> = HashMap::new();
    for ps in page_stats {
        page_map.entry(ps.file_id).or_default().push(ps);
    }

    // Accumulate session totals while building per-file JSON.
    let mut total_files = 0_i64;
    let mut total_pages = 0_i64;
    let mut total_chunks = 0_i64;
    let mut total_bytes = 0_i64;

    let mut file_array: Vec<serde_json::Value> = files
        .iter()
        .map(|f| {
            let clean_path = strip_extended_length_prefix(&f.file_path);
            let file_name = std::path::Path::new(clean_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| clean_path.to_string());

            // Page statistics: empty pages, OCR pages, and total text bytes
            // are aggregated from the per-backend page stats. The extracted
            // page count comes from the file record (f.page_count) rather
            // than re-summing page stats, because the file record is the
            // authoritative source for extracted page count.
            let file_page_stats = page_map.get(&f.id);
            let mut empty_pages = 0_i64;
            let mut ocr_pages = 0_i64;
            let mut file_bytes = 0_i64;

            if let Some(stats) = file_page_stats {
                for ps in stats {
                    empty_pages += ps.empty_count;
                    file_bytes += ps.total_bytes;
                    if ps.backend == "ocr" {
                        ocr_pages += ps.page_count;
                    }
                }
            }

            // Determine extraction status by comparing extracted vs structural page count.
            let status = match f.pdf_page_count {
                Some(pdf_pc) if f.page_count < pdf_pc => "incomplete",
                Some(pdf_pc) if f.page_count > pdf_pc => "mismatch",
                _ => "complete",
            };

            // Determine the dominant text extraction method for this file.
            // "native" means all pages used the built-in PDF text extractor.
            // "ocr" means all pages required OCR fallback (scanned documents).
            // "mixed" means both methods were used (partially native, partially OCR).
            // "unknown" when no page statistics are available.
            let native_pages = f.page_count - ocr_pages;
            let extraction_method = if f.page_count == 0 {
                "unknown"
            } else if ocr_pages == 0 {
                "native"
            } else if native_pages == 0 {
                "ocr"
            } else {
                "mixed"
            };

            // Chunk statistics for this file.
            let cs = chunk_map.get(&f.id);
            let chunk_count = cs.map_or(0, |c| c.chunk_count);

            // Word count approximation from byte count (avg English word ~5.5 bytes + space).
            let total_words = file_bytes / 6;
            let avg_words_per_page = if f.page_count > 0 {
                total_words / f.page_count
            } else {
                0
            };

            // Accumulate session totals.
            total_files += 1;
            total_pages += f.page_count;
            total_chunks += chunk_count;
            total_bytes += file_bytes;

            let mut entry = serde_json::json!({
                "file_id": f.id,
                "file_path": clean_path,
                "file_name": file_name,
                "size": f.size,
                "source_type": f.source_type,
                "extraction_method": extraction_method,
                "pages": {
                    "extracted": f.page_count,
                    "structural": f.pdf_page_count,
                    "empty": empty_pages,
                    "ocr_fallback": ocr_pages,
                    "status": status,
                },
                "text": {
                    "total_bytes": file_bytes,
                    "total_words": total_words,
                    "avg_words_per_page": avg_words_per_page,
                },
                "chunks": {
                    "count": chunk_count,
                    "avg_bytes": cs.map_or(0.0, |c| c.avg_content_len),
                    "min_bytes": cs.map_or(0, |c| c.min_content_len),
                    "max_bytes": cs.map_or(0, |c| c.max_content_len),
                },
            });

            // For HTML sources, include the pre-fetched web_source metadata
            // (URL, title, domain, author, language) in the file entry.
            if let Some(ws) = web_source_map.get(&f.id) {
                entry["web_source"] = ws.clone();
            }

            entry
        })
        .collect();

    // Sort the file array based on the requested field.
    match sort_by {
        "size" => file_array.sort_by(|a, b| b["size"].as_i64().cmp(&a["size"].as_i64())),
        "pages" => file_array.sort_by(|a, b| {
            b["pages"]["extracted"]
                .as_i64()
                .cmp(&a["pages"]["extracted"].as_i64())
        }),
        "chunks" => file_array.sort_by(|a, b| {
            b["chunks"]["count"]
                .as_i64()
                .cmp(&a["chunks"]["count"].as_i64())
        }),
        // "name" or any unrecognized value: files are already sorted by file_path
        // from the SQL ORDER BY clause.
        _ => {}
    }

    Ok(serde_json::json!({
        "session_id": session_id,
        "file_count": file_array.len(),
        "files": file_array,
        "session_totals": {
            "total_files": total_files,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_content_bytes": total_bytes,
            "total_words": total_bytes / 6,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Re-export rusqlite through r2d2_sqlite so the Connection type is
    // available without adding a direct rusqlite dependency to this crate.
    use r2d2_sqlite::rusqlite;

    /// Creates an in-memory SQLite connection pool with foreign keys enabled
    /// and all neuroncite-store migrations applied. The pool has a max size
    /// of 2 connections, sufficient for single-threaded test usage.
    fn test_pool() -> r2d2::Pool<r2d2_sqlite::SqliteConnectionManager> {
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }
        pool
    }

    /// Constructs an AppState backed by a stub embedding backend and the
    /// given connection pool. The stub backend returns zero-vectors for all
    /// embedding requests.
    fn test_state(pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>) -> Arc<AppState> {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed")
    }

    /// Returns a standard IndexConfig used across all tests in this module.
    /// Points to "/test/dir" as the session directory with a word-based
    /// chunking strategy (chunk_size=300, overlap=50).
    fn test_index_config() -> neuroncite_core::IndexConfig {
        neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/test/dir"),
            model_name: "test-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        }
    }

    /// Inserts a file record, its pages, and its chunks into the database.
    /// Returns the file_id. The file has 3 extracted pages (2 via pdf-extract,
    /// 1 via ocr) and 2 chunks. Page 3 is empty (zero bytes). The structural
    /// PDF page count matches the extracted count (3), so extraction status
    /// is "complete".
    fn seed_file_with_data(conn: &rusqlite::Connection, session_id: i64) -> i64 {
        let file_id = neuroncite_store::insert_file(
            conn,
            session_id,
            "/test/dir/paper.pdf",
            "abc123hash",
            1_700_000_000,
            8192,
            3,
            Some(3),
        )
        .expect("insert_file");

        // Three pages: two with text content via pdf-extract, one empty via ocr.
        let pages: Vec<(i64, &str, &str)> = vec![
            (1, "First page has some text content here", "pdf-extract"),
            (2, "Second page also contains text", "pdf-extract"),
            (3, "", "ocr"),
        ];
        neuroncite_store::bulk_insert_pages(conn, file_id, &pages).expect("bulk_insert_pages");

        // Two chunks referencing this file and session.
        let chunks = vec![
            neuroncite_store::ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 39,
                content: "First page has some text content here",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "chunk_hash_0",
                simhash: None,
            },
            neuroncite_store::ChunkInsert {
                file_id,
                session_id,
                page_start: 2,
                page_end: 2,
                chunk_index: 1,
                doc_text_offset_start: 39,
                doc_text_offset_end: 70,
                content: "Second page also contains text",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "chunk_hash_1",
                simhash: None,
            },
        ];
        neuroncite_store::bulk_insert_chunks(conn, &chunks).expect("bulk_insert_chunks");

        file_id
    }

    /// T-MCP-FILES-001: Calling handle without a session_id parameter returns
    /// an error indicating the required parameter is missing.
    #[tokio::test]
    async fn t_mcp_files_001_missing_session_id_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({});
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handle must fail when session_id is absent"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("session_id"),
            "error message must reference the missing parameter name, got: {err}"
        );
    }

    /// T-MCP-FILES-002: Providing a session_id that does not exist in the
    /// database returns a "not found" error. The handler verifies session
    /// existence before querying files.
    #[tokio::test]
    async fn t_mcp_files_002_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "session_id": 999999 });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handle must fail for a non-existent session"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("not found"),
            "error message must indicate session was not found, got: {err}"
        );
    }

    /// T-MCP-FILES-003: A session containing one file returns file_count=1
    /// with correctly populated pages, text, and chunks sub-objects. Verifies
    /// the structure and specific field values produced by the handler for a
    /// known data set (3 pages, 2 chunks, 1 ocr page, 1 empty page).
    #[tokio::test]
    async fn t_mcp_files_003_single_file_returns_correct_structure() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id, _file_id) = {
            let conn = pool.get().expect("get conn");
            let config = test_index_config();
            let sid =
                neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create_session");
            let fid = seed_file_with_data(&conn, sid);
            (sid, fid)
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handle must succeed: {:?}", result.err());

        let response = result.unwrap();

        // Top-level fields.
        assert_eq!(response["session_id"], session_id);
        assert_eq!(response["file_count"], 1);

        // Per-file entry.
        let files = response["files"]
            .as_array()
            .expect("files must be an array");
        assert_eq!(files.len(), 1);

        let file = &files[0];
        assert_eq!(file["file_name"], "paper.pdf");
        assert_eq!(file["size"], 8192);

        // Pages sub-object: 3 extracted, 3 structural, 1 empty, 1 ocr, status=complete.
        let pages = &file["pages"];
        assert_eq!(pages["extracted"], 3);
        assert_eq!(pages["structural"], 3);
        assert_eq!(pages["empty"], 1);
        assert_eq!(pages["ocr_fallback"], 1);
        assert_eq!(pages["status"], "complete");

        // Text sub-object: total_bytes is the sum of byte lengths of all page contents.
        // Page 1: "First page has some text content here" = 37 bytes
        // Page 2: "Second page also contains text" = 30 bytes
        // Page 3: "" = 0 bytes
        // Total = 67 bytes
        let text = &file["text"];
        assert_eq!(text["total_bytes"], 67);
        // total_words = 67 / 6 = 11 (integer division)
        assert_eq!(text["total_words"], 11);
        // avg_words_per_page = 11 / 3 = 3 (integer division)
        assert_eq!(text["avg_words_per_page"], 3);

        // Chunks sub-object: 2 chunks inserted.
        let chunks = &file["chunks"];
        assert_eq!(chunks["count"], 2);

        // min_bytes = min(37, 30) = 30, max_bytes = max(37, 30) = 37
        assert_eq!(chunks["min_bytes"], 30);
        assert_eq!(chunks["max_bytes"], 37);
    }

    /// T-MCP-FILES-004: The session_totals fields in the response match the
    /// summed values from all per-file entries. This test inserts two files
    /// in the same session and verifies that session_totals.total_files,
    /// total_pages, total_chunks, and total_content_bytes equal the sums
    /// of the respective per-file values.
    #[tokio::test]
    async fn t_mcp_files_004_session_totals_match_summed_per_file_values() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let session_id = {
            let conn = pool.get().expect("get conn");
            let config = test_index_config();
            let sid =
                neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create_session");

            // First file: seeded with 3 pages and 2 chunks via the helper.
            seed_file_with_data(&conn, sid);

            // Second file: 1 page, 1 chunk, different content.
            let file_id_2 = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/notes.pdf",
                "def456hash",
                1_700_001_000,
                2048,
                1,
                Some(1),
            )
            .expect("insert second file");

            let pages_2: Vec<(i64, &str, &str)> =
                vec![(1, "Notes page content with several words", "pdf-extract")];
            neuroncite_store::bulk_insert_pages(&conn, file_id_2, &pages_2)
                .expect("bulk_insert_pages file 2");

            let chunks_2 = vec![neuroncite_store::ChunkInsert {
                file_id: file_id_2,
                session_id: sid,
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 38,
                content: "Notes page content with several words",
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "chunk_hash_notes",
                simhash: None,
            }];
            neuroncite_store::bulk_insert_chunks(&conn, &chunks_2)
                .expect("bulk_insert_chunks file 2");

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handle must succeed: {:?}", result.err());

        let response = result.unwrap();
        let files = response["files"]
            .as_array()
            .expect("files must be an array");
        assert_eq!(files.len(), 2, "session contains two files");

        // Sum per-file values for cross-checking against session_totals.
        let mut sum_pages: i64 = 0;
        let mut sum_chunks: i64 = 0;
        let mut sum_bytes: i64 = 0;
        for f in files {
            sum_pages += f["pages"]["extracted"].as_i64().expect("extracted pages");
            sum_chunks += f["chunks"]["count"].as_i64().expect("chunk count");
            sum_bytes += f["text"]["total_bytes"].as_i64().expect("total_bytes");
        }

        let totals = &response["session_totals"];
        assert_eq!(
            totals["total_files"], 2,
            "total_files must equal the number of file entries"
        );
        assert_eq!(
            totals["total_pages"].as_i64().expect("total_pages"),
            sum_pages,
            "total_pages must equal sum of per-file extracted pages"
        );
        assert_eq!(
            totals["total_chunks"].as_i64().expect("total_chunks"),
            sum_chunks,
            "total_chunks must equal sum of per-file chunk counts"
        );
        assert_eq!(
            totals["total_content_bytes"]
                .as_i64()
                .expect("total_content_bytes"),
            sum_bytes,
            "total_content_bytes must equal sum of per-file text total_bytes"
        );

        // total_words = total_content_bytes / 6 (integer division)
        let expected_words = sum_bytes / 6;
        assert_eq!(
            totals["total_words"].as_i64().expect("total_words"),
            expected_words,
            "total_words must equal total_content_bytes / 6"
        );
    }

    /// T-MCP-FILES-005: Passing the file_id parameter returns a single-file
    /// detail response (file_count=1) containing only the specified file,
    /// rather than all files in the session.
    #[tokio::test]
    async fn t_mcp_files_005_file_id_returns_single_file_detail() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id, file_id_1, _file_id_2) = {
            let conn = pool.get().expect("get conn");
            let config = test_index_config();
            let sid =
                neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create_session");

            // Insert two files so we can verify only one is returned.
            let fid1 = seed_file_with_data(&conn, sid);

            let fid2 = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/other.pdf",
                "xyz789hash",
                1_700_002_000,
                4096,
                1,
                Some(1),
            )
            .expect("insert second file");

            let pages_2: Vec<(i64, &str, &str)> = vec![(1, "Other file content", "pdf-extract")];
            neuroncite_store::bulk_insert_pages(&conn, fid2, &pages_2)
                .expect("bulk_insert_pages file 2");

            (sid, fid1, fid2)
        };

        // Request only the first file by file_id.
        let params = serde_json::json!({
            "session_id": session_id,
            "file_id": file_id_1
        });
        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handle must succeed: {:?}", result.err());

        let response = result.unwrap();
        assert_eq!(
            response["file_count"], 1,
            "file_count must be 1 for single-file detail"
        );

        let files = response["files"]
            .as_array()
            .expect("files must be an array");
        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0]["file_id"].as_i64().expect("file_id"),
            file_id_1,
            "returned file_id must match the requested file_id"
        );
        assert_eq!(files[0]["file_name"], "paper.pdf");
    }

    /// T-MCP-FILES-006: Passing a file_id that belongs to a different session
    /// returns an error. The handler verifies session ownership of the file
    /// before returning its details.
    #[tokio::test]
    async fn t_mcp_files_006_file_id_wrong_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id_a, session_id_b, file_id_in_a) = {
            let conn = pool.get().expect("get conn");
            let config = test_index_config();

            // Session A: contains the file.
            let sid_a = neuroncite_store::create_session(&conn, &config, "0.1.0")
                .expect("create session A");
            let fid = seed_file_with_data(&conn, sid_a);

            // Session B: a separate session with no files. Uses a different
            // directory path to avoid the UNIQUE constraint on session directory.
            let config_b = neuroncite_core::IndexConfig {
                directory: std::path::PathBuf::from("/test/other_dir"),
                model_name: "test-model".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(300),
                chunk_overlap: Some(50),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
                vector_dimension: 384,
            };
            let sid_b = neuroncite_store::create_session(&conn, &config_b, "0.1.0")
                .expect("create session B");

            (sid_a, sid_b, fid)
        };

        // Request file from session A using session B's session_id.
        let params = serde_json::json!({
            "session_id": session_id_b,
            "file_id": file_id_in_a
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handle must fail when file_id belongs to a different session"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("does not belong to session"),
            "error message must indicate session mismatch, got: {err}"
        );

        // Verify the file is accessible from the correct session (session A).
        let params_correct = serde_json::json!({
            "session_id": session_id_a,
            "file_id": file_id_in_a
        });
        let result_correct = handle(&state, &params_correct).await;
        assert!(
            result_correct.is_ok(),
            "handle must succeed when file_id belongs to the specified session: {:?}",
            result_correct.err()
        );
    }
}
