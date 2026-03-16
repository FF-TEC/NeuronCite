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

//! Handler for the `neuroncite_chunks` MCP tool.
//!
//! Provides direct chunk browsing for a specific file with pagination and
//! optional page-number filtering. Returns chunk content, page range, and
//! word count without requiring a search query.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Browses chunks of a file with pagination.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Session containing the file.
/// - `file_id` (required): File to browse chunks for.
/// - `page_number` (optional): Filter to chunks spanning this page (1-indexed).
/// - `offset` (optional): Number of chunks to skip (default: 0).
/// - `limit` (optional): Maximum chunks to return (default: 20, max: 100).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;
    let file_id = params["file_id"]
        .as_i64()
        .ok_or("missing required parameter: file_id")?;
    let page_filter = params["page_number"].as_i64();
    let offset = params["offset"].as_i64().unwrap_or(0).max(0);
    let limit = params["limit"].as_i64().unwrap_or(20).clamp(1, 100);

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Verify the session exists.
    neuroncite_store::get_session(&conn, session_id)
        .map_err(|_| format!("session {session_id} not found"))?;

    // Verify the file exists and belongs to the session.
    let file = neuroncite_store::get_file(&conn, file_id)
        .map_err(|_| format!("file {file_id} not found"))?;
    if file.session_id != session_id {
        return Err(format!(
            "file {file_id} does not belong to session {session_id}"
        ));
    }

    let (chunks, total) =
        neuroncite_store::browse_chunks(&conn, file_id, page_filter, offset, limit)
            .map_err(|e| format!("browsing chunks: {e}"))?;

    let chunk_array: Vec<serde_json::Value> = chunks
        .iter()
        .map(|c| {
            let word_count = c.content.split_whitespace().count();
            serde_json::json!({
                "chunk_id": c.id,
                "chunk_index": c.chunk_index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "word_count": word_count,
                "byte_count": c.content.len(),
                "content": c.content,
            })
        })
        .collect();

    let file_name = std::path::Path::new(&file.file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    Ok(serde_json::json!({
        "session_id": session_id,
        "file_id": file_id,
        "file_name": file_name,
        "total_chunks": total,
        "offset": offset,
        "limit": limit,
        "returned": chunk_array.len(),
        "chunks": chunk_array,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates an in-memory SQLite pool with migrations applied.
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

    /// Helper: creates an AppState with a stub embedding backend and the given pool.
    fn test_state(pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>) -> Arc<AppState> {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        /// Minimal embedding backend that returns zero vectors. Used exclusively
        /// in tests where embedding values are irrelevant.
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

    /// Helper: creates a session, a file, pages, and the specified number of chunks.
    /// Returns (session_id, file_id).
    ///
    /// Each chunk spans a single page (page_start == page_end == chunk_index + 1),
    /// which allows page_number filtering tests to select specific chunks. The
    /// content of chunk i is "word_a word_b word_c chunk_{i}" so word counts and
    /// byte counts are deterministic.
    fn seed_session_with_chunks(
        pool: &r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
        chunk_count: usize,
    ) -> (i64, i64) {
        let conn = pool.get().expect("get conn");

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/test/dir"),
            model_name: "test-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        // Insert one file record associated with this session.
        let file_id = neuroncite_store::insert_file(
            &conn,
            session_id,
            "/test/dir/paper.pdf",
            "abc123hash",
            1_700_000_000,
            50_000,
            chunk_count as i64,
            Some(chunk_count as i64),
        )
        .expect("insert file");

        // Insert one page per chunk so page_number filtering is testable.
        let pages: Vec<(i64, &str, &str)> = (0..chunk_count)
            .map(|i| ((i + 1) as i64, "page content placeholder", "stub"))
            .collect();
        neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).expect("insert pages");

        // Build chunk content strings with deterministic word counts.
        let contents: Vec<String> = (0..chunk_count)
            .map(|i| format!("word_a word_b word_c chunk_{i}"))
            .collect();

        let chunks: Vec<neuroncite_store::ChunkInsert<'_>> = (0..chunk_count)
            .map(|i| {
                let page = (i + 1) as i64;
                neuroncite_store::ChunkInsert {
                    file_id,
                    session_id,
                    page_start: page,
                    page_end: page,
                    chunk_index: i as i64,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: contents[i].len() as i64,
                    content: &contents[i],
                    embedding: None,
                    ext_offset: None,
                    ext_length: None,
                    content_hash: "deadbeef",
                    simhash: None,
                }
            })
            .collect();

        neuroncite_store::bulk_insert_chunks(&conn, &chunks).expect("insert chunks");

        (session_id, file_id)
    }

    /// T-MCP-CHUNK-001: Calling the handler without session_id returns an error
    /// because session_id is a required parameter.
    #[tokio::test]
    async fn t_mcp_chunk_001_missing_session_id_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "file_id": 1 });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handler must return Err when session_id is absent"
        );
        assert!(
            result.unwrap_err().contains("session_id"),
            "error message must reference session_id"
        );
    }

    /// T-MCP-CHUNK-002: Calling the handler without file_id returns an error
    /// because file_id is a required parameter.
    #[tokio::test]
    async fn t_mcp_chunk_002_missing_file_id_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "session_id": 1 });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handler must return Err when file_id is absent"
        );
        assert!(
            result.unwrap_err().contains("file_id"),
            "error message must reference file_id"
        );
    }

    /// T-MCP-CHUNK-003: Providing a session_id that does not exist in the database
    /// returns an error. The handler validates session existence before querying chunks.
    #[tokio::test]
    async fn t_mcp_chunk_003_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "session_id": 999999, "file_id": 1 });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handler must return Err for a non-existent session"
        );
        assert!(
            result.unwrap_err().contains("not found"),
            "error message must indicate the session was not found"
        );
    }

    /// T-MCP-CHUNK-004: Providing a file_id that belongs to a different session
    /// returns an error. The handler checks file.session_id == requested session_id.
    #[tokio::test]
    async fn t_mcp_chunk_004_file_not_belonging_to_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        // Seed two sessions: chunk data exists only in the first.
        let (session_a, file_a) = seed_session_with_chunks(&state.pool, 1);

        // Create a second session with no files.
        let conn = state.pool.get().unwrap();
        let config_b = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/test/other"),
            model_name: "test-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let session_b =
            neuroncite_store::create_session(&conn, &config_b, "0.1.0").expect("create session B");
        drop(conn);

        // Request file_a under session_b -- should be rejected.
        let params = serde_json::json!({
            "session_id": session_b,
            "file_id": file_a,
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handler must reject a file that belongs to a different session"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("does not belong"),
            "error message must indicate the file does not belong to the session, got: {err}"
        );

        // Sanity check: the file is accessible under its actual session.
        let params_ok = serde_json::json!({
            "session_id": session_a,
            "file_id": file_a,
        });
        let result_ok = handle(&state, &params_ok).await;
        assert!(
            result_ok.is_ok(),
            "file must be accessible under its own session"
        );
    }

    /// T-MCP-CHUNK-005: Browsing chunks returns correct pagination metadata.
    /// Creates a session with 5 chunks, requests offset=0 limit=3, and verifies
    /// that `returned` is 3 and `total_chunks` is 5.
    #[tokio::test]
    async fn t_mcp_chunk_005_pagination_returns_correct_counts() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id, file_id) = seed_session_with_chunks(&state.pool, 5);

        let params = serde_json::json!({
            "session_id": session_id,
            "file_id": file_id,
            "offset": 0,
            "limit": 3,
        });
        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handler must succeed for valid parameters");

        let response = result.unwrap();
        assert_eq!(
            response["returned"], 3,
            "returned count must equal the requested limit when enough chunks exist"
        );
        assert_eq!(
            response["total_chunks"], 5,
            "total_chunks must reflect the full count regardless of limit"
        );
        assert_eq!(
            response["offset"], 0,
            "offset in response must match the requested offset"
        );
        assert_eq!(
            response["limit"], 3,
            "limit in response must match the requested limit"
        );

        // Verify the chunks array length matches the returned count.
        let chunks = response["chunks"]
            .as_array()
            .expect("chunks must be an array");
        assert_eq!(
            chunks.len(),
            3,
            "chunks array length must equal the returned count"
        );
    }

    /// T-MCP-CHUNK-006: The page_number filter narrows results to chunks whose
    /// page range includes the specified page. With each chunk spanning exactly
    /// one page (page_start == page_end == chunk_index + 1), filtering by
    /// page_number=3 returns only the chunk on page 3.
    #[tokio::test]
    async fn t_mcp_chunk_006_page_filter_narrows_results() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id, file_id) = seed_session_with_chunks(&state.pool, 5);

        let params = serde_json::json!({
            "session_id": session_id,
            "file_id": file_id,
            "page_number": 3,
        });
        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "handler must succeed with page_number filter"
        );

        let response = result.unwrap();
        assert_eq!(
            response["total_chunks"], 1,
            "only one chunk spans page 3 in the test data"
        );
        assert_eq!(
            response["returned"], 1,
            "returned count must equal 1 for single-page filter match"
        );

        let chunks = response["chunks"]
            .as_array()
            .expect("chunks must be an array");
        assert_eq!(chunks.len(), 1, "exactly one chunk must match the filter");
        assert_eq!(
            chunks[0]["page_start"], 3,
            "the matching chunk must start on the filtered page"
        );
        assert_eq!(
            chunks[0]["page_end"], 3,
            "the matching chunk must end on the filtered page"
        );
    }

    /// T-MCP-CHUNK-007: Each chunk object in the response contains word_count and
    /// byte_count fields. The test content "word_a word_b word_c chunk_{i}" has
    /// exactly 4 whitespace-separated words. byte_count equals the UTF-8 byte
    /// length of the content string.
    #[tokio::test]
    async fn t_mcp_chunk_007_chunks_contain_word_count_and_byte_count() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let (session_id, file_id) = seed_session_with_chunks(&state.pool, 3);

        let params = serde_json::json!({
            "session_id": session_id,
            "file_id": file_id,
        });
        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "handler must succeed for valid parameters");

        let response = result.unwrap();
        let chunks = response["chunks"]
            .as_array()
            .expect("chunks must be an array");
        assert_eq!(chunks.len(), 3, "all 3 chunks must be returned");

        for (i, chunk) in chunks.iter().enumerate() {
            // Verify word_count field exists and is a number.
            assert!(
                chunk["word_count"].is_number(),
                "chunk {i} must have a numeric word_count field"
            );
            // The seeded content "word_a word_b word_c chunk_{i}" has 4 words.
            assert_eq!(
                chunk["word_count"], 4,
                "chunk {i} word_count must be 4 for the seeded content"
            );

            // Verify byte_count field exists and is a number.
            assert!(
                chunk["byte_count"].is_number(),
                "chunk {i} must have a numeric byte_count field"
            );

            // Compute expected byte count from the known content pattern.
            let expected_content = format!("word_a word_b word_c chunk_{i}");
            let expected_bytes = expected_content.len() as i64;
            assert_eq!(
                chunk["byte_count"], expected_bytes,
                "chunk {i} byte_count must equal the UTF-8 byte length of its content"
            );
        }
    }
}
