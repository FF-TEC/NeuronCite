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

//! Handler for the `neuroncite_file_compare` MCP tool.
//!
//! Compares the same PDF across different index sessions, showing per-session
//! statistics (pages, chunks, text volume, chunk strategy). Enables evaluation
//! of which chunking configuration produces the best coverage.

use std::collections::HashMap;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;

/// Compares a file across sessions by path or name pattern.
///
/// # Parameters (from MCP tool call)
///
/// - `file_path` (optional): Exact file path to match.
/// - `file_name_pattern` (optional): SQL LIKE pattern to match file paths.
///
/// At least one of `file_path` or `file_name_pattern` must be provided.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let file_path = params["file_path"].as_str();
    let file_name_pattern = params["file_name_pattern"].as_str();

    let pattern = match (file_path, file_name_pattern) {
        (Some(path), _) => format!("%{path}"),
        (None, Some(pat)) => pat.to_string(),
        (None, None) => {
            return Err(
                "missing required parameter: provide either 'file_path' or 'file_name_pattern'"
                    .to_string(),
            );
        }
    };

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let files = neuroncite_store::find_files_by_path_pattern(&conn, &pattern)
        .map_err(|e| format!("searching files: {e}"))?;

    if files.is_empty() {
        return Ok(serde_json::json!({
            "pattern": pattern,
            "matched_files": 0,
            "comparisons": [],
        }));
    }

    // Group files by their canonical file name (last path component).
    let mut groups: HashMap<String, Vec<&neuroncite_store::FileRow>> = HashMap::new();
    for f in &files {
        let name = std::path::Path::new(strip_extended_length_prefix(&f.file_path))
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        groups.entry(name).or_default().push(f);
    }

    // Cache session metadata to avoid repeated queries.
    let mut session_cache: HashMap<i64, neuroncite_store::SessionRow> = HashMap::new();

    let mut comparisons: Vec<serde_json::Value> = Vec::new();

    for (file_name, file_group) in &groups {
        let mut sessions_data: Vec<serde_json::Value> = Vec::new();
        let mut chunk_counts: Vec<i64> = Vec::new();
        let mut hashes: Vec<&str> = Vec::new();

        for f in file_group {
            // Look up session metadata (cached after first lookup).
            if let std::collections::hash_map::Entry::Vacant(e) = session_cache.entry(f.session_id)
                && let Ok(session) = neuroncite_store::get_session(&conn, f.session_id)
            {
                e.insert(session);
            }

            let cs = neuroncite_store::single_file_chunk_stats(&conn, f.id)
                .map_err(|e| format!("chunk stats for file {}: {e}", f.id))?;

            chunk_counts.push(cs.chunk_count);
            hashes.push(&f.file_hash);

            let session_info = session_cache.get(&f.session_id);

            // Compute a human-readable label for the active chunking parameter.
            let chunk_param_label = session_info.map(|si| match si.chunk_strategy.as_str() {
                "sentence" => si
                    .max_words
                    .map(|w| format!("{w} max words per sentence chunk"))
                    .unwrap_or_else(|| "sentence (default)".to_string()),
                "page" => "one chunk per page".to_string(),
                strategy => si
                    .chunk_size
                    .map(|sz| format!("{sz} {strategy}s per chunk"))
                    .unwrap_or_else(|| format!("{strategy} (default)")),
            });

            sessions_data.push(serde_json::json!({
                "session_id": f.session_id,
                "file_id": f.id,
                "label": session_info.and_then(|s| s.label.as_deref()),
                "model_name": session_info.map(|s| s.model_name.as_str()).unwrap_or(""),
                "chunk_strategy": session_info.map(|s| s.chunk_strategy.as_str()).unwrap_or(""),
                "chunk_size": session_info.and_then(|s| s.chunk_size),
                "chunk_param_label": chunk_param_label,
                "pages_extracted": f.page_count,
                "pdf_page_count": f.pdf_page_count,
                "chunks": cs.chunk_count,
                "avg_chunk_bytes": cs.avg_content_len,
                "file_hash": f.file_hash,
            }));
        }

        // Determine if all instances have the same content hash.
        let same_content = hashes.windows(2).all(|w| w[0] == w[1]);

        let min_chunks = chunk_counts.iter().copied().min().unwrap_or(0);
        let max_chunks = chunk_counts.iter().copied().max().unwrap_or(0);

        comparisons.push(serde_json::json!({
            "file_name": file_name,
            "instances": file_group.len(),
            "sessions": sessions_data,
            "same_content": same_content,
            "chunk_count_range": [min_chunks, max_chunks],
        }));
    }

    Ok(serde_json::json!({
        "pattern": pattern,
        "matched_files": groups.len(),
        "comparisons": comparisons,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2d2_sqlite::rusqlite;

    /// Creates an in-memory SQLite connection pool with foreign keys enabled
    /// and the neuroncite schema migrated. The pool has a max_size of 2,
    /// sufficient for single-threaded test execution.
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

    /// Creates an AppState backed by a stub embedding backend. The stub
    /// returns zero-vectors of dimension 384 for any input text. This
    /// satisfies the AppState constructor without requiring a real model.
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

    /// Returns an IndexConfig suitable for creating test sessions. The
    /// directory, model name, and chunk strategy are arbitrary test values.
    fn make_config(directory: &str) -> neuroncite_core::IndexConfig {
        neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from(directory),
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

    /// Inserts a file record and a single chunk for that file. Returns the
    /// file ID. The chunk is required because `single_file_chunk_stats`
    /// aggregates over the chunk table; without at least one chunk row the
    /// statistics would be zero but the query still succeeds.
    fn insert_file_with_chunk(
        conn: &rusqlite::Connection,
        session_id: i64,
        file_path: &str,
        file_hash: &str,
    ) -> i64 {
        let file_id = neuroncite_store::insert_file(
            conn,
            session_id,
            file_path,
            file_hash,
            1_000_000, // mtime
            5000,      // size in bytes
            10,        // page_count (extracted pages)
            Some(10),  // pdf_page_count (structural pages)
        )
        .expect("insert_file");

        let chunk = neuroncite_store::ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 50,
            content: "test content for compare handler",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: &format!("chunkhash_{file_id}"),
            simhash: None,
        };
        neuroncite_store::bulk_insert_chunks(conn, &[chunk]).expect("bulk_insert_chunks");

        file_id
    }

    /// T-MCP-CMP-001: Calling handle without file_path or file_name_pattern
    /// returns an error. The handler requires at least one of these two
    /// parameters to construct a LIKE pattern for the database query.
    #[tokio::test]
    async fn t_mcp_cmp_001_missing_both_params_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({});
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "handle must return Err when both file_path and file_name_pattern are absent"
        );
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("missing required parameter"),
            "error message must mention the missing parameter requirement, got: {err_msg}"
        );
    }

    /// T-MCP-CMP-002: When no files in the database match the given pattern,
    /// the handler returns matched_files=0 and an empty comparisons array.
    /// This verifies the early-return path for zero matches.
    #[tokio::test]
    async fn t_mcp_cmp_002_no_matching_files_returns_empty() {
        let pool = test_pool();
        let state = test_state(pool);

        // Query a file path that does not exist in the empty database.
        let params = serde_json::json!({ "file_path": "/nonexistent/paper.pdf" });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "handle must succeed even with zero matches");
        let response = result.unwrap();
        assert_eq!(
            response["matched_files"], 0,
            "matched_files must be 0 when no files match the pattern"
        );
        let comparisons = response["comparisons"]
            .as_array()
            .expect("comparisons must be a JSON array");
        assert!(
            comparisons.is_empty(),
            "comparisons array must be empty when no files match"
        );
    }

    /// T-MCP-CMP-003: The same file (identical hash) indexed in two separate
    /// sessions results in same_content=true in the comparison output. Both
    /// sessions contain the same file path and content hash, so the handler
    /// correctly identifies the content as identical across sessions.
    #[tokio::test]
    async fn t_mcp_cmp_003_same_hash_returns_same_content_true() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let conn = pool.get().expect("get conn");

        let config_a = make_config("/test/dir_a");
        let config_b = make_config("/test/dir_b");
        let session_a =
            neuroncite_store::create_session(&conn, &config_a, "0.1.0").expect("create session A");
        let session_b =
            neuroncite_store::create_session(&conn, &config_b, "0.1.0").expect("create session B");

        // Both sessions index the same file with the same content hash.
        let shared_hash = "abc123deadbeef";
        let file_path = "/test/dir/paper.pdf";
        insert_file_with_chunk(&conn, session_a, file_path, shared_hash);
        insert_file_with_chunk(&conn, session_b, file_path, shared_hash);

        drop(conn);

        let params = serde_json::json!({ "file_path": "paper.pdf" });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "handle must succeed with matching files");
        let response = result.unwrap();
        assert_eq!(
            response["matched_files"], 1,
            "one distinct file name should be matched across both sessions"
        );

        let comparisons = response["comparisons"]
            .as_array()
            .expect("comparisons must be a JSON array");
        assert_eq!(comparisons.len(), 1, "one comparison group expected");

        let comparison = &comparisons[0];
        assert_eq!(
            comparison["same_content"], true,
            "same_content must be true when both sessions have the same file hash"
        );
        assert_eq!(
            comparison["instances"], 2,
            "two file instances expected (one per session)"
        );
    }

    /// T-MCP-CMP-004: The same file name indexed in two sessions with
    /// different content hashes results in same_content=false. This
    /// scenario occurs when a PDF is re-scanned or modified between
    /// indexing sessions.
    #[tokio::test]
    async fn t_mcp_cmp_004_different_hash_returns_same_content_false() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let conn = pool.get().expect("get conn");

        let config_a = make_config("/test/dir_a");
        let config_b = make_config("/test/dir_b");
        let session_a =
            neuroncite_store::create_session(&conn, &config_a, "0.1.0").expect("create session A");
        let session_b =
            neuroncite_store::create_session(&conn, &config_b, "0.1.0").expect("create session B");

        // Same file path, but different content hashes across sessions.
        let file_path = "/test/dir/report.pdf";
        insert_file_with_chunk(&conn, session_a, file_path, "hash_version_1");
        insert_file_with_chunk(&conn, session_b, file_path, "hash_version_2");

        drop(conn);

        let params = serde_json::json!({ "file_path": "report.pdf" });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "handle must succeed with matching files");
        let response = result.unwrap();

        let comparisons = response["comparisons"]
            .as_array()
            .expect("comparisons must be a JSON array");
        assert_eq!(comparisons.len(), 1, "one comparison group expected");

        let comparison = &comparisons[0];
        assert_eq!(
            comparison["same_content"], false,
            "same_content must be false when file hashes differ across sessions"
        );
        assert_eq!(
            comparison["instances"], 2,
            "two file instances expected (one per session)"
        );
    }

    /// T-MCP-CMP-005: The file_name_pattern parameter supports SQL LIKE
    /// wildcards. A pattern with `%` matches files whose paths contain the
    /// given substring. This verifies that the pattern is forwarded to
    /// `find_files_by_path_pattern` without additional wrapping.
    #[tokio::test]
    async fn t_mcp_cmp_005_wildcard_pattern_matches_file() {
        let pool = test_pool();
        let state = test_state(pool.clone());

        let conn = pool.get().expect("get conn");

        let config = make_config("/test/wildcard_dir");
        let session_id =
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

        let file_path = "/test/wildcard_dir/statistics_intro.pdf";
        insert_file_with_chunk(&conn, session_id, file_path, "wildcard_hash_abc");

        drop(conn);

        // Use a LIKE wildcard pattern that matches the file name substring.
        let params = serde_json::json!({ "file_name_pattern": "%statistics%" });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "handle must succeed when wildcard pattern matches"
        );
        let response = result.unwrap();
        assert_eq!(
            response["matched_files"], 1,
            "wildcard pattern must match the inserted file"
        );

        let comparisons = response["comparisons"]
            .as_array()
            .expect("comparisons must be a JSON array");
        assert_eq!(comparisons.len(), 1, "one comparison group expected");
        assert_eq!(
            comparisons[0]["file_name"], "statistics_intro.pdf",
            "file_name must be the last path component of the matched file"
        );
    }
}
