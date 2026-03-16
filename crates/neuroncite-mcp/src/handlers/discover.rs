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

//! Handler for the `neuroncite_discover` MCP tool.
//!
//! Scans a filesystem directory for indexable documents and queries all existing
//! index sessions for that directory. Reports per-session coverage, unindexed
//! files, and aggregate statistics.

use std::collections::HashSet;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;

/// Discovers what is indexed at a given directory path.
///
/// # Parameters (from MCP tool call)
///
/// - `directory` (required): Absolute path to the directory to discover.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let directory = params["directory"]
        .as_str()
        .ok_or("missing required parameter: directory")?;

    // Canonicalize the path for consistent DB comparison. This is a read-only
    // inspection endpoint; when the path does not exist on disk the fallback to
    // the raw input allows the response to report zero indexed files.
    let canonical = neuroncite_core::paths::canonicalize_directory(std::path::Path::new(directory))
        .unwrap_or_else(|_| std::path::PathBuf::from(directory));
    let canonical_str = canonical.to_string_lossy().to_string();

    // Check if the directory exists on disk.
    let dir_exists = canonical.is_dir();

    // Scan for indexable documents on disk (non-recursive, metadata only).
    // Tracks per-type file counts for the response.
    let mut files_on_disk: Vec<(String, u64)> = Vec::new();
    let mut total_size_on_disk: u64 = 0;
    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    if dir_exists && let Ok(entries) = std::fs::read_dir(&canonical) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ft) = neuroncite_pdf::file_type_for_path(&path)
                && let Ok(meta) = entry.metadata()
            {
                let size = meta.len();
                total_size_on_disk += size;
                files_on_disk.push((path.to_string_lossy().into_owned(), size));
                *type_counts.entry(ft.to_string()).or_insert(0) += 1;
            }
        }
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Find all sessions for this directory.
    let sessions = neuroncite_store::find_sessions_by_directory(&conn, &canonical_str)
        .map_err(|e| format!("finding sessions: {e}"))?;

    // Fetch aggregates for all sessions.
    let all_aggs = neuroncite_store::all_session_aggregates(&conn)
        .map_err(|e| format!("fetching aggregates: {e}"))?;
    let agg_map: std::collections::HashMap<i64, neuroncite_store::SessionAggregates> =
        all_aggs.into_iter().map(|a| (a.session_id, a)).collect();

    // Collect all indexed file paths across sessions to find unindexed files.
    let mut all_indexed_paths: HashSet<String> = HashSet::new();

    let mut session_array: Vec<serde_json::Value> = Vec::new();
    for s in &sessions {
        let agg = agg_map.get(&s.id);

        // Fetch indexed files for this session to build the coverage set.
        let files = neuroncite_store::list_files_by_session(&conn, s.id)
            .map_err(|e| format!("listing files for session {}: {e}", s.id))?;

        // Strip the Windows extended-length path prefix (`\\?\`) from indexed
        // file paths before inserting into the comparison set. On Windows,
        // `std::fs::canonicalize()` (called during indexing in `discover_pdfs_flat`)
        // adds the `\\?\` prefix to paths, but `std::fs::read_dir()` entries
        // (used to scan the filesystem above) do NOT include this prefix.
        // Without stripping, the HashSet comparison always fails and every
        // PDF is reported as "unindexed" even when it is fully indexed.
        for f in &files {
            all_indexed_paths.insert(strip_extended_length_prefix(&f.file_path).to_string());
        }

        let total_bytes = agg.map_or(0, |a| a.total_content_bytes);

        // Compute a human-readable label for the active chunking parameter.
        let chunk_param_label = match s.chunk_strategy.as_str() {
            "sentence" => s
                .max_words
                .map(|w| format!("{w} max words per sentence chunk"))
                .unwrap_or_else(|| "sentence (default)".to_string()),
            "page" => "one chunk per page".to_string(),
            strategy => s
                .chunk_size
                .map(|sz| format!("{sz} {strategy}s per chunk"))
                .unwrap_or_else(|| format!("{strategy} (default)")),
        };

        // Parse tags and metadata from stored JSON strings back to JSON
        // values for the response.
        let tags_json = s
            .tags
            .as_deref()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
            .unwrap_or(serde_json::Value::Null);
        let metadata_json = s
            .metadata
            .as_deref()
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .unwrap_or(serde_json::Value::Null);

        session_array.push(serde_json::json!({
            "session_id": s.id,
            "label": s.label,
            "tags": tags_json,
            "metadata": metadata_json,
            "model_name": s.model_name,
            "chunk_strategy": s.chunk_strategy,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "chunk_param_label": chunk_param_label,
            "file_count": agg.map_or(0, |a| a.file_count),
            "total_chunks": agg.map_or(0, |a| a.total_chunks),
            "total_pages": agg.map_or(0, |a| a.total_pages),
            "total_content_bytes": total_bytes,
            "total_words": total_bytes / 6,
            "created_at": s.created_at,
        }));
    }

    // Compute unindexed files: documents on disk that are not in any session.
    // Both sides of the comparison are stripped of the `\\?\` prefix to
    // ensure consistent matching regardless of how paths were obtained.
    let unindexed: Vec<&str> = files_on_disk
        .iter()
        .filter(|(path, _)| {
            let stripped = strip_extended_length_prefix(path);
            !all_indexed_paths.contains(stripped)
        })
        .map(|(path, _)| strip_extended_length_prefix(path))
        .collect();

    let mut response = serde_json::json!({
        "directory": strip_extended_length_prefix(&canonical_str),
        "directory_exists": dir_exists,
        "filesystem": {
            "type_counts": type_counts,
            "total_size_bytes": total_size_on_disk,
        },
        "sessions": session_array,
        "unindexed_files": unindexed,
    });

    // Add a warning when the directory does not exist on disk. This makes
    // the inconsistency with neuroncite_index (which returns an error for
    // non-existent directories) self-documenting: discover is a pre-check
    // tool that returns structured data, while index requires the directory
    // to exist before starting work.
    if !dir_exists {
        response["warning"] = serde_json::json!(format!(
            "directory does not exist: {}. neuroncite_index will reject this path. \
             Verify the path or create the directory before indexing.",
            strip_extended_length_prefix(&canonical_str)
        ));
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates an in-memory SQLite connection pool with all NeuronCite
    /// migrations applied. Foreign key enforcement is enabled via PRAGMA.
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
    /// returns zero-vectors of dimension 384, which is sufficient for
    /// handler tests that do not perform actual embedding operations.
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

    /// T-MCP-DISC-001: Calling the discover handler without the required
    /// `directory` parameter returns an error. The handler validates that
    /// `params["directory"]` is a string; an empty JSON object causes the
    /// indexing operation on `serde_json::Value::Null` to fail `as_str()`.
    #[tokio::test]
    async fn t_mcp_disc_001_missing_directory_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({});
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "calling discover without 'directory' must return an error"
        );
        assert!(
            result.unwrap_err().contains("missing required parameter"),
            "error message must indicate the missing parameter"
        );
    }

    /// T-MCP-DISC-002: Passing a directory path that does not exist on the
    /// filesystem returns a valid JSON response with `directory_exists` set
    /// to `false`. The filesystem, sessions, and unindexed_files arrays are
    /// all empty because no directory scan or session lookup can find results
    /// for a path that does not exist.
    #[tokio::test]
    async fn t_mcp_disc_002_nonexistent_directory_returns_exists_false() {
        let pool = test_pool();
        let state = test_state(pool);

        // Use a path that is extremely unlikely to exist on any system.
        let params = serde_json::json!({
            "directory": "/nonexistent_path_8f3a2b1c_discover_test"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "discover on a non-existent directory must still return Ok with a structured response"
        );

        let response = result.unwrap();

        assert_eq!(
            response["directory_exists"], false,
            "directory_exists must be false for a non-existent path"
        );

        // Filesystem block reports empty type_counts and zero bytes.
        let type_counts = response["filesystem"]["type_counts"]
            .as_object()
            .expect("type_counts must be a JSON object");
        assert!(
            type_counts.is_empty(),
            "type_counts must be empty for a non-existent directory"
        );
        assert_eq!(
            response["filesystem"]["total_size_bytes"], 0,
            "total_size_bytes must be 0 for a non-existent directory"
        );

        // No sessions exist in the database for this fabricated path.
        assert!(
            response["sessions"].as_array().unwrap().is_empty(),
            "sessions array must be empty when no sessions match the directory"
        );

        // No files on disk means the unindexed list is also empty.
        assert!(
            response["unindexed_files"].as_array().unwrap().is_empty(),
            "unindexed_files array must be empty for a non-existent directory"
        );
    }

    /// T-MCP-DISC-003: The response JSON structure contains all five required
    /// top-level fields: `directory`, `directory_exists`, `filesystem`,
    /// `sessions`, and `unindexed_files`. The `filesystem` sub-object contains
    /// `type_counts` and `total_size_bytes`. This test validates the response
    /// schema regardless of whether the directory exists on disk.
    #[tokio::test]
    async fn t_mcp_disc_003_response_has_all_required_fields() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({
            "directory": "/nonexistent_path_schema_check_disc_003"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "handler must return Ok for schema validation"
        );

        let response = result.unwrap();
        let obj = response
            .as_object()
            .expect("response must be a JSON object");

        // Verify all five top-level fields are present.
        assert!(
            obj.contains_key("directory"),
            "response must contain 'directory' field"
        );
        assert!(
            obj.contains_key("directory_exists"),
            "response must contain 'directory_exists' field"
        );
        assert!(
            obj.contains_key("filesystem"),
            "response must contain 'filesystem' field"
        );
        assert!(
            obj.contains_key("sessions"),
            "response must contain 'sessions' field"
        );
        assert!(
            obj.contains_key("unindexed_files"),
            "response must contain 'unindexed_files' field"
        );

        // Verify the filesystem sub-object contains its required fields.
        let fs_obj = response["filesystem"]
            .as_object()
            .expect("'filesystem' must be a JSON object");
        assert!(
            fs_obj.contains_key("type_counts"),
            "filesystem must contain 'type_counts' field"
        );
        assert!(
            fs_obj.contains_key("total_size_bytes"),
            "filesystem must contain 'total_size_bytes' field"
        );

        // Verify type correctness of structural fields.
        assert!(
            response["directory"].is_string(),
            "'directory' must be a string"
        );
        assert!(
            response["directory_exists"].is_boolean(),
            "'directory_exists' must be a boolean"
        );
        assert!(
            response["sessions"].is_array(),
            "'sessions' must be an array"
        );
        assert!(
            response["unindexed_files"].is_array(),
            "'unindexed_files' must be an array"
        );
        assert!(
            response["filesystem"]["type_counts"].is_object(),
            "'type_counts' must be an object"
        );
        assert!(
            response["filesystem"]["total_size_bytes"].is_number(),
            "'total_size_bytes' must be a number"
        );
    }

    /// T-MCP-DISC-004: Files indexed with the Windows extended-length prefix
    /// (`\\?\`) in their database path are correctly matched against filesystem
    /// paths that lack the prefix. This is the regression test for BUG-007
    /// where `neuroncite_discover` reported all PDFs as "unindexed" on Windows
    /// because indexed file paths had the `\\?\` prefix from
    /// `std::fs::canonicalize()` while filesystem paths from `read_dir()` did not.
    ///
    /// The test inserts a file into the database with a `\\?\`-prefixed path
    /// and a corresponding file on disk without the prefix, then verifies that
    /// the discover handler correctly identifies the file as indexed (not in
    /// `unindexed_pdfs`).
    #[tokio::test]
    async fn t_mcp_disc_004_windows_prefix_stripped_for_path_matching() {
        let pool = test_pool();
        let state = test_state(pool);

        // Create a temporary directory with a PDF file on disk. The path is
        // canonicalized to resolve 8.3 short names (e.g., `RUNNER~1`) on Windows
        // CI runners. Without canonicalization, the session's stored directory_path
        // would use the short-name form while the handler's `canonicalize_directory`
        // call would resolve to the full form, causing the SQL exact-match lookup
        // in `find_sessions_by_directory` to fail.
        let tmp = tempfile::tempdir().expect("create temp dir");
        let canonical_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let pdf_path = canonical_dir.join("test_paper.pdf");
        std::fs::write(&pdf_path, b"%PDF-1.0 minimal").expect("write test pdf");

        let dir_str = canonical_dir.to_string_lossy().to_string();

        // Create a session whose directory_path matches the temp directory.
        {
            let conn = state.pool.get().expect("get conn");
            let config = neuroncite_core::IndexConfig {
                directory: canonical_dir.clone(),
                model_name: "test-model".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(256),
                chunk_overlap: Some(32),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
                vector_dimension: 384,
            };
            let session_id =
                neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

            // Insert a file record with the `\\?\` prefix in the path,
            // simulating what `discover_pdfs_flat` stores on Windows when
            // `std::fs::canonicalize()` is called.
            let prefixed_path = format!(r"\\?\{}", pdf_path.to_string_lossy());
            neuroncite_store::insert_file(
                &conn,
                session_id,
                &prefixed_path,
                "abc123",
                1000,
                100,
                1,
                Some(1),
            )
            .expect("insert file");
        }

        let params = serde_json::json!({
            "directory": dir_str,
        });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "discover handler must succeed");

        let response = result.unwrap();

        // The PDF exists on disk.
        assert_eq!(
            response["filesystem"]["type_counts"]["pdf"], 1,
            "type_counts must report 1 PDF on disk"
        );

        // The session must be found.
        assert_eq!(
            response["sessions"].as_array().unwrap().len(),
            1,
            "one session must be found for the directory"
        );

        // The PDF must NOT appear in unindexed_files because the `\\?\` prefix
        // is stripped before comparison. Before BUG-007 fix, this would show 1.
        let unindexed = response["unindexed_files"].as_array().unwrap();
        assert!(
            unindexed.is_empty(),
            "unindexed_files must be empty when the file is indexed \
             (even if the DB path has \\\\?\\\\ prefix). Got: {unindexed:?}"
        );
    }

    /// T-MCP-DISC-005: Files indexed without the `\\?\` prefix are also
    /// correctly detected as indexed. This verifies that the prefix stripping
    /// does not break the common case where paths are stored without prefix.
    #[tokio::test]
    async fn t_mcp_disc_005_non_prefixed_paths_still_match() {
        let pool = test_pool();
        let state = test_state(pool);

        // Canonicalize the temp path to resolve 8.3 short names on Windows CI.
        // See T-MCP-DISC-004 comment for the full explanation.
        let tmp = tempfile::tempdir().expect("create temp dir");
        let canonical_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let pdf_path = canonical_dir.join("normal_paper.pdf");
        std::fs::write(&pdf_path, b"%PDF-1.0 minimal").expect("write test pdf");

        let dir_str = canonical_dir.to_string_lossy().to_string();

        {
            let conn = state.pool.get().expect("get conn");
            let config = neuroncite_core::IndexConfig {
                directory: canonical_dir.clone(),
                model_name: "test-model".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(256),
                chunk_overlap: Some(32),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
                vector_dimension: 384,
            };
            let session_id =
                neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");

            // Insert a file record WITHOUT the `\\?\` prefix (normal case).
            let normal_path = pdf_path.to_string_lossy().to_string();
            neuroncite_store::insert_file(
                &conn,
                session_id,
                &normal_path,
                "def456",
                1000,
                100,
                1,
                Some(1),
            )
            .expect("insert file");
        }

        let params = serde_json::json!({
            "directory": dir_str,
        });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "discover handler must succeed");

        let response = result.unwrap();
        let unindexed = response["unindexed_files"].as_array().unwrap();
        assert!(
            unindexed.is_empty(),
            "unindexed_files must be empty for normally-indexed files. Got: {unindexed:?}"
        );
    }

    /// T-MCP-DISC-006: An unindexed PDF file on disk is correctly reported
    /// in `unindexed_files` when no matching file record exists in any session.
    #[tokio::test]
    async fn t_mcp_disc_006_truly_unindexed_file_reported() {
        let pool = test_pool();
        let state = test_state(pool);

        // Canonicalize the temp path to resolve 8.3 short names on Windows CI.
        // See T-MCP-DISC-004 comment for the full explanation.
        let tmp = tempfile::tempdir().expect("create temp dir");
        let canonical_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let pdf_path = canonical_dir.join("unindexed_paper.pdf");
        std::fs::write(&pdf_path, b"%PDF-1.0 minimal").expect("write test pdf");

        let dir_str = canonical_dir.to_string_lossy().to_string();

        // Create a session but do NOT insert any files.
        {
            let conn = state.pool.get().expect("get conn");
            let config = neuroncite_core::IndexConfig {
                directory: canonical_dir.clone(),
                model_name: "test-model".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(256),
                chunk_overlap: Some(32),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
                vector_dimension: 384,
            };
            neuroncite_store::create_session(&conn, &config, "0.1.0").expect("create session");
        }

        let params = serde_json::json!({
            "directory": dir_str,
        });
        let result = handle(&state, &params).await;

        assert!(result.is_ok(), "discover handler must succeed");

        let response = result.unwrap();
        let unindexed = response["unindexed_files"].as_array().unwrap();
        assert_eq!(
            unindexed.len(),
            1,
            "unindexed_files must contain the one unindexed PDF. Got: {unindexed:?}"
        );
    }

    /// T-MCP-DISC-007: A non-existent directory produces a response with a
    /// `warning` field explaining that the path does not exist and that
    /// neuroncite_index will reject it. This addresses the inconsistency
    /// between discover (returns structured data) and index (returns error)
    /// for non-existent directories by making the behavior self-documenting.
    ///
    /// Regression test for BUG B-03: Before the fix, discover silently
    /// returned directory_exists=false without any explicit guidance, while
    /// index threw an error for the same path, creating a confusing UX.
    #[tokio::test]
    async fn t_mcp_disc_007_nonexistent_directory_has_warning() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({
            "directory": "/nonexistent_path_disc_007_warning_test"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "discover must still return Ok for non-existent directories"
        );

        let response = result.unwrap();
        assert_eq!(
            response["directory_exists"], false,
            "directory_exists must be false"
        );
        assert!(
            response["warning"].is_string(),
            "response must contain a 'warning' field when directory does not exist"
        );
        let warning = response["warning"].as_str().unwrap();
        assert!(
            warning.contains("does not exist"),
            "warning must explain that the directory does not exist, got: {warning}"
        );
        assert!(
            warning.contains("neuroncite_index"),
            "warning must mention that neuroncite_index will reject this path, got: {warning}"
        );
    }

    /// T-MCP-DISC-008: An existing directory produces a response without a
    /// `warning` field. The warning is only added for non-existent directories.
    #[tokio::test]
    async fn t_mcp_disc_008_existing_directory_has_no_warning() {
        let pool = test_pool();
        let state = test_state(pool);

        // Canonicalize for consistent path representation on Windows CI.
        let tmp = tempfile::tempdir().expect("create temp dir");
        let canonical_dir = neuroncite_core::paths::canonicalize_directory(tmp.path())
            .unwrap_or_else(|_| tmp.path().to_path_buf());
        let dir_str = canonical_dir.to_string_lossy().to_string();

        let params = serde_json::json!({
            "directory": dir_str,
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "discover must succeed for existing directories"
        );

        let response = result.unwrap();
        assert_eq!(
            response["directory_exists"], true,
            "directory_exists must be true for an existing directory"
        );
        assert!(
            response.get("warning").is_none() || response["warning"].is_null(),
            "response must not contain a 'warning' field when directory exists"
        );
    }

    /// T-MCP-DISC-009: The warning message includes the canonicalized
    /// directory path (with Windows extended-length prefix stripped) so
    /// the caller can verify which path was checked.
    #[tokio::test]
    async fn t_mcp_disc_009_warning_includes_path() {
        let pool = test_pool();
        let state = test_state(pool);

        let test_path = "/nonexistent_disc_009_path_check";
        let params = serde_json::json!({
            "directory": test_path,
        });
        let result = handle(&state, &params).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        let warning = response["warning"].as_str().unwrap_or("");

        // The warning must contain a recognizable part of the input path.
        assert!(
            warning.contains("nonexistent_disc_009"),
            "warning must reference the directory path, got: {warning}"
        );
    }
}
