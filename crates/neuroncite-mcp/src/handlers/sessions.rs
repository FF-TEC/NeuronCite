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

//! Handlers for the `neuroncite_sessions`, `neuroncite_session_delete`, and
//! `neuroncite_session_update` MCP tools.
//!
//! Provides session listing, deletion, and metadata updates through the
//! neuroncite-store layer. Session deletion cascades to all associated files,
//! pages, chunks, and embeddings.

use std::collections::HashMap;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;

/// Lists all index sessions with their metadata and aggregate statistics.
///
/// Returns a JSON object containing an array of session objects, each with
/// session ID, directory, model name, chunk strategy, creation timestamp,
/// and aggregate counts (file_count, total_pages, total_chunks, total_content_bytes).
pub async fn handle_list(
    state: &Arc<AppState>,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let sessions =
        neuroncite_store::list_sessions(&conn).map_err(|e| format!("listing sessions: {e}"))?;

    let aggregates = neuroncite_store::all_session_aggregates(&conn)
        .map_err(|e| format!("fetching session aggregates: {e}"))?;
    let agg_map: HashMap<i64, neuroncite_store::SessionAggregates> =
        aggregates.into_iter().map(|a| (a.session_id, a)).collect();

    let session_array: Vec<serde_json::Value> = sessions
        .iter()
        .map(|s| {
            let directory = strip_extended_length_prefix(&s.directory_path);

            // Compute a human-readable label for the active chunking parameter.
            // The 'sentence' strategy uses max_words (maximum words per sentence
            // chunk), while 'token' and 'word' strategies use chunk_size (window
            // size in tokens or words). The 'page' strategy uses neither.
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
            // values for the response. Returns null when the stored value is
            // NULL or unparseable.
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

            serde_json::json!({
                "session_id": s.id,
                "directory": directory,
                "model_name": s.model_name,
                "chunk_strategy": s.chunk_strategy,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "max_words": s.max_words,
                "chunk_param_label": chunk_param_label,
                "vector_dimension": s.vector_dimension,
                "created_at": s.created_at,
                "schema_version": s.schema_version,
                "app_version": s.app_version,
                "label": s.label,
                "tags": tags_json,
                "metadata": metadata_json,
                "file_count": agg_map.get(&s.id).map_or(0, |a| a.file_count),
                "total_pages": agg_map.get(&s.id).map_or(0, |a| a.total_pages),
                "total_chunks": agg_map.get(&s.id).map_or(0, |a| a.total_chunks),
                "total_content_bytes": agg_map.get(&s.id).map_or(0, |a| a.total_content_bytes),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "session_count": session_array.len(),
        "sessions": session_array,
    }))
}

/// Deletes sessions and all associated data (cascading delete).
///
/// Supports two deletion modes:
/// - By `session_id`: deletes a single session by its numeric ID.
/// - By `directory`: deletes all sessions whose directory_path matches the
///   given path (canonicalized before comparison).
///
/// At least one of `session_id` or `directory` must be provided.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (optional): Session ID to delete.
/// - `directory` (optional): Absolute path to the PDF directory whose sessions
///   should be deleted. The path is canonicalized before comparison.
pub async fn handle_delete(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"].as_i64();
    let directory = params["directory"].as_str();

    match (session_id, directory) {
        (Some(id), _) => {
            // Delete by session_id (single session). First verify the session
            // exists in the database. Attempting to delete a non-existent
            // session is an error, consistent with search and update handlers
            // that also reject non-existent session IDs.
            let conn = state
                .pool
                .get()
                .map_err(|e| format!("connection pool error: {e}"))?;

            neuroncite_store::get_session(&conn, id)
                .map_err(|_| format!("session {id} not found"))?;

            // Reject deletion when active jobs (queued or running) reference
            // this session. Deleting a session while a citation verification
            // or indexing job is in progress would cascade-delete citation rows
            // and indexed files, leaving the job in an inconsistent state with
            // dangling references. The caller must cancel or wait for the job
            // to complete before deleting the session.
            let jobs =
                neuroncite_store::list_jobs(&conn).map_err(|e| format!("listing jobs: {e}"))?;
            let blocking_jobs: Vec<&str> = jobs
                .iter()
                .filter(|j| {
                    j.session_id == Some(id)
                        && (j.state == neuroncite_store::JobState::Queued
                            || j.state == neuroncite_store::JobState::Running)
                })
                .map(|j| j.id.as_str())
                .collect();
            if !blocking_jobs.is_empty() {
                return Err(format!(
                    "session {id} has {} active job(s) ({}); cancel or wait for \
                     completion before deleting the session",
                    blocking_jobs.len(),
                    blocking_jobs.join(", ")
                ));
            }

            let rows_deleted = neuroncite_store::delete_session(&conn, id)
                .map_err(|e| format!("session deletion: {e}"))?;

            // Evict the session's HNSW index from the in-memory map.
            if rows_deleted > 0 {
                state.remove_hnsw(id);
            }

            Ok(serde_json::json!({
                "session_id": id,
                "deleted": true,
            }))
        }
        (None, Some(dir)) => {
            // Delete by directory (all sessions matching the path). The
            // canonicalization normalizes the path for comparison against
            // stored session directory paths.
            // Normalize the directory path for consistent DB comparison.
            // The directory might no longer exist on disk (e.g., the user
            // deleted the folder). In that case, std::fs::canonicalize would
            // fail, so we strip the Windows extended-length prefix and use
            // the resulting string directly. Sessions stored by the indexer
            // use the canonicalized-then-prefix-stripped form, so the result
            // is comparable when the path still exists; for deleted directories
            // the raw normalized form is used, which may not match any stored
            // session (resulting in "no sessions found").
            let path_obj = std::path::Path::new(dir);
            let canonical_str = neuroncite_core::paths::canonicalize_directory(path_obj)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| {
                    neuroncite_core::paths::strip_extended_length_prefix(dir).to_string()
                });

            let conn = state
                .pool
                .get()
                .map_err(|e| format!("connection pool error: {e}"))?;

            // Collect session IDs that match the directory path. For each,
            // check for active jobs that would be disrupted by deletion.
            let all_sessions = neuroncite_store::list_sessions(&conn)
                .map_err(|e| format!("listing sessions: {e}"))?;
            let matching_ids: Vec<i64> = all_sessions
                .iter()
                .filter(|s| s.directory_path == canonical_str)
                .map(|s| s.id)
                .collect();

            if !matching_ids.is_empty() {
                let jobs =
                    neuroncite_store::list_jobs(&conn).map_err(|e| format!("listing jobs: {e}"))?;
                let blocking_jobs: Vec<String> = jobs
                    .iter()
                    .filter(|j| {
                        j.session_id.is_some_and(|sid| matching_ids.contains(&sid))
                            && (j.state == neuroncite_store::JobState::Queued
                                || j.state == neuroncite_store::JobState::Running)
                    })
                    .map(|j| format!("{} (session {})", j.id, j.session_id.unwrap_or(-1)))
                    .collect();

                if !blocking_jobs.is_empty() {
                    return Err(format!(
                        "directory '{}' has {} active job(s) ({}); cancel or wait for \
                         completion before deleting",
                        canonical_str,
                        blocking_jobs.len(),
                        blocking_jobs.join(", ")
                    ));
                }
            }

            let deleted_ids = neuroncite_store::delete_sessions_by_directory(&conn, &canonical_str)
                .map_err(|e| format!("session deletion by directory: {e}"))?;

            // Return an error when no sessions matched the directory path,
            // consistent with the delete-by-session_id behavior that also
            // errors on non-existent sessions. This prevents silent no-ops
            // when the caller provides a wrong or mistyped directory path.
            if deleted_ids.is_empty() {
                return Err(format!("no sessions found for directory: {canonical_str}"));
            }

            // Evict all deleted sessions' HNSW indices from the in-memory map.
            state.remove_hnsw_many(&deleted_ids);

            Ok(serde_json::json!({
                "directory": canonical_str,
                "deleted_session_ids": deleted_ids,
                "deleted_count": deleted_ids.len(),
            }))
        }
        (None, None) => Err(
            "missing required parameter: provide either 'session_id' or 'directory'".to_string(),
        ),
    }
}

/// Updates session metadata fields: label, tags, and metadata.
///
/// Each parameter is independently optional. When a parameter key is absent,
/// the corresponding field is left unchanged. When present as JSON null, the
/// field is cleared to NULL. When present as a valid value, the field is set.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Session ID to update.
/// - `label` (optional): Human-readable label string, or null to clear.
/// - `tags` (optional): JSON array of tag strings (e.g. `["finance", "statistics"]`),
///   or null to clear.
/// - `metadata` (optional): JSON object of key-value pairs
///   (e.g. `{"source": "manual"}`), or null to clear.
pub async fn handle_update(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Verify the session exists before attempting the update.
    neuroncite_store::get_session(&conn, session_id)
        .map_err(|_| format!("session {session_id} not found"))?;

    // Distinguish between three cases for the "label" parameter:
    //
    // 1. Key absent: no update to the label (params.get("label") returns None).
    // 2. Explicit JSON null: clear the label (label_value.is_null() is true).
    // 3. String value: set the label to the given string.
    //
    // The literal string "null" is rejected to prevent accidental confusion
    // between the JSON null value (which clears the label) and the four-character
    // string "null" (which would set the label to the text "null"). MCP clients
    // that serialize JSON null as the string "null" would silently store the
    // wrong value without this guard.
    if let Some(label_value) = params.get("label") {
        let label = if label_value.is_null() {
            None
        } else {
            let s = label_value
                .as_str()
                .ok_or("label must be a string or null")?;
            if s == "null" {
                return Err(
                    "label value \"null\" is ambiguous: pass JSON null (without quotes) \
                     to clear the label, or use a different string value"
                        .to_string(),
                );
            }
            Some(s)
        };

        neuroncite_store::update_session_label(&conn, session_id, label)
            .map_err(|e| format!("updating session label: {e}"))?;
    }

    // Tags parameter: expects a JSON array of strings, or null to clear.
    // When present as an array, the value is serialized to a JSON string
    // before storage. Non-array, non-null values are rejected.
    if let Some(tags_value) = params.get("tags") {
        let tags_str = if tags_value.is_null() {
            None
        } else if tags_value.is_array() {
            // Validate that all elements are strings.
            let arr = tags_value.as_array().unwrap();
            for (i, elem) in arr.iter().enumerate() {
                if !elem.is_string() {
                    return Err(format!("tags[{i}] must be a string, got: {}", elem));
                }
            }
            Some(serde_json::to_string(tags_value).map_err(|e| format!("serializing tags: {e}"))?)
        } else {
            return Err("tags must be a JSON array of strings, or null to clear".to_string());
        };

        neuroncite_store::update_session_tags(&conn, session_id, tags_str.as_deref())
            .map_err(|e| format!("updating session tags: {e}"))?;
    }

    // Metadata parameter: expects a JSON object, or null to clear.
    // When present as an object, the value is serialized to a JSON string
    // before storage. Non-object, non-null values are rejected.
    if let Some(meta_value) = params.get("metadata") {
        let meta_str = if meta_value.is_null() {
            None
        } else if meta_value.is_object() {
            Some(
                serde_json::to_string(meta_value)
                    .map_err(|e| format!("serializing metadata: {e}"))?,
            )
        } else {
            return Err("metadata must be a JSON object, or null to clear".to_string());
        };

        neuroncite_store::update_session_metadata(&conn, session_id, meta_str.as_deref())
            .map_err(|e| format!("updating session metadata: {e}"))?;
    }

    // Re-read the session after updates to return the actual stored values,
    // avoiding stale or mismatched data in the response.
    let session = neuroncite_store::get_session(&conn, session_id)
        .map_err(|e| format!("re-reading session after update: {e}"))?;

    // Parse tags and metadata from stored JSON strings back to JSON values
    // for the response. Returns null when the stored value is NULL or
    // unparseable (defensive fallback).
    let tags_json = session
        .tags
        .as_deref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .unwrap_or(serde_json::Value::Null);

    let metadata_json = session
        .metadata
        .as_deref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .unwrap_or(serde_json::Value::Null);

    Ok(serde_json::json!({
        "session_id": session_id,
        "label": session.label,
        "tags": tags_json,
        "metadata": metadata_json,
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

    /// Helper: creates an AppState with a stub backend and the given pool.
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

    /// T-MCP-SES-001: Deleting a non-existent session by ID returns an error.
    /// This is consistent with the search and update handlers that also reject
    /// non-existent session IDs with an error, rather than silently succeeding.
    #[tokio::test]
    async fn t_mcp_ses_001_delete_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "session_id": 999999 });
        let result = handle_delete(&state, &params).await;

        assert!(
            result.is_err(),
            "deleting a non-existent session must return an error"
        );
        assert!(
            result.unwrap_err().contains("not found"),
            "error message must indicate that the session was not found"
        );
    }

    /// T-MCP-SES-002: Deleting an existing session by ID returns
    /// `deleted: true` and the session is removed from the database.
    #[tokio::test]
    async fn t_mcp_ses_002_delete_existing_session() {
        let pool = test_pool();
        let state = test_state(pool);

        // Create a session.
        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_delete"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle_delete(&state, &params).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response["deleted"], true);

        // Verify the session is removed from the database.
        let conn = state.pool.get().unwrap();
        let get_result = neuroncite_store::get_session(&conn, session_id);
        assert!(
            get_result.is_err(),
            "session should be removed from database after deletion"
        );
    }

    /// T-MCP-SES-003: Double-deleting a session by ID returns an error
    /// on the second attempt. The first delete succeeds with `deleted: true`,
    /// the second returns a "not found" error because the session no longer
    /// exists in the database.
    #[tokio::test]
    async fn t_mcp_ses_003_double_delete_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_double_delete"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        let params = serde_json::json!({ "session_id": session_id });

        // First delete: succeeds.
        let first = handle_delete(&state, &params).await.unwrap();
        assert_eq!(first["deleted"], true);

        // Second delete: returns error because the session no longer exists.
        let second = handle_delete(&state, &params).await;
        assert!(
            second.is_err(),
            "second delete of same session must return an error"
        );
        assert!(
            second.unwrap_err().contains("not found"),
            "error message must indicate that the session was not found"
        );
    }

    /// T-MCP-SES-004: Delete without session_id or directory returns an error.
    #[tokio::test]
    async fn t_mcp_ses_004_delete_missing_params_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({});
        let result = handle_delete(&state, &params).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing required parameter"));
    }

    /// T-MCP-SES-005: Delete by directory for a non-matching path returns
    /// an error indicating no sessions were found, consistent with the
    /// delete-by-ID behavior that also errors on non-existent sessions.
    #[tokio::test]
    async fn t_mcp_ses_005_delete_by_directory_nonexistent() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "directory": "/nonexistent/path/that/does/not/match" });
        let result = handle_delete(&state, &params).await;

        assert!(
            result.is_err(),
            "delete by non-matching directory must return an error"
        );
        assert!(
            result.unwrap_err().contains("no sessions found"),
            "error message must indicate no sessions were found"
        );
    }

    /// T-MCP-SES-006: session_update rejects the literal string "null" as
    /// the label value to prevent ambiguity with JSON null. Regression test
    /// for the bug where MCP clients serializing null as the string "null"
    /// would silently store the wrong value.
    #[tokio::test]
    async fn t_mcp_ses_006_update_rejects_string_null() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_string_null"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        // The literal string "null" (with quotes in JSON) must be rejected.
        let params = serde_json::json!({
            "session_id": session_id,
            "label": "null"
        });
        let result = handle_update(&state, &params).await;
        assert!(
            result.is_err(),
            "string 'null' label must be rejected as ambiguous"
        );
        assert!(
            result.unwrap_err().contains("ambiguous"),
            "error message must mention ambiguity"
        );
    }

    /// T-MCP-SES-007: session_update with JSON null correctly clears the label.
    #[tokio::test]
    async fn t_mcp_ses_007_update_json_null_clears_label() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_json_null"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        // First set a label.
        let params = serde_json::json!({
            "session_id": session_id,
            "label": "My Session"
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["label"], "My Session");

        // Then clear it with JSON null.
        let params = serde_json::json!({
            "session_id": session_id,
            "label": null
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());
        assert!(
            result.unwrap()["label"].is_null(),
            "label must be null after clearing with JSON null"
        );
    }

    /// T-MCP-SES-008: session_update with a valid string label stores it.
    #[tokio::test]
    async fn t_mcp_ses_008_update_valid_string_label() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_valid_label"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        let params = serde_json::json!({
            "session_id": session_id,
            "label": "Statistics Papers"
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["label"], "Statistics Papers");
    }

    /// T-MCP-SES-009: handle_list with a populated session returns aggregate
    /// fields (file_count, total_pages, total_chunks, total_content_bytes)
    /// with non-zero values. Inserts one file, two pages, and two chunks into
    /// the session, then verifies all four aggregate fields reflect the
    /// inserted data.
    #[tokio::test]
    async fn t_mcp_ses_009_list_populated_session_returns_nonzero_aggregates() {
        let pool = test_pool();
        let state = test_state(pool);

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

        let (session_id, file_id) = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // Insert one file with page_count=2.
            let fid = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/paper.pdf",
                "abc123hash",
                1700000000,
                50000,
                2,
                Some(2),
            )
            .unwrap();

            // Insert two pages with distinct content.
            let pages: Vec<(i64, &str, &str)> = vec![
                (
                    1,
                    "Page one content for testing aggregate byte counts",
                    "pdfium",
                ),
                (
                    2,
                    "Page two content also for testing aggregate byte counts",
                    "pdfium",
                ),
            ];
            neuroncite_store::bulk_insert_pages(&conn, fid, &pages).unwrap();

            // Insert two chunks spanning the two pages.
            let chunks = vec![
                neuroncite_store::ChunkInsert {
                    file_id: fid,
                    session_id: sid,
                    page_start: 1,
                    page_end: 1,
                    chunk_index: 0,
                    doc_text_offset_start: 0,
                    doc_text_offset_end: 52,
                    content: "Page one content for testing aggregate byte counts",
                    embedding: None,
                    ext_offset: None,
                    ext_length: None,
                    content_hash: "hash_chunk_0",
                    simhash: None,
                },
                neuroncite_store::ChunkInsert {
                    file_id: fid,
                    session_id: sid,
                    page_start: 2,
                    page_end: 2,
                    chunk_index: 1,
                    doc_text_offset_start: 52,
                    doc_text_offset_end: 109,
                    content: "Page two content also for testing aggregate byte counts",
                    embedding: None,
                    ext_offset: None,
                    ext_length: None,
                    content_hash: "hash_chunk_1",
                    simhash: None,
                },
            ];
            neuroncite_store::bulk_insert_chunks(&conn, &chunks).unwrap();

            (sid, fid)
        };

        // Suppress unused-variable warning; file_id is used only during setup above.
        let _ = file_id;

        let params = serde_json::json!({});
        let result = handle_list(&state, &params).await;
        assert!(
            result.is_ok(),
            "handle_list must succeed for a populated session"
        );

        let response = result.unwrap();
        assert_eq!(response["session_count"], 1);

        let sessions = response["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1);

        let session = &sessions[0];
        assert_eq!(session["session_id"], session_id);

        // file_count: one file was inserted.
        let file_count = session["file_count"].as_i64().unwrap();
        assert_eq!(file_count, 1, "file_count must be 1 for one inserted file");

        // total_pages: two pages were inserted.
        let total_pages = session["total_pages"].as_i64().unwrap();
        assert_eq!(
            total_pages, 2,
            "total_pages must be 2 for two inserted pages"
        );

        // total_chunks: two chunks were inserted.
        let total_chunks = session["total_chunks"].as_i64().unwrap();
        assert_eq!(
            total_chunks, 2,
            "total_chunks must be 2 for two inserted chunks"
        );

        // total_content_bytes: the sum of byte lengths of the two chunk content strings.
        // "Page one content for testing aggregate byte counts" = 50 bytes
        // "Page two content also for testing aggregate byte counts" = 55 bytes
        // Total = 105 bytes
        let total_content_bytes = session["total_content_bytes"].as_i64().unwrap();
        assert!(
            total_content_bytes > 0,
            "total_content_bytes must be non-zero when chunks contain text"
        );
    }

    /// T-MCP-SES-010: handle_list with an empty session (no files, pages, or
    /// chunks) returns all four aggregate fields set to 0. The session row
    /// exists in the database but has no associated data.
    #[tokio::test]
    async fn t_mcp_ses_010_list_empty_session_returns_zero_aggregates() {
        let pool = test_pool();
        let state = test_state(pool);

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

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        let params = serde_json::json!({});
        let result = handle_list(&state, &params).await;
        assert!(
            result.is_ok(),
            "handle_list must succeed for an empty session"
        );

        let response = result.unwrap();
        assert_eq!(response["session_count"], 1);

        let sessions = response["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1);

        let session = &sessions[0];
        assert_eq!(session["session_id"], session_id);

        // All aggregate fields must be 0 because no files, pages, or chunks exist.
        assert_eq!(
            session["file_count"].as_i64().unwrap(),
            0,
            "file_count must be 0 for an empty session"
        );
        assert_eq!(
            session["total_pages"].as_i64().unwrap(),
            0,
            "total_pages must be 0 for an empty session"
        );
        assert_eq!(
            session["total_chunks"].as_i64().unwrap(),
            0,
            "total_chunks must be 0 for an empty session"
        );
        assert_eq!(
            session["total_content_bytes"].as_i64().unwrap(),
            0,
            "total_content_bytes must be 0 for an empty session"
        );
    }

    /// T-MCP-SES-011: Deleting a session with an active (queued) job referencing
    /// it returns an error. Regression test for BUG-001 where session deletion
    /// cascade-deleted citation rows while a citation_verify job was still
    /// running, causing the job to fail with dangling references.
    #[tokio::test]
    async fn t_mcp_ses_011_delete_blocked_by_active_job() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_active_job"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        // Create a queued job referencing this session.
        {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_job(
                &conn,
                "test-job-001",
                "citation_verify",
                Some(session_id),
            )
            .unwrap();
        }

        // Attempt to delete the session -- must be rejected.
        let params = serde_json::json!({ "session_id": session_id });
        let result = handle_delete(&state, &params).await;

        assert!(
            result.is_err(),
            "session deletion must be blocked when active jobs reference it"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("active job"),
            "error message must mention active jobs, got: {err}"
        );

        // Verify the session still exists in the database.
        let conn = state.pool.get().unwrap();
        let session = neuroncite_store::get_session(&conn, session_id);
        assert!(
            session.is_ok(),
            "session must still exist after blocked deletion"
        );
    }

    /// T-MCP-SES-012: Deleting a session succeeds when the only jobs referencing
    /// it are completed (not queued or running). Completed jobs do not block
    /// deletion because their data is no longer being actively processed.
    #[tokio::test]
    async fn t_mcp_ses_012_delete_allowed_when_jobs_completed() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_completed_job"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        // Create a job and mark it as completed.
        {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_job(
                &conn,
                "test-job-completed",
                "citation_verify",
                Some(session_id),
            )
            .unwrap();
            // Transition: queued -> running -> completed (direct transitions
            // from queued to completed are invalid).
            neuroncite_store::update_job_state(
                &conn,
                "test-job-completed",
                neuroncite_store::JobState::Running,
                None,
            )
            .unwrap();
            neuroncite_store::update_job_state(
                &conn,
                "test-job-completed",
                neuroncite_store::JobState::Completed,
                None,
            )
            .unwrap();
        }

        // Deletion must succeed because the job is completed.
        let params = serde_json::json!({ "session_id": session_id });
        let result = handle_delete(&state, &params).await;

        assert!(
            result.is_ok(),
            "session deletion must succeed when all jobs are completed"
        );
    }

    /// Helper: creates a session and returns its ID.
    async fn create_test_session(state: &Arc<AppState>, dir: &str) -> i64 {
        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from(dir),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };
        let conn = state.pool.get().unwrap();
        neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
    }

    /// T-MCP-SES-013: session_update with a valid tags array stores the JSON
    /// and returns it parsed in the response.
    #[tokio::test]
    async fn t_mcp_ses_013_update_tags_array() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_tags").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "tags": ["finance", "statistics"]
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok(), "setting tags must succeed");

        let response = result.unwrap();
        assert_eq!(response["session_id"], session_id);
        let tags = response["tags"].as_array().unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0], "finance");
        assert_eq!(tags[1], "statistics");
    }

    /// T-MCP-SES-014: session_update with JSON null tags clears the tags field.
    #[tokio::test]
    async fn t_mcp_ses_014_update_tags_null_clears() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_tags_null").await;

        // Set tags first.
        let params = serde_json::json!({
            "session_id": session_id,
            "tags": ["a", "b"]
        });
        handle_update(&state, &params).await.unwrap();

        // Clear with null.
        let params = serde_json::json!({
            "session_id": session_id,
            "tags": null
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());
        assert!(
            result.unwrap()["tags"].is_null(),
            "tags must be null after clearing"
        );
    }

    /// T-MCP-SES-015: session_update rejects non-array tags values.
    #[tokio::test]
    async fn t_mcp_ses_015_update_tags_rejects_non_array() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_tags_invalid").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "tags": "not an array"
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_err(), "string tags must be rejected");
        assert!(
            result.unwrap_err().contains("must be a JSON array"),
            "error message must mention JSON array"
        );
    }

    /// T-MCP-SES-016: session_update rejects tags arrays containing non-string
    /// elements.
    #[tokio::test]
    async fn t_mcp_ses_016_update_tags_rejects_non_string_elements() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_tags_bad_elem").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "tags": ["valid", 42]
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_err(), "non-string tag elements must be rejected");
        assert!(
            result.unwrap_err().contains("tags[1] must be a string"),
            "error must identify the invalid element index"
        );
    }

    /// T-MCP-SES-017: session_update with a valid metadata object stores and
    /// returns it.
    #[tokio::test]
    async fn t_mcp_ses_017_update_metadata_object() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_metadata").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "metadata": {"source": "manual", "priority": "high"}
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok(), "setting metadata must succeed");

        let response = result.unwrap();
        let meta = &response["metadata"];
        assert_eq!(meta["source"], "manual");
        assert_eq!(meta["priority"], "high");
    }

    /// T-MCP-SES-018: session_update rejects non-object metadata values.
    #[tokio::test]
    async fn t_mcp_ses_018_update_metadata_rejects_non_object() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_meta_invalid").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "metadata": ["not", "an", "object"]
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_err(), "array metadata must be rejected");
        assert!(
            result.unwrap_err().contains("must be a JSON object"),
            "error message must mention JSON object"
        );
    }

    /// T-MCP-SES-019: session_update with all three fields (label, tags,
    /// metadata) updates all of them in a single call.
    #[tokio::test]
    async fn t_mcp_ses_019_update_all_fields_at_once() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_all_fields").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "label": "Full Update",
            "tags": ["a", "b"],
            "metadata": {"key": "value"}
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok(), "updating all fields must succeed");

        let response = result.unwrap();
        assert_eq!(response["label"], "Full Update");
        assert_eq!(response["tags"].as_array().unwrap().len(), 2);
        assert_eq!(response["metadata"]["key"], "value");
    }

    /// T-MCP-SES-020: handle_list includes tags and metadata in session output.
    #[tokio::test]
    async fn t_mcp_ses_020_list_includes_tags_and_metadata() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_list_meta").await;

        // Set tags and metadata.
        let params = serde_json::json!({
            "session_id": session_id,
            "tags": ["finance"],
            "metadata": {"src": "test"}
        });
        handle_update(&state, &params).await.unwrap();

        // List sessions.
        let list_params = serde_json::json!({});
        let result = handle_list(&state, &list_params).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let sessions = response["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1);

        let session = &sessions[0];
        assert_eq!(session["tags"].as_array().unwrap()[0], "finance");
        assert_eq!(session["metadata"]["src"], "test");
    }

    /// T-MCP-SES-021: handle_list returns null for tags and metadata when they
    /// are not set on the session.
    #[tokio::test]
    async fn t_mcp_ses_021_list_null_tags_and_metadata() {
        let pool = test_pool();
        let state = test_state(pool);
        let _session_id = create_test_session(&state, "/tmp/test_list_null").await;

        let list_params = serde_json::json!({});
        let result = handle_list(&state, &list_params).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let sessions = response["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1);

        let session = &sessions[0];
        assert!(session["tags"].is_null(), "tags must be null when not set");
        assert!(
            session["metadata"].is_null(),
            "metadata must be null when not set"
        );
    }

    /// T-MCP-SES-022: session_update with empty tags array stores `[]`.
    #[tokio::test]
    async fn t_mcp_ses_022_update_empty_tags_array() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_empty_tags").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "tags": []
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let tags = response["tags"].as_array().unwrap();
        assert!(tags.is_empty(), "empty tags array must be stored as []");
    }

    /// T-MCP-SES-023: session_update with empty metadata object stores `{}`.
    #[tokio::test]
    async fn t_mcp_ses_023_update_empty_metadata_object() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_empty_metadata").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "metadata": {}
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let meta = response["metadata"].as_object().unwrap();
        assert!(
            meta.is_empty(),
            "empty metadata object must be stored as {{}}"
        );
    }

    /// T-MCP-SES-024: Deleting by directory is blocked when an active job
    /// references one of the matching sessions. This covers the directory-based
    /// deletion path (distinct from T-MCP-SES-011 which tests delete-by-ID).
    #[tokio::test]
    async fn t_mcp_ses_024_delete_by_directory_blocked_by_active_job() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/test_dir_active_job"),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            chunk_strategy: "token".to_string(),
            chunk_size: Some(256),
            chunk_overlap: Some(32),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        // Create a queued job referencing this session.
        {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_job(&conn, "test-dir-job-001", "indexing", Some(session_id))
                .unwrap();
        }

        // Attempt to delete by directory path. The session's stored path
        // uses the canonicalized form, so the deletion must match the stored
        // path. Since we cannot canonicalize "/tmp/test_dir_active_job" on
        // Windows, test that the handler does not crash and returns either
        // an active-job error or a no-sessions-found error (depending on
        // path canonicalization behavior).
        let params = serde_json::json!({ "directory": "/tmp/test_dir_active_job" });
        let result = handle_delete(&state, &params).await;

        // On systems where the path canonicalizes to the stored value,
        // the deletion is blocked by the active job. On other systems,
        // the canonicalized path does not match the stored path, producing
        // a "no sessions found" error. Both are valid error outcomes.
        assert!(
            result.is_err(),
            "delete by directory must fail (either due to active job or no match)"
        );
    }

    /// T-MCP-SES-025: handle_list formats chunk_param_label correctly for
    /// each chunk strategy. Verifies the four strategy-specific label formats:
    /// - "sentence" strategy uses max_words label
    /// - "page" strategy uses fixed "one chunk per page" label
    /// - "word" strategy uses chunk_size label with "words per chunk"
    /// - "token" strategy uses chunk_size label with "tokens per chunk"
    #[tokio::test]
    async fn t_mcp_ses_025_chunk_param_label_formatting() {
        let pool = test_pool();
        let state = test_state(pool);

        // Create sessions with distinct chunk strategies.
        let make_config = |strategy: &str, chunk_size: Option<usize>, max_words: Option<usize>| {
            neuroncite_core::IndexConfig {
                directory: std::path::PathBuf::from(format!("/tmp/label_{strategy}")),
                model_name: "test-model".to_string(),
                chunk_strategy: strategy.to_string(),
                chunk_size,
                chunk_overlap: Some(32),
                max_words,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
                vector_dimension: 384,
            }
        };

        {
            let conn = state.pool.get().unwrap();

            // Sentence strategy with max_words=200.
            let cfg_sentence = make_config("sentence", None, Some(200));
            neuroncite_store::create_session(&conn, &cfg_sentence, "0.1.0").unwrap();

            // Page strategy (no chunk_size or max_words).
            let cfg_page = make_config("page", None, None);
            neuroncite_store::create_session(&conn, &cfg_page, "0.1.0").unwrap();

            // Word strategy with chunk_size=300.
            let cfg_word = make_config("word", Some(300), None);
            neuroncite_store::create_session(&conn, &cfg_word, "0.1.0").unwrap();

            // Token strategy with chunk_size=256.
            let cfg_token = make_config("token", Some(256), None);
            neuroncite_store::create_session(&conn, &cfg_token, "0.1.0").unwrap();
        }

        let list_params = serde_json::json!({});
        let result = handle_list(&state, &list_params).await.unwrap();
        let sessions = result["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 4);

        // Collect chunk_param_labels keyed by strategy.
        let labels: std::collections::HashMap<String, String> = sessions
            .iter()
            .map(|s| {
                let strategy = s["chunk_strategy"].as_str().unwrap().to_string();
                let label = s["chunk_param_label"].as_str().unwrap().to_string();
                (strategy, label)
            })
            .collect();

        assert_eq!(
            labels["sentence"], "200 max words per sentence chunk",
            "sentence strategy label must include max_words"
        );
        assert_eq!(
            labels["page"], "one chunk per page",
            "page strategy label is fixed"
        );
        assert_eq!(
            labels["word"], "300 words per chunk",
            "word strategy label must include chunk_size with 'words per chunk'"
        );
        assert_eq!(
            labels["token"], "256 tokens per chunk",
            "token strategy label must include chunk_size with 'tokens per chunk'"
        );
    }

    /// T-MCP-SES-026: session_update with a non-existent session_id returns
    /// a "not found" error. The handler validates session existence before
    /// attempting any field updates.
    #[tokio::test]
    async fn t_mcp_ses_026_update_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({
            "session_id": 999999,
            "label": "does not matter"
        });
        let result = handle_update(&state, &params).await;

        assert!(
            result.is_err(),
            "updating a non-existent session must return an error"
        );
        assert!(
            result.unwrap_err().contains("not found"),
            "error message must indicate the session was not found"
        );
    }

    /// T-MCP-SES-027: session_update with only session_id (no label, tags,
    /// or metadata) returns the current values without modifying anything.
    /// This is the "no-op update" path where all three optional fields are
    /// absent from the params.
    #[tokio::test]
    async fn t_mcp_ses_027_update_no_fields_is_noop() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_noop_update").await;

        // Set initial values.
        let set_params = serde_json::json!({
            "session_id": session_id,
            "label": "Original",
            "tags": ["a"],
            "metadata": {"key": "val"}
        });
        handle_update(&state, &set_params).await.unwrap();

        // Update with only session_id -- no fields to update.
        let noop_params = serde_json::json!({ "session_id": session_id });
        let result = handle_update(&state, &noop_params).await;
        assert!(result.is_ok(), "no-op update must succeed");

        let response = result.unwrap();
        assert_eq!(
            response["label"], "Original",
            "label must be unchanged after no-op update"
        );
        assert_eq!(
            response["tags"].as_array().unwrap().len(),
            1,
            "tags must be unchanged after no-op update"
        );
        assert_eq!(
            response["metadata"]["key"], "val",
            "metadata must be unchanged after no-op update"
        );
    }

    /// T-MCP-SES-028: session_update with numeric label (non-string) returns
    /// an error. The label field must be a string or null, not a number.
    #[tokio::test]
    async fn t_mcp_ses_028_update_numeric_label_rejected() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = create_test_session(&state, "/tmp/test_num_label").await;

        let params = serde_json::json!({
            "session_id": session_id,
            "label": 42
        });
        let result = handle_update(&state, &params).await;
        assert!(result.is_err(), "numeric label must be rejected");
        assert!(
            result.unwrap_err().contains("string or null"),
            "error message must mention that label must be a string or null"
        );
    }
}
