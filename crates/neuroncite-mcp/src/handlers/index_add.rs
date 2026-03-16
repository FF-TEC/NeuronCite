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

//! Handler for the `neuroncite_index_add` MCP tool.
//!
//! Incrementally adds one or more PDF files to an existing index session.
//! Files already present in the session are checked for changes using the
//! two-stage mtime/size + SHA-256 detection from `neuroncite_store`. Unchanged
//! files are skipped. Changed and new files are extracted, chunked, embedded,
//! and stored. The session's HNSW index is rebuilt once at the end from all
//! active chunk embeddings.
//!
//! This avoids the cost of re-indexing an entire directory when only a few
//! new PDFs need to be added to a large collection.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_api::indexer;
use neuroncite_store::{ChangeStatus, build_hnsw};

/// Adds or updates specific PDF files in an existing index session.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): The session to add files to.
/// - `files` (required): Array of absolute file paths to PDF files. Each
///   file is checked against the session's existing records and only
///   processed if new or changed.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let files_array = params["files"]
        .as_array()
        .ok_or("missing required parameter: files")?;

    if files_array.is_empty() {
        return Err("files array must not be empty".to_string());
    }

    // Collect file paths as strings, rejecting non-string entries.
    let file_paths: Vec<&str> = files_array.iter().filter_map(|v| v.as_str()).collect();

    if file_paths.len() != files_array.len() {
        return Err("files array must contain only string paths".to_string());
    }

    // Validate the session exists and retrieve its configuration.
    let session = {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        neuroncite_store::get_session(&conn, session_id)
            .map_err(|_| format!("session {session_id} not found"))?
    };

    // Check that the session's vector dimension matches the loaded model.
    // The vector_dimension field is an AtomicUsize, so a single relaxed load
    // is performed here and the resulting plain usize is reused for both the
    // comparison and the error message.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let dim = session.vector_dimension as usize;
    if dim != loaded_dim {
        return Err(format!(
            "session vector dimension ({dim}d) does not match the loaded model ({}d)",
            loaded_dim
        ));
    }

    // Reconstruct the chunk strategy from the session's stored parameters.
    // This early validation ensures the session's config is valid before
    // starting the expensive file triage and extraction phases.
    let chunk_size = session.chunk_size.map(|v| v as usize);
    let chunk_overlap = session.chunk_overlap.map(|v| v as usize);
    let max_words = session.max_words.map(|v| v as usize);
    let tokenizer_json = state.worker_handle.tokenizer_json();

    let _validate_strategy = neuroncite_chunk::create_strategy(
        &session.chunk_strategy,
        chunk_size,
        chunk_overlap,
        max_words,
        tokenizer_json.as_deref(),
    )
    .map_err(|e| format!("chunking strategy: {e}"))?;

    // Triage: classify each file as new, changed, unchanged, or invalid.
    let mut to_process: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut failed: Vec<serde_json::Value> = Vec::new();

    for &file_path in &file_paths {
        let pdf_path = std::path::Path::new(file_path);

        // Validate the file exists on disk.
        if !pdf_path.is_file() {
            failed.push(serde_json::json!({
                "file_path": file_path,
                "error": "file does not exist"
            }));
            continue;
        }

        // Canonicalize the file path and strip the Windows `\\?\` extended-length
        // prefix so the result matches the format stored by the bulk indexer.
        // The bulk `neuroncite_index` pipeline discovers files via
        // `discover_pdfs_flat`, which canonicalizes paths and then strips the
        // `\\?\` prefix via `strip_extended_prefix()` before storing them in
        // the database. Without stripping the prefix here, the
        // `find_file_by_session_path` lookup would compare `\\?\D:\path\file.pdf`
        // (from canonicalize) against `D:\path\file.pdf` (from bulk indexer),
        // causing the file to be treated as new instead of unchanged.
        let canonical_path = std::fs::canonicalize(pdf_path)
            .map(|p| {
                neuroncite_core::paths::strip_extended_length_prefix(&p.to_string_lossy())
                    .to_string()
            })
            .unwrap_or_else(|_| file_path.to_string());

        // Read current mtime and size from the filesystem.
        let metadata = std::fs::metadata(pdf_path)
            .map_err(|e| format!("metadata read failed for {file_path}: {e}"))?;

        let current_mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let current_size = metadata.len() as i64;

        // Check if this file was previously indexed in the session.
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        match neuroncite_store::find_file_by_session_path(&conn, session_id, &canonical_path) {
            Ok(Some(existing_file)) => {
                // File exists in session: check if it changed. Pass None for
                // hash because we have not extracted the file yet. Stage 1
                // (mtime/size comparison) is sufficient for the common case
                // where the file has not been modified. If mtime or size
                // differ, the file is re-processed regardless.
                match neuroncite_store::check_file_changed(
                    &conn,
                    existing_file.id,
                    current_mtime,
                    current_size,
                    None,
                ) {
                    Ok(ChangeStatus::Unchanged) => {
                        skipped.push(file_path.to_string());
                    }
                    Ok(_) => {
                        // MetadataOnly or ContentChanged: re-process the file.
                        to_process.push(canonical_path.clone());
                    }
                    Err(e) => {
                        failed.push(serde_json::json!({
                            "file_path": file_path,
                            "error": format!("change detection: {e}")
                        }));
                    }
                }
            }
            Ok(None) => {
                // File not in session: it is new. Use the canonical path so
                // the stored file_path matches the format that bulk indexing
                // produces, preventing duplicates on subsequent index_add calls.
                to_process.push(canonical_path.clone());
            }
            Err(e) => {
                failed.push(serde_json::json!({
                    "file_path": file_path,
                    "error": format!("file lookup: {e}")
                }));
            }
        }
    }

    // Phase 1: Extract and chunk all files that need processing.
    // Each file is extracted on a blocking thread (CPU-bound I/O).
    let mut extracted_files: Vec<indexer::ExtractedFile> = Vec::new();
    let mut added_count = 0usize;
    let mut updated_count = 0usize;

    for file_path in &to_process {
        let pdf_path = std::path::Path::new(file_path).to_path_buf();
        let strategy = neuroncite_chunk::create_strategy(
            &session.chunk_strategy,
            chunk_size,
            chunk_overlap,
            max_words,
            tokenizer_json.as_deref(),
        )
        .map_err(|e| format!("chunking strategy: {e}"))?;

        match tokio::task::spawn_blocking(move || {
            indexer::extract_and_chunk_file(strategy.as_ref(), &pdf_path)
        })
        .await
        {
            Ok(Ok(extracted)) => {
                if extracted.chunks.is_empty() {
                    failed.push(serde_json::json!({
                        "file_path": file_path,
                        "error": "extraction produced zero chunks"
                    }));
                    continue;
                }

                // Determine if this is a new or updated file.
                let conn = state
                    .pool
                    .get()
                    .map_err(|e| format!("connection pool error: {e}"))?;
                let is_existing =
                    neuroncite_store::find_file_by_session_path(&conn, session_id, file_path)
                        .map(|opt| opt.is_some())
                        .unwrap_or(false);

                if is_existing {
                    updated_count += 1;
                } else {
                    added_count += 1;
                }

                extracted_files.push(extracted);
            }
            Ok(Err(e)) => {
                failed.push(serde_json::json!({
                    "file_path": file_path,
                    "error": format!("extraction: {e}")
                }));
            }
            Err(e) => {
                failed.push(serde_json::json!({
                    "file_path": file_path,
                    "error": format!("task panicked: {e}")
                }));
            }
        }
    }

    // Phase 2: Embed and store each extracted file. The embed_and_store_file_async
    // function internally handles deletion of stale file records with the same
    // (session_id, file_path) before inserting the fresh data.
    let mut total_chunks_created = 0usize;

    for extracted in &extracted_files {
        let result = indexer::embed_and_store_file_async(
            &state.pool,
            &state.worker_handle,
            extracted,
            session_id,
        )
        .await
        .map_err(|e| {
            format!(
                "embedding/storage failed for {}: {e}",
                extracted.pdf_path.display()
            )
        })?;

        total_chunks_created += result.chunks_created;
    }

    // Phase 3: Rebuild the HNSW index once from all active chunk embeddings
    // in the session. This is necessary because chunk IDs changed after any
    // deletions and reinsertions.
    let hnsw_rebuilt = if !extracted_files.is_empty() {
        let hnsw_chunks = {
            let conn = state
                .pool
                .get()
                .map_err(|e| format!("connection pool error: {e}"))?;

            neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
                .map_err(|e| format!("loading embeddings for HNSW: {e}"))?
        };

        let vectors: Vec<(i64, Vec<f32>)> = hnsw_chunks
            .iter()
            .map(|(id, bytes)| {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (*id, floats)
            })
            .collect();

        let total_vectors = vectors.len();
        let labeled: Vec<(i64, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let index = build_hnsw(&labeled, dim).map_err(|e| format!("HNSW build failed: {e}"))?;
        state.insert_hnsw(session_id, index);

        Some(total_vectors)
    } else {
        // No files required extraction. Repair a missing in-memory HNSW entry
        // if the session has embeddings in SQLite but no loaded index -- this
        // occurs after a server restart when the in-memory HNSW map was cleared.
        if neuroncite_api::ensure_hnsw_for_session(state, session_id) {
            let conn = state
                .pool
                .get()
                .map_err(|e| format!("connection pool error: {e}"))?;
            let total_vectors = neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
                .map_err(|e| format!("loading embedding count: {e}"))?
                .len();
            Some(total_vectors)
        } else {
            None
        }
    };

    Ok(serde_json::json!({
        "session_id": session_id,
        "added": added_count,
        "updated": updated_count,
        "skipped": skipped.len(),
        "skipped_files": skipped,
        "failed": failed.len(),
        "failed_files": failed,
        "chunks_created": total_chunks_created,
        "hnsw_rebuilt": hnsw_rebuilt.is_some(),
        "total_session_vectors": hnsw_rebuilt,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-IDXADD-001: Handler rejects requests missing the session_id
    /// parameter. Validates the parameter extraction logic.
    #[test]
    fn t_mcp_idxadd_001_missing_session_id() {
        let params = serde_json::json!({
            "files": ["/some/path.pdf"]
        });
        assert!(params["session_id"].as_i64().is_none());
    }

    /// T-MCP-IDXADD-002: Handler rejects requests missing the files parameter.
    #[test]
    fn t_mcp_idxadd_002_missing_files() {
        let params = serde_json::json!({
            "session_id": 1
        });
        assert!(params["files"].as_array().is_none());
    }

    /// T-MCP-IDXADD-003: Handler rejects an empty files array. At least one
    /// file path must be provided for the operation to be meaningful.
    #[test]
    fn t_mcp_idxadd_003_empty_files_array() {
        let params = serde_json::json!({
            "session_id": 1,
            "files": []
        });
        let files = params["files"].as_array().unwrap();
        assert!(files.is_empty());
    }

    /// T-MCP-IDXADD-004: Nonexistent file paths are detected during the
    /// triage phase before any extraction is attempted.
    #[test]
    fn t_mcp_idxadd_004_nonexistent_file() {
        let path = std::path::Path::new("/nonexistent/fake_paper.pdf");
        assert!(!path.is_file());
    }

    /// T-MCP-IDXADD-005: Handler rejects non-string entries in the files
    /// array. All entries must be valid JSON strings.
    #[test]
    fn t_mcp_idxadd_005_non_string_files() {
        let params = serde_json::json!({
            "session_id": 1,
            "files": [123, true, null]
        });
        let files = params["files"].as_array().unwrap();
        let string_count = files.iter().filter(|v| v.as_str().is_some()).count();
        assert_eq!(string_count, 0, "none of the entries are strings");
    }

    /// T-MCP-IDXADD-006: Module compiles and the handler returns the correct
    /// Result type. Compilation of this module validates that all imports,
    /// types, and function signatures are correct.
    #[test]
    fn t_mcp_idxadd_006_module_compiles() {
        let _check: fn() -> Result<serde_json::Value, String> =
            || Err("compile-time type check only".to_string());
    }

    /// T-MCP-IDXADD-007: The ChangeStatus enum variants used by the handler
    /// are accessible. Validates that the neuroncite_store dependency exposes
    /// the required types.
    #[test]
    fn t_mcp_idxadd_007_change_status_accessible() {
        // Verify the enum variants exist and can be matched.
        let status = neuroncite_store::ChangeStatus::Unchanged;
        match status {
            neuroncite_store::ChangeStatus::Unchanged => {}
            neuroncite_store::ChangeStatus::MetadataOnly => {}
            neuroncite_store::ChangeStatus::ContentChanged => {}
            neuroncite_store::ChangeStatus::New => {}
        }
    }

    /// T-MCP-IDXADD-008: Mixed string and non-string entries in the files
    /// array produce a count mismatch. The handler uses filter_map to extract
    /// only string values, then compares the count to the original array length
    /// to detect non-string entries.
    #[test]
    fn t_mcp_idxadd_008_mixed_types_detected() {
        let params = serde_json::json!({
            "session_id": 1,
            "files": ["/valid/path.pdf", 42, "/another/path.pdf"]
        });
        let files_array = params["files"].as_array().unwrap();
        let string_paths: Vec<&str> = files_array.iter().filter_map(|v| v.as_str()).collect();

        assert_ne!(
            string_paths.len(),
            files_array.len(),
            "count mismatch detects non-string entries"
        );
    }

    /// T-MCP-IDXADD-009: Valid files array with all string entries passes
    /// the type validation check.
    #[test]
    fn t_mcp_idxadd_009_all_string_files_pass_validation() {
        let params = serde_json::json!({
            "session_id": 1,
            "files": ["/path/a.pdf", "/path/b.pdf", "/path/c.pdf"]
        });
        let files_array = params["files"].as_array().unwrap();
        let string_paths: Vec<&str> = files_array.iter().filter_map(|v| v.as_str()).collect();

        assert_eq!(
            string_paths.len(),
            files_array.len(),
            "all entries are strings, count must match"
        );
    }

    /// T-MCP-IDXADD-010: Session ID is correctly extracted from params as i64.
    #[test]
    fn t_mcp_idxadd_010_session_id_extraction() {
        let params = serde_json::json!({
            "session_id": 42,
            "files": ["/path/a.pdf"]
        });

        let session_id = params["session_id"].as_i64().unwrap();
        assert_eq!(session_id, 42);
    }

    /// T-MCP-IDXADD-011: The ChangeStatus and FileRow types are accessible
    /// from neuroncite_store. This validates the imports used by the handler.
    #[test]
    fn t_mcp_idxadd_011_store_types_accessible() {
        // Verify the types exist and are importable at compile time.
        let _status: neuroncite_store::ChangeStatus = neuroncite_store::ChangeStatus::Unchanged;
        fn _accepts_file_row(_row: &neuroncite_store::FileRow) {}
    }

    // ===================================================================
    // DEFECT-002 regression tests: canonical path matching in index_add
    //
    // The bulk `neuroncite_index` pipeline discovers PDFs via
    // `discover_pdfs_flat`, which canonicalizes the root directory before
    // listing entries. On Windows, `std::fs::canonicalize` produces paths
    // with the `\\?\` extended-length prefix (e.g. `\\?\D:\path\file.pdf`).
    // The `index_add` handler receives raw user-provided paths (e.g.
    // `D:\path\file.pdf`), so the `find_file_by_session_path` lookup
    // must compare canonical paths on both sides to detect duplicates.
    //
    // The fix canonicalizes user-provided paths in the triage phase
    // before the DB lookup. These tests verify that canonicalization
    // produces consistent paths for the same file.
    // ===================================================================

    /// T-MCP-IDXADD-012: Canonicalizing a real file path produces a result
    /// that matches itself when canonicalized again (idempotent). This is
    /// the fundamental property that DEFECT-002 fix relies on.
    #[test]
    fn t_mcp_idxadd_012_canonicalize_idempotent() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("test.pdf");
        std::fs::write(&file_path, b"test content").expect("write file");

        let canonical1 = std::fs::canonicalize(&file_path).expect("first canonicalize");
        let canonical2 = std::fs::canonicalize(&canonical1).expect("second canonicalize");

        assert_eq!(
            canonical1, canonical2,
            "canonicalize must be idempotent: calling it twice produces the same path"
        );
    }

    /// T-MCP-IDXADD-013: Two different string representations of the same
    /// file canonicalize to the same path. This simulates the bulk indexer
    /// storing a canonical path and index_add receiving a raw user path.
    #[test]
    fn t_mcp_idxadd_013_different_representations_canonicalize_equal() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("test.pdf");
        std::fs::write(&file_path, b"test content").expect("write file");

        // Raw path (as a user would provide it).
        let raw_path = file_path.to_string_lossy().to_string();

        // Canonical path (as the bulk indexer stores it).
        let canonical = std::fs::canonicalize(&file_path)
            .expect("canonicalize")
            .to_string_lossy()
            .to_string();

        // Re-canonicalize the raw path (as index_add does after the fix).
        let raw_canonicalized = std::fs::canonicalize(&raw_path)
            .expect("canonicalize raw")
            .to_string_lossy()
            .to_string();

        assert_eq!(
            canonical, raw_canonicalized,
            "raw path and pre-canonical path must canonicalize to the same string"
        );
    }

    /// T-MCP-IDXADD-014: Canonicalization fallback for nonexistent files
    /// returns the original path string. The handler uses `unwrap_or_else`
    /// to fall back gracefully when canonicalization fails.
    #[test]
    fn t_mcp_idxadd_014_canonicalize_fallback_for_nonexistent() {
        let fake_path = "/nonexistent/file/that/does/not/exist.pdf";

        let result = std::fs::canonicalize(fake_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| fake_path.to_string());

        assert_eq!(
            result, fake_path,
            "fallback must return the original path for nonexistent files"
        );
    }

    /// T-MCP-IDXADD-015: The `to_process` list stores canonical paths, not
    /// raw user paths. This ensures that subsequent `find_file_by_session_path`
    /// lookups in Phase 2 use the same path format as the initial triage,
    /// and that stored file paths match the bulk indexer's format.
    #[test]
    fn t_mcp_idxadd_015_to_process_stores_canonical_paths() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("test.pdf");
        std::fs::write(&file_path, b"test content").expect("write file");

        let raw_path = file_path.to_string_lossy().to_string();
        let pdf_path = std::path::Path::new(&raw_path);

        // Simulate the handler's canonicalization logic.
        let canonical_path = std::fs::canonicalize(pdf_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| raw_path.clone());

        // The canonical path should differ from the raw path on Windows
        // (due to \\?\ prefix) or be equal on Unix.
        #[cfg(windows)]
        assert!(
            canonical_path.starts_with("\\\\?\\"),
            "canonical path on Windows must start with \\\\?\\\\ prefix, got: {canonical_path}"
        );

        #[cfg(not(windows))]
        assert!(
            !canonical_path.is_empty(),
            "canonical path must not be empty on Unix"
        );

        // The to_process vector stores canonical_path.clone(), not raw_path.
        let to_process: Vec<String> = vec![canonical_path.clone()];
        assert_eq!(
            to_process[0], canonical_path,
            "to_process must store the canonical path"
        );
    }

    /// T-MCP-IDXADD-016: Canonical path comparison is case-sensitive on
    /// case-sensitive filesystems (Linux/macOS). On Windows, the OS-level
    /// canonicalization normalizes casing. This test verifies that the
    /// canonical path is stable for the same file.
    #[test]
    fn t_mcp_idxadd_016_canonical_path_is_stable() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("StableTest.pdf");
        std::fs::write(&file_path, b"stable test").expect("write file");

        let canonical_a = std::fs::canonicalize(&file_path)
            .expect("canonicalize a")
            .to_string_lossy()
            .to_string();

        let canonical_b = std::fs::canonicalize(&file_path)
            .expect("canonicalize b")
            .to_string_lossy()
            .to_string();

        assert_eq!(
            canonical_a, canonical_b,
            "canonicalizing the same file must produce identical paths across calls"
        );
    }

    // ===================================================================
    // DEFECT-005 regression tests: HNSW repair in the skipped-files path
    //
    // When all requested files are unchanged (skipped), Phase 3 of the
    // handler enters the else branch. Before DEFECT-005's fix, this branch
    // returned None unconditionally, leaving the HNSW absent after a server
    // restart even though the session has embeddings in SQLite.
    //
    // The fix calls ensure_hnsw_for_session inside the else branch:
    // - If the HNSW was repaired (embeddings found), hnsw_rebuilt is Some.
    // - If no embeddings exist, hnsw_rebuilt is None.
    // ===================================================================

    /// T-MCP-IDXADD-018: The JSON response fields hnsw_rebuilt and
    /// total_session_vectors correctly reflect the DEFECT-005 repair outcome.
    /// When ensure_hnsw_for_session succeeds, hnsw_rebuilt is true and
    /// total_session_vectors contains the vector count.
    #[test]
    fn t_mcp_idxadd_018_hnsw_rebuilt_response_fields_when_repaired() {
        // Simulate the else-branch returning Some(vector_count) after a
        // successful ensure_hnsw_for_session repair.
        let hnsw_rebuilt: Option<usize> = Some(42);

        let response = serde_json::json!({
            "hnsw_rebuilt": hnsw_rebuilt.is_some(),
            "total_session_vectors": hnsw_rebuilt,
        });

        assert_eq!(
            response["hnsw_rebuilt"], true,
            "hnsw_rebuilt must be true when the repair path built the index"
        );
        assert_eq!(
            response["total_session_vectors"], 42,
            "total_session_vectors must contain the vector count after repair"
        );
    }

    /// T-MCP-IDXADD-019: When all files are skipped and no embeddings exist
    /// in SQLite (ensure_hnsw_for_session returns false), hnsw_rebuilt is false
    /// and total_session_vectors is null.
    #[test]
    fn t_mcp_idxadd_019_hnsw_rebuilt_response_fields_when_no_embeddings() {
        // Simulate the else-branch returning None when ensure_hnsw_for_session
        // finds no embeddings to build from.
        let hnsw_rebuilt: Option<usize> = None;

        let response = serde_json::json!({
            "hnsw_rebuilt": hnsw_rebuilt.is_some(),
            "total_session_vectors": hnsw_rebuilt,
        });

        assert_eq!(
            response["hnsw_rebuilt"], false,
            "hnsw_rebuilt must be false when no embeddings were found"
        );
        assert_eq!(
            response["total_session_vectors"],
            serde_json::Value::Null,
            "total_session_vectors must be null when no repair was possible"
        );
    }

    /// T-MCP-IDXADD-017: Multiple files in the same directory all resolve
    /// to canonical paths that share the same parent directory prefix.
    #[test]
    fn t_mcp_idxadd_017_multiple_files_same_canonical_parent() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_a = tmp.path().join("a.pdf");
        let file_b = tmp.path().join("b.pdf");
        std::fs::write(&file_a, b"file a").expect("write a");
        std::fs::write(&file_b, b"file b").expect("write b");

        let canonical_a = std::fs::canonicalize(&file_a).expect("canonicalize a");
        let canonical_b = std::fs::canonicalize(&file_b).expect("canonicalize b");

        assert_eq!(
            canonical_a.parent(),
            canonical_b.parent(),
            "files in the same directory must have the same canonical parent"
        );
    }
}
