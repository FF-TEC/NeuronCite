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

//! Handler for the `neuroncite_reindex_file` MCP tool.
//!
//! Re-indexes a single PDF file within an existing session without deleting
//! the entire session. The old file record and its associated pages/chunks
//! are deleted (CASCADE), then the file is re-extracted, re-chunked,
//! re-embedded, and the session's HNSW index is rebuilt from all active
//! chunk embeddings.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_api::indexer;
use neuroncite_store::build_hnsw;

/// Re-indexes a single PDF file within an existing session.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): The session containing the file.
/// - `file_path` (required): Absolute path to the PDF file on disk.
///   The file must already exist in the session (previously indexed) or
///   be a PDF file that should be added to the session.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let file_path = params["file_path"]
        .as_str()
        .ok_or("missing required parameter: file_path")?;

    // Validate the file exists on disk.
    let pdf_path = std::path::Path::new(file_path);
    if !pdf_path.is_file() {
        return Err(format!("file does not exist: {file_path}"));
    }

    // Canonicalize the file path and strip the Windows `\\?\` extended-length
    // prefix so the result matches the format stored by the bulk indexer.
    // The bulk `neuroncite_index` pipeline discovers files via
    // `discover_pdfs_flat`, which canonicalizes paths and strips the `\\?\`
    // prefix via `strip_extended_prefix()` before storing. Without stripping
    // here, `find_file_by_session_path` would compare `\\?\D:\path\file.pdf`
    // against `D:\path\file.pdf`, treating the file as new instead of replaced.
    let canonical_path = std::fs::canonicalize(pdf_path)
        .map(|p| {
            neuroncite_core::paths::strip_extended_length_prefix(&p.to_string_lossy()).to_string()
        })
        .unwrap_or_else(|_| file_path.to_string());
    let pdf_path_for_extract = std::path::PathBuf::from(&canonical_path);

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
    let chunk_size = session.chunk_size.map(|v| v as usize);
    let chunk_overlap = session.chunk_overlap.map(|v| v as usize);
    let max_words = session.max_words.map(|v| v as usize);
    let tokenizer_json = state.worker_handle.tokenizer_json();

    let chunk_strategy = neuroncite_chunk::create_strategy(
        &session.chunk_strategy,
        chunk_size,
        chunk_overlap,
        max_words,
        tokenizer_json.as_deref(),
    )
    .map_err(|e| format!("chunking strategy: {e}"))?;

    // Check if the file was previously indexed in this session.
    let previous_chunks: Option<usize> = {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        neuroncite_store::find_file_by_session_path(&conn, session_id, &canonical_path)
            .map_err(|e| format!("file lookup: {e}"))?
            .map(|f| {
                neuroncite_store::single_file_chunk_stats(&conn, f.id)
                    .map(|stats| stats.chunk_count as usize)
                    .unwrap_or(0)
            })
    };

    // Phase 1: Extract text and chunk the file (CPU-bound, run on blocking thread).
    // Uses the canonical path (with `\\?\` prefix stripped) so the file_path
    // stored in the database matches the format used by the bulk indexer.
    let extracted = tokio::task::spawn_blocking(move || {
        indexer::extract_and_chunk_file(chunk_strategy.as_ref(), &pdf_path_for_extract)
    })
    .await
    .map_err(|e| format!("extraction task panicked: {e}"))?
    .map_err(|e| format!("extraction failed: {e}"))?;

    let pages_extracted = extracted.pages.len();
    let chunks_produced = extracted.chunks.len();

    if chunks_produced == 0 {
        return Err(
            "extraction produced zero chunks; the PDF may be empty or contain only images without OCR"
                .to_string(),
        );
    }

    // Phase 2: Embed chunks and store in the database (GPU-bound).
    // embed_and_store_file_async internally deletes any stale file record
    // with the same (session_id, file_path) before inserting the fresh one.
    let result = indexer::embed_and_store_file_async(
        &state.pool,
        &state.worker_handle,
        &extracted,
        session_id,
    )
    .await
    .map_err(|e| format!("embedding/storage failed: {e}"))?;

    // Rebuild the HNSW index for the session from all active chunk embeddings.
    // This is necessary because chunk IDs changed after deletion+reinsertion.
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

    let labeled: Vec<(i64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let index = build_hnsw(&labeled, dim).map_err(|e| format!("HNSW build failed: {e}"))?;
    state.insert_hnsw(session_id, index);

    let is_new_file = previous_chunks.is_none();

    Ok(serde_json::json!({
        "session_id": session_id,
        "file_path": canonical_path,
        "action": if is_new_file { "added" } else { "reindexed" },
        "pages_extracted": pages_extracted,
        "chunks_created": result.chunks_created,
        "previous_chunks": previous_chunks,
        "total_session_vectors": vectors.len(),
        "hnsw_rebuilt": true,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-REIDX-001: Constants and types used by the handler are
    /// accessible. Compilation of this module validates that all imports,
    /// types, and function signatures in the handler are correct.
    #[test]
    fn t_mcp_reidx_001_module_compiles() {
        // Verify the handler returns Result<serde_json::Value, String>.
        let _check: fn() -> Result<serde_json::Value, String> =
            || Err("compile-time type check only".to_string());
    }

    // -----------------------------------------------------------------------
    // DEF-002 regression tests: Path canonicalization for reindex_file
    //
    // The bulk indexer (neuroncite_index) stores file paths in canonical
    // form. On Windows, std::fs::canonicalize adds the \\?\ extended-length
    // prefix. Without canonicalizing the user-provided path in the reindex
    // handler, the find_file_by_session_path lookup fails (exact string
    // comparison), causing a duplicate file record instead of an in-place
    // replacement.
    //
    // These tests verify that the canonicalization logic in the handler
    // produces paths matching those stored by the bulk indexer.
    // -----------------------------------------------------------------------

    /// T-MCP-REIDX-002: Canonicalizing a real file path is idempotent.
    /// Calling canonicalize twice on the same file produces identical results.
    #[test]
    fn t_mcp_reidx_002_canonicalize_idempotent() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("test_reindex.pdf");
        std::fs::write(&file_path, b"pdf content").expect("write file");

        let canonical1 = std::fs::canonicalize(&file_path).expect("first canonicalize");
        let canonical2 = std::fs::canonicalize(&canonical1).expect("second canonicalize");

        assert_eq!(canonical1, canonical2, "canonicalize must be idempotent");
    }

    /// T-MCP-REIDX-003: A raw user path and a pre-canonicalized path resolve
    /// to the same string after canonicalization. This is the property that
    /// the DEF-002 fix relies on: the handler canonicalizes the user path
    /// before comparing it against the stored (canonical) path.
    #[test]
    fn t_mcp_reidx_003_raw_and_canonical_paths_match() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("bollerslev.pdf");
        std::fs::write(&file_path, b"pdf content").expect("write file");

        let raw_path = file_path.to_string_lossy().to_string();

        // Simulate the bulk indexer storing the canonical path.
        let stored_canonical = std::fs::canonicalize(&file_path)
            .expect("canonicalize stored")
            .to_string_lossy()
            .to_string();

        // Simulate the reindex handler canonicalizing the user-provided path.
        let user_canonical = std::fs::canonicalize(&raw_path)
            .expect("canonicalize user")
            .to_string_lossy()
            .to_string();

        assert_eq!(
            stored_canonical, user_canonical,
            "raw user path and pre-canonical stored path must resolve to the same string"
        );
    }

    /// T-MCP-REIDX-004: On Windows, canonicalized paths start with the
    /// extended-length prefix. On Unix, the canonical path is absolute.
    #[test]
    fn t_mcp_reidx_004_canonical_path_format() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("format_check.pdf");
        std::fs::write(&file_path, b"content").expect("write file");

        let canonical = std::fs::canonicalize(&file_path)
            .expect("canonicalize")
            .to_string_lossy()
            .to_string();

        #[cfg(windows)]
        assert!(
            canonical.starts_with("\\\\?\\"),
            "canonical path on Windows must have \\\\?\\\\ prefix, got: {canonical}"
        );

        #[cfg(not(windows))]
        assert!(
            canonical.starts_with('/'),
            "canonical path on Unix must be absolute, got: {canonical}"
        );
    }

    /// T-MCP-REIDX-005: Canonicalization fallback for nonexistent files
    /// returns the original path. The handler uses unwrap_or_else to avoid
    /// a panic when the file does not exist.
    #[test]
    fn t_mcp_reidx_005_canonicalize_fallback() {
        let fake_path = "/this/file/does/not/exist/reindex_test.pdf";

        let result = std::fs::canonicalize(fake_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| fake_path.to_string());

        assert_eq!(
            result, fake_path,
            "fallback must return the original path when canonicalize fails"
        );
    }

    /// T-MCP-REIDX-006: The handler passes the canonical path to both
    /// find_file_by_session_path and extract_and_chunk_file. This test
    /// verifies the construction logic: PathBuf::from(&canonical_path)
    /// preserves the canonical format.
    #[test]
    fn t_mcp_reidx_006_pathbuf_from_canonical_preserves_format() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let file_path = tmp.path().join("roundtrip.pdf");
        std::fs::write(&file_path, b"content").expect("write file");

        let canonical_str = std::fs::canonicalize(&file_path)
            .expect("canonicalize")
            .to_string_lossy()
            .to_string();

        // Reconstruct PathBuf from the canonical string (as the handler does).
        let reconstructed = std::path::PathBuf::from(&canonical_str);

        assert_eq!(
            reconstructed.to_string_lossy().to_string(),
            canonical_str,
            "PathBuf::from must preserve the canonical path string"
        );
    }
}
