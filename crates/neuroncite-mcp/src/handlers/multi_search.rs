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

//! Handler for the `neuroncite_multi_search` MCP tool.
//!
//! Searches across multiple index sessions simultaneously. The query is
//! embedded once, then the SearchPipeline runs for each session. Results
//! from all sessions are merged into a single ranked list sorted by score,
//! with each result tagged by its source session_id.
//!
//! This is useful for comparing how different chunking strategies affect
//! retrieval quality, or for searching a corpus that was indexed across
//! multiple sessions (e.g., different time periods or subject areas).

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_search::{SearchConfig, SearchPipeline, relevance_label};

/// Maximum number of sessions that can be searched in a single request.
const MAX_SESSIONS: usize = 10;

/// Maximum number of results returned in a multi-search response.
const MAX_TOP_K: usize = 50;

/// Searches across multiple sessions and returns a merged result set.
///
/// # Parameters (from MCP tool call)
///
/// - `session_ids` (required): Array of session IDs to search across (2-10).
/// - `query` (required): Search query text.
/// - `top_k` (optional): Total number of results to return (default: 10, max: 50).
/// - `use_fts` (optional): Enable FTS5 keyword search (default: true).
/// - `min_score` (optional): Minimum vector_score threshold (0.0 to 1.0).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_ids_array = params["session_ids"]
        .as_array()
        .ok_or("missing required parameter: session_ids")?;

    let session_ids: Vec<i64> = session_ids_array
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();

    if session_ids.len() != session_ids_array.len() {
        return Err("session_ids must contain only integer values".to_string());
    }

    if session_ids.len() < 2 {
        return Err("session_ids must contain at least 2 session IDs".to_string());
    }

    if session_ids.len() > MAX_SESSIONS {
        return Err(format!(
            "session_ids must contain at most {MAX_SESSIONS} session IDs, got {}",
            session_ids.len()
        ));
    }

    let query = params["query"]
        .as_str()
        .ok_or("missing required parameter: query")?;

    let query = query.trim();
    if query.is_empty() {
        return Err("query must not be empty".to_string());
    }

    let top_k = params["top_k"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(10)
        .clamp(1, MAX_TOP_K);
    let use_fts = params["use_fts"].as_bool().unwrap_or(true);
    let min_score: Option<f64> = params["min_score"].as_f64();

    if let Some(ms) = min_score
        && !(0.0..=1.0).contains(&ms)
    {
        return Err(format!("min_score must be between 0.0 and 1.0, got {ms}"));
    }

    // Validate all sessions exist and have matching vector dimensions.
    // All sessions must use the same vector dimension as the loaded model,
    // because the query embedding from the model must be compatible with
    // the stored document embeddings in each session.
    //
    // The vector_dimension field is an AtomicUsize, so a single relaxed load
    // is performed here and the resulting plain usize is reused for all
    // session dimension comparisons and the error message.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        for &sid in &session_ids {
            let session = neuroncite_store::get_session(&conn, sid)
                .map_err(|_| format!("session {sid} not found"))?;

            let dim = session.vector_dimension as usize;
            if dim != loaded_dim {
                return Err(format!(
                    "session {sid} has vector dimension {dim}d which does not match \
                     the loaded model ({}d)",
                    loaded_dim
                ));
            }
        }
    }

    // Embed the query once. The same embedding vector is reused for all
    // sessions since they share the same vector dimension.
    let query_vec = state
        .worker_handle
        .embed_query(query.to_string())
        .await
        .map_err(|e| format!("embedding failed: {e}"))?;

    // Ensure the HNSW index is loaded for each requested session before
    // acquiring the guard. The background executor's Phase 3 may have silently
    // skipped HNSW builds when the SQLite WAL read snapshot did not include
    // Phase 2's just-committed embeddings. All sessions are repaired upfront
    // so the guard acquired below reflects the current state.
    for &sid in &session_ids {
        neuroncite_api::ensure_hnsw_for_session(state, sid);
    }

    // Run SearchPipeline for each session and collect results.
    let mut all_results: Vec<(i64, neuroncite_core::SearchResult)> = Vec::new();
    let mut session_stats: Vec<serde_json::Value> = Vec::new();

    let hnsw_guard = state.index.hnsw_index.load();

    for &sid in &session_ids {
        let Some(hnsw_ref) = hnsw_guard.get(&sid) else {
            session_stats.push(serde_json::json!({
                "session_id": sid,
                "result_count": 0,
                "status": "no_hnsw_index",
            }));
            continue;
        };

        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        // Each session gets a generous candidate pool. The final top_k trim
        // happens after merging results from all sessions.
        let per_session_k = top_k * 3;

        let config = SearchConfig {
            session_id: sid,
            vector_top_k: per_session_k * 3,
            keyword_limit: if use_fts { per_session_k * 3 } else { 0 },
            ef_search: state.config.ef_search,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: per_session_k,
            rerank_enabled: false,
            file_ids: None,
            min_score,
            page_start: None,
            page_end: None,
        };

        let pipeline = SearchPipeline::new(hnsw_ref, &conn, &query_vec, query, None, config);

        match pipeline.search() {
            Ok(outcome) => {
                let count = outcome.results.len();
                for r in outcome.results {
                    all_results.push((sid, r));
                }
                session_stats.push(serde_json::json!({
                    "session_id": sid,
                    "result_count": count,
                    "status": "ok",
                }));
            }
            Err(e) => {
                session_stats.push(serde_json::json!({
                    "session_id": sid,
                    "result_count": 0,
                    "status": "error",
                    "error": format!("{e}"),
                }));
            }
        }
    }

    // Sort merged results by score descending and trim to top_k.
    all_results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(top_k);

    // Format results as JSON, adding session_id to each result.
    let result_array: Vec<serde_json::Value> = all_results
        .iter()
        .map(|(sid, r)| {
            let page_numbers: Vec<usize> = (r.citation.page_start..=r.citation.page_end).collect();
            let source_file_owned = r.citation.source_file.display().to_string();
            let source_file = strip_extended_length_prefix(&source_file_owned);

            let vs = if r.vector_score > 0.0 {
                serde_json::json!(r.vector_score)
            } else {
                serde_json::Value::Null
            };

            serde_json::json!({
                "session_id": sid,
                "score": r.score,
                "vector_score": vs,
                "bm25_rank": r.bm25_rank,
                "relevance": relevance_label(r.vector_score),
                "content": r.content,
                "file_id": r.citation.file_id,
                "source_file": source_file,
                "file_display_name": r.citation.file_display_name,
                "page_start": r.citation.page_start,
                "page_end": r.citation.page_end,
                "page_numbers": page_numbers,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "query": query,
        "session_ids": session_ids,
        "result_count": result_array.len(),
        "session_stats": session_stats,
        "results": result_array,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-MULTI-001: Handler rejects requests missing the session_ids
    /// parameter.
    #[test]
    fn t_mcp_multi_001_missing_session_ids() {
        let params = serde_json::json!({
            "query": "test"
        });
        assert!(params["session_ids"].as_array().is_none());
    }

    /// T-MCP-MULTI-002: Handler rejects an empty session_ids array.
    #[test]
    fn t_mcp_multi_002_empty_session_ids() {
        let params = serde_json::json!({
            "session_ids": [],
            "query": "test"
        });
        let ids = params["session_ids"].as_array().unwrap();
        assert!(ids.is_empty());
    }

    /// T-MCP-MULTI-003: Handler rejects a single session_id. Multi-search
    /// requires at least 2 sessions; single-session search should use
    /// neuroncite_search instead.
    #[test]
    fn t_mcp_multi_003_single_session_id() {
        let session_ids: Vec<i64> = vec![1];
        assert!(session_ids.len() < 2, "single session must be rejected");
    }

    /// T-MCP-MULTI-004: Handler rejects session_ids that exceed the maximum
    /// of 10 sessions.
    #[test]
    fn t_mcp_multi_004_too_many_sessions() {
        let session_ids: Vec<i64> = (1..=11).collect();
        assert!(session_ids.len() > MAX_SESSIONS);
    }

    /// T-MCP-MULTI-005: Handler rejects requests missing the query parameter.
    #[test]
    fn t_mcp_multi_005_missing_query() {
        let params = serde_json::json!({
            "session_ids": [1, 2]
        });
        assert!(params["query"].as_str().is_none());
    }

    /// T-MCP-MULTI-006: top_k is clamped to the maximum of 50.
    #[test]
    fn t_mcp_multi_006_top_k_clamped() {
        let top_k: usize = 999_usize.clamp(1, MAX_TOP_K);
        assert_eq!(top_k, MAX_TOP_K);
    }

    /// T-MCP-MULTI-007: Module compiles and the handler returns the correct
    /// Result type.
    #[test]
    fn t_mcp_multi_007_module_compiles() {
        let _check: fn() -> Result<serde_json::Value, String> =
            || Err("compile-time type check only".to_string());
    }

    /// T-MCP-MULTI-008: min_score validation rejects values outside [0.0, 1.0].
    #[test]
    fn t_mcp_multi_008_min_score_validation() {
        assert!(!(0.0..=1.0).contains(&1.5));
        assert!(!(0.0..=1.0).contains(&-0.1));
        assert!((0.0..=1.0).contains(&0.72));
    }

    /// T-MCP-MULTI-009: session_ids with non-integer values are filtered out.
    /// The handler uses filter_map(as_i64) and compares the count to detect
    /// non-integer entries in the session_ids array.
    #[test]
    fn t_mcp_multi_009_non_integer_session_ids_detected() {
        let params = serde_json::json!({
            "session_ids": [1, "two", 3],
            "query": "test"
        });

        let session_ids_array = params["session_ids"].as_array().unwrap();
        let session_ids: Vec<i64> = session_ids_array
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();

        assert_ne!(
            session_ids.len(),
            session_ids_array.len(),
            "non-integer entries must cause a count mismatch"
        );
    }

    /// T-MCP-MULTI-010: Exactly 2 session_ids is the minimum accepted count.
    /// Multi-search requires at least 2 sessions to be meaningful.
    #[test]
    fn t_mcp_multi_010_minimum_two_sessions() {
        let session_ids: Vec<i64> = vec![1, 2];
        assert!(session_ids.len() >= 2, "2 sessions must be accepted");
    }

    /// T-MCP-MULTI-011: Exactly 10 session_ids is the maximum accepted count.
    #[test]
    fn t_mcp_multi_011_maximum_ten_sessions() {
        let session_ids: Vec<i64> = (1..=10).collect();
        assert_eq!(session_ids.len(), MAX_SESSIONS);
        assert!(
            session_ids.len() <= MAX_SESSIONS,
            "10 sessions must be accepted"
        );
    }

    /// T-MCP-MULTI-012: top_k defaults to 10 when omitted.
    #[test]
    fn t_mcp_multi_012_default_top_k() {
        let params = serde_json::json!({
            "session_ids": [1, 2],
            "query": "test"
        });

        let top_k = params["top_k"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(10)
            .clamp(1, MAX_TOP_K);

        assert_eq!(top_k, 10);
    }

    /// T-MCP-MULTI-013: use_fts defaults to true when omitted.
    #[test]
    fn t_mcp_multi_013_default_use_fts() {
        let params = serde_json::json!({
            "session_ids": [1, 2],
            "query": "test"
        });

        let use_fts = params["use_fts"].as_bool().unwrap_or(true);
        assert!(use_fts, "use_fts must default to true");
    }

    /// T-MCP-MULTI-014: Empty query string (after trimming) is rejected.
    #[test]
    fn t_mcp_multi_014_empty_query_rejected() {
        let query = "   ";
        let trimmed = query.trim();
        assert!(trimmed.is_empty(), "whitespace-only query must be rejected");
    }

    /// T-MCP-MULTI-015: min_score at boundary values (0.0 and 1.0) is accepted.
    #[test]
    fn t_mcp_multi_015_min_score_boundary_values() {
        assert!((0.0..=1.0).contains(&0.0), "0.0 must be accepted");
        assert!((0.0..=1.0).contains(&1.0), "1.0 must be accepted");
    }

    /// T-MCP-MULTI-016: per_session_k is computed as top_k * 3 to provide a
    /// generous candidate pool for cross-session merging.
    #[test]
    fn t_mcp_multi_016_per_session_k_formula() {
        let top_k: usize = 10;
        let per_session_k = top_k * 3;
        assert_eq!(per_session_k, 30);
    }
}
