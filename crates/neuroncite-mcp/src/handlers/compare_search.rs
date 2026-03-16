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

//! Handler for the `neuroncite_compare_search` MCP tool.
//!
//! Runs the same query against two different index sessions and returns
//! side-by-side results for comparing search quality between different
//! chunking strategies, embedding models, or indexing configurations.
//! This complements `neuroncite_file_compare` which compares chunk
//! statistics; this tool compares actual search result quality.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_search::{SearchConfig, SearchOutcome, SearchPipeline, relevance_label};

/// Maximum number of results per session in a comparison query.
const MAX_TOP_K: usize = 20;

/// Runs a search query against two sessions and returns side-by-side results.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id_a` (required): First session to search.
/// - `session_id_b` (required): Second session to search.
/// - `query` (required): Search query text (same query is run against both sessions).
/// - `top_k` (optional): Number of results per session (1-20, default: 5).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id_a = params["session_id_a"]
        .as_i64()
        .ok_or("missing required parameter: session_id_a")?;
    let session_id_b = params["session_id_b"]
        .as_i64()
        .ok_or("missing required parameter: session_id_b")?;
    let query = params["query"]
        .as_str()
        .ok_or("missing required parameter: query")?;

    let query = query.trim();
    if query.is_empty() {
        return Err("query must not be empty".to_string());
    }

    if session_id_a == session_id_b {
        return Err("session_id_a and session_id_b must be different sessions".to_string());
    }

    let top_k = params["top_k"].as_u64().unwrap_or(5) as usize;
    if !(1..=MAX_TOP_K).contains(&top_k) {
        return Err(format!(
            "top_k must be between 1 and {MAX_TOP_K}, got {top_k}"
        ));
    }

    // Validate both sessions exist and have compatible vector dimensions.
    // The vector_dimension field is an AtomicUsize, so a single relaxed load
    // is performed here and the resulting plain usize is reused for the
    // comparison and the error message.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let (session_a_info, session_b_info) = {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let sa = neuroncite_store::get_session(&conn, session_id_a)
            .map_err(|_| format!("session {session_id_a} not found"))?;
        let sb = neuroncite_store::get_session(&conn, session_id_b)
            .map_err(|_| format!("session {session_id_b} not found"))?;

        if sa.vector_dimension != sb.vector_dimension {
            return Err(format!(
                "sessions have different vector dimensions: session {} has {}d, session {} has {}d; \
                 cross-session comparison requires the same embedding model",
                session_id_a, sa.vector_dimension, session_id_b, sb.vector_dimension
            ));
        }

        let dim = sa.vector_dimension as usize;
        if dim != loaded_dim {
            return Err(format!(
                "session vector dimension ({dim}d) does not match the loaded model ({}d)",
                loaded_dim
            ));
        }

        (sa, sb)
    };

    // Embed the query once (same embedding is used for both sessions).
    let query_vec = state
        .worker_handle
        .embed_query(query.to_string())
        .await
        .map_err(|e| format!("embedding failed: {e}"))?;

    // Run search against both sessions.
    let hnsw_guard = state.index.hnsw_index.load();
    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let run_search = |session_id: i64| -> Result<(Vec<serde_json::Value>, usize), String> {
        let Some(hnsw_ref) = hnsw_guard.get(&session_id) else {
            return Ok((Vec::new(), 0));
        };

        let config = SearchConfig {
            session_id,
            vector_top_k: top_k * 5,
            keyword_limit: top_k * 5,
            ef_search: state.config.ef_search,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: top_k,
            rerank_enabled: false,
            file_ids: None,
            min_score: None,
            page_start: None,
            page_end: None,
        };

        let pipeline = SearchPipeline::new(hnsw_ref, &conn, &query_vec, query, None, config);
        let SearchOutcome {
            results,
            total_candidates,
        } = pipeline
            .search()
            .map_err(|e| format!("search failed for session {session_id}: {e}"))?;

        let result_json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                let source_owned = r.citation.source_file.display().to_string();
                let source = strip_extended_length_prefix(&source_owned);
                let is_bm25_only = r.vector_score == 0.0;
                serde_json::json!({
                    "score": r.score,
                    "vector_score": if is_bm25_only { serde_json::Value::Null } else { serde_json::json!(r.vector_score) },
                    "relevance": if is_bm25_only { "bm25_only" } else { relevance_label(r.vector_score) },
                    "content": r.content,
                    "file_id": r.citation.file_id,
                    "source_file": source,
                    "page_start": r.citation.page_start,
                    "page_end": r.citation.page_end,
                })
            })
            .collect();

        Ok((result_json, total_candidates))
    };

    let (results_a, total_a) = run_search(session_id_a)?;
    let (results_b, total_b) = run_search(session_id_b)?;

    // Compute summary statistics for comparison.
    let avg_score = |results: &[serde_json::Value]| -> Option<f64> {
        let scores: Vec<f64> = results
            .iter()
            .filter_map(|r| r["vector_score"].as_f64())
            .collect();
        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    };

    Ok(serde_json::json!({
        "query": query,
        "top_k": top_k,
        "session_a": {
            "session_id": session_id_a,
            "model": session_a_info.model_name,
            "chunk_strategy": session_a_info.chunk_strategy,
            "result_count": results_a.len(),
            "total_candidates": total_a,
            "avg_vector_score": avg_score(&results_a),
            "results": results_a,
        },
        "session_b": {
            "session_id": session_id_b,
            "model": session_b_info.model_name,
            "chunk_strategy": session_b_info.chunk_strategy,
            "result_count": results_b.len(),
            "total_candidates": total_b,
            "avg_vector_score": avg_score(&results_b),
            "results": results_b,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-CSRCH-001: MAX_TOP_K is set to 20 for comparison queries.
    #[test]
    fn t_mcp_csrch_001_max_top_k() {
        assert_eq!(MAX_TOP_K, 20);
    }
}
