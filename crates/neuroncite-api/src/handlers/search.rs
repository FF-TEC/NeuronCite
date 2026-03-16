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

//! Search endpoint handlers.
//!
//! POST /api/v1/search       -- standard search (vector + optional FTS + optional reranking)
//! POST /api/v1/search/hybrid -- alias for the same handler
//! POST /api/v1/search/multi  -- multi-session search (merges results from 1-10 sessions)
//!
//! Accepts a search query and parameters, embeds the query via the GPU worker,
//! runs the full SearchPipeline (vector search, FTS5 keyword search, RRF fusion,
//! deduplication, citation assembly), and optionally applies cross-encoder
//! reranking through the GPU worker.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use neuroncite_search::{
    SearchConfig, SearchPipeline, apply_refinement, generate_sub_chunks, parse_divisors,
};

use crate::dto::{
    API_VERSION, MultiSearchRequest, MultiSearchResponse, MultiSearchSessionStat, SearchRequest,
    SearchResponse, SearchResultDto, SessionSearchStatus,
};
use crate::error::ApiError;
use crate::state::AppState;

/// POST /api/v1/search
///
/// Performs a hybrid search (vector + FTS5 keyword) using the full SearchPipeline.
/// When `use_fts` is false, keyword results are empty and fusion degrades to
/// vector-only ranking. When `rerank` is true, cross-encoder reranking is
/// applied to the pipeline results via the GPU worker.
#[utoipa::path(
    post,
    path = "/api/v1/search",
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request"),
    )
)]
pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    // Apply server-configured defaults to optional fields that the caller
    // omitted. This reads default values from the config file instead of
    // hardcoding them in the handler.
    req.with_defaults(&state.config.defaults);

    // Validate request fields against the configured input limits. This covers
    // query emptiness, query max length, and refine_divisors count. Must run
    // before any connection acquisition or embedding work.
    req.validate(&state.config.limits)?;

    // Validate top_k bounds (1-200). The lower bound prevents ambiguous
    // "zero results requested" semantics. The upper bound prevents resource
    // exhaustion from unbounded allocation. After with_defaults(), top_k
    // is guaranteed to be Some when it was originally absent.
    let top_k = req.top_k.unwrap_or(10);
    if !(1..=200).contains(&top_k) {
        return Err(ApiError::BadRequest {
            reason: format!("top_k must be between 1 and 200, got {top_k}"),
        });
    }
    let use_fts = req.use_fts.unwrap_or(true);
    let rerank = req.rerank.unwrap_or(false);
    let refine = req.refine.unwrap_or(true);
    let divisors = req
        .refine_divisors
        .as_deref()
        .map(parse_divisors)
        .unwrap_or_else(|| vec![4, 8, 16]);

    // Embed the query text via the GPU worker.
    let query_vec = state
        .worker_handle
        .embed_query(req.query.clone())
        .await
        .map_err(|e| ApiError::Internal {
            reason: format!("embedding failed: {e}"),
        })?;

    // Extract the HNSW index Arc for this session. The ArcSwap guard is !Send
    // and must not cross the spawn_blocking boundary, so we clone the Arc here
    // and let the guard drop on the async thread.
    let hnsw_arc = {
        let guard = state.index.hnsw_index.load();
        guard.get(&req.session_id).cloned()
    };

    // When no index exists for this session (e.g., before indexing completes),
    // return an empty result set instead of a 500 error.
    let Some(hnsw_arc) = hnsw_arc else {
        return Ok(Json(SearchResponse {
            api_version: API_VERSION.to_string(),
            results: Vec::new(),
        }));
    };

    // Capture values needed by the blocking closure. The pool is Clone+Send,
    // and the atomic dimension value is loaded before the closure to avoid
    // passing the full AppState into the blocking thread.
    let pool = state.pool.clone();
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let ef_search = state.config.ef_search;
    let session_id = req.session_id;
    let query_for_pipeline = req.query.clone();
    let query_vec_for_pipeline = query_vec.clone();

    // Run the synchronous search pipeline on a blocking thread. This prevents
    // pool.get() (which waits for a connection) and the CPU-bound HNSW search
    // from stalling the tokio async executor. The rusqlite Connection is !Send,
    // so it must be created and dropped entirely within the blocking thread.
    // Returns a tuple of (results, file_metadata_map) where the map contains
    // created_at and updated_at timestamps keyed by file_id.
    let (mut results, file_meta) = tokio::task::spawn_blocking(move || {
        let conn = pool.get().map_err(ApiError::from)?;

        // Guard against dimension mismatch: verify that the session was indexed
        // with the same model dimensionality as the currently loaded model before
        // executing the search. A mismatch means the query vector has a different
        // length than the HNSW vectors, which would produce incorrect results or
        // a panic inside the HNSW search.
        {
            let session = neuroncite_store::get_session(&conn, session_id).map_err(|e| {
                ApiError::BadRequest {
                    reason: format!("session {session_id} not found: {e}"),
                }
            })?;
            let session_dim = session.vector_dimension as usize;
            if session_dim != loaded_dim {
                tracing::info!(
                    session_id,
                    session_model = session.model_name,
                    session_dim,
                    loaded_dim,
                    "search rejected: vector dimension mismatch between session and loaded model"
                );
                return Err(ApiError::BadRequest {
                    reason: format!(
                        "model dimension mismatch: session {session_id} was indexed with a \
                         {session_dim}d model but the server has a {loaded_dim}d model loaded. \
                         Re-index the session with the current model.",
                    ),
                });
            }
        }

        // Configure the search pipeline. When use_fts is false, keyword_limit is
        // set to 0 so the keyword search stage returns no results and fusion
        // degrades to vector-only ranking.
        let config = SearchConfig {
            session_id,
            vector_top_k: top_k * 5,
            keyword_limit: if use_fts { top_k * 5 } else { 0 },
            ef_search,
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

        let pipeline = SearchPipeline::new(
            &hnsw_arc,
            &conn,
            &query_vec_for_pipeline,
            &query_for_pipeline,
            None, // Reranker is applied post-pipeline through the async GPU worker.
            config,
        );

        let results = pipeline
            .search()
            .map_err(|e| ApiError::Internal {
                reason: format!("search pipeline failed: {e}"),
            })
            .map(|outcome| outcome.results)?;

        // Collect file_ids from results and fetch their metadata (created_at,
        // updated_at) from the indexed_file table. This is done on the blocking
        // thread because it requires the rusqlite Connection.
        let mut file_meta: HashMap<i64, (i64, i64)> = HashMap::new();
        for r in &results {
            let fid = r.citation.file_id;
            if let std::collections::hash_map::Entry::Vacant(e) = file_meta.entry(fid)
                && let Ok(file_row) = neuroncite_store::get_file(&conn, fid)
            {
                e.insert((file_row.created_at, file_row.updated_at));
            }
        }

        Ok::<_, ApiError>((results, file_meta))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })??;

    // Apply cross-encoder reranking via the GPU worker if requested.
    // The GPU worker's rerank_batch is async, so reranking is done outside
    // the synchronous SearchPipeline. Errors are propagated to the caller
    // so that a user requesting reranking receives a clear error when no
    // reranker model is configured.
    if rerank && !results.is_empty() {
        let content_refs: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        let scores = state
            .worker_handle
            .rerank_batch(&req.query, &content_refs)
            .await
            .map_err(|e| ApiError::Internal {
                reason: format!("reranking failed: {e}"),
            })?;

        for (result, score) in results.iter_mut().zip(scores.iter()) {
            result.reranker_score = Some(*score);
            result.score = *score;
        }
        // Re-sort by descending reranker score.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Apply sub-chunk refinement via the GPU worker if requested. Refinement
    // splits each result's content into token-based sub-chunks at multiple
    // scales (defined by divisors), embeds them, and replaces the content
    // with the highest-scoring sub-chunk when it beats the original.
    // Uses the pre-deserialized tokenizer from WorkerHandle instead of
    // parsing the tokenizer JSON per request.
    if refine
        && !results.is_empty()
        && !divisors.is_empty()
        && let Some(ref tokenizer) = *state.worker_handle.cached_tokenizer()
    {
        let candidates = generate_sub_chunks(&results, tokenizer, &divisors).map_err(|e| {
            ApiError::Internal {
                reason: format!("refinement sub-chunk generation failed: {e}"),
            }
        })?;

        if !candidates.is_empty() {
            let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
            let embeddings = state
                .worker_handle
                .embed_batch_search(&texts)
                .await
                .map_err(|e| ApiError::Internal {
                    reason: format!("refinement embedding failed: {e}"),
                })?;
            // The returned Vec<bool> indicates which results had their score
            // adjusted by sub-chunk refinement. Logged at debug level for diagnostics.
            let refined_flags =
                apply_refinement(&mut results, &candidates, &embeddings, &query_vec);
            let refined_count = refined_flags.iter().filter(|&&f| f).count();
            if refined_count > 0 {
                tracing::debug!(
                    refined = refined_count,
                    total = refined_flags.len(),
                    "sub-chunk refinement applied"
                );
            }
        }
    }

    // Convert SearchResult objects to the API response DTO format.
    // source_file, page_start, page_end are accessed through the citation
    // to avoid duplicating data that Citation already holds.
    // session_id is None for single-session search responses.
    // Document metadata (doc_created_at, doc_modified_at) is populated from
    // the file_meta map collected during the blocking search phase.
    let result_dtos: Vec<SearchResultDto> = results
        .into_iter()
        .map(|r| {
            let fid = r.citation.file_id;
            let (doc_created_at, doc_modified_at) = match file_meta.get(&fid) {
                Some(&(created, updated)) => (Some(created), Some(updated)),
                None => (None, None),
            };
            SearchResultDto {
                score: r.score,
                content: r.content,
                vector_score: r.vector_score,
                bm25_rank: r.bm25_rank,
                reranker_score: r.reranker_score,
                file_id: fid,
                source_file: r.citation.source_file.display().to_string(),
                page_start: r.citation.page_start,
                page_end: r.citation.page_end,
                chunk_index: r.chunk_index,
                citation: r.citation.formatted,
                session_id: None,
                doc_created_at,
                doc_modified_at,
                doc_author: None,
            }
        })
        .collect();

    Ok(Json(SearchResponse {
        api_version: API_VERSION.to_string(),
        results: result_dtos,
    }))
}

/// POST /api/v1/search/hybrid
///
/// Alias for the standard search handler. The hybrid label indicates that
/// both vector and keyword search are used when `use_fts` is true.
#[utoipa::path(
    post,
    path = "/api/v1/search/hybrid",
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Hybrid search results", body = SearchResponse),
    )
)]
pub async fn hybrid_search(
    state: State<Arc<AppState>>,
    body: Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    search(state, body).await
}

/// Maximum number of results returned in a multi-search response.
const MULTI_SEARCH_MAX_TOP_K: usize = 50;

/// POST /api/v1/search/multi
///
/// Searches across multiple sessions simultaneously. The query is embedded once,
/// then the SearchPipeline runs for each session. Results from all sessions are
/// merged into a single ranked list sorted by score, with each result tagged by
/// its source session_id.
///
/// All requested sessions must have the same vector dimension as the loaded model.
/// Reranking and refinement are applied to the merged result set (not per-session).
#[utoipa::path(
    post,
    path = "/api/v1/search/multi",
    request_body = MultiSearchRequest,
    responses(
        (status = 200, description = "Multi-session search results", body = MultiSearchResponse),
        (status = 400, description = "Invalid request"),
    )
)]
pub async fn multi_search(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<MultiSearchRequest>,
) -> Result<Json<MultiSearchResponse>, ApiError> {
    // Apply server-configured defaults to optional fields that the caller
    // omitted. This reads default values from the config file instead of
    // hardcoding them in the handler.
    req.with_defaults(&state.config.defaults);

    // Validate request fields against the configured input limits. This covers
    // query emptiness, query max length, session_ids emptiness, and session_ids
    // max count. The limit defaults to max_session_ids from AppConfig (default: 10).
    req.validate(&state.config.limits)?;

    let top_k = req.top_k.unwrap_or(10).clamp(1, MULTI_SEARCH_MAX_TOP_K);
    let use_fts = req.use_fts.unwrap_or(true);
    let rerank = req.rerank.unwrap_or(false);
    let refine = req.refine.unwrap_or(true);
    let divisors = req
        .refine_divisors
        .as_deref()
        .map(parse_divisors)
        .unwrap_or_else(|| vec![4, 8, 16]);

    // Embed the query first (async). The same embedding vector is reused for
    // all sessions since they share the same vector dimension.
    let query_vec = state
        .worker_handle
        .embed_query(req.query.clone())
        .await
        .map_err(|e| ApiError::Internal {
            reason: format!("embedding failed: {e}"),
        })?;

    // Move all synchronous work (validation, HNSW loading, pipeline execution)
    // to a blocking thread. This prevents pool.get() from blocking the tokio
    // async executor and keeps rusqlite Connections (which are !Send) confined
    // to the blocking thread. A single connection is reused for all sessions
    // to maintain a consistent WAL snapshot across the multi-session search.
    // Returns a tuple of (tagged_results, session_stats, file_metadata_map).
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let state_clone = state.clone();
    let session_ids = req.session_ids.clone();
    let query_for_pipeline = req.query.clone();
    let query_vec_for_pipeline = query_vec.clone();
    let ef_search = state.config.ef_search;

    let (mut all_results, session_stats, file_meta) = tokio::task::spawn_blocking(move || {
        // Acquire a single connection for validation and all pipeline runs.
        let conn = state_clone.pool.get().map_err(ApiError::from)?;

        // Validate all sessions exist and have matching vector dimensions.
        for &sid in &session_ids {
            let session =
                neuroncite_store::get_session(&conn, sid).map_err(|e| ApiError::BadRequest {
                    reason: format!("session {sid} not found: {e}"),
                })?;
            let dim = session.vector_dimension as usize;
            if dim != loaded_dim {
                return Err(ApiError::BadRequest {
                    reason: format!(
                        "session {sid} has vector dimension {dim}d which does not match \
                         the loaded model ({loaded_dim}d). All sessions must use the same \
                         embedding model dimension."
                    ),
                });
            }
        }

        // Ensure HNSW indexes are loaded for all sessions. This repairs sessions
        // where Phase 3 silently skipped HNSW builds during indexing.
        for &sid in &session_ids {
            crate::ensure_hnsw_for_session(&state_clone, sid);
        }

        // Run SearchPipeline for each session and collect tagged results.
        // The ArcSwap guard is created inside the blocking closure where it
        // is safe (!Send guard never crosses thread boundaries).
        let hnsw_guard = state_clone.index.hnsw_index.load();
        let mut all_results: Vec<(i64, neuroncite_core::SearchResult)> = Vec::new();
        let mut session_stats: Vec<MultiSearchSessionStat> = Vec::new();

        for &sid in &session_ids {
            let Some(hnsw_ref) = hnsw_guard.get(&sid) else {
                session_stats.push(MultiSearchSessionStat {
                    session_id: sid,
                    result_count: 0,
                    status: SessionSearchStatus::NoHnswIndex,
                    error: None,
                });
                continue;
            };

            // Each session gets a moderate candidate pool (top_k * 2).
            // The 2x over-fetch provides enough candidates to survive merging,
            // deduplication, and score-based trimming while avoiding the 9x
            // total over-fetch of the previous (top_k * 3) * 3 formula.
            let per_session_k = top_k * 2;

            let config = SearchConfig {
                session_id: sid,
                vector_top_k: per_session_k * 2,
                keyword_limit: if use_fts { per_session_k * 2 } else { 0 },
                ef_search,
                rrf_k: 60,
                bm25_must_match: false,
                simhash_threshold: 3,
                max_results: per_session_k,
                rerank_enabled: false,
                file_ids: None,
                min_score: None,
                page_start: None,
                page_end: None,
            };

            let pipeline = SearchPipeline::new(
                hnsw_ref,
                &conn,
                &query_vec_for_pipeline,
                &query_for_pipeline,
                None,
                config,
            );

            match pipeline.search() {
                Ok(outcome) => {
                    let count = outcome.results.len();
                    for r in outcome.results {
                        all_results.push((sid, r));
                    }
                    session_stats.push(MultiSearchSessionStat {
                        session_id: sid,
                        result_count: count,
                        status: SessionSearchStatus::Ok,
                        error: None,
                    });
                }
                Err(e) => {
                    session_stats.push(MultiSearchSessionStat {
                        session_id: sid,
                        result_count: 0,
                        status: SessionSearchStatus::Error,
                        error: Some(format!("{e}")),
                    });
                }
            }
        }

        // Collect file metadata for document timestamps.
        let mut file_meta: HashMap<i64, (i64, i64)> = HashMap::new();
        for (_, r) in &all_results {
            let fid = r.citation.file_id;
            if let std::collections::hash_map::Entry::Vacant(e) = file_meta.entry(fid)
                && let Ok(file_row) = neuroncite_store::get_file(&conn, fid)
            {
                e.insert((file_row.created_at, file_row.updated_at));
            }
        }

        Ok::<_, ApiError>((all_results, session_stats, file_meta))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })??;

    // Sort merged results by score descending and trim to top_k.
    all_results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(top_k);

    // Convert to the DTO format. Each result carries session_id to
    // identify which session it originated from.
    let mut results: Vec<neuroncite_core::SearchResult> = Vec::with_capacity(all_results.len());
    let mut result_session_ids: Vec<i64> = Vec::with_capacity(all_results.len());
    for (sid, r) in all_results {
        result_session_ids.push(sid);
        results.push(r);
    }

    // Apply cross-encoder reranking to the merged result set if requested.
    if rerank && !results.is_empty() {
        let content_refs: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        let scores = state
            .worker_handle
            .rerank_batch(&req.query, &content_refs)
            .await
            .map_err(|e| ApiError::Internal {
                reason: format!("reranking failed: {e}"),
            })?;

        // Apply reranker scores and re-sort results + session_ids together.
        for (i, score) in scores.iter().enumerate() {
            results[i].reranker_score = Some(*score);
            results[i].score = *score;
        }

        // Build a permutation index, then apply it in-place by zipping results
        // and session_ids into a sortable vec, avoiding clone-for-reorder.
        let mut paired: Vec<(neuroncite_core::SearchResult, i64)> = results
            .into_iter()
            .zip(result_session_ids.into_iter())
            .collect();
        paired.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (sorted_results, sorted_sids): (Vec<_>, Vec<_>) = paired.into_iter().unzip();
        results = sorted_results;
        result_session_ids = sorted_sids;
    }

    // Apply sub-chunk refinement to the merged result set if requested.
    // Uses the pre-deserialized tokenizer from WorkerHandle instead of
    // parsing the tokenizer JSON per request.
    if refine
        && !results.is_empty()
        && !divisors.is_empty()
        && let Some(ref tokenizer) = *state.worker_handle.cached_tokenizer()
    {
        let candidates = generate_sub_chunks(&results, tokenizer, &divisors).map_err(|e| {
            ApiError::Internal {
                reason: format!("refinement sub-chunk generation failed: {e}"),
            }
        })?;

        if !candidates.is_empty() {
            let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
            let embeddings = state
                .worker_handle
                .embed_batch_search(&texts)
                .await
                .map_err(|e| ApiError::Internal {
                    reason: format!("refinement embedding failed: {e}"),
                })?;
            // The returned Vec<bool> indicates which results had their score
            // adjusted by sub-chunk refinement. Logged at debug level for diagnostics.
            let refined_flags =
                apply_refinement(&mut results, &candidates, &embeddings, &query_vec);
            let refined_count = refined_flags.iter().filter(|&&f| f).count();
            if refined_count > 0 {
                tracing::debug!(
                    refined = refined_count,
                    total = refined_flags.len(),
                    "sub-chunk refinement applied"
                );
            }
        }
    }

    // Build the final DTO list with session_id tags and document metadata.
    let result_dtos: Vec<SearchResultDto> = results
        .into_iter()
        .zip(result_session_ids.iter())
        .map(|(r, &sid)| {
            let fid = r.citation.file_id;
            let (doc_created_at, doc_modified_at) = match file_meta.get(&fid) {
                Some(&(created, updated)) => (Some(created), Some(updated)),
                None => (None, None),
            };
            SearchResultDto {
                score: r.score,
                content: r.content,
                vector_score: r.vector_score,
                bm25_rank: r.bm25_rank,
                reranker_score: r.reranker_score,
                file_id: fid,
                source_file: r.citation.source_file.display().to_string(),
                page_start: r.citation.page_start,
                page_end: r.citation.page_end,
                chunk_index: r.chunk_index,
                citation: r.citation.formatted,
                session_id: Some(sid),
                doc_created_at,
                doc_modified_at,
                doc_author: None,
            }
        })
        .collect();

    Ok(Json(MultiSearchResponse {
        api_version: API_VERSION.to_string(),
        session_ids: req.session_ids,
        session_stats,
        results: result_dtos,
    }))
}
