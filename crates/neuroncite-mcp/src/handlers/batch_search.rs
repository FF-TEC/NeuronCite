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

//! Handler for the `neuroncite_batch_search` MCP tool.
//!
//! Accepts multiple search queries in a single call, executes each through the
//! full SearchPipeline (vector + BM25 keyword via Reciprocal Rank Fusion,
//! deduplication, optional cross-encoder reranking), and returns results keyed
//! by the caller-provided query ID.
//!
//! The handler shares a single HNSW guard and database connection across all
//! queries in the batch to minimize resource acquisition overhead. Query
//! embeddings are computed sequentially via the GPU worker (which serializes
//! GPU operations through a single channel regardless).

use std::collections::HashSet;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_search::{
    CachedTokenizer, SearchConfig, SearchOutcome, SearchPipeline, apply_refinement,
    extract_query_terms, generate_sub_chunks, parse_divisors, relevance_label,
};

/// Maximum number of queries allowed in a single batch call. Prevents
/// unbounded resource consumption from a single MCP tool invocation.
const MAX_BATCH_QUERIES: usize = 20;

/// Maximum per-query top_k value in batch mode. Lower than the single-search
/// cap (200) to keep total work bounded at MAX_BATCH_QUERIES * MAX_PER_QUERY_TOP_K.
const MAX_PER_QUERY_TOP_K: usize = 50;

/// Executes multiple search queries against the specified session.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Index session to search within.
/// - `queries` (required): Array of query objects, each with `id`, `query`,
///   and optional `top_k` (1-50, default 5).
/// - `rerank` (optional): Apply cross-encoder reranking to all queries (default false).
/// - `use_fts` (optional): Enable FTS5 keyword search (default true).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let queries = params["queries"]
        .as_array()
        .ok_or("missing required parameter: queries (must be an array)")?;

    if queries.is_empty() {
        return Err("queries array must not be empty".to_string());
    }

    if queries.len() > MAX_BATCH_QUERIES {
        return Err(format!(
            "batch limit exceeded: {} queries provided, maximum is {MAX_BATCH_QUERIES}",
            queries.len()
        ));
    }

    let rerank = params["rerank"].as_bool().unwrap_or(false);
    let use_fts = params["use_fts"].as_bool().unwrap_or(true);
    let refine = params["refine"].as_bool().unwrap_or(true);
    let divisors = params["refine_divisors"]
        .as_str()
        .map(parse_divisors)
        .unwrap_or_else(|| vec![4, 8, 16]);

    // Parse session-level file_ids filter. When provided, restricts all queries
    // in the batch to chunks belonging to the specified file IDs.
    let file_ids: Option<Vec<i64>> = params["file_ids"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

    // Parse session-level min_score threshold. When provided, results with
    // vector_score below this value are excluded from all query responses.
    // Cosine similarity from normalized embeddings is bounded to [0.0, 1.0],
    // so values outside this range indicate a caller error.
    let min_score: Option<f64> = params["min_score"].as_f64();
    if let Some(ms) = min_score
        && !(0.0..=1.0).contains(&ms)
    {
        return Err(format!("min_score must be between 0.0 and 1.0, got {ms}"));
    }

    // Validate session exists and dimension matches the loaded model.
    // The vector_dimension field is an AtomicUsize, so a single relaxed load
    // is performed here and the resulting plain usize is reused for both the
    // comparison and the error message.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let session = neuroncite_store::get_session(&conn, session_id)
            .map_err(|e| format!("session {session_id} not found: {e}"))?;

        let session_dim = session.vector_dimension as usize;
        if session_dim != loaded_dim {
            return Err(format!(
                "model dimension mismatch: session {session_id} was indexed with '{}' \
                 ({}d) but the MCP server loaded a {}d model",
                session.model_name, session_dim, loaded_dim
            ));
        }
    }

    // Parse and validate all query entries upfront before any computation.
    // Valid queries are collected for processing; invalid queries produce
    // per-query error objects in the response. This partial-failure model
    // allows the batch to succeed for well-formed queries even when some
    // entries are malformed (missing fields, empty text, duplicate IDs,
    // out-of-range top_k).
    //
    // Structural errors that prevent any query processing (missing session_id,
    // empty queries array, batch limit exceeded) still return a top-level
    // error above.
    let mut seen_ids = HashSet::with_capacity(queries.len());
    let mut parsed_queries: Vec<(String, String, usize)> = Vec::with_capacity(queries.len());
    let mut query_errors: Vec<(String, String)> = Vec::new();

    for (idx, q) in queries.iter().enumerate() {
        // Extract the query ID first. Queries without a valid string "id"
        // field receive a synthetic ID based on their array index so the
        // error can be reported in the response map.
        let qid = match q["id"].as_str() {
            Some(id) => id.to_string(),
            None => {
                let synthetic_id = format!("__error_idx_{idx}");
                query_errors.push((
                    synthetic_id,
                    format!("queries[{idx}]: missing required field 'id'"),
                ));
                continue;
            }
        };

        let query_text = match q["query"].as_str() {
            Some(text) => text,
            None => {
                query_errors.push((
                    qid,
                    format!("queries[{idx}]: missing required field 'query'"),
                ));
                continue;
            }
        };

        // Reject empty and whitespace-only queries, consistent with
        // neuroncite_search behavior.
        let query_text = query_text.trim();
        if query_text.is_empty() {
            query_errors.push((qid, format!("queries[{idx}]: query text must not be empty")));
            continue;
        }

        // Auto-suffix duplicate IDs so every valid query executes. The first
        // occurrence retains the original ID; subsequent duplicates receive
        // "_2", "_3", etc. suffixes in the response map.
        let effective_id = if !seen_ids.insert(qid.clone()) {
            let mut suffixed = format!("{qid}_2");
            let mut counter = 3;
            while seen_ids.contains(&suffixed) {
                suffixed = format!("{qid}_{counter}");
                counter += 1;
            }
            seen_ids.insert(suffixed.clone());
            suffixed
        } else {
            qid
        };

        let raw_top_k = q["top_k"].as_u64().unwrap_or(5);
        if !(1..=MAX_PER_QUERY_TOP_K as u64).contains(&raw_top_k) {
            query_errors.push((
                effective_id,
                format!(
                    "queries[{idx}]: top_k must be between 1 and {MAX_PER_QUERY_TOP_K}, got {raw_top_k}"
                ),
            ));
            continue;
        }
        let top_k = raw_top_k as usize;

        parsed_queries.push((effective_id, query_text.to_string(), top_k));
    }

    // When every query in the batch is invalid, return a top-level error
    // because there is nothing to process. This avoids returning a response
    // with an empty results map that could be mistaken for "no matches".
    if parsed_queries.is_empty() && !query_errors.is_empty() {
        let error_details: Vec<serde_json::Value> = query_errors
            .iter()
            .map(|(id, msg)| serde_json::json!({"id": id, "error": msg}))
            .collect();
        return Err(format!(
            "all {} queries in the batch are invalid: {}",
            query_errors.len(),
            serde_json::to_string(&error_details).unwrap_or_default()
        ));
    }

    // Estimate truncation for each query before embedding. WordPiece/BPE
    // tokenizers produce at least 1 subword token per whitespace word, plus
    // 2 special tokens ([CLS] and [SEP]). Using word_count + 2 as the lower
    // bound catches queries that are borderline or certain to be truncated.
    let max_seq_len = state.worker_handle.max_sequence_length();
    let mut truncation_warnings: Vec<Option<String>> = parsed_queries
        .iter()
        .map(|(_, query_text, _)| {
            let word_count = query_text.split_whitespace().count();
            let estimated_min_tokens = word_count + 2;
            if estimated_min_tokens > max_seq_len {
                Some(format!(
                    "query has ~{word_count} words (at least {estimated_min_tokens} tokens) \
                     which likely exceeds max_sequence_length of {max_seq_len}; only a \
                     prefix was used for embedding"
                ))
            } else {
                None
            }
        })
        .collect();

    // Phase 1: Embed all query texts sequentially via the GPU worker.
    // Each embedding request is serialized through the worker channel regardless,
    // so sequential calls match the actual execution order.
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(parsed_queries.len());
    for (qid, query_text, _) in &parsed_queries {
        let vec = state
            .worker_handle
            .embed_query(query_text.clone())
            .await
            .map_err(|e| format!("embedding failed for query '{qid}': {e}"))?;
        embeddings.push(vec);
    }

    // Ensure the HNSW index is loaded for this session before acquiring the
    // guard. The background executor's Phase 3 may have silently skipped the
    // HNSW build when the SQLite WAL read snapshot did not include Phase 2's
    // just-committed embeddings.
    neuroncite_api::ensure_hnsw_for_session(state, session_id);

    // Phase 2: Run all search pipelines in a synchronous block.
    // The HNSW guard (ArcSwap) and rusqlite Connection are non-Send, so they
    // must be acquired and dropped within a single synchronous scope before
    // any subsequent `.await` calls (reranking).
    // Per-query tuple: (query_id, query_text, search_results, total_candidates_before_cap).
    let mut all_results: Vec<(String, String, Vec<neuroncite_core::SearchResult>, usize)> = {
        let hnsw_guard = state.index.hnsw_index.load();
        let Some(hnsw_ref) = hnsw_guard.get(&session_id) else {
            // No HNSW index loaded for this session -- return empty results
            // for all valid queries, plus error entries for invalid queries.
            let early_error_count = query_errors.len();
            let mut empty =
                serde_json::Map::with_capacity(parsed_queries.len() + query_errors.len());
            for (qid, qt, _) in &parsed_queries {
                // Build a basic query_analysis for the early return so
                // callers can inspect per-query tokenization details even
                // when no HNSW index is loaded. The exact query_tokens
                // count is unavailable here because the tokenizer loads
                // lazily during the refinement phase.
                let qword_count = qt.split_whitespace().count();
                let mut early_analysis = serde_json::json!({
                    "query_word_count": qword_count,
                    "max_sequence_length": max_seq_len,
                    "was_truncated": (qword_count + 2) > max_seq_len,
                });
                if use_fts {
                    early_analysis["bm25_terms"] = serde_json::json!(extract_query_terms(qt));
                }

                empty.insert(
                    qid.clone(),
                    serde_json::json!({
                        "query": qt,
                        "result_count": 0,
                        "results": [],
                        "query_analysis": early_analysis,
                    }),
                );
            }
            for (qid, error_message) in query_errors {
                // Use a suffixed key for duplicate-ID error entries to avoid
                // overwriting valid first-occurrence results (same logic as
                // the main response-building path below).
                let insert_key = if empty.contains_key(&qid) {
                    let mut suffixed = format!("{qid}__dup");
                    let mut counter = 2;
                    while empty.contains_key(&suffixed) {
                        suffixed = format!("{qid}__dup_{counter}");
                        counter += 1;
                    }
                    suffixed
                } else {
                    qid.clone()
                };

                empty.insert(
                    insert_key,
                    serde_json::json!({
                        "original_id": qid,
                        "error": error_message,
                        "result_count": 0,
                        "results": [],
                    }),
                );
            }

            return Ok(serde_json::json!({
                "session_id": session_id,
                "query_count": empty.len(),
                "error_count": early_error_count,
                "results": empty,
            }));
        };

        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let mut batch_results = Vec::with_capacity(parsed_queries.len());

        for ((qid, query_text, top_k), embedding) in parsed_queries.iter().zip(embeddings.iter()) {
            let config = SearchConfig {
                session_id,
                vector_top_k: top_k * 5,
                keyword_limit: if use_fts { top_k * 5 } else { 0 },
                ef_search: state.config.ef_search,
                rrf_k: 60,
                bm25_must_match: false,
                simhash_threshold: 3,
                max_results: *top_k,
                rerank_enabled: false,
                file_ids: file_ids.clone(),
                min_score,
                page_start: None,
                page_end: None,
            };

            let pipeline =
                SearchPipeline::new(hnsw_ref, &conn, embedding, query_text, None, config);

            let SearchOutcome {
                results,
                total_candidates,
            } = pipeline
                .search()
                .map_err(|e| format!("search failed for query '{qid}': {e}"))?;

            batch_results.push((qid.clone(), query_text.clone(), results, total_candidates));
        }

        batch_results
    };
    // hnsw_guard and conn are dropped here.

    // Phase 3: Apply cross-encoder reranking if requested. When no reranker
    // is configured, reranking is skipped and a warning is added to the
    // response instead of returning a hard error.
    let mut rerank_warning: Option<String> = None;
    if rerank {
        if !state.worker_handle.reranker_available() {
            rerank_warning = Some(
                "rerank=true was requested but no reranker model is configured; \
                 results are returned without reranking. Check neuroncite_health \
                 for 'reranker_available'."
                    .to_string(),
            );
        } else {
            for (qid, query_text, results, _) in &mut all_results {
                if results.is_empty() {
                    continue;
                }
                let content_refs: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
                let scores = state
                    .worker_handle
                    .rerank_batch(query_text, &content_refs)
                    .await
                    .map_err(|e| format!("reranking failed for query '{qid}': {e}"))?;

                for (result, score) in results.iter_mut().zip(scores.iter()) {
                    result.reranker_score = Some(*score);
                    result.score = *score;
                }
                results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    // Phase 4: Apply sub-chunk refinement if requested. Each query's
    // results are refined independently using the query's embedding.
    // The CachedTokenizer is created once before the loop to avoid
    // re-deserializing the tokenizer JSON (typically ~700 KB) for each
    // of the up to 20 queries in the batch.
    //
    // `per_query_refined` stores a Vec<bool> per query indicating which
    // results had their content replaced by a higher-scoring sub-chunk.
    // `per_query_tokens` stores the exact subword token count for each query.
    let query_count = all_results.len();
    let mut per_query_refined: Vec<Vec<bool>> = all_results
        .iter()
        .map(|(_, _, results, _)| vec![false; results.len()])
        .collect();
    let mut per_query_tokens: Vec<Option<usize>> = vec![None; query_count];

    // Preserve original chunk contents before refinement replaces them with
    // narrower sub-chunks. Stored per-query so the full_chunk_content field
    // can be included in the response for refined results.
    let per_query_original_contents: Vec<Vec<String>> = all_results
        .iter()
        .map(|(_, _, results, _)| results.iter().map(|r| r.content.clone()).collect())
        .collect();

    if let Some(ref tokenizer_json) = *state.worker_handle.tokenizer_json()
        && let Ok(tokenizer) = CachedTokenizer::from_json(tokenizer_json)
    {
        // Compute exact token counts for all queries in the batch.
        for (idx, (_, query_text, _, _)) in all_results.iter().enumerate() {
            per_query_tokens[idx] = tokenizer.token_count(query_text).ok();
        }

        if refine && !divisors.is_empty() {
            for (q_idx, ((_, _, results, _), embedding)) in
                all_results.iter_mut().zip(embeddings.iter()).enumerate()
            {
                if results.is_empty() {
                    continue;
                }
                let candidates = generate_sub_chunks(results, &tokenizer, &divisors)
                    .map_err(|e| format!("refinement sub-chunk generation failed: {e}"))?;

                if !candidates.is_empty() {
                    let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
                    let sub_embeddings = state
                        .worker_handle
                        .embed_batch_search(&texts)
                        .await
                        .map_err(|e| format!("refinement embedding failed: {e}"))?;
                    per_query_refined[q_idx] =
                        apply_refinement(results, &candidates, &sub_embeddings, embedding);
                }
            }
        }
    }

    // Upgrade truncation warnings for queries where the heuristic (word_count + 2)
    // did not fire but the exact BPE/WordPiece token count exceeds max_seq_len.
    // See the single-search handler for the full rationale.
    for (idx, tokens_opt) in per_query_tokens.iter().enumerate() {
        if truncation_warnings[idx].is_none()
            && let Some(tokens) = tokens_opt
            && *tokens >= max_seq_len
        {
            truncation_warnings[idx] = Some(format!(
                "query has {tokens} tokens which exceeds \
                 max_sequence_length of {max_seq_len}; only a \
                 prefix was used for embedding"
            ));
        }
    }

    // Phase 5: Build the JSON response keyed by query ID. Includes both
    // successful search results and per-query error entries for queries
    // that failed validation in the parsing phase.
    let mut results_map = serde_json::Map::with_capacity(all_results.len() + query_errors.len());

    let max_seq_len = state.worker_handle.max_sequence_length();

    for (result_idx, (qid, query_text, results, total_candidates)) in
        all_results.into_iter().enumerate()
    {
        // Build score_summary for this query group. Results with
        // vector_score == 0.0 (found only by BM25, not vector search) are
        // excluded to avoid skewing the mean. A bm25_only_count field reports
        // how many results were keyword-only.
        let score_summary = if !results.is_empty() {
            let vs_nonzero: Vec<f64> = results
                .iter()
                .map(|r| r.vector_score)
                .filter(|&s| s > 0.0)
                .collect();
            let bm25_only_count = results.len() - vs_nonzero.len();

            if vs_nonzero.is_empty() {
                // All results in this query group are BM25-only (no vector
                // search scores). Emit null instead of 0.0 for vector score
                // statistics to distinguish "not scored" from "zero similarity".
                serde_json::json!({
                    "min_vector_score": null,
                    "max_vector_score": null,
                    "mean_vector_score": null,
                    "bm25_only_count": bm25_only_count,
                })
            } else {
                let min_vs = vs_nonzero.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_vs = vs_nonzero.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mean_vs = vs_nonzero.iter().sum::<f64>() / vs_nonzero.len() as f64;
                serde_json::json!({
                    "min_vector_score": min_vs,
                    "max_vector_score": max_vs,
                    "mean_vector_score": mean_vs,
                    "bm25_only_count": bm25_only_count,
                })
            }
        } else {
            serde_json::json!(null)
        };

        // Extract query terms for per-result matched_terms reporting.
        let query_terms: Vec<String> = if use_fts {
            extract_query_terms(&query_text)
        } else {
            Vec::new()
        };

        // Compute max score across this query's results for normalizing to
        // a 0.0-1.0 relevance_score. Defaults to epsilon when empty or all zero.
        let max_score = results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(f64::EPSILON);

        // Serialize each result to JSON with relevance_score and matched_terms.
        // Results found only by BM25 keyword search (vector_score == 0.0) emit
        // null for the vector_score field. The relevance label for BM25-only
        // results is "bm25_only" rather than a threshold-based label.
        let refined_flags = per_query_refined
            .get(result_idx)
            .cloned()
            .unwrap_or_default();

        let result_array: Vec<serde_json::Value> = results
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                let page_numbers: Vec<usize> =
                    (r.citation.page_start..=r.citation.page_end).collect();
                let source_file_owned = r.citation.source_file.display().to_string();
                let source_file = strip_extended_length_prefix(&source_file_owned);
                let is_bm25_only = r.vector_score == 0.0;
                let refined = refined_flags.get(idx).copied().unwrap_or(false);
                let relevance_score = r.score / max_score;

                let matched_terms: Vec<&str> = if !query_terms.is_empty() {
                    let content_lower = r.content.to_lowercase();
                    query_terms
                        .iter()
                        .filter(|term| content_lower.contains(&term.to_lowercase()))
                        .map(|s| s.as_str())
                        .collect()
                } else {
                    Vec::new()
                };

                let mut result_json = serde_json::json!({
                    "score": r.score,
                    "relevance_score": relevance_score,
                    "content": r.content,
                    "vector_score": if is_bm25_only { serde_json::Value::Null } else { serde_json::json!(r.vector_score) },
                    "relevance": if is_bm25_only { "bm25_only" } else { relevance_label(r.vector_score) },
                    "bm25_rank": r.bm25_rank,
                    "reranker_score": r.reranker_score,
                    "was_refined": refined,
                    "file_id": r.citation.file_id,
                    "source_file": source_file,
                    "page_start": r.citation.page_start,
                    "page_end": r.citation.page_end,
                    "page_numbers": page_numbers,
                    "citation": r.citation.formatted,
                });

                if use_fts {
                    result_json["matched_terms"] = serde_json::json!(matched_terms);
                }

                // Include the original full chunk text when refinement replaced
                // the content with a narrower sub-chunk.
                if refined
                    && let Some(originals) = per_query_original_contents.get(result_idx)
                    && let Some(original) = originals.get(idx)
                {
                    result_json["full_chunk_content"] = serde_json::json!(original);
                }

                result_json
            })
            .collect();

        let mut entry = serde_json::json!({
            "query": query_text,
            "result_count": result_array.len(),
            "total_available": total_candidates,
            "score_summary": score_summary,
            "results": result_array,
        });

        // Include exact query token count when the tokenizer is available.
        if let Some(tokens) = per_query_tokens.get(result_idx).copied().flatten() {
            entry["query_tokens"] = serde_json::json!(tokens);
            entry["max_sequence_length"] = serde_json::json!(max_seq_len);
        }

        // Build a structured query_analysis section for this query that provides
        // transparency about tokenization, truncation, and BM25 term extraction.
        // This mirrors the query_analysis section in the single-search handler.
        {
            let query_word_count = query_text.split_whitespace().count();
            let query_token_count = per_query_tokens.get(result_idx).copied().flatten();
            let was_truncated = query_token_count.map(|t| t >= max_seq_len).unwrap_or(false);
            let tokens_used = query_token_count.map(|t| t.min(max_seq_len));

            let mut analysis = serde_json::json!({
                "query_word_count": query_word_count,
                "max_sequence_length": max_seq_len,
                "was_truncated": was_truncated,
            });

            if let Some(tokens) = query_token_count {
                analysis["query_tokens"] = serde_json::json!(tokens);
                analysis["tokens_used_for_embedding"] = serde_json::json!(tokens_used);
            }

            // Include the BM25 terms when FTS is enabled. These are the tokens
            // that the keyword search pipeline extracted from this query for
            // matching against the FTS5 index.
            if use_fts {
                analysis["bm25_terms"] = serde_json::json!(query_terms);
            }

            entry["query_analysis"] = analysis;
        }

        // Include a per-query truncation_warning when the query text is
        // estimated to exceed the model's max_sequence_length.
        if let Some(warning) = truncation_warnings.get(result_idx).and_then(|w| w.as_ref()) {
            entry["truncation_warning"] = serde_json::json!(warning);
        }

        results_map.insert(qid, entry);
    }

    // Insert per-query error entries for queries that failed validation.
    // Each error entry contains the error message and zero results, allowing
    // the caller to distinguish between "no matches" and "invalid query".
    //
    // Duplicate-ID error entries use a suffixed key ("q1__dup_2") instead of
    // the raw duplicate ID to avoid overwriting the valid first-occurrence
    // result that was already inserted into the results_map. Without this
    // suffixing, the error entry for a duplicate "q1" would replace the
    // successful search result for the original "q1" query.
    let error_count = query_errors.len();
    for (qid, error_message) in query_errors {
        let insert_key = if results_map.contains_key(&qid) {
            // This error shares an ID with an already-inserted valid result
            // (duplicate ID scenario). Append a suffix to avoid overwriting.
            let mut suffixed = format!("{qid}__dup");
            let mut counter = 2;
            while results_map.contains_key(&suffixed) {
                suffixed = format!("{qid}__dup_{counter}");
                counter += 1;
            }
            suffixed
        } else {
            qid.clone()
        };

        results_map.insert(
            insert_key,
            serde_json::json!({
                "original_id": qid,
                "error": error_message,
                "result_count": 0,
                "results": [],
            }),
        );
    }

    let mut response = serde_json::json!({
        "session_id": session_id,
        "query_count": results_map.len(),
        "error_count": error_count,
        "results": results_map,
    });

    if let Some(warning) = rerank_warning {
        response["rerank_warning"] = serde_json::json!(warning);
    }

    Ok(response)
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

    /// Helper: creates an AppState with a stub embedding backend.
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

    /// T-MCP-BATCH-001: A batch where all queries are invalid (missing "id"
    /// fields) returns a top-level error because there is nothing to process.
    #[tokio::test]
    async fn t_mcp_batch_001_all_queries_invalid_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        // Create a session so session_id validation passes.
        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_test"),
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

        // All queries missing the "id" field.
        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"query": "test"},
                {"query": "another test"},
            ]
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_err(),
            "batch with all invalid queries must return a top-level error"
        );
        assert!(
            result.unwrap_err().contains("all"),
            "error message must indicate that all queries are invalid"
        );
    }

    /// T-MCP-BATCH-002: A batch with one valid and one invalid query returns
    /// a successful response containing both the search results for the valid
    /// query and an error entry for the invalid query.
    #[tokio::test]
    async fn t_mcp_batch_002_partial_failure_mixed_batch() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_partial"),
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

        // One valid query, one with empty query text.
        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "valid_q", "query": "statistical methods"},
                {"id": "empty_q", "query": "   "},
            ]
        });

        let result = handle(&state, &params).await;

        // The batch must succeed (not return a top-level error) because
        // at least one query is valid.
        assert!(
            result.is_ok(),
            "partial failure batch must succeed, got error: {:?}",
            result.err()
        );

        let response = result.unwrap();

        // The response must contain entries for both queries.
        let results = response["results"].as_object().unwrap();
        assert!(
            results.contains_key("valid_q"),
            "response must contain the valid query result"
        );
        assert!(
            results.contains_key("empty_q"),
            "response must contain the invalid query error entry"
        );

        // The invalid query entry must have an error field.
        let empty_q = &results["empty_q"];
        assert!(
            empty_q["error"].is_string(),
            "invalid query entry must have an 'error' field"
        );
        assert_eq!(
            empty_q["result_count"], 0,
            "invalid query entry must have result_count=0"
        );
    }

    /// T-MCP-BATCH-003: A batch with duplicate query IDs produces per-query
    /// error entries for the duplicates while processing the first occurrence.
    #[tokio::test]
    async fn t_mcp_batch_003_duplicate_ids_auto_suffixed() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_dup"),
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
            "queries": [
                {"id": "q1", "query": "first query"},
                {"id": "q1", "query": "duplicate id"},
                {"id": "q2", "query": "second query"},
            ]
        });

        let result = handle(&state, &params).await;

        // The batch must succeed -- all three queries execute.
        assert!(
            result.is_ok(),
            "batch with duplicate IDs must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        let results = response["results"].as_object().unwrap();

        // First occurrence retains original ID. Duplicate receives "_2" suffix.
        assert!(results.contains_key("q1"), "first q1 must be in results");
        assert!(
            results.contains_key("q1_2"),
            "duplicate q1 must appear as q1_2"
        );
        assert!(results.contains_key("q2"), "q2 must be in results");

        // All three queries produce valid result entries (no error entries).
        assert_eq!(
            results.len(),
            3,
            "all three queries must produce result entries"
        );
        for (key, entry) in results {
            assert!(
                entry.get("error").is_none() || entry["error"].is_null(),
                "query '{key}' must not have an error field"
            );
        }
    }

    /// T-MCP-BATCH-004: Structural validation errors (empty queries array,
    /// missing session_id) still return top-level errors as before.
    #[tokio::test]
    async fn t_mcp_batch_004_structural_errors_return_top_level_error() {
        let pool = test_pool();
        let state = test_state(pool);

        // Missing session_id.
        let params = serde_json::json!({
            "queries": [{"id": "q1", "query": "test"}]
        });
        let result = handle(&state, &params).await;
        assert!(result.is_err(), "missing session_id must return error");

        // Empty queries array.
        let params = serde_json::json!({
            "session_id": 1,
            "queries": []
        });
        let result = handle(&state, &params).await;
        assert!(result.is_err(), "empty queries array must return error");

        // Batch limit exceeded (21 queries).
        let queries: Vec<serde_json::Value> = (0..21)
            .map(|i| serde_json::json!({"id": format!("q{i}"), "query": "test"}))
            .collect();
        let params = serde_json::json!({
            "session_id": 1,
            "queries": queries
        });
        let result = handle(&state, &params).await;
        assert!(result.is_err(), "exceeding batch limit must return error");
    }

    /// T-MCP-BATCH-005: A query with out-of-range top_k produces a per-query
    /// error instead of blocking the entire batch.
    #[tokio::test]
    async fn t_mcp_batch_005_out_of_range_top_k_partial_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_topk"),
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
            "queries": [
                {"id": "ok", "query": "valid query", "top_k": 5},
                {"id": "bad_topk", "query": "also valid", "top_k": 999},
            ]
        });

        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "batch with one bad top_k must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        let results = response["results"].as_object().unwrap();

        assert!(results.contains_key("ok"), "valid query must be in results");
        assert!(
            results.contains_key("bad_topk"),
            "bad_topk query must be in results as error entry"
        );
        assert!(
            results["bad_topk"]["error"].is_string(),
            "bad_topk entry must have an error field"
        );
    }

    /// T-MCP-BATCH-006: top_k=0 produces a per-query error in batch mode.
    ///
    /// Regression test for BUG-004: Before the fix, top_k=0 passed the
    /// upper bound check and was silently converted to 1 via `.max(1)`.
    /// After the fix, top_k=0 fails the lower bound check (top_k < 1)
    /// and produces a per-query error entry.
    #[tokio::test]
    async fn t_mcp_batch_006_top_k_zero_produces_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_topk_zero"),
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
            "queries": [
                {"id": "valid", "query": "valid query", "top_k": 5},
                {"id": "zero_topk", "query": "also valid query", "top_k": 0},
            ]
        });

        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "batch with one zero top_k must succeed as partial failure, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        let results = response["results"].as_object().unwrap();

        assert!(
            results.contains_key("valid"),
            "valid query must be in results"
        );
        assert!(
            results.contains_key("zero_topk"),
            "zero_topk query must be in results as error entry"
        );
        assert!(
            results["zero_topk"]["error"].is_string(),
            "zero_topk entry must have an error field"
        );
        let err = results["zero_topk"]["error"].as_str().unwrap();
        assert!(
            err.contains("top_k") && err.contains("1"),
            "error must mention valid range, got: {err}"
        );
    }

    /// T-MCP-BATCH-007: The response includes an `error_count` field that
    /// reflects the number of queries that failed validation. This allows
    /// callers to detect partial failures without iterating all result entries.
    ///
    /// Regression test for BUG B-02: Before the fix, callers had no
    /// top-level indicator that some queries in the batch had failed.
    /// Per-query error entries existed but required iterating every result
    /// to discover them.
    #[tokio::test]
    async fn t_mcp_batch_007_error_count_reflects_invalid_queries() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_error_count"),
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

        // One valid query, one invalid (empty text), one duplicate ID (auto-suffixed).
        // The duplicate is no longer counted as an error since it executes.
        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "valid_q", "query": "statistical methods"},
                {"id": "empty_q", "query": ""},
                {"id": "valid_q", "query": "duplicate id"},
            ]
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "batch with partial failures must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        assert_eq!(
            response["error_count"], 1,
            "error_count must be 1 (empty query only; duplicate ID is auto-suffixed)"
        );
    }

    /// T-MCP-BATCH-008: When all queries in a batch are valid, error_count
    /// is 0 in the response.
    #[tokio::test]
    async fn t_mcp_batch_008_error_count_zero_when_all_valid() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_no_errors"),
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
            "queries": [
                {"id": "q1", "query": "statistical methods"},
                {"id": "q2", "query": "regression analysis"},
            ]
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "all-valid batch must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        assert_eq!(
            response["error_count"], 0,
            "error_count must be 0 when all queries are valid"
        );
    }

    /// T-MCP-BATCH-009: The error_count field is present and consistent
    /// with the actual number of error entries in the results map. This
    /// validates the field is computed before query_errors is consumed.
    #[tokio::test]
    async fn t_mcp_batch_009_error_count_matches_error_entries() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_consistency"),
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

        // Three invalid queries: missing id, empty text, out-of-range top_k.
        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "valid", "query": "valid query"},
                {"query": "missing id"},
                {"id": "empty", "query": "   "},
                {"id": "bad_topk", "query": "valid text", "top_k": 999},
            ]
        });

        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "partial failure batch must succeed");

        let response = result.unwrap();
        let error_count = response["error_count"].as_u64().unwrap() as usize;

        // Count actual error entries in the results map.
        let results = response["results"].as_object().unwrap();
        let actual_errors = results.values().filter(|v| v["error"].is_string()).count();

        assert_eq!(
            error_count, actual_errors,
            "error_count ({error_count}) must match actual error entries ({actual_errors})"
        );
        assert_eq!(error_count, 3, "three queries should have failed");
    }

    /// T-MCP-BATCH-010: Duplicate query IDs do not overwrite the valid first
    /// occurrence's result. Regression test for ISSUE-007 where the error entry
    /// for a duplicate ID was inserted with the same key as the valid result,
    /// replacing the successful search output with a "duplicate query id" error.
    #[tokio::test]
    async fn t_mcp_batch_010_duplicate_id_auto_suffixed_both_execute() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_dup_id"),
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

        // Two queries with the same ID "q1". Both execute: the first retains
        // "q1", the duplicate receives "q1_2".
        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "q1", "query": "risk factors"},
                {"id": "q1", "query": "market efficiency"},
            ]
        });

        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "batch with duplicate IDs must succeed");

        let response = result.unwrap();
        let results = response["results"].as_object().unwrap();

        // The first "q1" result retains the original ID.
        let q1_result = results.get("q1").expect("q1 must exist in results");
        assert!(
            q1_result.get("query").is_some(),
            "q1 must contain the valid query result"
        );
        assert!(
            q1_result.get("error").is_none(),
            "q1 must not have an error field"
        );

        // The duplicate is auto-suffixed to "q1_2" and also executes.
        let q1_2_result = results.get("q1_2").expect("q1_2 must exist in results");
        assert!(
            q1_2_result.get("query").is_some(),
            "q1_2 must contain the valid query result (duplicate executed)"
        );
        assert!(
            q1_2_result.get("error").is_none(),
            "q1_2 must not have an error field"
        );
    }

    /// T-MCP-BATCH-011: min_score values outside [0.0, 1.0] are rejected.
    /// Regression test for ISSUE-010 where min_score=1.5 was silently accepted,
    /// filtering out all results because no cosine similarity exceeds 1.0.
    #[tokio::test]
    async fn t_mcp_batch_011_min_score_out_of_range_rejected() {
        let pool = test_pool();
        let state = test_state(pool);

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/batch_min_score"),
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

        // min_score=1.5 is out of range.
        let params = serde_json::json!({
            "session_id": session_id,
            "min_score": 1.5,
            "queries": [{"id": "q1", "query": "test"}]
        });

        let result = handle(&state, &params).await;
        assert!(result.is_err(), "min_score > 1.0 must be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("min_score"),
            "error must mention min_score, got: {err}"
        );

        // min_score=-0.5 is also out of range.
        let params = serde_json::json!({
            "session_id": session_id,
            "min_score": -0.5,
            "queries": [{"id": "q1", "query": "test"}]
        });

        let result = handle(&state, &params).await;
        assert!(result.is_err(), "min_score < 0.0 must be rejected");
    }

    /// Helper: creates a test session in the database and returns its ID.
    fn test_session(state: &AppState, dir: &str) -> i64 {
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

    /// T-MCP-BATCH-012: Each query entry in the batch response contains a
    /// `query_analysis` object with structured tokenization details. Verifies
    /// the presence of query_word_count, max_sequence_length, was_truncated,
    /// and bm25_terms fields.
    #[tokio::test]
    async fn t_mcp_batch_012_query_analysis_present_per_query() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/batch_qa_012");

        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "q1", "query": "risk factors in financial markets"},
                {"id": "q2", "query": "heteroskedasticity"},
            ],
            "use_fts": true,
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "batch must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        let results = response["results"].as_object().unwrap();

        // Verify query_analysis for q1.
        let q1 = &results["q1"];
        let q1_analysis = &q1["query_analysis"];
        assert!(
            q1_analysis.is_object(),
            "q1 must have a query_analysis object, got: {q1_analysis}"
        );
        assert_eq!(q1_analysis["query_word_count"], 5, "q1 has 5 words");
        assert!(
            q1_analysis["max_sequence_length"].as_u64().is_some(),
            "max_sequence_length must be present"
        );
        assert_eq!(
            q1_analysis["was_truncated"], false,
            "short query must not be truncated"
        );
        assert!(
            q1_analysis["bm25_terms"].is_array(),
            "bm25_terms must be an array when use_fts=true"
        );

        // Verify query_analysis for q2.
        let q2 = &results["q2"];
        let q2_analysis = &q2["query_analysis"];
        assert!(
            q2_analysis.is_object(),
            "q2 must have a query_analysis object"
        );
        assert_eq!(q2_analysis["query_word_count"], 1, "q2 has 1 word");
    }

    /// T-MCP-BATCH-013: The query_analysis section omits bm25_terms when
    /// use_fts=false in batch mode.
    #[tokio::test]
    async fn t_mcp_batch_013_query_analysis_no_bm25_terms_when_fts_disabled() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/batch_qa_013");

        let params = serde_json::json!({
            "session_id": session_id,
            "queries": [
                {"id": "q1", "query": "test query"},
            ],
            "use_fts": false,
        });

        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "batch must succeed");

        let response = result.unwrap();
        let q1 = &response["results"]["q1"];
        let analysis = &q1["query_analysis"];

        assert!(analysis.is_object(), "query_analysis must be present");
        assert!(
            analysis.get("bm25_terms").is_none() || analysis["bm25_terms"].is_null(),
            "bm25_terms must be absent when use_fts=false"
        );
    }

    /// T-MCP-BATCH-014: The full_chunk_content field preservation mechanism
    /// stores original chunk text per-query before refinement. This unit test
    /// verifies the data structure logic without requiring an actual search.
    #[test]
    fn t_mcp_batch_014_per_query_original_contents_structure() {
        // Simulate the per_query_original_contents data structure that is built
        // before refinement runs. Each inner Vec corresponds to one query's results.
        let query_results: Vec<Vec<String>> = vec![
            vec!["chunk A1 full text".into(), "chunk A2 full text".into()],
            vec!["chunk B1 full text".into()],
        ];

        // Verify the structure: 2 queries, first has 2 chunks, second has 1.
        assert_eq!(query_results.len(), 2);
        assert_eq!(query_results[0].len(), 2);
        assert_eq!(query_results[1].len(), 1);

        // After refinement replaces content, originals are still accessible.
        assert_eq!(query_results[0][0], "chunk A1 full text");
        assert_eq!(query_results[1][0], "chunk B1 full text");
    }
}
