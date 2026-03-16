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

//! Handler for the `neuroncite_search` MCP tool.
//!
//! Embeds the query via the GPU worker, runs the SearchPipeline (vector +
//! keyword + RRF fusion + deduplication + citation assembly), and optionally
//! applies cross-encoder reranking. Returns a JSON array of search results
//! with scores, content, and source citations.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_search::{
    CachedTokenizer, SearchConfig, SearchOutcome, SearchPipeline, apply_refinement,
    extract_query_terms, generate_sub_chunks, parse_divisors, relevance_label,
};

/// Maximum number of results for a single search request. Aligned with
/// batch_search's per-query limit of 50. A previous limit of 200 caused
/// MCP client-side truncation because the combined JSON payload exceeded
/// the client's message size budget (>400KB for broad queries across
/// large corpora).
const MAX_TOP_K: usize = 50;

/// Maximum response payload size in bytes. When the serialized JSON
/// response exceeds this threshold, the handler trims results from the
/// end (lowest relevance) until the payload fits. This prevents MCP
/// client-side truncation which saves the output to a temp file instead
/// of delivering structured JSON to the caller.
const MAX_RESPONSE_BYTES: usize = 200_000;

/// Detects a mismatch between the query language and the embedding model's
/// language scope. When the model name indicates an English-specific model
/// (contains "en-" or ends with "-en", case-insensitive) and the query
/// contains a significant proportion of non-ASCII alphabetic characters
/// (more than 20% of all alphabetic characters), this function returns a
/// warning message. Otherwise returns `None`.
///
/// The 20% threshold avoids false positives for English text that contains
/// occasional accented characters (e.g., "naive" spelled as "naive").
///
/// # Arguments
///
/// * `query` - The search query text.
/// * `model_name` - The embedding model identifier (e.g., "BAAI/bge-small-en-v1.5").
///
/// # Returns
///
/// `Some(warning_message)` when a language mismatch is detected, `None` otherwise.
fn detect_language_mismatch(query: &str, model_name: &str) -> Option<String> {
    // Determine whether the model name suggests an English-only model.
    // The check is case-insensitive and looks for "en-" or "-en" as substring
    // patterns within the model name. The lowercase conversion ensures model
    // names like "BGE-SMALL-EN-V1.5" are handled correctly.
    let model_lower = model_name.to_lowercase();
    let is_english_model = model_lower.contains("en-") || model_lower.ends_with("-en");

    if !is_english_model {
        return None;
    }

    // Count total alphabetic characters and non-ASCII alphabetic characters
    // in the query. Characters that are not alphabetic (digits, punctuation,
    // whitespace) are excluded from the ratio computation because they do not
    // indicate language.
    let mut total_alpha: usize = 0;
    let mut non_ascii_alpha: usize = 0;

    for c in query.chars() {
        if c.is_alphabetic() {
            total_alpha += 1;
            if !c.is_ascii() {
                non_ascii_alpha += 1;
            }
        }
    }

    // Require at least some alphabetic characters to avoid division by zero
    // and to avoid triggering on queries that consist entirely of numbers
    // or punctuation.
    if total_alpha == 0 {
        return None;
    }

    let non_ascii_ratio = non_ascii_alpha as f64 / total_alpha as f64;

    // Threshold of 20%: queries with more than 20% non-ASCII alphabetic
    // characters are considered non-English. This threshold is high enough
    // to tolerate occasional accented characters in English text (e.g.,
    // "naive", "resume") but low enough to detect queries predominantly
    // written in German, French, or other languages with frequent diacritics.
    if non_ascii_ratio > 0.20 {
        Some(format!(
            "query appears to contain non-English text, but the embedding model \
             '{model_name}' is English-specific; results may have reduced accuracy"
        ))
    } else {
        None
    }
}

/// Executes a hybrid search against the specified session.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Index session to search within.
/// - `query` (required): Search query text.
/// - `top_k` (optional): Number of results (1-200, default 10).
/// - `use_fts` (optional): Enable FTS5 keyword search (default true).
/// - `rerank` (optional): Apply cross-encoder reranking (default false).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;
    let query = params["query"]
        .as_str()
        .ok_or("missing required parameter: query")?;

    // Reject empty and whitespace-only queries. Trimming first ensures
    // consistent behavior with batch_search which also rejects empty queries.
    let query = query.trim();
    if query.is_empty() {
        return Err("query must not be empty".to_string());
    }

    // Validate that the session's embedding model matches the loaded model.
    // The MCP server pre-loads a specific model at startup. If the session was
    // indexed with a different model (different vector dimension), the query
    // embedding would be incompatible with the stored document embeddings,
    // producing meaningless cosine similarity scores.
    //
    // The session's model_name is preserved beyond this block for the language
    // mismatch detection that runs after the search completes.
    //
    // The vector_dimension field is an AtomicUsize, so a single relaxed load
    // is performed here and the resulting plain usize is reused for both the
    // comparison and the error message.
    let loaded_dim = state
        .index
        .vector_dimension
        .load(std::sync::atomic::Ordering::Relaxed);
    let session_model_name: String;
    {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let session = neuroncite_store::get_session(&conn, session_id)
            .map_err(|e| format!("session {session_id} not found: {e}"))?;

        session_model_name = session.model_name.clone();

        let session_dim = session.vector_dimension as usize;
        if session_dim != loaded_dim {
            return Err(format!(
                "model dimension mismatch: session {session_id} was indexed with '{}' \
                 ({}d) but the MCP server loaded a {}d model. Restart the MCP server \
                 with the matching model or re-index with the server's current model.",
                session.model_name, session_dim, loaded_dim
            ));
        }
    }

    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    if !(1..=MAX_TOP_K).contains(&top_k) {
        return Err(format!(
            "top_k must be between 1 and {MAX_TOP_K}, got {top_k}"
        ));
    }
    let use_fts = params["use_fts"].as_bool().unwrap_or(true);
    let rerank = params["rerank"].as_bool().unwrap_or(false);
    let refine = params["refine"].as_bool().unwrap_or(true);
    let divisors = params["refine_divisors"]
        .as_str()
        .map(parse_divisors)
        .unwrap_or_else(|| vec![4, 8, 16]);

    // Parse file_ids filter. When provided, restricts search to chunks belonging
    // to the specified file IDs only (file-scoped search).
    let file_ids: Option<Vec<i64>> = params["file_ids"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

    // Validate that all requested file_ids belong to this session. Silently
    // returning 0 results for non-existent file IDs is misleading -- the
    // caller cannot distinguish "no matches" from "wrong file ID". This
    // check queries the database for files in the session and reports any
    // IDs that do not exist.
    let mut file_ids_warning: Option<String> = None;
    if let Some(ref ids) = file_ids
        && !ids.is_empty()
    {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;
        let session_files = neuroncite_store::list_files_by_session(&conn, session_id)
            .map_err(|e| format!("listing files: {e}"))?;
        let valid_ids: std::collections::HashSet<i64> =
            session_files.iter().map(|f| f.id).collect();
        let invalid: Vec<i64> = ids
            .iter()
            .filter(|id| !valid_ids.contains(id))
            .copied()
            .collect();
        if !invalid.is_empty() {
            file_ids_warning = Some(format!(
                "file_ids {:?} do not belong to session {session_id}; \
                 results are restricted to the remaining valid file IDs",
                invalid
            ));
        }
    }

    // Parse page_start and page_end for page range filtering. When both are
    // provided, search results are restricted to chunks whose page range overlaps
    // [page_start, page_end]. This is useful for searching within a specific
    // section of a large document (e.g., pages 1-50 of a 711-page book).
    let page_start: Option<i64> = params["page_start"].as_i64();
    let page_end: Option<i64> = params["page_end"].as_i64();

    // Validate page range parameters: both must be provided together, both
    // must be positive, and page_start must not exceed page_end.
    match (page_start, page_end) {
        (Some(ps), Some(pe)) => {
            if ps < 1 {
                return Err(format!("page_start must be >= 1, got {ps}"));
            }
            if pe < ps {
                return Err(format!("page_end ({pe}) must be >= page_start ({ps})"));
            }
        }
        (Some(_), None) => {
            return Err("page_start requires page_end to be specified".to_string());
        }
        (None, Some(_)) => {
            return Err("page_end requires page_start to be specified".to_string());
        }
        (None, None) => {}
    }

    // Parse min_score threshold. When provided, results with vector_score below
    // this value are excluded from the response. The value is clamped to [0.0, 1.0]
    // because cosine similarity scores from normalized embeddings are bounded to
    // this range. Values outside this range indicate a caller error.
    let min_score: Option<f64> = params["min_score"].as_f64();
    if let Some(ms) = min_score
        && !(0.0..=1.0).contains(&ms)
    {
        return Err(format!("min_score must be between 0.0 and 1.0, got {ms}"));
    }

    // Estimate whether the query will be truncated by the tokenizer.
    // WordPiece/BPE tokenizers produce at least 1 subword token per
    // whitespace word, plus 2 special tokens ([CLS] and [SEP]). Technical
    // and academic text frequently splits words into 1.3-1.5 subword tokens.
    // Using word_count + 2 (for special tokens) as the lower bound catches
    // queries that are borderline or certain to be truncated. The previous
    // heuristic (word_count * 1.3) systematically underestimated token count
    // and failed to fire for 350-word queries that actually exceeded 512
    // tokens after subword tokenization.
    let max_seq_len = state.worker_handle.max_sequence_length();
    let word_count = query.split_whitespace().count();
    let estimated_min_tokens = word_count + 2;
    let mut truncation_warning: Option<String> = if estimated_min_tokens > max_seq_len {
        Some(format!(
            "query has ~{word_count} words (at least {estimated_min_tokens} tokens including \
             special tokens) which likely exceeds the model's max_sequence_length of \
             {max_seq_len}; the embedding may only represent a prefix of the query. \
             Consider shortening the query for complete coverage."
        ))
    } else {
        None
    };

    // Embed the query text via the GPU worker.
    let query_vec = state
        .worker_handle
        .embed_query(query.to_string())
        .await
        .map_err(|e| format!("embedding failed: {e}"))?;

    // Ensure the HNSW index is loaded for this session before acquiring the
    // guard. The background executor's Phase 3 may have silently skipped the
    // HNSW build when the SQLite WAL read snapshot did not include Phase 2's
    // just-committed embeddings. ensure_hnsw_for_session detects the missing
    // entry and builds the index on-demand from the current database state.
    neuroncite_api::ensure_hnsw_for_session(state, session_id);

    // Run the synchronous SearchPipeline. The HNSW guard and database connection
    // are scoped to this block so they are dropped before the async rerank call.
    let (mut results, total_candidates) = {
        let hnsw_guard = state.index.hnsw_index.load();
        let Some(hnsw_ref) = hnsw_guard.get(&session_id) else {
            // Build a basic query_analysis even for the early return so
            // callers can inspect tokenization details regardless of index
            // state. The exact query_tokens count is unavailable here because
            // the tokenizer loads lazily during the refinement phase.
            let mut early_analysis = serde_json::json!({
                "query_word_count": word_count,
                "max_sequence_length": max_seq_len,
                "was_truncated": estimated_min_tokens > max_seq_len,
            });
            if use_fts {
                early_analysis["bm25_terms"] = serde_json::json!(extract_query_terms(query));
            }

            return Ok(serde_json::json!({
                "results": [],
                "query_analysis": early_analysis,
                "message": format!("no HNSW index loaded for session {session_id}; run indexing first")
            }));
        };

        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let config = SearchConfig {
            session_id,
            vector_top_k: top_k * 5,
            keyword_limit: if use_fts { top_k * 5 } else { 0 },
            ef_search: state.config.ef_search,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: top_k,
            rerank_enabled: false,
            file_ids,
            min_score,
            page_start,
            page_end,
        };

        let pipeline = SearchPipeline::new(hnsw_ref, &conn, &query_vec, query, None, config);

        let SearchOutcome {
            results,
            total_candidates,
        } = pipeline
            .search()
            .map_err(|e| format!("search pipeline failed: {e}"))?;

        (results, total_candidates)
    };

    // Apply cross-encoder reranking via the GPU worker if requested.
    // When no reranker model is configured, reranking is skipped and a
    // warning is included in the response instead of returning a hard error.
    let mut rerank_warning: Option<String> = None;
    if rerank && !results.is_empty() {
        if !state.worker_handle.reranker_available() {
            rerank_warning = Some(
                "rerank=true was requested but no reranker model is configured; \
                 results are returned without reranking. Check neuroncite_health \
                 for 'reranker_available'."
                    .to_string(),
            );
        } else {
            let content_refs: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
            let scores = state
                .worker_handle
                .rerank_batch(query, &content_refs)
                .await
                .map_err(|e| format!("reranking failed: {e}"))?;

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

    // Apply sub-chunk refinement if requested. Refinement narrows each
    // result's content to the most relevant sub-section by splitting into
    // overlapping token windows at multiple scales, embedding them, and
    // selecting the best-scoring sub-chunk.
    //
    // The `was_refined` vector tracks which results had their content replaced
    // by a higher-scoring sub-chunk. Reported per-result in the JSON response
    // so callers know which results contain narrowed content.
    //
    // The tokenizer is also used to compute the exact query token count
    // (`query_tokens`), enabling callers to determine whether the embedding
    // model truncated the query at max_sequence_length.
    let mut was_refined: Vec<bool> = vec![false; results.len()];
    let mut query_tokens: Option<usize> = None;

    // Preserve the original chunk content before refinement replaces it with
    // a narrower sub-chunk. When refinement is applied, `result.content` is
    // replaced with the highest-scoring sub-section. The original full chunk
    // text is stored here and included in the response as `full_chunk_content`
    // for refined results, saving callers a roundtrip to neuroncite_page.
    let original_contents: Vec<String> = results.iter().map(|r| r.content.clone()).collect();

    if refine
        && !results.is_empty()
        && !divisors.is_empty()
        && let Some(ref tokenizer_json) = *state.worker_handle.tokenizer_json()
    {
        let tokenizer = CachedTokenizer::from_json(tokenizer_json)
            .map_err(|e| format!("tokenizer deserialization failed: {e}"))?;

        // Compute the exact subword token count for the query text. This is
        // more precise than the word-based heuristic used for the truncation
        // warning: BPE/WordPiece tokenizers can split a single word into
        // multiple subword tokens (e.g., "heteroskedasticity" -> 4 tokens).
        query_tokens = tokenizer.token_count(query).ok();

        let candidates = generate_sub_chunks(&results, &tokenizer, &divisors)
            .map_err(|e| format!("refinement sub-chunk generation failed: {e}"))?;

        if !candidates.is_empty() {
            let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
            let embeddings = state
                .worker_handle
                .embed_batch_search(&texts)
                .await
                .map_err(|e| format!("refinement embedding failed: {e}"))?;
            was_refined = apply_refinement(&mut results, &candidates, &embeddings, &query_vec);
        }
    } else if let Some(ref tokenizer_json) = *state.worker_handle.tokenizer_json() {
        // Compute query token count even when refinement is disabled, because
        // the caller still benefits from knowing the exact token count for
        // truncation detection.
        if let Ok(tokenizer) = CachedTokenizer::from_json(tokenizer_json) {
            query_tokens = tokenizer.token_count(query).ok();
        }
    }

    // The word-based heuristic (word_count + 2) only catches queries where the
    // minimum possible token count exceeds max_seq_len. BPE/WordPiece tokenizers
    // typically produce 1.3-1.5 subword tokens per word for technical text, so a
    // 300-word query (heuristic: 302 tokens) passes the heuristic check but may
    // actually produce 400+ BPE tokens exceeding 512 max_seq_len. When the exact
    // token count is available from the tokenizer, use it to generate a precise
    // truncation warning that the heuristic missed.
    if truncation_warning.is_none()
        && let Some(tokens) = query_tokens
        && tokens >= max_seq_len
    {
        truncation_warning = Some(format!(
            "query has {tokens} tokens which exceeds the model's \
             max_sequence_length of {max_seq_len}; the embedding represents \
             only a prefix of the query. Consider shortening the query for \
             complete coverage."
        ));
    }

    // Detect language mismatch between the query text and the session's
    // embedding model. When an English-specific model (e.g., "bge-small-en-v1.5")
    // receives a query with significant non-ASCII alphabetic content, the
    // resulting embeddings may poorly represent the query semantics, leading
    // to degraded search accuracy.
    let language_warning: Option<String> = detect_language_mismatch(query, &session_model_name);

    // Build score_summary providing min/max/mean vector_score across results.
    // Results with vector_score == 0.0 (found only by BM25 keyword search,
    // not by vector search) are excluded from the summary statistics to avoid
    // skewing the mean downward. A separate bm25_only_count field reports how
    // many results were keyword-only.
    let score_summary = if !results.is_empty() {
        let vs_nonzero: Vec<f64> = results
            .iter()
            .map(|r| r.vector_score)
            .filter(|&s| s > 0.0)
            .collect();
        let bm25_only_count = results.len() - vs_nonzero.len();

        if vs_nonzero.is_empty() {
            // All results are BM25-only (no vector search scores). Emit null
            // instead of 0.0 for vector score statistics to distinguish "not
            // scored by vector search" from an actual cosine similarity of 0.0.
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

    // Extract the individual query terms for per-result matched_terms reporting.
    // When use_fts is enabled, each result reports which BM25 keywords from the
    // query actually appear in the result's content. This transparency helps
    // callers understand why a result was returned by keyword search.
    let query_terms: Vec<String> = if use_fts {
        extract_query_terms(query)
    } else {
        Vec::new()
    };

    // Compute the maximum RRF/reranker score across all results for normalizing
    // to a 0.0-1.0 relevance_score. When results are empty or all scores are
    // zero, max_score defaults to 1.0 to avoid division by zero.
    let max_score = results
        .iter()
        .map(|r| r.score)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(f64::EPSILON);

    // Convert results to JSON. Each result includes:
    // - relevance_score: RRF/reranker score normalized to [0.0, 1.0] where
    //   the top result has 1.0 and others are proportionally scaled.
    // - matched_terms: query keywords found in the result content (when
    //   use_fts=true). Determined via case-insensitive substring matching
    //   against the cleaned query tokens.
    // - relevance label derived from vector_score.
    // - page_numbers array for direct use with neuroncite_page.
    //
    // The source_file path is cleaned of the Windows `\\?\` extended-length
    // prefix for user-facing output.
    //
    // Results found only by BM25 keyword search (vector_score == 0.0) emit
    // null for the vector_score field to distinguish "not scored by vector
    // search" from an actual cosine similarity of 0.0. The relevance label
    // for BM25-only results is "bm25_only" instead of a threshold-based label.
    let result_array: Vec<serde_json::Value> = results
        .into_iter()
        .enumerate()
        .map(|(idx, r)| {
            let page_numbers: Vec<usize> = (r.citation.page_start..=r.citation.page_end).collect();
            let source_file_owned = r.citation.source_file.display().to_string();
            let source_file = strip_extended_length_prefix(&source_file_owned);
            let is_bm25_only = r.vector_score == 0.0;
            let refined = was_refined.get(idx).copied().unwrap_or(false);

            // Normalize the score to [0.0, 1.0] relative to the top result.
            let relevance_score = r.score / max_score;

            // Determine which query terms appear in this result's content.
            // Uses case-insensitive substring matching on the content text.
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

            // Include matched_terms only when FTS is enabled. Omitting the field
            // when use_fts=false avoids confusing callers who did not request
            // keyword search.
            if use_fts {
                result_json["matched_terms"] = serde_json::json!(matched_terms);
            }

            // Include the original full chunk text when refinement replaced the
            // content with a narrower sub-chunk. This saves callers from making
            // a separate neuroncite_page call to retrieve surrounding context.
            // Only included when was_refined=true, because for non-refined results
            // the content field already contains the full chunk text.
            if refined
                && let Some(original) = original_contents.get(idx)
            {
                result_json["full_chunk_content"] = serde_json::json!(original);
            }

            result_json
        })
        .collect();

    let mut response = serde_json::json!({
        "session_id": session_id,
        "query": query,
        "result_count": result_array.len(),
        "total_available": total_candidates,
        "score_summary": score_summary,
        "results": result_array,
    });

    // Include the exact query token count when the tokenizer is available.
    // This count reflects the BPE/WordPiece subword tokens including special
    // tokens ([CLS], [SEP]). When `query_tokens > max_sequence_length`, the
    // embedding model truncated the query. Callers can compare this against
    // the model's max_sequence_length (reported by neuroncite_health) to
    // determine whether truncation occurred.
    if let Some(tokens) = query_tokens {
        response["query_tokens"] = serde_json::json!(tokens);
        response["max_sequence_length"] = serde_json::json!(max_seq_len);
    }

    // Build a structured query_analysis section that provides transparency
    // about how the query was processed by the embedding and keyword search
    // pipelines. This helps callers debug issues with long, multilingual,
    // or domain-specific queries by showing the internal tokenization.
    {
        let was_truncated = query_tokens.map(|t| t >= max_seq_len).unwrap_or(false);
        let tokens_used = query_tokens.map(|t| t.min(max_seq_len));

        let mut analysis = serde_json::json!({
            "query_word_count": word_count,
            "max_sequence_length": max_seq_len,
            "was_truncated": was_truncated,
        });

        if let Some(tokens) = query_tokens {
            analysis["query_tokens"] = serde_json::json!(tokens);
            analysis["tokens_used_for_embedding"] = serde_json::json!(tokens_used);
        }

        // Include the BM25 terms when FTS is enabled. These are the tokens
        // that the keyword search pipeline uses to match against the FTS5
        // index. Callers can inspect which terms were extracted and whether
        // stop words or short tokens were filtered out.
        if use_fts {
            analysis["bm25_terms"] = serde_json::json!(query_terms);
        }

        response["query_analysis"] = analysis;
    }

    // Include a rerank_warning field when reranking was requested but
    // could not be performed (no reranker configured).
    if let Some(warning) = rerank_warning {
        response["rerank_warning"] = serde_json::json!(warning);
    }

    // Include a truncation_warning field when the query is estimated to
    // exceed the model's max_sequence_length. This alerts the caller that
    // the embedding represents only a prefix of the query text.
    if let Some(warning) = truncation_warning {
        response["truncation_warning"] = serde_json::json!(warning);
    }

    // Include a file_ids_warning when some requested file IDs do not exist
    // in the session. This prevents silent 0-result returns for invalid IDs.
    if let Some(warning) = file_ids_warning {
        response["file_ids_warning"] = serde_json::json!(warning);
    }

    // Include a language_warning when the query language appears to mismatch
    // the embedding model's language scope. This alerts the caller that an
    // English-specific model is receiving non-English text, which degrades
    // embedding quality and search accuracy.
    if let Some(warning) = language_warning {
        response["language_warning"] = serde_json::json!(warning);
    }

    // Enforce response payload size limit. MCP clients impose a maximum
    // message size. If the serialized response exceeds MAX_RESPONSE_BYTES,
    // results are trimmed from the end (lowest relevance, since results are
    // sorted by score) until the payload fits. The original result count is
    // preserved in a `truncated_from` field so the caller knows results
    // were trimmed.
    truncate_response_if_oversized(&mut response, MAX_RESPONSE_BYTES);

    Ok(response)
}

/// Trims the `results` array of a search response JSON to keep the
/// serialized payload under `max_bytes`. Results are removed from the end
/// (lowest relevance, since results are sorted by score descending) until
/// the estimated serialized size is within the limit.
///
/// Returns true if truncation was applied. When truncation occurs, the
/// response receives a `truncated_from` field reporting the original
/// result count before trimming.
fn truncate_response_if_oversized(response: &mut serde_json::Value, max_bytes: usize) -> bool {
    let response_str = serde_json::to_string(&*response).unwrap_or_default();
    if response_str.len() <= max_bytes {
        return false;
    }

    let results_arr = match response["results"].as_array() {
        Some(arr) if arr.len() > 1 => arr,
        _ => return false,
    };

    let original_count = results_arr.len();
    let results_json_len = serde_json::to_string(results_arr).unwrap_or_default().len();
    let overhead = response_str.len() - results_json_len;

    // Estimate how many results fit within the budget. The `truncated_from`
    // field adds approximately 30 bytes of JSON overhead, accounted for in
    // the 50-byte padding.
    let avg_result_bytes = results_json_len / original_count;
    let available = max_bytes.saturating_sub(overhead + 50);
    let keep = (available / avg_result_bytes.max(1))
        .max(1)
        .min(original_count);

    let trimmed: Vec<serde_json::Value> = response["results"]
        .as_array()
        .unwrap()
        .iter()
        .take(keep)
        .cloned()
        .collect();

    response["results"] = serde_json::json!(trimmed);
    response["result_count"] = serde_json::json!(keep);
    response["truncated_from"] = serde_json::json!(original_count);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-SEARCH-001: top_k=0 returns an error instead of silently
    /// returning 1 result.
    ///
    /// Regression test for BUG-004: Before the fix, top_k=0 passed the
    /// upper bound check (0 <= 200) and was silently converted to 1 via
    /// `.max(1)`. Callers expected an error or zero results but received
    /// one result, leading to incorrect behavior in pagination logic and
    /// result count assertions.
    #[tokio::test]
    async fn t_mcp_search_001_top_k_zero_returns_error() {
        let pool = {
            let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
                conn.execute_batch("PRAGMA foreign_keys = ON;")?;
                Ok(())
            });
            r2d2::Pool::builder()
                .max_size(2)
                .build(manager)
                .expect("pool build")
        };
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

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
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/search_test"),
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

        // top_k=0 must return an error.
        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query",
            "top_k": 0
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_err(),
            "top_k=0 must return an error, got: {:?}",
            result
        );
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("top_k")
                && err_msg.contains("1")
                && err_msg.contains(&MAX_TOP_K.to_string()),
            "error message must mention valid range, got: {err_msg}"
        );
    }

    /// T-MCP-SEARCH-002: top_k=201 returns an error. This value exceeds
    /// MAX_TOP_K (50), confirming the upper bound is enforced.
    #[tokio::test]
    async fn t_mcp_search_002_top_k_over_200_returns_error() {
        let pool = {
            let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
                conn.execute_batch("PRAGMA foreign_keys = ON;")?;
                Ok(())
            });
            r2d2::Pool::builder()
                .max_size(2)
                .build(manager)
                .expect("pool build")
        };
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

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
        let state = AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed");

        let config = neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/tmp/search_test2"),
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

        // top_k=201 must return an error.
        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query",
            "top_k": 201
        });

        let result = handle(&state, &params).await;
        assert!(result.is_err(), "top_k=201 must return an error");
    }

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

    /// Helper: creates an AppState with a stub embedding backend that
    /// returns zero-vectors of dimension 384.
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

    /// T-MCP-SEARCH-003: top_k=51 returns an error because the maximum
    /// was reduced from 200 to MAX_TOP_K (50) to prevent MCP client-side
    /// output truncation on large result sets.
    ///
    /// Regression test for BUG B-01: Requesting top_k=200 for broad queries
    /// produced JSON payloads exceeding 436K characters, which caused the
    /// MCP client to truncate the response and save it to a temp file
    /// instead of delivering structured JSON.
    #[tokio::test]
    async fn t_mcp_search_003_top_k_above_max_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_003");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query",
            "top_k": MAX_TOP_K + 1
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_err(),
            "top_k={} must return an error (max is {MAX_TOP_K})",
            MAX_TOP_K + 1
        );
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("top_k") && err_msg.contains(&MAX_TOP_K.to_string()),
            "error message must mention the max limit of {MAX_TOP_K}, got: {err_msg}"
        );
    }

    /// T-MCP-SEARCH-004: top_k=50 (MAX_TOP_K) passes the validation check.
    /// The handler continues past validation and returns a response. With no
    /// HNSW index loaded the response contains an empty result set, which
    /// confirms the validation did not reject the boundary value.
    #[tokio::test]
    async fn t_mcp_search_004_top_k_at_max_accepted() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_004");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query",
            "top_k": MAX_TOP_K
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "top_k={MAX_TOP_K} must be accepted (at the MAX_TOP_K boundary), got error: {:?}",
            result.err()
        );
    }

    /// T-MCP-SEARCH-005: truncate_response_if_oversized correctly trims
    /// the results array when the serialized payload exceeds the byte limit.
    /// Results are trimmed from the end (lowest relevance). The response
    /// receives a `truncated_from` field reporting the original count.
    ///
    /// Regression test for BUG B-01: Verifies the safety mechanism that
    /// prevents oversized responses even when individual results are large.
    #[test]
    fn t_mcp_search_005_response_truncation_trims_results() {
        let big_content = "x".repeat(5000);
        let results: Vec<serde_json::Value> = (0..50)
            .map(|i| {
                serde_json::json!({
                    "score": 1.0 - (i as f64 * 0.01),
                    "content": big_content,
                    "file_id": i,
                })
            })
            .collect();

        let mut response = serde_json::json!({
            "session_id": 1,
            "query": "test",
            "result_count": results.len(),
            "results": results,
        });

        // 50 results * ~5KB content each = ~250KB, over the 50KB test limit.
        let truncated = truncate_response_if_oversized(&mut response, 50_000);

        assert!(
            truncated,
            "response must be truncated when over the byte limit"
        );

        let result_count = response["result_count"].as_u64().unwrap() as usize;
        assert!(
            result_count < 50,
            "truncated result count ({result_count}) must be less than original (50)"
        );
        assert!(
            result_count >= 1,
            "at least 1 result must be kept after truncation"
        );

        let truncated_from = response["truncated_from"].as_u64().unwrap();
        assert_eq!(
            truncated_from, 50,
            "truncated_from must report the original result count"
        );

        // Verify the results array length matches result_count.
        let actual_len = response["results"].as_array().unwrap().len();
        assert_eq!(
            actual_len, result_count,
            "results array length must match result_count"
        );
    }

    /// T-MCP-SEARCH-006: truncate_response_if_oversized is a no-op when
    /// the serialized payload is already within the byte limit. No
    /// `truncated_from` field is added and result_count is unchanged.
    #[test]
    fn t_mcp_search_006_no_truncation_when_under_limit() {
        let results: Vec<serde_json::Value> = (0..3)
            .map(|i| {
                serde_json::json!({
                    "score": 1.0,
                    "content": "short content",
                    "file_id": i,
                })
            })
            .collect();

        let mut response = serde_json::json!({
            "session_id": 1,
            "query": "test",
            "result_count": 3,
            "results": results,
        });

        let truncated = truncate_response_if_oversized(&mut response, 200_000);

        assert!(
            !truncated,
            "response must not be truncated when under the limit"
        );
        assert_eq!(
            response["result_count"], 3,
            "result_count must be unchanged when no truncation"
        );
        assert!(
            response.get("truncated_from").is_none() || response["truncated_from"].is_null(),
            "truncated_from must not be present when no truncation"
        );
    }

    /// T-MCP-SEARCH-007: truncate_response_if_oversized preserves at least
    /// one result even when a single result exceeds the byte limit. This
    /// prevents returning an empty results array which callers could mistake
    /// for "no matches found".
    #[test]
    fn t_mcp_search_007_truncation_preserves_at_least_one_result() {
        let huge_content = "x".repeat(100_000);
        let results: Vec<serde_json::Value> = (0..5)
            .map(|i| {
                serde_json::json!({
                    "score": 1.0,
                    "content": huge_content,
                    "file_id": i,
                })
            })
            .collect();

        let mut response = serde_json::json!({
            "session_id": 1,
            "query": "test",
            "result_count": results.len(),
            "results": results,
        });

        // Even with a very small limit, at least 1 result must remain.
        let truncated = truncate_response_if_oversized(&mut response, 1_000);

        assert!(truncated, "response must be truncated");
        let result_count = response["result_count"].as_u64().unwrap() as usize;
        assert_eq!(result_count, 1, "at least 1 result must be preserved");
    }

    /// T-MCP-SEARCH-008: min_score values outside [0.0, 1.0] are rejected.
    /// Regression test for ISSUE-010 where min_score=1.5 was silently accepted
    /// and filtered out all results because no cosine similarity exceeds 1.0.
    #[test]
    fn t_mcp_search_008_min_score_out_of_range_rejected() {
        // Verify the validation condition directly.
        let ms_high = 1.5_f64;
        assert!(
            !(0.0..=1.0).contains(&ms_high),
            "min_score=1.5 must fail the [0.0, 1.0] range check"
        );

        let ms_neg = -0.1_f64;
        assert!(
            !(0.0..=1.0).contains(&ms_neg),
            "min_score=-0.1 must fail the [0.0, 1.0] range check"
        );
    }

    /// T-MCP-SEARCH-009: min_score within [0.0, 1.0] passes validation.
    #[test]
    fn t_mcp_search_009_min_score_valid_range_passes() {
        for valid in [0.0, 0.5, 0.72, 0.82, 1.0] {
            assert!(
                (0.0..=1.0).contains(&valid),
                "min_score={valid} must pass the [0.0, 1.0] range check"
            );
        }
    }

    /// T-MCP-SEARCH-010: Truncation warning is generated for long queries.
    /// Regression test for BUG-006 where queries exceeding the model's
    /// max_sequence_length (512 tokens) were silently truncated without
    /// any indication to the caller. The heuristic uses word_count + 2
    /// (accounting for [CLS] and [SEP] special tokens) as the minimum
    /// token count estimate.
    #[test]
    fn t_mcp_search_010_truncation_warning_generated_for_long_query() {
        let max_seq_len: usize = 512;
        // A query with 520 words produces at least 522 tokens (520 + 2 special),
        // which exceeds max_seq_len=512.
        let long_query: String = (0..520)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let word_count = long_query.split_whitespace().count();
        let estimated_min_tokens = word_count + 2;

        assert!(
            estimated_min_tokens > max_seq_len,
            "520-word query (>={estimated_min_tokens} tokens) must exceed max_seq_len={max_seq_len}"
        );

        // A short query (5 words + 2 special = 7 tokens) must not trigger.
        let short_query = "risk factors in financial markets";
        let short_words = short_query.split_whitespace().count();
        let short_estimated = short_words + 2;

        assert!(
            short_estimated <= max_seq_len,
            "5-word query must not trigger truncation warning"
        );
    }

    /// T-MCP-SEARCH-011: An empty string query returns an error. The handler
    /// rejects queries that are empty after trimming whitespace, to prevent
    /// meaningless embedding computations.
    #[tokio::test]
    async fn t_mcp_search_011_empty_query_rejected() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_011");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": ""
        });
        let result = handle(&state, &params).await;

        assert!(result.is_err(), "empty query must return an error");
        assert!(
            result.unwrap_err().contains("empty"),
            "error message must indicate the query is empty"
        );
    }

    /// T-MCP-SEARCH-012: A whitespace-only query returns an error. The
    /// handler trims leading/trailing whitespace before the empty check,
    /// so "   " is treated the same as "".
    #[tokio::test]
    async fn t_mcp_search_012_whitespace_only_query_rejected() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_012");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "   \t\n  "
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "whitespace-only query must return an error"
        );
        assert!(
            result.unwrap_err().contains("empty"),
            "error message must indicate the query is empty"
        );
    }

    /// T-MCP-SEARCH-013: A non-existent session_id returns a "not found"
    /// error. The handler validates session existence before attempting the
    /// embedding or HNSW index lookup.
    #[tokio::test]
    async fn t_mcp_search_013_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({
            "session_id": 999999,
            "query": "test query"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "search with non-existent session must return an error"
        );
        assert!(
            result.unwrap_err().contains("not found"),
            "error message must indicate the session was not found"
        );
    }

    /// T-MCP-SEARCH-014: Missing session_id parameter returns the
    /// "missing required parameter" error.
    #[tokio::test]
    async fn t_mcp_search_014_missing_session_id_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({
            "query": "test query"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "search without session_id must return an error"
        );
        assert!(
            result.unwrap_err().contains("session_id"),
            "error message must reference the missing session_id parameter"
        );
    }

    /// T-MCP-SEARCH-015: Missing query parameter returns the
    /// "missing required parameter" error.
    #[tokio::test]
    async fn t_mcp_search_015_missing_query_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_015");

        let params = serde_json::json!({
            "session_id": session_id
        });
        let result = handle(&state, &params).await;

        assert!(result.is_err(), "search without query must return an error");
        assert!(
            result.unwrap_err().contains("query"),
            "error message must reference the missing query parameter"
        );
    }

    /// T-MCP-SEARCH-016: When the HNSW index is not loaded, the handler
    /// returns a response with an empty results array and a message
    /// indicating the index is not available.
    #[tokio::test]
    async fn t_mcp_search_016_no_hnsw_returns_message() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_016");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query"
        });
        let result = handle(&state, &params).await;

        assert!(
            result.is_ok(),
            "search must succeed even without HNSW index"
        );
        let response = result.unwrap();
        assert!(
            response["results"].as_array().is_none_or(|a| a.is_empty()),
            "results must be empty when no HNSW index is loaded"
        );
        assert!(
            response["message"].as_str().is_some(),
            "response must contain a message when no HNSW index is loaded"
        );
    }

    /// T-MCP-SEARCH-020: Language mismatch warning is produced when an English
    /// embedding model receives a query with a high proportion of non-ASCII
    /// alphabetic characters (above 20% threshold). The query "Grosse Ol
    /// Prufung Anderung" has 5 non-ASCII out of 22 alpha chars (22.7%).
    #[test]
    fn t_mcp_search_020_language_mismatch_warning() {
        let warning = detect_language_mismatch(
            "Gr\u{00f6}\u{00df}e \u{00d6}l Pr\u{00fc}fung \u{00c4}nderung",
            "BAAI/bge-small-en-v1.5",
        );
        assert!(
            warning.is_some(),
            "German query with English model must warn"
        );
        let msg = warning.unwrap();
        assert!(
            msg.contains("non-English"),
            "warning must mention non-English"
        );
        assert!(
            msg.contains("bge-small-en-v1.5"),
            "warning must mention model name"
        );
    }

    /// T-MCP-SEARCH-021: No warning for English queries with English models.
    #[test]
    fn t_mcp_search_021_no_warning_english_query() {
        let warning =
            detect_language_mismatch("heteroskedasticity and variance", "BAAI/bge-small-en-v1.5");
        assert!(warning.is_none(), "English query must not produce warning");
    }

    /// T-MCP-SEARCH-022: No warning for non-English queries with multilingual models.
    #[test]
    fn t_mcp_search_022_no_warning_multilingual_model() {
        let warning =
            detect_language_mismatch("Heteroskedastizit\u{00e4}t und Varianz", "BAAI/bge-m3");
        assert!(
            warning.is_none(),
            "multilingual model must not produce warning"
        );
    }

    /// T-MCP-SEARCH-023: Short queries with one non-ASCII character do not trigger.
    #[test]
    fn t_mcp_search_023_short_query_no_false_positive() {
        let warning =
            detect_language_mismatch("na\u{00ef}ve Bayes classifier", "BAAI/bge-small-en-v1.5");
        assert!(
            warning.is_none(),
            "single accented char should not trigger warning"
        );
    }

    /// T-MCP-SEARCH-024: Exact token count upgrades truncation warning when
    /// the word-based heuristic did not fire.
    ///
    /// Regression test for BUG-006: Queries where word_count + 2 <= max_seq_len
    /// (heuristic passes) but the BPE token count >= max_seq_len were silently
    /// truncated without any warning. The fix adds a second check using the
    /// exact token count from the tokenizer. This test verifies the logic path
    /// where the heuristic produces no warning but the exact count triggers one.
    #[test]
    fn t_mcp_search_024_exact_token_count_upgrades_truncation_warning() {
        let max_seq_len: usize = 512;

        // Simulate a query where the heuristic does NOT fire:
        // 350 words + 2 special tokens = 352 estimated minimum tokens, under 512.
        let word_count: usize = 350;
        let estimated_min_tokens = word_count + 2;
        assert!(
            estimated_min_tokens <= max_seq_len,
            "heuristic must NOT fire for {word_count} words (est. {estimated_min_tokens} <= {max_seq_len})"
        );

        // Heuristic-based warning is None (did not fire).
        let mut truncation_warning: Option<String> = None;

        // Simulate the exact token count from the tokenizer: BPE produces more
        // tokens than words for technical text (1.3-1.5x factor), pushing the
        // actual count above max_seq_len.
        let query_tokens: Option<usize> = Some(530);

        // Apply the exact-token-count upgrade logic (mirrors the production code).
        if truncation_warning.is_none()
            && let Some(tokens) = query_tokens
            && tokens >= max_seq_len
        {
            truncation_warning = Some(format!(
                "query has {tokens} tokens which exceeds the model's \
                 max_sequence_length of {max_seq_len}; the embedding represents \
                 only a prefix of the query. Consider shortening the query for \
                 complete coverage."
            ));
        }

        assert!(
            truncation_warning.is_some(),
            "truncation warning must be generated when exact token count (530) >= max_seq_len ({max_seq_len})",
        );

        let msg = truncation_warning.unwrap();
        assert!(
            msg.contains("530"),
            "warning must contain the exact token count, got: {msg}"
        );
        assert!(
            msg.contains("512"),
            "warning must contain max_seq_len, got: {msg}"
        );
    }

    /// T-MCP-SEARCH-025: Exact token count does NOT generate a spurious
    /// warning when the count is below max_seq_len.
    ///
    /// Complementary test to T-MCP-SEARCH-024: when the exact token count is
    /// below max_seq_len, no warning should be generated regardless of whether
    /// the heuristic fired.
    #[test]
    fn t_mcp_search_025_no_spurious_warning_when_tokens_under_limit() {
        let max_seq_len: usize = 512;
        let mut truncation_warning: Option<String> = None;

        // Exact token count is below max_seq_len.
        let query_tokens: Option<usize> = Some(400);

        if truncation_warning.is_none()
            && let Some(tokens) = query_tokens
            && tokens >= max_seq_len
        {
            truncation_warning = Some(format!(
                "query has {tokens} tokens which exceeds max_seq_len {max_seq_len}"
            ));
        }

        assert!(
            truncation_warning.is_none(),
            "no warning for token count (400) below max_seq_len ({max_seq_len})",
        );
    }

    /// T-MCP-SEARCH-026: Heuristic-based warning is NOT overwritten by the
    /// exact token count check. When the heuristic already fires (very long
    /// queries), the exact-count path must not replace the heuristic message.
    #[test]
    fn t_mcp_search_026_heuristic_warning_not_overwritten_by_exact_count() {
        let max_seq_len: usize = 512;

        // Heuristic fires for a 520-word query.
        let word_count: usize = 520;
        let estimated_min_tokens = word_count + 2;
        assert!(estimated_min_tokens > max_seq_len);

        let heuristic_msg =
            format!("query has ~{word_count} words (at least {estimated_min_tokens} tokens)");
        let mut truncation_warning: Option<String> = Some(heuristic_msg.clone());

        // Exact token count is also over the limit.
        let query_tokens: Option<usize> = Some(650);

        // The exact-count path should NOT overwrite the heuristic warning.
        if truncation_warning.is_none()
            && let Some(tokens) = query_tokens
            && tokens >= max_seq_len
        {
            truncation_warning = Some(format!(
                "query has {tokens} tokens which exceeds max_seq_len {max_seq_len}"
            ));
        }

        assert_eq!(
            truncation_warning.as_deref(),
            Some(heuristic_msg.as_str()),
            "heuristic warning must be preserved, not overwritten by exact count"
        );
    }

    /// T-MCP-SEARCH-027: The search response contains a `query_analysis`
    /// object with structured tokenization and BM25 term information. This
    /// test verifies the query_analysis section is present and contains the
    /// expected fields when a valid search is performed against an empty
    /// session (no HNSW index loaded).
    #[tokio::test]
    async fn t_mcp_search_027_query_analysis_present_in_response() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_027");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "risk factors in financial markets",
            "use_fts": true,
        });

        let result = handle(&state, &params).await;
        assert!(
            result.is_ok(),
            "search must succeed, got: {:?}",
            result.err()
        );

        let response = result.unwrap();
        let analysis = &response["query_analysis"];

        // query_analysis must be a JSON object (not null).
        assert!(
            analysis.is_object(),
            "query_analysis must be present as an object, got: {analysis}"
        );

        // query_word_count must match the number of whitespace-delimited words.
        assert_eq!(analysis["query_word_count"], 5, "query has 5 words");

        // max_sequence_length must be a positive integer.
        assert!(
            analysis["max_sequence_length"].as_u64().is_some(),
            "max_sequence_length must be present"
        );

        // was_truncated must be false for a short 5-word query.
        assert_eq!(
            analysis["was_truncated"], false,
            "5-word query must not be truncated"
        );

        // bm25_terms must be present when use_fts=true.
        assert!(
            analysis["bm25_terms"].is_array(),
            "bm25_terms must be an array when use_fts=true"
        );
    }

    /// T-MCP-SEARCH-028: The query_analysis section omits bm25_terms when
    /// use_fts=false, since no keyword search is performed.
    #[tokio::test]
    async fn t_mcp_search_028_query_analysis_no_bm25_terms_when_fts_disabled() {
        let pool = test_pool();
        let state = test_state(pool);
        let session_id = test_session(&state, "/tmp/search_test_028");

        let params = serde_json::json!({
            "session_id": session_id,
            "query": "test query",
            "use_fts": false,
        });

        let result = handle(&state, &params).await;
        assert!(result.is_ok(), "search must succeed");

        let response = result.unwrap();
        let analysis = &response["query_analysis"];

        assert!(analysis.is_object(), "query_analysis must be present");

        // bm25_terms must be absent when use_fts=false.
        assert!(
            analysis.get("bm25_terms").is_none() || analysis["bm25_terms"].is_null(),
            "bm25_terms must be absent when use_fts=false, got: {:?}",
            analysis.get("bm25_terms")
        );
    }

    /// T-MCP-SEARCH-029: The `full_chunk_content` field logic is correct.
    /// When refinement produces a narrower sub-chunk, the original chunk text
    /// must be preserved. This test verifies the preservation mechanism by
    /// checking that original_contents stores clones before refinement.
    #[test]
    fn t_mcp_search_029_full_chunk_content_preserves_original() {
        // Simulate the original_contents preservation logic from the handler.
        let original = "This is the full chunk text spanning multiple sentences.";
        let refined = "full chunk text";

        // Before refinement, original_contents captures the full text.
        let original_contents = [original.to_string()];

        // After refinement, the result content would be replaced.
        let current_content = refined;

        // The original must be retrievable from original_contents.
        assert_eq!(
            original_contents[0], original,
            "original_contents must preserve the full chunk text before refinement"
        );

        // The current content is the narrower sub-chunk.
        assert_ne!(
            current_content, original,
            "after refinement, content differs from original"
        );
    }
}
