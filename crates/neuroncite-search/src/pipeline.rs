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

//! Top-level search pipeline orchestration.
//!
//! Coordinates the execution of vector search, keyword search, fusion,
//! deduplication, optional reranking, and citation assembly into a single
//! `search()` function that accepts a query string and search parameters and
//! returns a ranked list of `SearchResult` objects.

use rusqlite::Connection;

use neuroncite_core::{Reranker, SearchResult};
use neuroncite_store::HnswIndex;

use crate::citation::{batch_load_chunk_meta, citation_from_meta};
use crate::dedup::deduplicate;
use crate::error::SearchError;
use crate::fusion::{FusedCandidate, reciprocal_rank_fusion};
use crate::keyword::keyword_search;
use crate::vector::vector_search;

/// Configuration parameters for a single search pipeline execution.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// The session ID that scopes this search. Both vector search (via per-session
    /// HNSW indexes) and keyword search (via FTS5 + session_id filter) are
    /// restricted to chunks belonging to this session. Required because the
    /// FTS5 virtual table is shared across all sessions.
    pub session_id: i64,
    /// Maximum number of vector search results to retrieve from the HNSW index.
    pub vector_top_k: usize,
    /// Maximum number of keyword search results to retrieve from FTS5.
    pub keyword_limit: usize,
    /// HNSW ef_search parameter controlling the recall/latency trade-off.
    pub ef_search: usize,
    /// Reciprocal Rank Fusion constant k.
    pub rrf_k: usize,
    /// If true, only candidates that appear in both vector and keyword results
    /// are retained after fusion.
    pub bm25_must_match: bool,
    /// Maximum SimHash Hamming distance (out of 64 bits) for near-duplicate
    /// detection. Default: 3.
    pub simhash_threshold: u32,
    /// Maximum number of final results to return after all pipeline stages.
    pub max_results: usize,
    /// If true and a reranker is provided, reranking replaces the RRF score
    /// with the cross-encoder score.
    pub rerank_enabled: bool,
    /// When `Some`, restricts search results to chunks belonging to these file
    /// IDs. Applies to both vector and keyword search stages. File IDs
    /// correspond to `indexed_file.id` values from `neuroncite_files`.
    pub file_ids: Option<Vec<i64>>,
    /// When `Some`, filters out results whose `vector_score` (cosine similarity)
    /// is below this threshold. Applied after all pipeline stages (fusion,
    /// dedup, reranking) to the final result list.
    pub min_score: Option<f64>,
    /// When `Some`, restricts search results to chunks that overlap with the
    /// specified page range [page_start, page_end]. A chunk overlaps a page
    /// range when chunk.page_start <= page_end AND chunk.page_end >= page_start.
    /// Both bounds are 1-indexed and inclusive. Applied at the vector search and
    /// keyword search stages via SQL WHERE clauses.
    pub page_start: Option<i64>,
    /// Upper bound (inclusive, 1-indexed) of the page range filter. Must be
    /// provided together with `page_start`. See `page_start` for overlap logic.
    pub page_end: Option<i64>,
}

impl SearchConfig {
    /// Creates a SearchConfig for the given session with all other fields at
    /// their defaults. The session_id is required because the FTS5 virtual
    /// table is shared across all sessions and keyword search must be scoped.
    pub fn for_session(session_id: i64) -> Self {
        Self {
            session_id,
            vector_top_k: 50,
            keyword_limit: 50,
            ef_search: 100,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: 10,
            rerank_enabled: false,
            file_ids: None,
            min_score: None,
            page_start: None,
            page_end: None,
        }
    }
}

/// Assigns a human-readable relevance label based on the vector_score (cosine
/// similarity). Thresholds are calibrated for BERT-family embedding models
/// (BGE, MiniLM) on academic PDF text.
///
/// - `"high"`:     vector_score >= 0.82 -- strong semantic match
/// - `"medium"`:   vector_score >= 0.72 -- moderate semantic overlap
/// - `"low"`:      vector_score >= 0.60 -- weak semantic connection
/// - `"marginal"`: vector_score <  0.60 -- minimal relevance
pub fn relevance_label(vector_score: f64) -> &'static str {
    if vector_score >= 0.82 {
        "high"
    } else if vector_score >= 0.72 {
        "medium"
    } else if vector_score >= 0.60 {
        "low"
    } else {
        "marginal"
    }
}

/// The result of a search pipeline execution. Contains the ranked search
/// results along with metadata about the pipeline run that is not derivable
/// from the results alone.
pub struct SearchOutcome {
    /// Ranked search results after all pipeline stages (fusion, dedup,
    /// reranking, min_score filtering). Sorted by descending final score.
    pub results: Vec<SearchResult>,
    /// Total number of unique candidates after deduplication but before
    /// the `max_results` cap was applied. When `total_candidates > results.len()`,
    /// the caller knows that increasing `top_k` would yield more results.
    /// This field enables the MCP response to report `total_available` so
    /// callers can decide whether to widen their search.
    pub total_candidates: usize,
}

/// The `SearchPipeline` holds references to the resources needed for search:
/// the HNSW index, a database connection, and an optional reranker. It provides
/// a `search()` method that executes the full pipeline.
pub struct SearchPipeline<'a> {
    /// The HNSW approximate nearest neighbor index.
    index: &'a HnswIndex,
    /// Database connection for FTS5 queries, chunk metadata lookups, and
    /// citation assembly.
    conn: &'a Connection,
    /// The query embedding vector produced by the embedding backend.
    query_embedding: &'a [f32],
    /// The original query string for keyword search and reranking.
    query_text: &'a str,
    /// Optional cross-encoder reranker for fine-grained relevance scoring.
    reranker: Option<&'a dyn Reranker>,
    /// Search configuration parameters.
    config: SearchConfig,
}

impl<'a> SearchPipeline<'a> {
    /// Creates a search pipeline with the given resources and configuration.
    #[must_use]
    pub fn new(
        index: &'a HnswIndex,
        conn: &'a Connection,
        query_embedding: &'a [f32],
        query_text: &'a str,
        reranker: Option<&'a dyn Reranker>,
        config: SearchConfig,
    ) -> Self {
        Self {
            index,
            conn,
            query_embedding,
            query_text,
            reranker,
            config,
        }
    }

    /// Executes the full search pipeline:
    /// 1. Vector search (HNSW ANN)
    /// 2. Keyword search (FTS5 BM25)
    /// 3. Reciprocal Rank Fusion
    /// 4. Deduplication (exact hash + SimHash)
    /// 5. Reranking (optional cross-encoder)
    /// 6. Citation assembly
    ///
    /// Returns a `SearchOutcome` containing the ranked results (capped at
    /// `config.max_results`) and the total candidate count before the cap.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any pipeline stage fails.
    pub fn search(&self) -> Result<SearchOutcome, SearchError> {
        // Prepare the file_ids filter reference for vector and keyword stages.
        let file_ids_ref = self.config.file_ids.as_deref();

        // Prepare page range filter as a tuple for vector and keyword stages.
        let page_range = match (self.config.page_start, self.config.page_end) {
            (Some(ps), Some(pe)) => Some((ps, pe)),
            _ => None,
        };

        // Stage 1: Vector search
        tracing::debug!(top_k = self.config.vector_top_k, "starting vector search");
        let vector_hits = vector_search(
            self.index,
            self.conn,
            self.query_embedding,
            self.config.vector_top_k,
            self.config.ef_search,
            file_ids_ref,
            page_range,
        )?;

        // Stage 2: Keyword search (scoped to session_id via JOIN on chunk table)
        tracing::debug!(
            limit = self.config.keyword_limit,
            session_id = self.config.session_id,
            "starting keyword search"
        );
        let keyword_hits = keyword_search(
            self.conn,
            self.query_text,
            self.config.keyword_limit,
            self.config.session_id,
            file_ids_ref,
            page_range,
        )?;

        // Log top keyword hit details for diagnostics.
        if let Some(top_hit) = keyword_hits.first() {
            tracing::debug!(
                chunk_id = top_hit.chunk_id,
                raw_bm25 = top_hit.raw_bm25_score,
                rank = top_hit.bm25_rank,
                total_keyword_hits = keyword_hits.len(),
                "keyword search complete"
            );
        }

        // Stage 3: Reciprocal Rank Fusion
        tracing::debug!(
            k = self.config.rrf_k,
            must_match = self.config.bm25_must_match,
            "starting RRF fusion"
        );
        let fused = reciprocal_rank_fusion(
            &vector_hits,
            &keyword_hits,
            self.config.rrf_k,
            self.config.bm25_must_match,
        );

        // Stage 4: Deduplication
        tracing::debug!(
            threshold = self.config.simhash_threshold,
            "starting deduplication"
        );
        let deduped = deduplicate(&fused, self.conn, self.config.simhash_threshold)?;

        // Record the total number of deduplicated candidates before the
        // max_results cap. Exposed via SearchOutcome.total_candidates so
        // callers can determine whether increasing top_k would yield more.
        let total_candidates = deduped.len();

        // Limit the candidate list before reranking (reranking is expensive).
        let top_candidates: Vec<FusedCandidate> =
            deduped.into_iter().take(self.config.max_results).collect();

        // Batch-load all chunk metadata + file paths in a single JOIN query.
        // This replaces the per-candidate citation + content queries (N+1 pattern)
        // with a single WHERE IN (...) query covering all candidates.
        let candidate_ids: Vec<i64> = top_candidates.iter().map(|c| c.chunk_id).collect();
        let chunk_metas = batch_load_chunk_meta(self.conn, &candidate_ids)?;

        // Stage 5: Optional reranking (uses content from the batch-loaded metadata)
        let scored_candidates = if self.config.rerank_enabled {
            if let Some(reranker) = self.reranker {
                self.apply_reranking(&top_candidates, reranker, &chunk_metas)?
            } else {
                top_candidates
                    .iter()
                    .map(|c| (c.clone(), c.rrf_score, None))
                    .collect()
            }
        } else {
            top_candidates
                .iter()
                .map(|c| (c.clone(), c.rrf_score, None))
                .collect()
        };

        // Stage 6: Citation assembly and SearchResult construction from
        // the batch-loaded metadata (no additional queries needed).
        tracing::debug!(count = scored_candidates.len(), "assembling citations");
        let mut results = Vec::with_capacity(scored_candidates.len());
        for (candidate, final_score, reranker_score) in &scored_candidates {
            let meta =
                chunk_metas
                    .get(&candidate.chunk_id)
                    .ok_or_else(|| SearchError::ChunkLookup {
                        reason: format!("batch metadata missing for chunk {}", candidate.chunk_id),
                    })?;

            let citation = citation_from_meta(meta);

            results.push(SearchResult {
                score: *final_score,
                vector_score: candidate.vector_score,
                bm25_rank: candidate.bm25_rank,
                reranker_score: *reranker_score,
                chunk_index: meta.chunk_index as usize,
                content: meta.content.clone(),
                citation,
            });
        }

        // Sort by descending final score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply the min_score threshold after final sorting.
        // When reranking is active, the `score` field holds the cross-encoder
        // score (set in apply_reranking). Filtering on `r.score` in that case
        // respects the reranker's assessment rather than the cosine similarity.
        // When reranking is not active, `r.score` equals the RRF fused score,
        // which may not be comparable to the caller's min_score expectation.
        // In that case, filter on `r.vector_score` (cosine similarity), which
        // is the only score that has a stable, interpretable range [-1, 1].
        if let Some(min) = self.config.min_score {
            let reranker_active = self.config.rerank_enabled && self.reranker.is_some();
            if reranker_active {
                // Use the reranker score (stored in r.score by apply_reranking).
                results.retain(|r| r.score >= min);
            } else {
                // Use cosine similarity, which has consistent semantics across
                // embedding models (higher = more similar, range -1 to 1).
                results.retain(|r| r.vector_score >= min);
            }
        }

        Ok(SearchOutcome {
            results,
            total_candidates,
        })
    }

    /// Applies cross-encoder reranking to the candidates using content from the
    /// batch-loaded `ChunkMeta` map (no additional queries). Returns tuples of
    /// (candidate, final_score, reranker_score). When reranking is active, the
    /// final score is replaced by the reranker score.
    fn apply_reranking(
        &self,
        candidates: &[FusedCandidate],
        reranker: &dyn Reranker,
        chunk_metas: &std::collections::HashMap<i64, crate::citation::ChunkMeta>,
    ) -> Result<Vec<(FusedCandidate, f64, Option<f64>)>, SearchError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Extract content strings from the batch-loaded metadata. All
        // candidates were loaded via batch_load_chunk_meta immediately before
        // this call, so a missing entry indicates a bug in the caller's
        // assembly logic (not a concurrent re-index). Return an error rather
        // than silently submitting an empty string to the cross-encoder, which
        // would produce a misleadingly low reranker score for that candidate.
        let contents: Vec<&str> = candidates
            .iter()
            .map(|c| {
                chunk_metas
                    .get(&c.chunk_id)
                    .map(|m| m.content.as_str())
                    .ok_or_else(|| SearchError::ChunkLookup {
                        reason: format!(
                            "chunk {} absent from reranker metadata — batch_load_chunk_meta returned incomplete results",
                            c.chunk_id
                        ),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let scores = reranker
            .rerank_batch(self.query_text, &contents)
            .map_err(|e| SearchError::Reranking {
                reason: format!("reranker batch scoring failed: {e}"),
            })?;

        let result: Vec<(FusedCandidate, f64, Option<f64>)> = candidates
            .iter()
            .zip(scores.iter())
            .map(|(candidate, &rerank_score)| {
                // When reranking is active, the reranker score replaces the RRF score.
                (candidate.clone(), rerank_score, Some(rerank_score))
            })
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{IndexConfig, NeuronCiteError, StorageMode};
    use neuroncite_store::{ChunkInsert, build_hnsw, bulk_insert_chunks};
    use rusqlite::Connection;
    use std::path::PathBuf;

    /// Generates a normalized vector from a seed for reproducible tests.
    fn make_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut state = seed;
        for _ in 0..dim {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = (state >> 33) as f32 / (u32::MAX as f32);
            v.push(val);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// A mock reranker that returns a fixed score for each candidate.
    /// Used to verify that reranking replaces the RRF score.
    struct MockReranker {
        fixed_scores: Vec<f64>,
    }

    impl Reranker for MockReranker {
        fn name(&self) -> &str {
            "mock-reranker"
        }

        fn rerank_batch(
            &self,
            _query: &str,
            candidates: &[&str],
        ) -> Result<Vec<f64>, NeuronCiteError> {
            // Return the fixed scores (one per candidate), cycling if needed.
            let scores: Vec<f64> = candidates
                .iter()
                .enumerate()
                .map(|(i, _)| self.fixed_scores[i % self.fixed_scores.len()])
                .collect();
            Ok(scores)
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
            Ok(())
        }
    }

    /// Sets up a full test environment with database, chunks, embeddings, and
    /// HNSW index. Returns all components needed for pipeline testing,
    /// including the session_id (required for keyword search scoping).
    fn setup_pipeline_env() -> (Connection, HnswIndex, Vec<Vec<f32>>, Vec<i64>, i64) {
        let dim = 4;

        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: dim,
        };

        let session_id = neuroncite_store::repo::session::create_session(&conn, &config, "0.1.0")
            .expect("create session");

        let file_id = neuroncite_store::repo::file::insert_file(
            &conn,
            session_id,
            "/docs/statistics.pdf",
            "hash123",
            1_700_000_000,
            4096,
            3,
            None,
        )
        .expect("insert file");

        let contents = [
            "statistical hypothesis testing and p-values in experimental research",
            "machine learning algorithms for classification and regression tasks",
            "bayesian statistical inference using prior and posterior distributions",
        ];

        let embeddings: Vec<Vec<f32>> = (1..=3).map(|i| make_vector(i, dim)).collect();

        // Pre-compute hash strings so they outlive the ChunkInsert references.
        let hash_strings: Vec<String> = (0..contents.len()).map(|i| format!("hash_{i}")).collect();

        let chunk_inserts: Vec<ChunkInsert<'_>> = contents
            .iter()
            .enumerate()
            .map(|(i, content)| ChunkInsert {
                file_id,
                session_id,
                page_start: (i as i64) + 1,
                page_end: (i as i64) + 1,
                chunk_index: i as i64,
                doc_text_offset_start: 0,
                doc_text_offset_end: content.len() as i64,
                content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &hash_strings[i],
                simhash: None,
            })
            .collect();

        let ids = bulk_insert_chunks(&conn, &chunk_inserts).expect("bulk insert");

        let labeled: Vec<(i64, &[f32])> = ids
            .iter()
            .zip(embeddings.iter())
            .map(|(id, emb)| (*id, emb.as_slice()))
            .collect();
        let index = build_hnsw(&labeled, dim).expect("build_hnsw");

        (conn, index, embeddings, ids, session_id)
    }

    /// T-SCH-023: relevance_label returns correct labels at exact threshold
    /// boundaries. The four tiers are:
    /// - >= 0.82 -> "high"
    /// - >= 0.72 -> "medium"
    /// - >= 0.60 -> "low"
    /// - <  0.60 -> "marginal"
    #[test]
    fn t_sch_023_relevance_label_boundary_values() {
        assert_eq!(relevance_label(1.0), "high");
        assert_eq!(relevance_label(0.82), "high");
        assert_eq!(relevance_label(0.819), "medium");
        assert_eq!(relevance_label(0.72), "medium");
        assert_eq!(relevance_label(0.719), "low");
        assert_eq!(relevance_label(0.60), "low");
        assert_eq!(relevance_label(0.599), "marginal");
        assert_eq!(relevance_label(0.0), "marginal");
    }

    /// T-SCH-024: relevance_label handles negative vector_score values. These
    /// can occur with cosine similarity for non-normalized vectors and must
    /// return "marginal" without panic.
    #[test]
    fn t_sch_024_relevance_label_negative_score() {
        assert_eq!(relevance_label(-0.5), "marginal");
        assert_eq!(relevance_label(-1.0), "marginal");
    }

    /// T-SCH-022: Reranker score replaces RRF score. When reranking is active,
    /// the final score of each result equals the reranker score, not the RRF score.
    #[test]
    fn t_sch_022_reranker_score_replaces_rrf() {
        let (conn, index, embeddings, _ids, session_id) = setup_pipeline_env();

        let reranker = MockReranker {
            fixed_scores: vec![0.99, 0.50, 0.75],
        };

        let config = SearchConfig {
            session_id,
            vector_top_k: 10,
            keyword_limit: 10,
            ef_search: 100,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: 10,
            rerank_enabled: true,
            file_ids: None,
            min_score: None,
            page_start: None,
            page_end: None,
        };

        let pipeline = SearchPipeline::new(
            &index,
            &conn,
            &embeddings[0],
            "statistical",
            Some(&reranker),
            config,
        );

        let outcome = pipeline.search().expect("search pipeline");

        // All results with reranking must have a reranker_score set.
        for result in &outcome.results {
            assert!(
                result.reranker_score.is_some(),
                "reranker_score must be set when reranking is active"
            );
            // The final score must equal the reranker score.
            let reranker_score = result.reranker_score.expect("reranker_score is Some");
            assert!(
                (result.score - reranker_score).abs() < 1e-10,
                "final score ({}) must equal reranker score ({reranker_score})",
                result.score,
            );
        }
    }
}
