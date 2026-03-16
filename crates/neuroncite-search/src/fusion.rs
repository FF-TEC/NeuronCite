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

//! Reciprocal Rank Fusion (RRF) for merging vector and keyword results.
//!
//! Combines the ranked result lists from vector search and keyword search into
//! a single ranked list using the RRF formula: `score(d) = sum(1 / (k + rank_i(d)))`
//! where `k` is a constant (typically 60) and `rank_i(d)` is document d's rank
//! in the i-th result list. Documents appearing in both lists receive contributions
//! from both ranks and therefore score higher than documents in only one list.

use std::collections::HashMap;

use crate::keyword::KeywordHit;
use crate::vector::VectorHit;

/// A candidate document after Reciprocal Rank Fusion. Contains the chunk ID,
/// the fused RRF score, the original vector similarity score, and the original
/// BM25 rank (if the chunk appeared in keyword results).
#[derive(Debug, Clone)]
pub struct FusedCandidate {
    /// Primary key of the chunk in the `chunk` table.
    pub chunk_id: i64,
    /// Fused RRF score. Higher values indicate higher relevance. The score
    /// is the sum of `1 / (k + rank)` contributions from each ranker.
    pub rrf_score: f64,
    /// Cosine similarity score from vector search. 0.0 if the chunk was not
    /// found by vector search.
    pub vector_score: f64,
    /// 1-indexed BM25 rank from keyword search, if the chunk appeared in
    /// FTS5 results.
    pub bm25_rank: Option<usize>,
}

/// Performs Reciprocal Rank Fusion on vector and keyword result lists.
///
/// The RRF formula for a document `d` is:
///   `RRF(d) = sum(1 / (k + rank_r(d)))` for each ranker `r`
///
/// where `rank_r(d)` is the 1-indexed rank of `d` in ranker `r`'s output.
/// Documents not present in a ranker's list receive no contribution from
/// that ranker.
///
/// If `bm25_must_match` is true, candidates that appear only in the vector
/// list (and not in the BM25 keyword list) are excluded from the fused output.
/// This acts as a hard filter ensuring all results have keyword relevance.
///
/// # Arguments
///
/// * `vector_hits` - Results from HNSW vector search, ordered by descending similarity.
/// * `keyword_hits` - Results from FTS5 keyword search, with 1-indexed ranks.
/// * `k` - The RRF constant. Typical values: 60 (default). Higher k reduces the
///   score difference between high-ranked and low-ranked documents.
/// * `bm25_must_match` - If true, candidates not in the keyword list are excluded.
///
/// # Returns
///
/// A vector of `FusedCandidate` sorted by descending RRF score.
pub fn reciprocal_rank_fusion(
    vector_hits: &[VectorHit],
    keyword_hits: &[KeywordHit],
    k: usize,
    bm25_must_match: bool,
) -> Vec<FusedCandidate> {
    // Clamp k to a minimum of 1 to prevent degenerate score distributions
    // when k = 0. With k = 0, rank 1 receives weight 1.0 and rank 2 receives
    // weight 0.5, producing an extremely aggressive winner-take-all ranking.
    // The standard RRF default of k = 60 is recommended for balanced scoring.
    let k_f64 = k.max(1) as f64;

    // Accumulate RRF score contributions from both rankers into a shared map.
    // Key: chunk_id, Value: (rrf_score, vector_score, bm25_rank)
    let mut scores: HashMap<i64, (f64, f64, Option<usize>)> = HashMap::new();

    // Contribute vector search ranks. The vector_hits are already sorted by
    // descending similarity, so position index 0 is rank 1.
    for (rank_0, hit) in vector_hits.iter().enumerate() {
        let rank = (rank_0 + 1) as f64;
        let entry = scores.entry(hit.chunk_id).or_insert((0.0, 0.0, None));
        entry.0 += 1.0 / (k_f64 + rank);
        entry.1 = hit.similarity; // already f64 (widened from f32 in vector_search)
    }

    // Contribute keyword search ranks. The bm25_rank field is already 1-indexed.
    for hit in keyword_hits {
        let rank = hit.bm25_rank as f64;
        let entry = scores.entry(hit.chunk_id).or_insert((0.0, 0.0, None));
        entry.0 += 1.0 / (k_f64 + rank);
        entry.2 = Some(hit.bm25_rank);
    }

    // Build the set of BM25 chunk IDs for must-match filtering.
    let bm25_ids: std::collections::HashSet<i64> =
        keyword_hits.iter().map(|h| h.chunk_id).collect();

    // Collect and sort candidates by descending RRF score.
    let mut candidates: Vec<FusedCandidate> = scores
        .into_iter()
        .filter(|(chunk_id, _)| {
            // If bm25_must_match is enabled, exclude candidates not in the BM25 list.
            if bm25_must_match {
                bm25_ids.contains(chunk_id)
            } else {
                true
            }
        })
        .map(
            |(chunk_id, (rrf_score, vector_score, bm25_rank))| FusedCandidate {
                chunk_id,
                rrf_score,
                vector_score,
                bm25_rank,
            },
        )
        .collect();

    candidates.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-SCH-005: RRF fusion combines both lists. A candidate in both lists
    /// scores higher than one in only one list.
    #[test]
    fn t_sch_005_rrf_fusion_combines_both_lists() {
        // chunk 1 appears in both vector and keyword results
        // chunk 2 appears only in vector results
        // chunk 3 appears only in keyword results
        let vector_hits = vec![
            VectorHit {
                chunk_id: 1,
                similarity: 0.95,
            },
            VectorHit {
                chunk_id: 2,
                similarity: 0.90,
            },
        ];

        let keyword_hits = vec![
            KeywordHit {
                chunk_id: 1,
                raw_bm25_score: -5.0,
                bm25_rank: 1,
            },
            KeywordHit {
                chunk_id: 3,
                raw_bm25_score: -3.0,
                bm25_rank: 2,
            },
        ];

        let fused = reciprocal_rank_fusion(&vector_hits, &keyword_hits, 60, false);

        // Find chunk 1 (in both lists) and chunk 2 (vector only)
        let score_chunk1 = fused
            .iter()
            .find(|c| c.chunk_id == 1)
            .expect("chunk 1")
            .rrf_score;
        let score_chunk2 = fused
            .iter()
            .find(|c| c.chunk_id == 2)
            .expect("chunk 2")
            .rrf_score;
        let score_chunk3 = fused
            .iter()
            .find(|c| c.chunk_id == 3)
            .expect("chunk 3")
            .rrf_score;

        // Chunk 1 (in both) must score higher than chunk 2 (vector only)
        assert!(
            score_chunk1 > score_chunk2,
            "candidate in both lists ({score_chunk1}) must score higher than vector-only ({score_chunk2})"
        );

        // Chunk 1 (in both) must score higher than chunk 3 (keyword only)
        assert!(
            score_chunk1 > score_chunk3,
            "candidate in both lists ({score_chunk1}) must score higher than keyword-only ({score_chunk3})"
        );
    }

    /// T-SCH-006: RRF parameter k effect. k=10 produces a larger score
    /// difference between rank 1 and rank 10 than k=200.
    #[test]
    fn t_sch_006_rrf_parameter_k_effect() {
        // Create 10 vector hits at different ranks
        // VectorHit.similarity is f64 (widened from f32 in P5.5 for score comparison precision).
        let vector_hits: Vec<VectorHit> = (1..=10)
            .map(|i| VectorHit {
                chunk_id: i,
                similarity: 1.0_f64 - (i as f64) * 0.05,
            })
            .collect();

        let keyword_hits: Vec<KeywordHit> = Vec::new();

        // Fuse with k=10
        let fused_k10 = reciprocal_rank_fusion(&vector_hits, &keyword_hits, 10, false);
        let score_rank1_k10 = fused_k10
            .iter()
            .find(|c| c.chunk_id == 1)
            .expect("rank 1")
            .rrf_score;
        let score_rank10_k10 = fused_k10
            .iter()
            .find(|c| c.chunk_id == 10)
            .expect("rank 10")
            .rrf_score;
        let diff_k10 = score_rank1_k10 - score_rank10_k10;

        // Fuse with k=200
        let fused_k200 = reciprocal_rank_fusion(&vector_hits, &keyword_hits, 200, false);
        let score_rank1_k200 = fused_k200
            .iter()
            .find(|c| c.chunk_id == 1)
            .expect("rank 1")
            .rrf_score;
        let score_rank10_k200 = fused_k200
            .iter()
            .find(|c| c.chunk_id == 10)
            .expect("rank 10")
            .rrf_score;
        let diff_k200 = score_rank1_k200 - score_rank10_k200;

        // k=10 amplifies rank differences more than k=200
        assert!(
            diff_k10 > diff_k200,
            "k=10 score difference ({diff_k10}) must exceed k=200 difference ({diff_k200})"
        );
    }

    /// T-SCH-007: BM25 must-match filter. A vector candidate not in the BM25
    /// list is excluded when must-match is enabled.
    #[test]
    fn t_sch_007_bm25_must_match_filter() {
        let vector_hits = vec![
            VectorHit {
                chunk_id: 1,
                similarity: 0.95,
            },
            VectorHit {
                chunk_id: 2,
                similarity: 0.90,
            },
            VectorHit {
                chunk_id: 3,
                similarity: 0.85,
            },
        ];

        // Only chunk 1 and chunk 3 appear in keyword results
        let keyword_hits = vec![
            KeywordHit {
                chunk_id: 1,
                raw_bm25_score: -5.0,
                bm25_rank: 1,
            },
            KeywordHit {
                chunk_id: 3,
                raw_bm25_score: -3.0,
                bm25_rank: 2,
            },
        ];

        let fused = reciprocal_rank_fusion(&vector_hits, &keyword_hits, 60, true);

        let fused_ids: Vec<i64> = fused.iter().map(|c| c.chunk_id).collect();

        // Chunk 2 is vector-only and must be excluded when must-match is enabled
        assert!(
            !fused_ids.contains(&2),
            "vector-only candidate must be excluded when bm25_must_match is true"
        );

        // Chunks 1 and 3 must still be present
        assert!(fused_ids.contains(&1), "chunk 1 in both lists must remain");
        assert!(
            fused_ids.contains(&3),
            "chunk 3 in keyword list must remain"
        );
    }
}
