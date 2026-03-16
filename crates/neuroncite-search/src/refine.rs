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

//! Sub-chunk refinement for search results.
//!
//! After the initial search pipeline returns ranked chunks, the refinement
//! stage narrows each result's content window to the most relevant sub-section.
//! It works in three phases:
//!
//! 1. **Generation**: Each result chunk is tokenized and split into overlapping
//!    sub-chunks at multiple scales (controlled by `divisors`, e.g. [4, 8, 16]).
//!    For a chunk with T tokens and divisor d, the sub-chunk window size is T/d
//!    with 25% overlap between consecutive windows.
//!
//! 2. **Embedding**: All generated sub-chunks are embedded in a single batch
//!    call via the GPU worker (done by the caller in the handler layer).
//!
//! 3. **Selection**: For each result, the cosine similarity of every sub-chunk
//!    embedding against the query embedding is computed. If the best sub-chunk
//!    scores higher than the original chunk's vector_score, the result's content
//!    is replaced with the sub-chunk text.
//!
//! The refinement operates exclusively on the `content` field of `SearchResult`.
//! The ranking order (determined by RRF or reranker scores) remains unchanged.
//! Citation metadata (page range, file path, offsets) is preserved from the
//! original chunk, since the sub-chunk is a substring of the original content.

#![forbid(unsafe_code)]

use crate::error::SearchError;
use neuroncite_core::SearchResult;
use neuroncite_core::cosine_similarity;

/// Maximum number of sub-chunk candidates generated across all results in
/// a single refinement call. Prevents unbounded memory growth when top_k
/// is large (e.g., 200 results with divisors \[4,8,16\] would produce ~7200
/// candidates without this cap). Once the limit is reached, generation
/// stops. Results are processed in ranking order, so higher-ranked results
/// are prioritized for refinement.
pub const MAX_REFINEMENT_CANDIDATES: usize = 512;

/// Opaque wrapper around `tokenizers::Tokenizer` for callers that do not
/// depend on the `tokenizers` crate directly. Created once from the
/// model's tokenizer JSON and reused across multiple `generate_sub_chunks`
/// calls to avoid repeated deserialization. The JSON string for a typical
/// BPE tokenizer (bge-small-en-v1.5) is ~700 KB; parsing it allocates
/// vocabulary tables, merge rules, and pre/post-processors each time.
/// Caching the deserialized instance eliminates this overhead.
pub struct CachedTokenizer(tokenizers::Tokenizer);

impl CachedTokenizer {
    /// Deserializes a tokenizer from its HuggingFace JSON representation.
    ///
    /// This constructor should be called once per request (or once per batch)
    /// and the resulting `CachedTokenizer` reused for all `generate_sub_chunks`
    /// calls within that scope. In batch search with 20 queries, creating the
    /// tokenizer once and passing it to each query's refinement avoids 19
    /// redundant deserialization cycles.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Refinement` if the JSON string is malformed or
    /// incompatible with the `tokenizers` crate's deserialization.
    pub fn from_json(json: &str) -> Result<Self, SearchError> {
        tokenizers::Tokenizer::from_bytes(json)
            .map(CachedTokenizer)
            .map_err(|e| SearchError::Refinement {
                reason: format!("failed to deserialize tokenizer JSON: {e}"),
            })
    }

    /// Tokenizes the input text and returns the number of subword tokens
    /// produced by the model's BPE/WordPiece tokenizer. This count includes
    /// special tokens (\[CLS\], \[SEP\]) that the tokenizer adds automatically.
    ///
    /// Used by the MCP search handler to report the exact token count of a
    /// query in the response, enabling callers to determine whether the query
    /// was truncated by the embedding model's max_sequence_length.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Refinement` if tokenization fails (e.g., the
    /// input contains byte sequences incompatible with the model's vocabulary).
    pub fn token_count(&self, text: &str) -> Result<usize, SearchError> {
        let encoding = self
            .0
            .encode(text, true)
            .map_err(|e| SearchError::Refinement {
                reason: format!("tokenization failed: {e}"),
            })?;
        Ok(encoding.get_ids().len())
    }
}

/// Configuration parameters for the sub-chunk refinement stage.
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Whether sub-chunk refinement is active. When false, the refinement
    /// stage is skipped entirely and results pass through unchanged.
    pub enabled: bool,
    /// Divisor values controlling the granularity of sub-chunk splitting.
    /// For each divisor d, the original chunk (T tokens) is split into
    /// windows of size T/d with 25% overlap. Multiple divisors enable
    /// multi-scale refinement (e.g., [4, 8, 16] produces sub-chunks at
    /// 1/4, 1/8, and 1/16 of the original chunk size).
    ///
    /// Divisors must be >= 2. A divisor of 1 would produce a single
    /// sub-chunk identical to the original, which is pointless.
    pub divisors: Vec<usize>,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            divisors: vec![4, 8, 16],
        }
    }
}

/// A sub-chunk candidate produced by splitting a parent result's content.
/// Tracks the parent result index so the caller can associate embeddings
/// back to their originating result after batch embedding.
#[derive(Debug, Clone)]
pub struct SubChunkCandidate {
    /// Index of the parent SearchResult in the original results vector.
    pub parent_index: usize,
    /// The sub-chunk text content (a substring of the parent's content).
    pub content: String,
}

/// Splits each search result's content into token-based sub-chunks at
/// multiple scales defined by `divisors`.
///
/// For each result and each divisor d:
/// - Tokenize the result content to determine the total token count T.
/// - Compute sub-chunk window size: `window = max(1, T / d)`.
/// - Compute overlap: `overlap = max(1, window / 4)` (25% overlap).
/// - Compute step: `step = window - overlap`.
/// - Slide the window across the token sequence, extracting sub-chunk text
///   using the tokenizer's byte offset mappings for verbatim substrings.
///
/// Sub-chunks that are identical to the full original content (when T <= window)
/// are skipped, since comparing a chunk against itself is wasteful.
///
/// Generation stops when `MAX_REFINEMENT_CANDIDATES` is reached. Results are
/// processed in ranking order, so higher-ranked results are refined first.
///
/// # Arguments
///
/// * `results` - The search results whose content will be split.
/// * `tokenizer` - Pre-deserialized tokenizer (via `CachedTokenizer::from_json`).
///   Accepting a `CachedTokenizer` instead of raw JSON avoids re-parsing the
///   tokenizer on every call. In batch search with 20 queries, this eliminates
///   19 redundant deserialization cycles (~700 KB JSON each).
/// * `divisors` - The divisor values (e.g., [4, 8, 16]).
///
/// # Returns
///
/// A vector of `SubChunkCandidate` structs ready for batch embedding. The
/// vector contains at most `MAX_REFINEMENT_CANDIDATES` entries. May be empty
/// if all results have too few tokens to split.
///
/// # Errors
///
/// Returns `SearchError::Refinement` if tokenization of any result's content
/// fails.
pub fn generate_sub_chunks(
    results: &[SearchResult],
    tokenizer: &CachedTokenizer,
    divisors: &[usize],
) -> Result<Vec<SubChunkCandidate>, SearchError> {
    if results.is_empty() || divisors.is_empty() {
        return Ok(Vec::new());
    }

    let mut candidates = Vec::new();

    for (result_idx, result) in results.iter().enumerate() {
        if candidates.len() >= MAX_REFINEMENT_CANDIDATES {
            break;
        }

        let content = &result.content;
        if content.is_empty() {
            continue;
        }

        // Tokenize the result content. The encoding provides token IDs and
        // byte offset mappings for each token back into the original text.
        let encoding =
            tokenizer
                .0
                .encode(content.as_str(), false)
                .map_err(|e| SearchError::Refinement {
                    reason: format!("tokenizer encoding failed for result {result_idx}: {e}"),
                })?;

        // Collect byte offset pairs, filtering special tokens with zero-width
        // offsets (CLS, SEP markers that have no text representation).
        let offsets: Vec<(usize, usize)> = encoding
            .get_offsets()
            .iter()
            .filter(|&&(s, e)| s != e)
            .copied()
            .collect();

        let total_tokens = offsets.len();

        for &divisor in divisors {
            if divisor < 2 || candidates.len() >= MAX_REFINEMENT_CANDIDATES {
                continue;
            }

            let window = total_tokens / divisor;
            if window == 0 {
                // The chunk has fewer tokens than the divisor; splitting would
                // produce empty or single-token sub-chunks. Skip this divisor.
                continue;
            }

            // Skip if the window covers the entire chunk (divisor too small
            // for this chunk size). Comparing the full chunk against itself
            // wastes embedding compute.
            if window >= total_tokens {
                continue;
            }

            // 25% overlap between consecutive windows, minimum 1 token.
            let overlap = (window / 4).max(1);
            let step = window - overlap;

            let mut token_offset = 0;
            while token_offset < total_tokens {
                if candidates.len() >= MAX_REFINEMENT_CANDIDATES {
                    break;
                }

                let window_end = (token_offset + window).min(total_tokens);
                let window_offsets = &offsets[token_offset..window_end];

                if window_offsets.is_empty() {
                    break;
                }

                // Extract the sub-chunk text as a verbatim substring of the
                // original content using the tokenizer's byte offset mappings.
                let byte_start = window_offsets[0].0;
                let byte_end = window_offsets[window_offsets.len() - 1].1;

                // Clamp byte offsets to content length to prevent panics from
                // tokenizers that report offsets beyond the input string.
                let byte_start = byte_start.min(content.len());
                let byte_end = byte_end.min(content.len());

                if byte_end > byte_start {
                    // Use str::get() rather than direct indexing to handle
                    // tokenizers that report byte offsets that do not align
                    // with UTF-8 character boundaries. BPE tokenizers may
                    // emit offsets that bisect a multi-byte character; direct
                    // indexing would panic in that case. get() returns None
                    // for any misaligned or out-of-bounds range, so the
                    // sub-chunk is silently skipped and generation continues.
                    if let Some(sub_text) = content.get(byte_start..byte_end) {
                        // Only include non-trivial sub-chunks (at least 2 characters).
                        if sub_text.len() >= 2 {
                            candidates.push(SubChunkCandidate {
                                parent_index: result_idx,
                                content: sub_text.to_string(),
                            });
                        }
                    } else {
                        tracing::debug!(
                            result_index = result_idx,
                            byte_start,
                            byte_end,
                            content_len = content.len(),
                            "tokenizer byte offsets do not align with UTF-8 boundaries — skipping sub-chunk"
                        );
                    }
                }

                if window_end >= total_tokens {
                    break;
                }
                token_offset += step;
            }
        }
    }

    Ok(candidates)
}

/// Applies sub-chunk refinement to the search results using pre-computed
/// sub-chunk embeddings.
///
/// For each result, finds the sub-chunk with the highest cosine similarity
/// to the query embedding. If this sub-chunk scores higher than the original
/// chunk's `vector_score`, the result's `content` field is replaced with
/// the sub-chunk text. The original ranking order and citation metadata
/// remain unchanged.
///
/// # Arguments
///
/// * `results` - Mutable reference to the search results to refine.
/// * `candidates` - The sub-chunk candidates (from `generate_sub_chunks`).
/// * `candidate_embeddings` - One embedding vector per candidate, in the
///   same order as `candidates`. Produced by batch-embedding the candidate
///   contents via the GPU worker.
/// * `query_embedding` - The query's embedding vector for cosine comparison.
///
/// # Returns
///
/// A `Vec<bool>` with one entry per result. `true` indicates that the
/// result's content was replaced by a higher-scoring sub-chunk; `false`
/// indicates the original content was retained.
///
/// # Panics
///
/// Debug-asserts that `candidates.len() == candidate_embeddings.len()`.
pub fn apply_refinement(
    results: &mut [SearchResult],
    candidates: &[SubChunkCandidate],
    candidate_embeddings: &[Vec<f32>],
    query_embedding: &[f32],
) -> Vec<bool> {
    // Runtime invariant check — not debug_assert (which compiles out in
    // release builds). A mismatch indicates a bug in the caller: the embedding
    // worker returned a different number of vectors than candidates submitted.
    if candidates.len() != candidate_embeddings.len() {
        tracing::error!(
            candidates = candidates.len(),
            embeddings = candidate_embeddings.len(),
            "candidate count does not match embedding count — refinement skipped"
        );
        return vec![false; results.len()];
    }

    let result_count = results.len();

    if candidates.is_empty() {
        return vec![false; result_count];
    }

    // For each result, track the best sub-chunk score and content.
    // Initialize with the original chunk's vector_score so the sub-chunk
    // must strictly beat the original to replace it.
    let mut best_scores: Vec<f64> = results.iter().map(|r| r.vector_score).collect();
    let mut best_contents: Vec<Option<String>> = vec![None; result_count];

    for (candidate, embedding) in candidates.iter().zip(candidate_embeddings.iter()) {
        if candidate.parent_index >= result_count {
            continue;
        }

        let sim = cosine_similarity(embedding, query_embedding);

        if sim > best_scores[candidate.parent_index] {
            best_scores[candidate.parent_index] = sim;
            best_contents[candidate.parent_index] = Some(candidate.content.clone());
        }
    }

    // Replace content for results where a sub-chunk scored higher.
    // Track which results were refined via the returned Vec<bool>.
    let mut was_refined = vec![false; result_count];
    for (idx, result) in results.iter_mut().enumerate() {
        if let Some(refined_content) = best_contents[idx].take() {
            tracing::debug!(
                result_index = idx,
                original_score = result.vector_score,
                refined_score = best_scores[idx],
                original_len = result.content.len(),
                refined_len = refined_content.len(),
                "sub-chunk refinement replaced content"
            );
            result.content = refined_content;
            was_refined[idx] = true;
        }
    }

    was_refined
}

/// Parses a comma-separated string of divisor values into a `Vec<usize>`.
/// Filters out values less than 2 (a divisor of 1 is the original chunk,
/// which is pointless to compare). Removes duplicates and sorts ascending.
///
/// # Examples
///
/// - "4,8,16" -> \[4, 8, 16\]
/// - "4" -> \[4\]
/// - "4, 8, 16, 32" -> \[4, 8, 16, 32\]
/// - "" -> \[\] (empty, refinement will be skipped)
/// - "1,4" -> \[4\] (1 is filtered out)
pub fn parse_divisors(input: &str) -> Vec<usize> {
    let mut divisors: Vec<usize> = input
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&d| d >= 2)
        .collect();

    divisors.sort_unstable();
    divisors.dedup();
    divisors
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{Citation, SearchResult};
    use std::path::PathBuf;

    /// Builds a minimal word-level tokenizer for testing. Each whitespace-
    /// delimited word becomes one token. The vocabulary covers enough words
    /// for the test cases below.
    fn build_test_tokenizer() -> CachedTokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let vocab_entries: Vec<(String, u32)> = vec![
            ("the".into(), 0),
            ("quick".into(), 1),
            ("brown".into(), 2),
            ("fox".into(), 3),
            ("jumps".into(), 4),
            ("over".into(), 5),
            ("lazy".into(), 6),
            ("dog".into(), 7),
            ("statistical".into(), 8),
            ("hypothesis".into(), 9),
            ("testing".into(), 10),
            ("and".into(), 11),
            ("p-values".into(), 12),
            ("in".into(), 13),
            ("experimental".into(), 14),
            ("research".into(), 15),
            ("methods".into(), 16),
            ("for".into(), 17),
            ("data".into(), 18),
            ("analysis".into(), 19),
            ("regression".into(), 20),
            ("models".into(), 21),
            ("bayesian".into(), 22),
            ("inference".into(), 23),
            ("using".into(), 24),
            ("prior".into(), 25),
            ("posterior".into(), 26),
            ("distributions".into(), 27),
            ("machine".into(), 28),
            ("learning".into(), 29),
            ("algorithms".into(), 30),
            ("classification".into(), 31),
            ("[UNK]".into(), 32),
        ];

        let vocab = vocab_entries.into_iter().collect();

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .expect("WordLevel model construction failed");

        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));

        CachedTokenizer(tokenizer)
    }

    /// Returns the test tokenizer's JSON representation for testing
    /// CachedTokenizer::from_json.
    fn build_test_tokenizer_json() -> String {
        let tok = build_test_tokenizer();
        tok.0
            .to_string(false)
            .expect("tokenizer JSON serialization failed")
    }

    /// Constructs a test SearchResult with the given content and vector_score.
    fn make_result(content: &str, vector_score: f64) -> SearchResult {
        SearchResult {
            score: vector_score,
            vector_score,
            bm25_rank: None,
            reranker_score: None,
            chunk_index: 0,
            content: content.to_string(),
            citation: Citation {
                file_id: 1,
                source_file: PathBuf::from("/docs/test.pdf"),
                file_display_name: "test.pdf".into(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: content.len(),
                formatted: format!("test.pdf, p. 1: '{}'", &content[..content.len().min(50)]),
            },
        }
    }

    // -------------------------------------------------------------------
    // parse_divisors tests
    // -------------------------------------------------------------------

    /// T-REF-001: parse_divisors parses comma-separated integers correctly.
    #[test]
    fn t_ref_001_parse_divisors_basic() {
        assert_eq!(parse_divisors("4,8,16"), vec![4, 8, 16]);
    }

    /// T-REF-002: parse_divisors handles whitespace around values.
    #[test]
    fn t_ref_002_parse_divisors_whitespace() {
        assert_eq!(parse_divisors("4 , 8 , 16"), vec![4, 8, 16]);
    }

    /// T-REF-003: parse_divisors filters out values less than 2.
    #[test]
    fn t_ref_003_parse_divisors_filters_small() {
        assert_eq!(parse_divisors("1,2,4"), vec![2, 4]);
        assert_eq!(parse_divisors("0,1"), Vec::<usize>::new());
    }

    /// T-REF-004: parse_divisors removes duplicates.
    #[test]
    fn t_ref_004_parse_divisors_dedup() {
        assert_eq!(parse_divisors("4,4,8,8"), vec![4, 8]);
    }

    /// T-REF-005: parse_divisors handles empty input.
    #[test]
    fn t_ref_005_parse_divisors_empty() {
        assert_eq!(parse_divisors(""), Vec::<usize>::new());
    }

    /// T-REF-006: parse_divisors handles single value.
    #[test]
    fn t_ref_006_parse_divisors_single() {
        assert_eq!(parse_divisors("4"), vec![4]);
    }

    /// T-REF-007: parse_divisors sorts ascending.
    #[test]
    fn t_ref_007_parse_divisors_sorted() {
        assert_eq!(parse_divisors("16,4,8"), vec![4, 8, 16]);
    }

    /// T-REF-008: parse_divisors ignores non-numeric values.
    #[test]
    fn t_ref_008_parse_divisors_non_numeric() {
        assert_eq!(parse_divisors("4,abc,8"), vec![4, 8]);
    }

    // -------------------------------------------------------------------
    // generate_sub_chunks tests
    // -------------------------------------------------------------------

    /// T-REF-009: generate_sub_chunks returns empty for empty results.
    #[test]
    fn t_ref_009_generate_empty_results() {
        let tok = build_test_tokenizer();
        let candidates = generate_sub_chunks(&[], &tok, &[4, 8]).unwrap();
        assert!(candidates.is_empty());
    }

    /// T-REF-010: generate_sub_chunks returns empty for empty divisors.
    #[test]
    fn t_ref_010_generate_empty_divisors() {
        let tok = build_test_tokenizer();
        let results = vec![make_result(
            "statistical hypothesis testing and p-values in experimental research",
            0.8,
        )];
        let candidates = generate_sub_chunks(&results, &tok, &[]).unwrap();
        assert!(candidates.is_empty());
    }

    /// T-REF-011: generate_sub_chunks produces sub-chunks that are substrings
    /// of the parent result's content.
    #[test]
    fn t_ref_011_sub_chunks_are_substrings() {
        let tok = build_test_tokenizer();
        let content = "statistical hypothesis testing and p-values in experimental research";
        let results = vec![make_result(content, 0.8)];
        let candidates = generate_sub_chunks(&results, &tok, &[2, 4]).unwrap();

        assert!(
            !candidates.is_empty(),
            "must produce at least one sub-chunk"
        );

        for candidate in &candidates {
            assert_eq!(
                candidate.parent_index, 0,
                "all candidates must reference the single result"
            );
            assert!(
                content.contains(&candidate.content),
                "sub-chunk '{}' must be a substring of the original content",
                candidate.content
            );
        }
    }

    /// T-REF-012: generate_sub_chunks respects parent_index for multiple results.
    #[test]
    fn t_ref_012_parent_index_tracking() {
        let tok = build_test_tokenizer();
        let results = vec![
            make_result(
                "statistical hypothesis testing and p-values in experimental research",
                0.8,
            ),
            make_result(
                "machine learning algorithms for classification and regression models",
                0.7,
            ),
        ];
        let candidates = generate_sub_chunks(&results, &tok, &[2]).unwrap();

        let parent_0_count = candidates.iter().filter(|c| c.parent_index == 0).count();
        let parent_1_count = candidates.iter().filter(|c| c.parent_index == 1).count();

        assert!(parent_0_count > 0, "must produce sub-chunks for result 0");
        assert!(parent_1_count > 0, "must produce sub-chunks for result 1");

        // Sub-chunks for result 0 must be substrings of result 0's content.
        for candidate in candidates.iter().filter(|c| c.parent_index == 0) {
            assert!(
                results[0].content.contains(&candidate.content),
                "sub-chunk for result 0 must be substring of result 0"
            );
        }

        // Sub-chunks for result 1 must be substrings of result 1's content.
        for candidate in candidates.iter().filter(|c| c.parent_index == 1) {
            assert!(
                results[1].content.contains(&candidate.content),
                "sub-chunk for result 1 must be substring of result 1"
            );
        }
    }

    /// T-REF-013: generate_sub_chunks skips chunks with too few tokens.
    /// A chunk with 3 tokens and divisor 4 has window=0, which is skipped.
    #[test]
    fn t_ref_013_skip_short_chunks() {
        let tok = build_test_tokenizer();
        let results = vec![make_result("the quick brown", 0.8)];
        // 3 tokens, divisor 4 => window = 0, skip. divisor 2 => window = 1.
        let candidates = generate_sub_chunks(&results, &tok, &[4]).unwrap();
        // With 3 tokens and divisor 4, window=0 => no candidates.
        assert!(
            candidates.is_empty(),
            "divisor 4 on a 3-token chunk must produce no candidates"
        );
    }

    /// T-REF-014: CachedTokenizer::from_json errors on invalid JSON.
    #[test]
    fn t_ref_014_invalid_tokenizer_json() {
        let err = CachedTokenizer::from_json("not valid json");
        assert!(err.is_err(), "must error on invalid tokenizer JSON");
    }

    /// T-REF-015: generate_sub_chunks handles single-word content gracefully.
    #[test]
    fn t_ref_015_single_word_content() {
        let tok = build_test_tokenizer();
        let results = vec![make_result("statistical", 0.8)];
        // 1 token, any divisor >= 2 => window = 0, skip.
        let candidates = generate_sub_chunks(&results, &tok, &[2, 4]).unwrap();
        assert!(
            candidates.is_empty(),
            "single-token chunk must produce no sub-chunks"
        );
    }

    // -------------------------------------------------------------------
    // cosine_similarity tests (via neuroncite_core re-export)
    // -------------------------------------------------------------------

    /// T-REF-016: cosine_similarity of identical vectors returns 1.0.
    #[test]
    fn t_ref_016_cosine_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "cosine similarity of identical vectors must be 1.0, got {sim}"
        );
    }

    /// T-REF-017: cosine_similarity of orthogonal vectors returns 0.0.
    #[test]
    fn t_ref_017_cosine_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "cosine similarity of orthogonal vectors must be 0.0, got {sim}"
        );
    }

    /// T-REF-018: cosine_similarity of zero vector returns 0.0.
    #[test]
    fn t_ref_018_cosine_zero_vector() {
        let zero = vec![0.0_f32; 4];
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&zero, &v);
        assert!(
            sim.abs() < 1e-10,
            "cosine similarity with zero vector must be 0.0, got {sim}"
        );
    }

    // -------------------------------------------------------------------
    // apply_refinement tests
    // -------------------------------------------------------------------

    /// T-REF-019: apply_refinement replaces content when a sub-chunk scores
    /// higher than the original.
    #[test]
    fn t_ref_019_refinement_replaces_content() {
        let mut results = vec![make_result("original long content here", 0.5)];

        let candidates = vec![SubChunkCandidate {
            parent_index: 0,
            content: "relevant part".to_string(),
        }];

        // Query embedding that is more similar to the sub-chunk's embedding
        // than to the original chunk's embedding.
        let query_emb = vec![1.0_f32, 0.0, 0.0, 0.0];
        // Sub-chunk embedding: nearly identical to query (high similarity).
        let sub_emb = vec![vec![0.99_f32, 0.01, 0.0, 0.0]];

        apply_refinement(&mut results, &candidates, &sub_emb, &query_emb);

        assert_eq!(
            results[0].content, "relevant part",
            "content must be replaced by the higher-scoring sub-chunk"
        );
    }

    /// T-REF-020: apply_refinement preserves original content when no
    /// sub-chunk scores higher.
    #[test]
    fn t_ref_020_refinement_preserves_original() {
        let mut results = vec![make_result("original content", 0.95)];

        let candidates = vec![SubChunkCandidate {
            parent_index: 0,
            content: "worse sub-chunk".to_string(),
        }];

        let query_emb = vec![1.0_f32, 0.0, 0.0, 0.0];
        // Sub-chunk embedding: low similarity to query (below original 0.95).
        let sub_emb = vec![vec![0.1_f32, 0.9, 0.0, 0.0]];

        apply_refinement(&mut results, &candidates, &sub_emb, &query_emb);

        assert_eq!(
            results[0].content, "original content",
            "content must be preserved when no sub-chunk scores higher"
        );
    }

    /// T-REF-021: apply_refinement handles empty candidates gracefully.
    #[test]
    fn t_ref_021_refinement_empty_candidates() {
        let mut results = vec![make_result("content", 0.8)];
        apply_refinement(&mut results, &[], &[], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(results[0].content, "content");
    }

    /// T-REF-022: apply_refinement handles multiple results and selects
    /// the best sub-chunk independently for each result.
    #[test]
    fn t_ref_022_refinement_multiple_results() {
        let mut results = vec![
            make_result("result zero original", 0.5),
            make_result("result one original", 0.5),
        ];

        let candidates = vec![
            SubChunkCandidate {
                parent_index: 0,
                content: "better zero".to_string(),
            },
            SubChunkCandidate {
                parent_index: 1,
                content: "worse one".to_string(),
            },
        ];

        let query_emb = vec![1.0_f32, 0.0, 0.0, 0.0];
        let candidate_embs = vec![
            vec![0.95_f32, 0.05, 0.0, 0.0], // better than 0.5
            vec![0.1_f32, 0.9, 0.0, 0.0],   // worse than 0.5
        ];

        apply_refinement(&mut results, &candidates, &candidate_embs, &query_emb);

        assert_eq!(
            results[0].content, "better zero",
            "result 0 must be refined"
        );
        assert_eq!(
            results[1].content, "result one original",
            "result 1 must keep original"
        );
    }

    /// T-REF-023: apply_refinement preserves citation metadata after
    /// content replacement.
    #[test]
    fn t_ref_023_refinement_preserves_citation() {
        let mut results = vec![make_result("original text content", 0.5)];
        let original_file_id = results[0].citation.file_id;
        let original_page_start = results[0].citation.page_start;
        let original_score = results[0].score;

        let candidates = vec![SubChunkCandidate {
            parent_index: 0,
            content: "refined".to_string(),
        }];

        let query_emb = vec![1.0_f32, 0.0, 0.0, 0.0];
        let sub_emb = vec![vec![0.99_f32, 0.01, 0.0, 0.0]];

        apply_refinement(&mut results, &candidates, &sub_emb, &query_emb);

        assert_eq!(results[0].content, "refined");
        assert_eq!(
            results[0].citation.file_id, original_file_id,
            "file_id must be preserved"
        );
        assert_eq!(
            results[0].citation.page_start, original_page_start,
            "page_start must be preserved"
        );
        assert!(
            (results[0].score - original_score).abs() < f64::EPSILON,
            "ranking score must be preserved"
        );
    }

    /// T-REF-024: RefinementConfig default values.
    #[test]
    fn t_ref_024_default_config() {
        let config = RefinementConfig::default();
        assert!(config.enabled, "refinement must be enabled by default");
        assert_eq!(
            config.divisors,
            vec![4, 8, 16],
            "default divisors must be [4, 8, 16]"
        );
    }

    /// T-REF-025: generate_sub_chunks with divisor 2 on an 8-token chunk
    /// produces sub-chunks with exactly 4 tokens each (window = 8/2 = 4).
    #[test]
    fn t_ref_025_divisor_produces_correct_window_size() {
        let tok = build_test_tokenizer();
        // 8 tokens: "the quick brown fox jumps over lazy dog"
        let content = "the quick brown fox jumps over lazy dog";
        let results = vec![make_result(content, 0.8)];
        let candidates = generate_sub_chunks(&results, &tok, &[2]).unwrap();

        assert!(
            !candidates.is_empty(),
            "divisor 2 on 8-token chunk must produce sub-chunks"
        );

        // Each sub-chunk should contain roughly 4 words (tokens).
        for candidate in &candidates {
            let word_count = candidate.content.split_whitespace().count();
            assert!(
                word_count <= 5,
                "sub-chunk '{}' has {} words, expected at most ~4 for divisor 2",
                candidate.content,
                word_count
            );
        }
    }

    /// T-REF-026: generate_sub_chunks with multiple divisors produces
    /// candidates at different granularities.
    #[test]
    fn t_ref_026_multiple_divisors_different_granularities() {
        let tok = build_test_tokenizer();
        // 8 tokens
        let content = "the quick brown fox jumps over lazy dog";
        let results = vec![make_result(content, 0.8)];

        let candidates_div2 = generate_sub_chunks(&results, &tok, &[2]).unwrap();
        let candidates_div4 = generate_sub_chunks(&results, &tok, &[4]).unwrap();
        let candidates_both = generate_sub_chunks(&results, &tok, &[2, 4]).unwrap();

        // More candidates with both divisors than with either alone.
        assert!(
            candidates_both.len() >= candidates_div2.len(),
            "both divisors must produce at least as many candidates as divisor 2 alone"
        );
        assert!(
            candidates_both.len() >= candidates_div4.len(),
            "both divisors must produce at least as many candidates as divisor 4 alone"
        );
        assert_eq!(
            candidates_both.len(),
            candidates_div2.len() + candidates_div4.len(),
            "combined candidates must equal sum of individual divisor candidates"
        );
    }

    /// T-REF-027: CachedTokenizer::from_json succeeds with valid tokenizer
    /// JSON and the resulting tokenizer can be used for generate_sub_chunks.
    #[test]
    fn t_ref_027_cached_tokenizer_from_json() {
        let json = build_test_tokenizer_json();
        let tok = CachedTokenizer::from_json(&json);
        assert!(
            tok.is_ok(),
            "CachedTokenizer::from_json must succeed with valid JSON"
        );

        // The deserialized tokenizer must produce the same results as the
        // directly-constructed one.
        let content = "the quick brown fox jumps over lazy dog";
        let results = vec![make_result(content, 0.8)];
        let candidates = generate_sub_chunks(&results, &tok.unwrap(), &[2]).unwrap();
        assert!(
            !candidates.is_empty(),
            "deserialized tokenizer must produce sub-chunks"
        );
    }

    /// T-REF-028: generate_sub_chunks enforces MAX_REFINEMENT_CANDIDATES.
    /// With enough results and fine-grained divisors, the candidate count
    /// must not exceed the hard limit.
    #[test]
    fn t_ref_028_max_refinement_candidates_enforced() {
        let tok = build_test_tokenizer();

        // Create many results with long content to generate many sub-chunks.
        // Each result has 32 tokens, with divisors [2,4,8,16] producing
        // roughly 4+9+20+41 = 74 sub-chunks per result. With 20 results,
        // the uncapped total would be ~1480, well above MAX_REFINEMENT_CANDIDATES.
        let long_content = (0..4)
            .map(|_| "the quick brown fox jumps over lazy dog")
            .collect::<Vec<_>>()
            .join(" ");
        let results: Vec<SearchResult> = (0..20).map(|_| make_result(&long_content, 0.8)).collect();

        let candidates = generate_sub_chunks(&results, &tok, &[2, 4, 8, 16]).unwrap();

        assert!(
            candidates.len() <= MAX_REFINEMENT_CANDIDATES,
            "candidate count {} must not exceed MAX_REFINEMENT_CANDIDATES ({})",
            candidates.len(),
            MAX_REFINEMENT_CANDIDATES
        );

        // The limit must be reached (i.e., it was actually capping).
        // With 20 results * ~74 sub-chunks, uncapped would produce >512.
        assert!(
            candidates.len() >= MAX_REFINEMENT_CANDIDATES / 2,
            "must produce a substantial number of candidates (got {})",
            candidates.len()
        );
    }
}
