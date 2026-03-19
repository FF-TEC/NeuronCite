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

// Property tests for the search subsystem (SimHash and RRF fusion).
//
// Since the `dedup` and `fusion` modules of `neuroncite-search` are not
// publicly exported, these tests verify the properties through independent
// reimplementations of the core algorithms and through the database-backed
// pipeline integration where applicable.
//
// T-PROP-005 verifies SimHash collision-freedom for identical texts by
// reimplementing the FNV-1a-based SimHash algorithm from dedup.rs and
// confirming that identical inputs always produce Hamming distance 0.
//
// T-PROP-006 verifies RRF score boundedness by reimplementing the RRF
// formula and checking that scores are strictly positive and bounded
// above by 2/k (the maximum possible score for a candidate appearing at
// rank 1 in both rankers).

// -- SimHash reimplementation for property testing ---------------------------
//
// This is a faithful copy of the algorithm in neuroncite-search::dedup, used
// here to verify properties without depending on the private module.

/// Computes a 64-bit SimHash fingerprint for the given text using FNV-1a
/// token hashing and bit-counting accumulation. Identical to the algorithm
/// in the dedup module of neuroncite-search.
fn compute_simhash(text: &str) -> u64 {
    let mut counters = [0_i32; 64];

    for token in text.split_whitespace() {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }

        for (bit, counter) in counters.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *counter += 1;
            } else {
                *counter -= 1;
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for (bit, &count) in counters.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1_u64 << bit;
        }
    }
    fingerprint
}

/// Returns the Hamming distance (number of differing bits) between two
/// 64-bit values.
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

// -- RRF score computation for property testing ------------------------------

/// Computes the RRF score for a candidate that appears at the given 1-indexed
/// ranks in the vector and keyword result lists. The formula is:
///   RRF(d) = sum(1 / (k + rank_r(d))) for each ranker r
///
/// If the candidate does not appear in one of the lists, that ranker's
/// contribution is 0.
fn rrf_score(vector_rank: Option<usize>, keyword_rank: Option<usize>, k: usize) -> f64 {
    let k_f64 = k as f64;
    let mut score = 0.0;
    if let Some(vr) = vector_rank {
        score += 1.0 / (k_f64 + vr as f64);
    }
    if let Some(kr) = keyword_rank {
        score += 1.0 / (k_f64 + kr as f64);
    }
    score
}

// -- Deterministic pseudo-random input generation ----------------------------

/// Deterministic 64-bit linear congruential generator.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns a pseudo-random value in the range [lo, hi] (inclusive).
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(hi >= lo);
        let span = hi - lo + 1;
        lo + (self.next_u64() % span)
    }
}

/// Word pool for generating random text strings.
const WORD_POOL: &[&str] = &[
    "the",
    "of",
    "and",
    "in",
    "to",
    "a",
    "is",
    "that",
    "for",
    "it",
    "was",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "at",
    "be",
    "this",
    "from",
    "have",
    "or",
    "by",
    "one",
    "had",
    "not",
    "but",
    "what",
    "all",
    "statistical",
    "regression",
    "variance",
    "distribution",
    "hypothesis",
    "probability",
    "sampling",
    "correlation",
    "experiment",
    "observation",
];

/// Generates a random whitespace-delimited text string.
fn random_text(rng: &mut Lcg, word_count: usize) -> String {
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let idx = rng.range(0, (WORD_POOL.len() - 1) as u64) as usize;
        words.push(WORD_POOL[idx]);
    }
    words.join(" ")
}

// -- T-PROP-007 ---------------------------------------------------------------

/// T-PROP-007: SimHash of identical texts always produces Hamming distance 0.
///
/// The SimHash algorithm is deterministic: given the same input text, it must
/// produce the same 64-bit fingerprint every time. Therefore, the Hamming
/// distance between the fingerprint of a text and itself must always be 0.
///
/// This property is tested with 100 random text strings of varying lengths
/// (1 to 200 words).
#[test]
fn t_prop_007_simhash_identical_texts_hamming_zero() {
    let mut rng = Lcg::new(0xA5FA_CE50_0000_0005);

    for trial in 0..100 {
        let word_count = rng.range(1, 200) as usize;
        let text = random_text(&mut rng, word_count);

        let hash1 = compute_simhash(&text);
        let hash2 = compute_simhash(&text);

        assert_eq!(
            hash1, hash2,
            "trial {trial}: SimHash of identical text must be equal"
        );

        let distance = hamming_distance(hash1, hash2);
        assert_eq!(
            distance, 0,
            "trial {trial}: Hamming distance of identical SimHash fingerprints must be 0"
        );
    }

    // Additional check: verify the property with specific edge cases.
    // Empty text (no whitespace tokens) produces a fingerprint of 0.
    let empty_hash = compute_simhash("");
    assert_eq!(
        hamming_distance(empty_hash, empty_hash),
        0,
        "empty text self-distance must be 0"
    );

    // Single-word text.
    let single_hash = compute_simhash("statistics");
    assert_eq!(
        hamming_distance(single_hash, single_hash),
        0,
        "single-word text self-distance must be 0"
    );

    // Unicode text with accented characters.
    let unicode_hash = compute_simhash("Stra\u{00DF}e Ubersicht Ubung");
    assert_eq!(
        hamming_distance(unicode_hash, unicode_hash),
        0,
        "unicode text self-distance must be 0"
    );
}

// -- T-PROP-008 ---------------------------------------------------------------

/// T-PROP-008: RRF fusion score is always positive and bounded.
///
/// For any candidate that appears in at least one result list at a valid
/// 1-indexed rank position, the RRF score must satisfy:
///   0 < RRF(d) <= 2 / (k + 1)
///
/// The upper bound occurs when a candidate appears at rank 1 in both rankers:
///   RRF_max = 1/(k+1) + 1/(k+1) = 2/(k+1)
///
/// The lower bound is strictly positive because 1/(k+rank) > 0 for all
/// positive k and rank values.
///
/// This property is tested with 200 random rank combinations across different
/// k values (10, 30, 60, 100, 200).
#[test]
fn t_prop_006_rrf_score_positive_and_bounded() {
    let mut rng = Lcg::new(0xBEEF_CAFE_0000_0006);
    let k_values = [10, 30, 60, 100, 200];

    for trial in 0..200 {
        let k = k_values[trial % k_values.len()];

        // Generate random rank positions. At least one must be present (Some).
        let max_rank = 1000_u64;
        let vector_rank: Option<usize> = if rng.range(0, 2) > 0 {
            Some(rng.range(1, max_rank) as usize)
        } else {
            None
        };
        let keyword_rank: Option<usize> = if rng.range(0, 2) > 0 || vector_rank.is_none() {
            // Ensure at least one rank is present.
            Some(rng.range(1, max_rank) as usize)
        } else {
            None
        };

        let score = rrf_score(vector_rank, keyword_rank, k);

        // The score must be strictly positive.
        assert!(
            score > 0.0,
            "trial {trial}: RRF score must be > 0, got {score} \
             (k={k}, vector_rank={vector_rank:?}, keyword_rank={keyword_rank:?})"
        );

        // The score must be bounded above by 2/(k+1).
        let max_score = 2.0 / (k as f64 + 1.0);
        assert!(
            score <= max_score + 1e-12, // small epsilon for floating-point comparison
            "trial {trial}: RRF score {score} exceeds maximum {max_score} \
             (k={k}, vector_rank={vector_rank:?}, keyword_rank={keyword_rank:?})"
        );

        // The score must be finite (not NaN or infinity).
        assert!(
            score.is_finite(),
            "trial {trial}: RRF score must be finite, got {score}"
        );
    }

    // Verify the exact maximum score: rank 1 in both lists.
    for &k in &k_values {
        let max_score = rrf_score(Some(1), Some(1), k);
        let expected = 2.0 / (k as f64 + 1.0);
        assert!(
            (max_score - expected).abs() < 1e-12,
            "maximum RRF score for k={k} must be {expected}, got {max_score}"
        );
    }

    // Verify monotonicity: better rank produces higher score.
    for &k in &k_values {
        let score_rank1 = rrf_score(Some(1), None, k);
        let score_rank10 = rrf_score(Some(10), None, k);
        let score_rank100 = rrf_score(Some(100), None, k);

        assert!(
            score_rank1 > score_rank10,
            "k={k}: rank 1 score ({score_rank1}) must exceed rank 10 score ({score_rank10})"
        );
        assert!(
            score_rank10 > score_rank100,
            "k={k}: rank 10 score ({score_rank10}) must exceed rank 100 score ({score_rank100})"
        );
    }
}
