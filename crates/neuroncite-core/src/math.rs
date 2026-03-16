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

// Mathematical utility functions shared across the NeuronCite workspace.
//
// This module provides vector similarity computations that multiple crates
// require (neuroncite-api for citation verification, neuroncite-search for
// sub-chunk refinement). Centralizing the implementation here eliminates
// code duplication and ensures consistent behavior across all call sites.

/// Computes the cosine similarity between two f32 slices, returning a value
/// in the range [-1.0, 1.0] as f64. The computation promotes each f32
/// element to f64 before accumulation to avoid precision loss during the
/// dot product and magnitude summation -- this matters for high-dimensional
/// embedding vectors (384-1024 dimensions) where f32 accumulation can
/// diverge noticeably from the true value.
///
/// Returns 0.0 in the following edge cases:
/// - Either slice is empty.
/// - Either slice has zero magnitude (all elements are 0.0).
/// - The slices have different lengths (only the overlapping prefix is
///   used for the dot product, but the magnitudes use the full slices,
///   so this case is ill-defined; returning 0.0 avoids misleading results).
///
/// # Examples
///
/// ```
/// use neuroncite_core::math::cosine_similarity;
///
/// let a = [1.0_f32, 0.0, 0.0];
/// let b = [0.0_f32, 1.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-10);
///
/// let c = [3.0_f32, 4.0];
/// assert!((cosine_similarity(&c, &c) - 1.0).abs() < 1e-10);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    // If the slices differ in length, the cosine similarity is undefined
    // because the vectors live in different-dimensional spaces. Return 0.0
    // to signal "no meaningful similarity" rather than panicking.
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xd = f64::from(x);
        let yd = f64::from(y);
        dot += xd * yd;
        norm_a += xd * xd;
        norm_b += yd * yd;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    let result = dot / denom;
    // NaN can arise if denom is subnormal and rounds to zero after the division,
    // or from pathological input values. Map NaN to 0.0 (no meaningful similarity).
    if result.is_nan() { 0.0 } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CORE-MATH-001: Identical vectors produce cosine similarity of 1.0.
    /// Two equal non-zero vectors are perfectly aligned, so their cosine is 1.0.
    #[test]
    fn t_core_math_001_identical_vectors() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "identical vectors must have similarity 1.0, got {sim}"
        );
    }

    /// T-CORE-MATH-002: Orthogonal vectors produce cosine similarity of 0.0.
    /// Two vectors whose dot product is zero are perpendicular.
    #[test]
    fn t_core_math_002_orthogonal_vectors() {
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "orthogonal vectors must have similarity 0.0, got {sim}"
        );
    }

    /// T-CORE-MATH-003: Opposite vectors (negated) produce cosine similarity
    /// of -1.0. A vector and its negation point in exactly opposite directions.
    #[test]
    fn t_core_math_003_opposite_vectors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [-1.0_f32, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-10,
            "opposite vectors must have similarity -1.0, got {sim}"
        );
    }

    /// T-CORE-MATH-004: A zero-magnitude vector (all elements 0.0) returns
    /// similarity 0.0 because the denominator is zero.
    #[test]
    fn t_core_math_004_zero_length_vector() {
        let zero = [0.0_f32; 4];
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        assert!(
            cosine_similarity(&zero, &v).abs() < 1e-10,
            "zero vector as first argument must return 0.0"
        );
        assert!(
            cosine_similarity(&v, &zero).abs() < 1e-10,
            "zero vector as second argument must return 0.0"
        );
        assert!(
            cosine_similarity(&zero, &zero).abs() < 1e-10,
            "two zero vectors must return 0.0"
        );
    }

    /// T-CORE-MATH-005: Vectors of different lengths return 0.0 because
    /// cosine similarity is undefined for vectors in different-dimensional
    /// spaces.
    #[test]
    fn t_core_math_005_different_length_vectors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "different-length vectors must return 0.0, got {sim}"
        );
    }

    /// T-CORE-MATH-006: Single-element vectors. Cosine similarity between
    /// two positive scalars is 1.0 (same direction on the number line).
    #[test]
    fn t_core_math_006_single_element_vectors() {
        let a = [5.0_f32];
        let b = [3.0_f32];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "two positive single-element vectors must have similarity 1.0, got {sim}"
        );

        // One positive, one negative: opposite directions.
        let c = [-3.0_f32];
        let sim_neg = cosine_similarity(&a, &c);
        assert!(
            (sim_neg + 1.0).abs() < 1e-10,
            "positive and negative single-element vectors must have similarity -1.0, got {sim_neg}"
        );
    }

    /// T-CORE-MATH-007: Empty slices return 0.0 because there are no
    /// dimensions to compare.
    #[test]
    fn t_core_math_007_empty_slices() {
        let empty: [f32; 0] = [];
        let v = [1.0_f32, 2.0];
        assert!(
            cosine_similarity(&empty, &v).abs() < 1e-10,
            "empty first slice must return 0.0"
        );
        assert!(
            cosine_similarity(&v, &empty).abs() < 1e-10,
            "empty second slice must return 0.0"
        );
        assert!(
            cosine_similarity(&empty, &empty).abs() < 1e-10,
            "two empty slices must return 0.0"
        );
    }
}
