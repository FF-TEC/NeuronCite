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

// Property tests for the embedding normalization subsystem.
//
// These tests verify that `l2_normalize` and `l2_normalize_batch` satisfy
// their mathematical invariants for all non-zero inputs: the resulting vectors
// have unit L2 norm, and batch normalization applies the same transformation
// independently to each vector in the batch.
//
// Input generation uses a deterministic linear congruential generator seeded
// with fixed values. Each property is tested across 100 random vectors with
// dimensions ranging from 1 to 1024.

use neuroncite_embed::{l2_normalize, l2_normalize_batch};

// -- Deterministic pseudo-random input generation ----------------------------

/// Deterministic 64-bit linear congruential generator for reproducible test
/// input generation. Uses Knuth's MMIX constants.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Creates a generator seeded with the given value.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns a pseudo-random f32 in the range [-max_abs, +max_abs].
    fn next_f32(&mut self, max_abs: f32) -> f32 {
        let raw = self.next_u64();
        // Map the upper 32 bits to a value in [-1.0, 1.0], then scale.
        let normalized = ((raw >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        normalized * max_abs
    }
}

/// Generates a random f32 vector of the given dimension with values in
/// [-max_abs, +max_abs]. Guarantees at least one non-zero element by
/// setting the first element to 1.0 if all generated values are below
/// f32::EPSILON.
fn random_vector(rng: &mut Lcg, dimension: usize, max_abs: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dimension).map(|_| rng.next_f32(max_abs)).collect();

    // Ensure the vector is non-zero. If all elements are near-zero, force
    // the first element to 1.0 so the normalization invariant is testable.
    let sum_sq: f32 = v.iter().map(|x| x * x).sum();
    if sum_sq < f32::EPSILON {
        v[0] = 1.0;
    }

    v
}

// -- T-PROP-003 ---------------------------------------------------------------

/// T-PROP-003: `l2_normalize` produces unit vectors for any non-zero input.
///
/// For 100 random non-zero vectors with dimensions ranging from 1 to 1024,
/// the L2 norm of the normalized vector must be within 1e-6 of 1.0. This
/// tolerance accounts for accumulated floating-point rounding in the sum of
/// squares and square root operations.
#[test]
fn t_prop_003_l2_normalize_produces_unit_vectors() {
    let mut rng = Lcg::new(0xABCD_1234_5678_0003);

    for trial in 0..100 {
        // Vary the dimension across trials to cover scalar, small, and large vectors.
        let dimension = match trial % 5 {
            0 => 1,
            1 => 3,
            2 => 128,
            3 => 384,
            _ => (rng.next_u64() % 1024 + 1) as usize,
        };

        // Vary the magnitude range to test with small, normal, and large values.
        let max_abs = match trial % 4 {
            0 => 0.01,
            1 => 1.0,
            2 => 100.0,
            _ => 10_000.0,
        };

        let mut v = random_vector(&mut rng, dimension, max_abs);
        l2_normalize(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-5,
            "trial {trial}: L2 norm after normalization is {norm}, expected 1.0 \
             (dimension={dimension}, max_abs={max_abs})"
        );

        // Verify that no element is NaN or infinite after normalization.
        for (i, &val) in v.iter().enumerate() {
            assert!(
                val.is_finite(),
                "trial {trial}: element {i} is not finite after normalization: {val}"
            );
        }
    }
}

// -- T-PROP-004 ---------------------------------------------------------------

/// T-PROP-004: `l2_normalize_batch` normalizes each vector independently.
///
/// For 50 random batches (each containing 2 to 20 vectors), the function
/// must produce the same result as calling `l2_normalize` on each vector
/// individually. This test verifies independence by comparing batch-normalized
/// vectors element-by-element against independently normalized copies.
#[test]
fn t_prop_004_l2_normalize_batch_normalizes_independently() {
    let mut rng = Lcg::new(0xBADC_AFFE_0000_0004);

    for trial in 0..50 {
        let batch_size = (rng.next_u64() % 19 + 2) as usize; // 2..=20 vectors
        let dimension = (rng.next_u64() % 512 + 1) as usize; // 1..=512 dimensions

        // Generate the batch of random vectors.
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| random_vector(&mut rng, dimension, 50.0))
            .collect();

        // Create two copies: one for batch normalization, one for individual.
        let mut batch_copy = vectors.clone();
        let mut individual_copies = vectors.clone();

        // Apply batch normalization.
        l2_normalize_batch(&mut batch_copy);

        // Apply individual normalization to each vector separately.
        for v in &mut individual_copies {
            l2_normalize(v);
        }

        // Compare element-by-element.
        for (vec_idx, (batch_vec, indiv_vec)) in
            batch_copy.iter().zip(individual_copies.iter()).enumerate()
        {
            assert_eq!(
                batch_vec.len(),
                indiv_vec.len(),
                "trial {trial}: vector {vec_idx} length mismatch"
            );

            for (elem_idx, (&batch_val, &indiv_val)) in
                batch_vec.iter().zip(indiv_vec.iter()).enumerate()
            {
                let diff = (batch_val - indiv_val).abs();
                assert!(
                    diff < 1e-7,
                    "trial {trial}: vector {vec_idx}, element {elem_idx}: \
                     batch={batch_val}, individual={indiv_val}, diff={diff}"
                );
            }
        }

        // Verify that each batch-normalized vector has unit norm.
        for (vec_idx, v) in batch_copy.iter().enumerate() {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "trial {trial}: batch vector {vec_idx} norm is {norm}, expected 1.0"
            );
        }
    }
}
