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

//! L2 normalization for embedding vectors.
//!
//! After the forward pass produces a raw embedding vector, this module normalizes
//! it to unit length (L2 norm = 1.0). Normalized embeddings allow cosine
//! similarity to be computed as a simple dot product, which is the operation
//! used by the HNSW index in neuroncite-store.

/// Normalizes a single embedding vector in-place to unit L2 norm.
///
/// Computes the Euclidean norm (square root of the sum of squares) of all
/// elements, then divides each element by that norm. If the vector is a
/// zero-vector (norm is zero or below `f32::EPSILON`), it is left unchanged
/// to avoid division by zero.
///
/// # Arguments
///
/// * `vector` - Mutable slice of `f32` values representing a single embedding.
///
/// # Examples
///
/// ```
/// use neuroncite_embed::normalize::l2_normalize;
///
/// let mut v = vec![3.0_f32, 4.0];
/// l2_normalize(&mut v);
/// let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn l2_normalize(vector: &mut [f32]) {
    let sum_of_squares: f32 = vector.iter().map(|x| x * x).sum();
    let norm = sum_of_squares.sqrt();

    // A zero-vector has no direction so normalization is undefined.
    // NaN arises when sum_of_squares is NaN (e.g., from a model that emits
    // NaN activations). Both conditions leave the vector unchanged to prevent
    // NaN from propagating through the embedding pipeline.
    if norm.is_nan() || norm < f32::EPSILON {
        return;
    }

    let inv_norm = 1.0 / norm;
    for element in vector.iter_mut() {
        *element *= inv_norm;
    }
}

/// Normalizes a batch of embedding vectors in-place to unit L2 norm.
///
/// Applies `l2_normalize` to each vector in the batch independently. This is
/// a convenience wrapper for post-processing the output of `embed_batch`.
///
/// # Arguments
///
/// * `vectors` - Mutable slice of `Vec<f32>` vectors to normalize.
pub fn l2_normalize_batch(vectors: &mut [Vec<f32>]) {
    for vector in vectors.iter_mut() {
        l2_normalize(vector);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-EMB-001: L2 normalization produces a unit vector.
    /// After normalization, the Euclidean norm of the vector must be within
    /// 1e-6 of 1.0 for any non-zero input.
    #[test]
    fn t_emb_001_l2_normalization_produces_unit_vector() {
        let mut v = vec![3.0_f32, 4.0, 0.0, 5.0, 1.0];
        l2_normalize(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "norm after L2 normalization should be 1.0, got {norm}"
        );
    }

    /// Verifies that a zero-vector remains unchanged after normalization.
    /// Division by zero must not produce NaN or infinity values.
    #[test]
    fn zero_vector_remains_unchanged() {
        let mut v = vec![0.0_f32; 5];
        l2_normalize(&mut v);

        for (i, &element) in v.iter().enumerate() {
            assert!(
                element.abs() < f32::EPSILON,
                "element {i} should remain zero, got {element}"
            );
        }
    }

    /// Verifies that batch normalization applies L2 normalization to each
    /// vector independently.
    #[test]
    fn batch_normalization_normalizes_each_vector() {
        let mut batch = vec![
            vec![3.0_f32, 4.0],
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![5.0, 12.0],
        ];
        l2_normalize_batch(&mut batch);

        // First vector: norm should be 1.0
        let norm_0: f32 = batch[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_0 - 1.0).abs() < 1e-6);

        // Second vector: [1, 0] normalized is still [1, 0]
        assert!((batch[1][0] - 1.0).abs() < 1e-6);
        assert!(batch[1][1].abs() < 1e-6);

        // Third vector: zero-vector stays zero
        assert!(batch[2][0].abs() < f32::EPSILON);
        assert!(batch[2][1].abs() < f32::EPSILON);

        // Fourth vector: [5, 12] / 13 = [0.3846, 0.9230]
        let norm_3: f32 = batch[3].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_3 - 1.0).abs() < 1e-6);
    }

    /// Verifies that a single-element vector normalizes to exactly 1.0 or -1.0.
    #[test]
    fn single_element_vector_normalizes_to_unit() {
        let mut v = vec![42.0_f32];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);

        let mut v_neg = vec![-7.5_f32];
        l2_normalize(&mut v_neg);
        assert!((v_neg[0] + 1.0).abs() < 1e-6);
    }
}
