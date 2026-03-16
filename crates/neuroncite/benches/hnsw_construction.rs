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

// Benchmark for HNSW index construction from random vectors.
//
// Measures the time to build an HNSW approximate nearest neighbor index
// from 500 deterministic pseudo-random vectors at dimension 384. This
// benchmark exercises the parallel insertion path in hnsw_rs and the
// index finalization (switching to search mode).

use criterion::{Criterion, criterion_group, criterion_main};
use neuroncite_store::build_hnsw;

/// Embedding vector dimensionality matching BAAI/bge-small-en-v1.5.
const DIMENSION: usize = 384;

/// Number of vectors used for the primary construction benchmark.
const NUM_VECTORS_500: usize = 500;

/// Number of vectors used for a smaller construction benchmark to
/// measure scaling behavior.
const NUM_VECTORS_100: usize = 100;

/// Generates a deterministic pseudo-random f32 vector from a seed value.
/// Uses a linear congruential generator for reproducibility. The output
/// is L2-normalized to unit length for cosine distance compatibility.
fn make_vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dimension);
    let mut state = seed;
    for _ in 0..dimension {
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

/// Measures the time to build an HNSW index from 500 vectors at dim 384.
/// The vector generation is performed inside the measured region because
/// build_hnsw requires references into the vector storage, making
/// separation impractical. The generation cost is small relative to
/// the index construction cost.
fn bench_hnsw_build_500(c: &mut Criterion) {
    c.bench_function("hnsw_build_500vecs_dim384", |b| {
        b.iter(|| {
            let vectors: Vec<Vec<f32>> = (1..=NUM_VECTORS_500 as u64)
                .map(|seed| make_vector(seed, DIMENSION))
                .collect();

            let labeled: Vec<(i64, &[f32])> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
                .collect();

            let index = build_hnsw(&labeled, DIMENSION).expect("build_hnsw");
            assert_eq!(index.len(), NUM_VECTORS_500);
        });
    });
}

/// Measures the time to build an HNSW index from 100 vectors at dim 384.
/// Provides a reference point for estimating how construction time scales
/// with the number of inserted vectors.
fn bench_hnsw_build_100(c: &mut Criterion) {
    c.bench_function("hnsw_build_100vecs_dim384", |b| {
        b.iter(|| {
            let vectors: Vec<Vec<f32>> = (1..=NUM_VECTORS_100 as u64)
                .map(|seed| make_vector(seed, DIMENSION))
                .collect();

            let labeled: Vec<(i64, &[f32])> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
                .collect();

            let index = build_hnsw(&labeled, DIMENSION).expect("build_hnsw");
            assert_eq!(index.len(), NUM_VECTORS_100);
        });
    });
}

criterion_group!(benches, bench_hnsw_build_500, bench_hnsw_build_100);
criterion_main!(benches);
