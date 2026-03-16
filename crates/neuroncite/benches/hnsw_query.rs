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

// Benchmark for HNSW nearest neighbor query latency.
//
// Builds an HNSW index with 1000 deterministic pseudo-random vectors at
// dimension 384 (matching the BGE-small model), then measures the time
// to query the index for the 10 nearest neighbors of a single vector.
// The index is built once during setup and reused across all iterations.

use criterion::{Criterion, criterion_group, criterion_main};
use neuroncite_store::build_hnsw;

/// Embedding vector dimensionality used throughout the benchmark.
/// Matches the output dimension of BAAI/bge-small-en-v1.5.
const DIMENSION: usize = 384;

/// Number of vectors inserted into the HNSW index for querying.
const NUM_VECTORS: usize = 1000;

/// Number of nearest neighbors requested per query.
const TOP_K: usize = 10;

/// ef_search parameter controlling the recall/latency trade-off.
/// Higher values increase recall at the cost of slower queries.
const EF_SEARCH: usize = 64;

/// Generates a deterministic pseudo-random f32 vector from a seed value.
/// Uses a linear congruential generator (LCG) for reproducibility across
/// runs. The output vector is L2-normalized to unit length.
fn make_vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dimension);
    let mut state = seed;
    for _ in 0..dimension {
        // LCG constants from Knuth's MMIX
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let val = (state >> 33) as f32 / (u32::MAX as f32);
        v.push(val);
    }
    // Normalize to unit length for cosine distance compatibility.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Measures the latency of a single HNSW nearest neighbor query on an index
/// containing 1000 vectors of dimension 384.
fn bench_hnsw_query(c: &mut Criterion) {
    // Build the index once outside the benchmark loop.
    let vectors: Vec<Vec<f32>> = (1..=NUM_VECTORS as u64)
        .map(|seed| make_vector(seed, DIMENSION))
        .collect();

    let labeled: Vec<(i64, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
        .collect();

    let index = build_hnsw(&labeled, DIMENSION).expect("build_hnsw");

    // Use a query vector that is not in the index to measure a realistic
    // search path through the graph.
    let query = make_vector(0, DIMENSION);

    c.bench_function("hnsw_query_top10_1000vecs_dim384", |b| {
        b.iter(|| {
            let results = index.search(&query, TOP_K, EF_SEARCH);
            // Prevent the compiler from optimizing away the search call.
            assert!(!results.is_empty());
        });
    });
}

/// Measures the latency of a nearest neighbor query with a larger ef_search
/// value (200), which increases recall at the cost of higher latency.
fn bench_hnsw_query_high_ef(c: &mut Criterion) {
    let vectors: Vec<Vec<f32>> = (1..=NUM_VECTORS as u64)
        .map(|seed| make_vector(seed, DIMENSION))
        .collect();

    let labeled: Vec<(i64, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
        .collect();

    let index = build_hnsw(&labeled, DIMENSION).expect("build_hnsw");
    let query = make_vector(0, DIMENSION);

    c.bench_function("hnsw_query_top10_1000vecs_dim384_ef200", |b| {
        b.iter(|| {
            let results = index.search(&query, TOP_K, 200);
            assert!(!results.is_empty());
        });
    });
}

criterion_group!(benches, bench_hnsw_query, bench_hnsw_query_high_ef);
criterion_main!(benches);
