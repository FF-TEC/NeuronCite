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

// Benchmark for CPU-bound embedding vector normalization throughput.
//
// Measures the performance of L2 normalization operations from the
// neuroncite-embed crate. These are pure floating-point computations
// that do not require a loaded ML model, making them suitable for
// benchmarking on any machine without GPU or model dependencies.

use criterion::{Criterion, criterion_group, criterion_main};
use neuroncite_embed::{l2_normalize, l2_normalize_batch};

/// Embedding vector dimensionality matching BAAI/bge-small-en-v1.5.
const DIMENSION: usize = 384;

/// Number of vectors in a batch normalization benchmark.
const BATCH_SIZE: usize = 32;

/// Generates a deterministic pseudo-random f32 vector from a seed value.
/// Uses a linear congruential generator for reproducible benchmark data.
/// The output is intentionally not normalized, since the benchmark measures
/// the normalization operation itself.
fn make_unnormalized_vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dimension);
    let mut state = seed;
    for _ in 0..dimension {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Map to range [-1.0, 1.0] to produce vectors with varying magnitudes.
        let val = ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
        v.push(val);
    }
    v
}

/// Measures the throughput of L2-normalizing a single 384-dimensional vector.
fn bench_l2_normalize_single(c: &mut Criterion) {
    let original = make_unnormalized_vector(42, DIMENSION);

    c.bench_function("l2_normalize_single_dim384", |b| {
        b.iter_batched(
            || original.clone(),
            |mut v| {
                l2_normalize(&mut v);
                v
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Measures the throughput of L2-normalizing a batch of 32 vectors, each
/// with 384 dimensions. This reflects the typical post-processing step
/// after a batch embedding call.
fn bench_l2_normalize_batch_32(c: &mut Criterion) {
    let originals: Vec<Vec<f32>> = (0..BATCH_SIZE as u64)
        .map(|seed| make_unnormalized_vector(seed + 100, DIMENSION))
        .collect();

    c.bench_function("l2_normalize_batch_32x384", |b| {
        b.iter_batched(
            || originals.clone(),
            |mut batch| {
                l2_normalize_batch(&mut batch);
                batch
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Measures the throughput of L2-normalizing a single vector with a larger
/// dimension (768), corresponding to models like BAAI/bge-base-en-v1.5.
fn bench_l2_normalize_dim768(c: &mut Criterion) {
    let original = make_unnormalized_vector(99, 768);

    c.bench_function("l2_normalize_single_dim768", |b| {
        b.iter_batched(
            || original.clone(),
            |mut v| {
                l2_normalize(&mut v);
                v
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Measures the throughput of L2-normalizing a batch of 32 vectors at
/// dimension 768 to compare scaling behavior across different model sizes.
fn bench_l2_normalize_batch_32_dim768(c: &mut Criterion) {
    let originals: Vec<Vec<f32>> = (0..BATCH_SIZE as u64)
        .map(|seed| make_unnormalized_vector(seed + 200, 768))
        .collect();

    c.bench_function("l2_normalize_batch_32x768", |b| {
        b.iter_batched(
            || originals.clone(),
            |mut batch| {
                l2_normalize_batch(&mut batch);
                batch
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_l2_normalize_single,
    bench_l2_normalize_batch_32,
    bench_l2_normalize_dim768,
    bench_l2_normalize_batch_32_dim768
);
criterion_main!(benches);
