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

// Benchmark for BM25 keyword search via the FTS5 virtual table.
//
// Creates an in-memory SQLite database with the full NeuronCite schema,
// inserts 1000 chunks of synthetic text, and measures the latency of
// FTS5 MATCH queries using direct rusqlite SQL. The database is built
// once during setup and reused across all benchmark iterations.
//
// Direct SQL is used instead of neuroncite_search::keyword because the
// keyword module is intentionally private to the search crate. The SQL
// statements here mirror the exact query pattern used by that module.

use criterion::{Criterion, criterion_group, criterion_main};
use neuroncite_core::{IndexConfig, StorageMode};
use neuroncite_store::{ChunkInsert, bulk_insert_chunks, migrate};
use rusqlite::{Connection, params};
use std::path::PathBuf;

/// Number of chunks inserted into the test database.
const NUM_CHUNKS: usize = 1000;

/// Maximum number of FTS5 results returned per query.
const FTS_LIMIT: i64 = 20;

/// Corpus of sentence fragments used to generate synthetic chunk content.
/// Each chunk is assembled from multiple fragments to create varied text
/// that exercises the FTS5 tokenizer and BM25 scoring across different
/// term distributions.
const FRAGMENTS: &[&str] = &[
    "statistical analysis of regression models",
    "the central limit theorem applies to large samples",
    "hypothesis testing requires significance levels",
    "Bayesian inference combines prior and likelihood",
    "machine learning algorithms classify data points",
    "neural networks approximate nonlinear functions",
    "gradient descent minimizes the loss function",
    "cross validation estimates generalization error",
    "random forests aggregate decision tree predictions",
    "principal component analysis reduces dimensionality",
    "variance estimation in experimental design",
    "confidence intervals quantify parameter uncertainty",
    "time series forecasting with autoregressive models",
    "probability distributions describe random variables",
    "maximum likelihood estimation fits parametric models",
    "kernel density estimation for nonparametric analysis",
    "multivariate regression with correlated predictors",
    "sampling distributions and the law of large numbers",
    "effect size measurement in clinical trials",
    "information criterion for model selection and comparison",
];

/// Creates an in-memory SQLite database with the NeuronCite schema, a test
/// session, a test file, and NUM_CHUNKS chunks of synthetic text. Returns
/// the open database connection ready for FTS5 queries.
fn setup_fts_database() -> Connection {
    let conn = Connection::open_in_memory().expect("failed to open in-memory db");
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .expect("failed to enable foreign keys");
    migrate(&conn).expect("schema migration failed");

    let config = IndexConfig {
        directory: PathBuf::from("/bench/docs"),
        model_name: "bench-model".into(),
        chunk_strategy: "word".into(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 384,
    };

    let session_id = neuroncite_store::repo::session::create_session(&conn, &config, "0.1.0")
        .expect("session creation failed");

    let file_id = neuroncite_store::repo::file::insert_file(
        &conn,
        session_id,
        "/bench/docs/corpus.pdf",
        "bench_hash_0000",
        1_700_000_000,
        1_048_576,
        100,
        None,
    )
    .expect("file insert failed");

    // Generate chunk content by cycling through fragments. Each chunk gets
    // 3 concatenated fragments, creating overlapping vocabulary across chunks
    // that exercises BM25's term frequency and inverse document frequency
    // weighting.
    let contents: Vec<String> = (0..NUM_CHUNKS)
        .map(|i| {
            let f0 = FRAGMENTS[i % FRAGMENTS.len()];
            let f1 = FRAGMENTS[(i + 7) % FRAGMENTS.len()];
            let f2 = FRAGMENTS[(i + 13) % FRAGMENTS.len()];
            format!("{f0}. {f1}. {f2}.")
        })
        .collect();

    let hash_strings: Vec<String> = (0..NUM_CHUNKS).map(|i| format!("hash_{i:04}")).collect();

    let inserts: Vec<ChunkInsert<'_>> = contents
        .iter()
        .enumerate()
        .map(|(i, content)| ChunkInsert {
            file_id,
            session_id,
            page_start: (i / 10 + 1) as i64,
            page_end: (i / 10 + 1) as i64,
            chunk_index: i as i64,
            doc_text_offset_start: (i * 200) as i64,
            doc_text_offset_end: ((i + 1) * 200) as i64,
            content: content.as_str(),
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: hash_strings[i].as_str(),
            simhash: None,
        })
        .collect();

    bulk_insert_chunks(&conn, &inserts).expect("bulk chunk insert failed");

    conn
}

/// Executes an FTS5 BM25 query and returns the number of result rows.
/// The SQL mirrors the exact query pattern used by neuroncite_search::keyword.
fn run_fts5_query(conn: &Connection, query: &str) -> usize {
    let mut stmt = conn
        .prepare(
            "SELECT rowid, bm25(chunk_fts) AS score
             FROM chunk_fts
             WHERE chunk_fts MATCH ?1
             ORDER BY score ASC
             LIMIT ?2",
        )
        .expect("FTS5 query preparation failed");

    let rows = stmt
        .query_map(params![query, FTS_LIMIT], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })
        .expect("FTS5 query execution failed");

    let mut count = 0;
    for row_result in rows {
        let _hit = row_result.expect("FTS5 row read failed");
        count += 1;
    }
    count
}

/// Measures the latency of a single-term FTS5 BM25 query against 1000 chunks.
fn bench_bm25_single_term(c: &mut Criterion) {
    let conn = setup_fts_database();

    c.bench_function("bm25_single_term_1000chunks", |b| {
        b.iter(|| {
            let count = run_fts5_query(&conn, "regression");
            assert!(count > 0);
        });
    });
}

/// Measures the latency of a multi-term FTS5 BM25 query that exercises
/// multi-token scoring across the inverted index.
fn bench_bm25_multi_term(c: &mut Criterion) {
    let conn = setup_fts_database();

    c.bench_function("bm25_multi_term_1000chunks", |b| {
        b.iter(|| {
            let count = run_fts5_query(&conn, "machine learning algorithms");
            assert!(count > 0);
        });
    });
}

/// Measures the latency of an FTS5 query that matches very few chunks,
/// exercising the fast-path where few posting list entries are traversed.
fn bench_bm25_rare_term(c: &mut Criterion) {
    let conn = setup_fts_database();

    c.bench_function("bm25_rare_term_1000chunks", |b| {
        b.iter(|| {
            // "autoregressive" appears in only one fragment, so it matches
            // a small subset of the 1000 chunks.
            let count = run_fts5_query(&conn, "autoregressive");
            assert!(count > 0);
        });
    });
}

criterion_group!(
    benches,
    bench_bm25_single_term,
    bench_bm25_multi_term,
    bench_bm25_rare_term
);
criterion_main!(benches);
