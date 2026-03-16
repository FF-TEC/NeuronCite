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

// End-to-end search pipeline integration tests.
//
// These tests exercise the full search pipeline from chunk insertion through
// HNSW index construction, vector search, FTS5 keyword search, and Reciprocal
// Rank Fusion, verifying that the correct chunks are returned with correct
// metadata. Each test creates an in-memory SQLite database with schema
// migrations, inserts chunks with deterministic embedding vectors, builds an
// HNSW index, and runs the SearchPipeline to verify results.
//
// The tests cover:
// - Vector-only search: correct chunks returned based on embedding similarity.
// - Hybrid search: vector + keyword fusion returns chunks with bm25_rank populated.
// - Hyphenated query regression: queries containing hyphens (previously caused
//   FTS5 "no such column" errors) complete without error and return results.
// - Citation assembly: returned results carry correct source file, page range,
//   and content fields from the database.
// - Result ordering: results are sorted by descending score.
// - Reranker integration: when a reranker is provided, its scores replace the
//   RRF fusion scores.

use std::path::PathBuf;

use neuroncite_core::{IndexConfig, NeuronCiteError, Reranker, StorageMode};
use neuroncite_search::{SearchConfig, SearchPipeline};
use neuroncite_store::{self as store, ChunkInsert, HnswIndex, build_hnsw, bulk_insert_chunks};
use rusqlite::Connection;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Generates a normalized vector from a seed value for reproducible tests.
/// Uses a simple linear congruential generator to produce pseudo-random
/// components, then L2-normalizes the result.
fn make_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
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

/// Deterministic mock reranker that assigns scores based on whether the
/// candidate text contains specified keywords. Used to verify that the
/// reranking stage integrates correctly with the pipeline and that reranker
/// scores replace RRF scores.
struct KeywordReranker {
    /// Keyword to look for in candidate texts. Candidates containing this
    /// keyword receive a score of 0.95; others receive 0.10.
    keyword: String,
}

impl Reranker for KeywordReranker {
    fn name(&self) -> &str {
        "keyword-reranker"
    }

    fn rerank_batch(&self, _query: &str, candidates: &[&str]) -> Result<Vec<f64>, NeuronCiteError> {
        Ok(candidates
            .iter()
            .map(|text| {
                if text.to_lowercase().contains(&self.keyword) {
                    0.95
                } else {
                    0.10
                }
            })
            .collect())
    }

    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }
}

/// Contents for the test chunks. Each entry is a distinct piece of text
/// that covers a different topic, enabling targeted keyword searches and
/// verifiable vector similarity patterns.
const CHUNK_CONTENTS: [&str; 6] = [
    "statistical hypothesis testing and p-values in experimental research design",
    "machine learning algorithms for classification and regression tasks in data science",
    "bayesian statistical inference using prior and posterior distributions for estimation",
    "short-term volatility estimation using range-based measures and realized variance",
    "the Box-Cox transformation family for normalizing non-Gaussian distributions",
    "autoregressive conditional heteroskedasticity models for financial time series",
];

/// Creates an in-memory SQLite database with schema, a session, a file with
/// 3 pages, 6 chunks with embeddings, and a built HNSW index. Returns all
/// components needed for search pipeline testing, including the session_id
/// required by SearchConfig.
fn setup_search_env(dim: usize) -> (Connection, HnswIndex, Vec<Vec<f32>>, Vec<i64>, i64) {
    let conn = Connection::open_in_memory().expect("open in-memory db");
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .expect("enable foreign keys");
    store::migrate(&conn).expect("migration");

    let config = IndexConfig {
        directory: PathBuf::from("/test/search"),
        model_name: "test-model".into(),
        chunk_strategy: "word".into(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: dim,
    };

    let session_id = store::create_session(&conn, &config, "0.1.0").expect("create session");

    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/search/statistics_and_finance.pdf",
        "hash_search_test",
        1_700_000_000,
        8192,
        3,
        None,
    )
    .expect("insert file");

    // Insert page records so citation assembly can resolve page content.
    let pages_data = vec![
        (
            1_i64,
            "Page 1: hypothesis testing and machine learning overview",
            "pdf-extract",
        ),
        (
            2,
            "Page 2: bayesian inference and volatility estimation",
            "pdf-extract",
        ),
        (
            3,
            "Page 3: Box-Cox transformations and ARCH models",
            "pdf-extract",
        ),
    ];
    store::bulk_insert_pages(&conn, file_id, &pages_data).expect("page insert failed");

    // Generate deterministic embeddings. Each chunk receives a vector seeded
    // from its index, producing distinct but reproducible embedding vectors.
    let embeddings: Vec<Vec<f32>> = (1..=CHUNK_CONTENTS.len() as u64)
        .map(|seed| make_vector(seed, dim))
        .collect();

    // Pre-compute hash strings so they outlive the ChunkInsert references.
    let hash_strings: Vec<String> = (0..CHUNK_CONTENTS.len())
        .map(|i| format!("search_hash_{i}"))
        .collect();

    let chunk_inserts: Vec<ChunkInsert<'_>> = CHUNK_CONTENTS
        .iter()
        .enumerate()
        .map(|(i, content)| {
            // Distribute chunks across 3 pages: chunks 0-1 on page 1, 2-3 on page 2, 4-5 on page 3.
            let page = (i / 2) as i64 + 1;
            ChunkInsert {
                file_id,
                session_id,
                page_start: page,
                page_end: page,
                chunk_index: i as i64,
                doc_text_offset_start: (i * 100) as i64,
                doc_text_offset_end: ((i + 1) * 100) as i64,
                content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &hash_strings[i],
                simhash: None,
            }
        })
        .collect();

    let ids = bulk_insert_chunks(&conn, &chunk_inserts).expect("bulk insert");

    // Build the HNSW index from the chunk embeddings.
    let labeled: Vec<(i64, &[f32])> = ids
        .iter()
        .zip(embeddings.iter())
        .map(|(id, emb)| (*id, emb.as_slice()))
        .collect();
    let index = build_hnsw(&labeled, dim).expect("build_hnsw");

    (conn, index, embeddings, ids, session_id)
}

// ---------------------------------------------------------------------------
// T-E2E-001: Vector-only search returns the nearest chunk
// ---------------------------------------------------------------------------

/// T-E2E-001: Queries the HNSW index with the embedding vector of the first
/// chunk (seed 1) and verifies that the first chunk is returned as the top
/// result. Keyword search is still active but has minimal impact because the
/// query text uses a generic term.
#[test]
fn t_e2e_001_vector_search_returns_nearest_chunk() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    // Use the first chunk's embedding as the query vector. The nearest
    // neighbor in HNSW must be the first chunk itself (distance = 0).
    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(
        &index,
        &conn,
        &embeddings[0],
        "hypothesis testing",
        None,
        config,
    );

    let results = pipeline
        .search()
        .expect("search pipeline execution")
        .results;

    assert!(
        !results.is_empty(),
        "search must return at least one result"
    );

    // The first chunk (chunk_id = ids[0]) must be present in the results.
    let result_ids: Vec<usize> = results.iter().map(|r| r.chunk_index).collect();
    assert!(
        result_ids.contains(&0),
        "chunk 0 ('hypothesis testing') must appear in results, got indices: {result_ids:?}"
    );

    // The top result must have a positive score.
    assert!(
        results[0].score > 0.0,
        "top result score must be positive, got {}",
        results[0].score,
    );
}

// ---------------------------------------------------------------------------
// T-E2E-002: Hybrid search returns chunks with bm25_rank populated
// ---------------------------------------------------------------------------

/// T-E2E-002: Queries with a term that appears in exactly one chunk
/// ("bayesian") and verifies that hybrid search returns that chunk with a
/// populated bm25_rank field, confirming that FTS5 keyword search contributed
/// to the fusion ranking.
#[test]
fn t_e2e_002_hybrid_search_populates_bm25_rank() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    // "bayesian" appears only in chunk index 2.
    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[2], "bayesian", None, config);

    let results = pipeline.search().expect("hybrid search execution").results;

    assert!(
        !results.is_empty(),
        "hybrid search must return at least one result"
    );

    // Find the chunk containing "bayesian" (chunk index 2, database ID ids[2]).
    let bayesian_result = results.iter().find(|r| r.chunk_index == 2);

    assert!(
        bayesian_result.is_some(),
        "chunk 2 ('bayesian statistical inference') must appear in hybrid results"
    );

    let bayesian = bayesian_result.expect("bayesian result exists");

    // The bm25_rank must be set because "bayesian" matches in FTS5.
    assert!(
        bayesian.bm25_rank.is_some(),
        "bm25_rank must be populated for a chunk matching the keyword query, got None"
    );

    // The bm25_rank must be 1 because "bayesian" appears in only one chunk.
    assert_eq!(
        bayesian.bm25_rank,
        Some(1),
        "bayesian chunk must have bm25_rank = 1 (sole keyword match)"
    );
}

// ---------------------------------------------------------------------------
// T-E2E-003: Hyphenated query does not cause FTS5 error (regression test)
// ---------------------------------------------------------------------------

/// T-E2E-003: Regression test for the FTS5 query sanitization fix. Queries
/// containing hyphens (e.g., "short-term") were previously interpreted as
/// the FTS5 NOT operator, causing "no such column: term" errors. This test
/// verifies that the sanitization layer wraps hyphenated tokens in double
/// quotes, allowing FTS5 to treat them as literal terms.
#[test]
fn t_e2e_003_hyphenated_query_does_not_error() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    // Chunk index 3 contains "short-term volatility estimation".
    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(
        &index,
        &conn,
        &embeddings[3],
        "short-term volatility",
        None,
        config,
    );

    // The critical assertion: this must not return an error. Before the
    // sanitization fix, this call would fail with SearchError::KeywordSearch.
    let results = pipeline
        .search()
        .expect("hyphenated query 'short-term volatility' must not cause FTS5 error")
        .results;

    assert!(
        !results.is_empty(),
        "hyphenated query must return at least one result"
    );

    // The chunk containing "short-term volatility" (index 3) must be present.
    let volatility_result = results.iter().find(|r| r.chunk_index == 3);

    assert!(
        volatility_result.is_some(),
        "chunk 3 ('short-term volatility estimation') must appear in results"
    );

    // Verify keyword search contributed (bm25_rank is populated).
    let vol = volatility_result.expect("volatility result exists");
    assert!(
        vol.bm25_rank.is_some(),
        "bm25_rank must be populated for 'short-term volatility' match"
    );
}

// ---------------------------------------------------------------------------
// T-E2E-004: Citation fields are correctly assembled
// ---------------------------------------------------------------------------

/// T-E2E-004: Verifies that the citation assembly stage populates the source
/// file path, file display name, page range, and formatted citation string
/// on each returned result. These fields are resolved from the chunk -> file
/// join in the database.
#[test]
fn t_e2e_004_citation_fields_correctly_assembled() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[0], "statistical", None, config);

    let results = pipeline.search().expect("search pipeline").results;

    assert!(
        !results.is_empty(),
        "search must return results for citation verification"
    );

    for result in &results {
        // Every result must have a source file path.
        assert!(
            !result.citation.source_file.as_os_str().is_empty(),
            "citation source_file must not be empty for chunk index {}",
            result.chunk_index,
        );

        // The source file path must contain the filename from the test setup.
        let source_str = result.citation.source_file.to_string_lossy();
        assert!(
            source_str.contains("statistics_and_finance.pdf"),
            "source_file must reference the test PDF, got: {source_str}"
        );

        // The file display name must be the filename without the directory path.
        assert!(
            !result.citation.file_display_name.is_empty(),
            "file_display_name must not be empty for chunk index {}",
            result.chunk_index,
        );

        // Page numbers must be valid positive integers within the document range.
        assert!(
            result.citation.page_start >= 1 && result.citation.page_start <= 3,
            "page_start must be between 1 and 3, got {}",
            result.citation.page_start,
        );
        assert!(
            result.citation.page_end >= result.citation.page_start,
            "page_end ({}) must be >= page_start ({})",
            result.citation.page_end,
            result.citation.page_start,
        );

        // The formatted citation string must not be empty.
        assert!(
            !result.citation.formatted.is_empty(),
            "formatted citation must not be empty for chunk index {}",
            result.chunk_index,
        );

        // The content field must match one of the known chunk contents.
        assert!(
            CHUNK_CONTENTS
                .iter()
                .any(|c| result.content.contains(c) || c.contains(&result.content)),
            "result content must match a known chunk, got: '{}'",
            &result.content[..result.content.len().min(60)],
        );
    }
}

// ---------------------------------------------------------------------------
// T-E2E-005: Results are sorted by descending score
// ---------------------------------------------------------------------------

/// T-E2E-005: Verifies that the search pipeline returns results sorted in
/// descending order by their final score. The first result has the highest
/// score and subsequent results have equal or lower scores.
#[test]
fn t_e2e_005_results_sorted_by_descending_score() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[0], "statistical", None, config);

    let results = pipeline.search().expect("search pipeline").results;

    assert!(
        results.len() >= 2,
        "need at least 2 results to verify sort order, got {}",
        results.len(),
    );

    for window in results.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "results must be in descending score order: {} >= {} violated (chunk {} vs chunk {})",
            window[0].score,
            window[1].score,
            window[0].chunk_index,
            window[1].chunk_index,
        );
    }
}

// ---------------------------------------------------------------------------
// T-E2E-006: Box-Cox query with special characters succeeds
// ---------------------------------------------------------------------------

/// T-E2E-006: Verifies that a query containing embedded quotes ("Box-Cox")
/// is handled correctly by the FTS5 sanitization layer. The quotes are
/// stripped and the hyphenated token is wrapped in double quotes for FTS5.
#[test]
fn t_e2e_006_special_char_query_succeeds() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    // Query with a term that contains both a hyphen and is typically quoted
    // in academic writing: "Box-Cox". Chunk index 4 contains this term.
    let pipeline = SearchPipeline::new(
        &index,
        &conn,
        &embeddings[4],
        "Box-Cox transformation",
        None,
        config,
    );

    let results = pipeline
        .search()
        .expect("query with 'Box-Cox' must not cause FTS5 error")
        .results;

    assert!(!results.is_empty(), "Box-Cox query must return results");

    // The chunk containing "Box-Cox" (index 4) must be among the results.
    let box_cox_result = results.iter().find(|r| r.chunk_index == 4);

    assert!(
        box_cox_result.is_some(),
        "chunk 4 ('Box-Cox transformation') must appear in results"
    );
}

// ---------------------------------------------------------------------------
// T-E2E-007: bm25_must_match filters vector-only results
// ---------------------------------------------------------------------------

/// T-E2E-007: When bm25_must_match is true, only candidates that appear in
/// both the vector and keyword result sets survive fusion. A query for
/// "heteroskedasticity" (appearing only in chunk 5) with the embedding of
/// chunk 5 should return chunk 5 (present in both sets) and exclude chunks
/// that only appear in the vector results.
#[test]
fn t_e2e_007_bm25_must_match_filters_vector_only() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: true,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    // "heteroskedasticity" appears only in chunk 5.
    let pipeline = SearchPipeline::new(
        &index,
        &conn,
        &embeddings[5],
        "heteroskedasticity",
        None,
        config,
    );

    let results = pipeline.search().expect("bm25_must_match search").results;

    // With bm25_must_match = true, only chunks matching the keyword should
    // survive. Chunk 5 contains "heteroskedasticity" and is the nearest vector
    // neighbor, so it must be the sole result.
    assert!(
        !results.is_empty(),
        "bm25_must_match search must return at least one result"
    );

    // Every returned result must have a bm25_rank (keyword match).
    for result in &results {
        assert!(
            result.bm25_rank.is_some(),
            "with bm25_must_match=true, all results must have bm25_rank, \
             but chunk {} has None",
            result.chunk_index,
        );
    }

    // Chunk 5 must be present.
    assert!(
        results.iter().any(|r| r.chunk_index == 5),
        "chunk 5 ('heteroskedasticity') must appear in bm25_must_match results"
    );
}

// ---------------------------------------------------------------------------
// T-E2E-008: Reranker integration replaces RRF scores
// ---------------------------------------------------------------------------

/// T-E2E-008: Verifies that when a reranker is provided and rerank_enabled
/// is true, the final scores in the search results reflect the reranker's
/// output rather than the RRF fusion scores.
#[test]
fn t_e2e_008_reranker_replaces_rrf_scores() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let reranker = KeywordReranker {
        keyword: "bayesian".to_string(),
    };

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: true,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(
        &index,
        &conn,
        &embeddings[0],
        "statistical inference",
        Some(&reranker),
        config,
    );

    let results = pipeline.search().expect("reranked search").results;

    assert!(!results.is_empty(), "reranked search must return results");

    // All results must have reranker_score set.
    for result in &results {
        assert!(
            result.reranker_score.is_some(),
            "reranker_score must be set for chunk {}",
            result.chunk_index,
        );

        // The final score must equal the reranker score.
        let rerank_score = result.reranker_score.expect("reranker_score is Some");
        assert!(
            (result.score - rerank_score).abs() < 1e-10,
            "final score ({}) must equal reranker score ({rerank_score}) for chunk {}",
            result.score,
            result.chunk_index,
        );
    }

    // The "bayesian" chunk (index 2) must be the top result because the
    // KeywordReranker gives it 0.95 while others get 0.10.
    assert_eq!(
        results[0].chunk_index, 2,
        "chunk 2 ('bayesian') must be top result after reranking, got chunk {}",
        results[0].chunk_index,
    );

    assert!(
        (results[0].score - 0.95).abs() < 1e-10,
        "top result score must be 0.95 from KeywordReranker, got {}",
        results[0].score,
    );
}

// ---------------------------------------------------------------------------
// T-E2E-009: Empty query returns empty results
// ---------------------------------------------------------------------------

/// T-E2E-009: Verifies that an empty or whitespace-only query text does not
/// cause errors. The vector search still runs (using the provided embedding),
/// but keyword search returns no matches. The pipeline completes without
/// error.
#[test]
fn t_e2e_009_empty_query_text_returns_results() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    // Empty query text: keyword search produces no results, but vector search
    // still returns neighbors based on the embedding.
    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[0], "", None, config);

    let results = pipeline
        .search()
        .expect("empty query must not error")
        .results;

    // Vector search should still return results even with empty query text.
    assert!(
        !results.is_empty(),
        "vector search should return results even with empty query text"
    );

    // No result should have a bm25_rank because keyword search returned nothing.
    for result in &results {
        assert!(
            result.bm25_rank.is_none(),
            "bm25_rank must be None when query text is empty, got {:?} for chunk {}",
            result.bm25_rank,
            result.chunk_index,
        );
    }
}

// ---------------------------------------------------------------------------
// T-E2E-010: max_results caps the output size
// ---------------------------------------------------------------------------

/// T-E2E-010: Verifies that the max_results configuration parameter
/// correctly limits the number of results returned, even when more
/// candidates are available from the HNSW index and FTS5.
#[test]
fn t_e2e_010_max_results_caps_output() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 2,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[0], "statistical", None, config);

    let results = pipeline.search().expect("search pipeline").results;

    assert!(
        results.len() <= 2,
        "max_results=2 must cap output to at most 2 results, got {}",
        results.len(),
    );
}

// ---------------------------------------------------------------------------
// T-E2E-011: Content field matches the original chunk text
// ---------------------------------------------------------------------------

/// T-E2E-011: Verifies that the content field of each search result matches
/// the text that was originally inserted into the database. This confirms
/// that the citation assembly stage correctly retrieves the full chunk text
/// from the chunk table.
#[test]
fn t_e2e_011_content_matches_inserted_text() {
    let dim = 4;
    let (conn, index, embeddings, _ids, session_id) = setup_search_env(dim);

    let config = SearchConfig {
        session_id,
        vector_top_k: 10,
        keyword_limit: 10,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 6,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = SearchPipeline::new(&index, &conn, &embeddings[0], "statistical", None, config);

    let results = pipeline.search().expect("search pipeline").results;

    for result in &results {
        let idx = result.chunk_index;
        assert!(
            idx < CHUNK_CONTENTS.len(),
            "chunk_index {} must be within the range of known chunks",
            idx,
        );
        assert_eq!(
            result.content, CHUNK_CONTENTS[idx],
            "content for chunk {} must match the original inserted text",
            idx,
        );
    }
}
