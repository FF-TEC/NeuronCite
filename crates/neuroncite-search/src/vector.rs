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

//! Vector similarity search via the HNSW index.
//!
//! Queries the HNSW approximate nearest neighbor index with a query embedding
//! vector and returns the top-K results ranked by cosine similarity (computed
//! as dot product on L2-normalized vectors). Labels returned by HNSW that do
//! not correspond to an active chunk row in the database are filtered out as
//! orphans via a single batch `WHERE id IN (...)` query.

use std::collections::HashSet;

use rusqlite::Connection;

use neuroncite_store::HnswIndex;

use crate::error::SearchError;

/// A single result from the HNSW vector search. Contains the chunk's database
/// row ID and its cosine similarity score (higher is more similar).
#[derive(Debug, Clone)]
pub struct VectorHit {
    /// Primary key of the chunk in the `chunk` table.
    pub chunk_id: i64,
    /// Cosine similarity score (1.0 - cosine_distance). For pre-normalized
    /// vectors, this equals the dot product. Range: [-1.0, 1.0], typically
    /// [0.0, 1.0] for positive embeddings. Stored as f64 so it can be
    /// directly compared against min_score thresholds and SearchResult scores
    /// without widening casts at the call site.
    pub similarity: f64,
}

/// Batch-checks which chunk IDs from the given list are active (exist in the
/// chunk table with `is_deleted = 0`). When `file_ids` is `Some`, only chunks
/// belonging to one of the specified files are considered active. When
/// `page_range` is `Some((start, end))`, only chunks whose page range overlaps
/// [start, end] are considered active. Returns the set of active chunk IDs.
///
/// Uses a single `WHERE id IN (...)` query instead of N individual queries,
/// reducing the query count from O(N) to O(1).
fn batch_active_chunk_ids(
    conn: &Connection,
    chunk_ids: &[i64],
    file_ids: Option<&[i64]>,
    page_range: Option<(i64, i64)>,
) -> Result<HashSet<i64>, SearchError> {
    if chunk_ids.is_empty() {
        return Ok(HashSet::new());
    }

    let placeholders: String = std::iter::repeat_n("?", chunk_ids.len())
        .collect::<Vec<_>>()
        .join(", ");

    // Build the SQL query with optional file_id and page_range filter clauses.
    let mut sql = format!("SELECT id FROM chunk WHERE id IN ({placeholders}) AND is_deleted = 0");

    if let Some(fids) = file_ids
        && !fids.is_empty()
    {
        let file_placeholders: String = std::iter::repeat_n("?", fids.len())
            .collect::<Vec<_>>()
            .join(", ");
        sql.push_str(&format!(" AND file_id IN ({file_placeholders})"));
    }

    // Page range overlap condition: chunk.page_start <= range_end AND
    // chunk.page_end >= range_start. This captures all chunks that have any
    // page overlap with the requested range.
    if page_range.is_some() {
        sql.push_str(" AND page_start <= ? AND page_end >= ?");
    }

    let mut stmt = conn.prepare(&sql).map_err(|e| SearchError::VectorSearch {
        reason: format!("batch orphan check preparation failed: {e}"),
    })?;

    // Combine chunk_ids, optional file_ids, and optional page_range bounds
    // into a single parameter slice. Uses trait object references instead of
    // Box<dyn ToSql> to avoid per-parameter heap allocation, following the
    // same pattern as the dedup module.
    let mut all_params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();

    if let Some(fids) = file_ids
        && !fids.is_empty()
    {
        for fid in fids {
            all_params.push(fid as &dyn rusqlite::types::ToSql);
        }
    }

    // Page range bounds must outlive the query execution, so they are stored
    // in local variables before borrowing into the parameter slice.
    let page_end_val: i64;
    let page_start_val: i64;
    if let Some((ps, pe)) = page_range {
        page_end_val = pe;
        page_start_val = ps;
        all_params.push(&page_end_val as &dyn rusqlite::types::ToSql); // page_start <= range_end
        all_params.push(&page_start_val as &dyn rusqlite::types::ToSql); // page_end >= range_start
    }

    let rows = stmt
        .query_map(all_params.as_slice(), |row| row.get::<_, i64>(0))
        .map_err(|e| SearchError::VectorSearch {
            reason: format!("batch orphan check query failed: {e}"),
        })?;

    let mut active = HashSet::with_capacity(chunk_ids.len());
    for row in rows {
        active.insert(row.map_err(|e| SearchError::VectorSearch {
            reason: format!("batch orphan check row read failed: {e}"),
        })?);
    }

    Ok(active)
}

/// Queries the HNSW index for the `top_k` nearest neighbors of `query_vec`,
/// then filters out orphaned labels (HNSW labels whose corresponding chunk
/// row does not exist or has `is_deleted = 1` in the database).
///
/// The `hnsw_rs` distance metric is cosine distance, where distance = 1 - similarity.
/// This function converts distance back to similarity for ranking purposes.
///
/// Orphan filtering uses a single batch query with `WHERE id IN (...)` to
/// verify all candidate chunk IDs at once, instead of one query per candidate.
/// When `file_ids` is `Some`, the orphan check additionally filters by file
/// membership, effectively scoping the search to specific PDF documents.
/// When `page_range` is `Some((start, end))`, the orphan check additionally
/// filters by page overlap, restricting results to chunks spanning the
/// specified page range.
///
/// # Arguments
///
/// * `index` - The HNSW index to search.
/// * `conn` - Database connection for verifying chunk existence.
/// * `query_vec` - The query embedding vector (must be L2-normalized).
/// * `top_k` - Maximum number of results to return.
/// * `ef_search` - HNSW ef_search parameter controlling recall/latency trade-off.
/// * `file_ids` - When `Some`, only chunks belonging to these file IDs are returned.
/// * `page_range` - When `Some((start, end))`, only chunks overlapping the page range
///   [start, end] (inclusive, 1-indexed) are returned.
///
/// # Errors
///
/// Returns `SearchError::VectorSearch` if the index query fails.
pub fn vector_search(
    index: &HnswIndex,
    conn: &Connection,
    query_vec: &[f32],
    top_k: usize,
    ef_search: usize,
    file_ids: Option<&[i64]>,
    page_range: Option<(i64, i64)>,
) -> Result<Vec<VectorHit>, SearchError> {
    if index.is_empty() {
        return Ok(Vec::new());
    }

    // A zero-magnitude query vector produces undefined cosine similarity.
    // Return empty results rather than propagating nonsensical distances from
    // the HNSW library (which may return any neighbor for a zero vector).
    let norm_sq: f32 = query_vec.iter().map(|x| x * x).sum();
    if norm_sq < 1e-18 {
        tracing::warn!("query embedding is a zero vector — returning empty vector search results");
        return Ok(Vec::new());
    }

    // Request more candidates than top_k to account for potential orphan filtering.
    // Fetch 2x top_k candidates so that enough active results remain after
    // orphan filtering. The redundant `.max(top_k)` is omitted because
    // `saturating_mul(2)` always produces a value >= top_k for any positive input.
    let fetch_count = top_k.saturating_mul(2);
    let raw_results = index.search(query_vec, fetch_count, ef_search);

    // Collect all candidate IDs and convert distances to similarities.
    // hnsw_rs DistCosine returns f32 distance = 1 - cosine_similarity.
    // Convert back to f64 similarity for ranking. Widening to f64 here
    // avoids narrowing casts in the fusion and pipeline stages.
    let candidates: Vec<(i64, f64)> = raw_results
        .iter()
        .map(|&(chunk_id, distance)| {
            // hnsw_rs DistCosine returns distance = 1 - cosine_similarity.
            // Convert back to similarity for ranking (higher = more relevant).
            (chunk_id, 1.0_f64 - f64::from(distance))
        })
        .collect();

    // Batch orphan check: single query verifies all candidate chunk IDs at once.
    let candidate_ids: Vec<i64> = candidates.iter().map(|(id, _)| *id).collect();
    let active_ids = batch_active_chunk_ids(conn, &candidate_ids, file_ids, page_range)?;

    let orphan_count = candidates.len() - active_ids.len();
    if orphan_count > 0 {
        tracing::debug!(
            orphan_count,
            total_candidates = candidates.len(),
            "filtered orphaned HNSW labels via batch query"
        );
    }

    // Filter to active chunks, take top_k, and sort by descending similarity.
    let mut hits: Vec<VectorHit> = candidates
        .into_iter()
        .filter(|(chunk_id, _)| active_ids.contains(chunk_id))
        .take(top_k)
        .map(|(chunk_id, similarity)| VectorHit {
            chunk_id,
            similarity,
        })
        .collect();

    hits.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal) // f64::NaN is treated as Equal
    });

    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{build_hnsw, migrate};
    use rusqlite::Connection;
    use std::path::PathBuf;

    /// Creates an in-memory database with schema applied, a session, a file,
    /// and the specified chunks inserted. Returns the connection and chunk IDs.
    fn setup_db_with_chunks(contents: &[&str], embeddings: &[Vec<f32>]) -> (Connection, Vec<i64>) {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        migrate(&conn).expect("migration");

        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };

        let session_id = neuroncite_store::repo::session::create_session(&conn, &config, "0.1.0")
            .expect("create session");

        let file_id = neuroncite_store::repo::file::insert_file(
            &conn,
            session_id,
            "/docs/test.pdf",
            "hash123",
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("insert file");

        // Pre-compute hash strings so they outlive the ChunkInsert references.
        let hash_strings: Vec<String> = (0..contents.len()).map(|i| format!("hash_{i}")).collect();

        let chunk_inserts: Vec<neuroncite_store::ChunkInsert<'_>> = contents
            .iter()
            .enumerate()
            .map(|(i, content)| neuroncite_store::ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: i as i64,
                doc_text_offset_start: 0,
                doc_text_offset_end: content.len() as i64,
                content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &hash_strings[i],
                simhash: None,
            })
            .collect();

        let ids = neuroncite_store::bulk_insert_chunks(&conn, &chunk_inserts).expect("bulk insert");

        // embeddings are used by the caller for HNSW construction, not stored in DB.
        let _ = embeddings;

        (conn, ids)
    }

    /// Generates a normalized vector from a seed for reproducible tests.
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

    /// T-SCH-001: Vector search returns top-K sorted by descending cosine similarity.
    #[test]
    fn t_sch_001_vector_search_returns_top_k_sorted() {
        let dim = 4;
        let contents = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let embeddings: Vec<Vec<f32>> = (1..=5).map(|i| make_vector(i, dim)).collect();

        let (conn, ids) = setup_db_with_chunks(&contents, &embeddings);

        let labeled: Vec<(i64, &[f32])> = ids
            .iter()
            .zip(embeddings.iter())
            .map(|(id, emb)| (*id, emb.as_slice()))
            .collect();
        let index = build_hnsw(&labeled, dim).expect("build_hnsw");

        let results = vector_search(&index, &conn, &embeddings[0], 3, 100, None, None)
            .expect("vector search");

        assert!(results.len() <= 3, "must return at most top_k results");
        assert!(!results.is_empty(), "must return at least one result");

        for window in results.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "results must be sorted by descending similarity: {} >= {}",
                window[0].similarity,
                window[1].similarity,
            );
        }

        assert_eq!(
            results[0].chunk_id, ids[0],
            "nearest neighbor of a vector is itself"
        );
    }

    /// T-SCH-002: Vector search filters orphaned labels (HNSW label with no
    /// chunk row is skipped) via batch query.
    #[test]
    fn t_sch_002_vector_search_filters_orphaned_labels() {
        let dim = 4;
        let contents = ["alpha", "beta", "gamma"];
        let embeddings: Vec<Vec<f32>> = (1..=3).map(|i| make_vector(i, dim)).collect();

        let (conn, ids) = setup_db_with_chunks(&contents, &embeddings);

        let labeled: Vec<(i64, &[f32])> = ids
            .iter()
            .zip(embeddings.iter())
            .map(|(id, emb)| (*id, emb.as_slice()))
            .collect();
        let index = build_hnsw(&labeled, dim).expect("build_hnsw");

        neuroncite_store::set_chunk_deleted(&conn, ids[1], true).expect("set_chunk_deleted");

        let results = vector_search(&index, &conn, &embeddings[1], 5, 100, None, None)
            .expect("vector search");

        let result_ids: Vec<i64> = results.iter().map(|h| h.chunk_id).collect();
        assert!(
            !result_ids.contains(&ids[1]),
            "orphaned/deleted chunk must be filtered from results"
        );
    }

    /// T-PERF-001: Batch orphan check identifies active vs. deleted chunk IDs
    /// in a single query round-trip.
    #[test]
    fn t_perf_001_batch_orphan_check_single_query() {
        let dim = 4;
        let contents = ["alpha", "beta", "gamma", "delta"];
        let embeddings: Vec<Vec<f32>> = (1..=4).map(|i| make_vector(i, dim)).collect();

        let (conn, ids) = setup_db_with_chunks(&contents, &embeddings);

        // Soft-delete chunks 1 and 3
        neuroncite_store::set_chunk_deleted(&conn, ids[1], true).expect("delete 1");
        neuroncite_store::set_chunk_deleted(&conn, ids[3], true).expect("delete 3");

        let active = batch_active_chunk_ids(&conn, &ids, None, None).expect("batch check");
        assert_eq!(active.len(), 2, "only 2 active chunks expected");
        assert!(active.contains(&ids[0]), "chunk 0 must be active");
        assert!(active.contains(&ids[2]), "chunk 2 must be active");
        assert!(!active.contains(&ids[1]), "chunk 1 is deleted");
        assert!(!active.contains(&ids[3]), "chunk 3 is deleted");
    }

    /// T-PERF-002: Batch orphan check with empty input returns empty set.
    #[test]
    fn t_perf_002_batch_orphan_check_empty_input() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;").expect("fk");
        migrate(&conn).expect("migration");

        let active = batch_active_chunk_ids(&conn, &[], None, None).expect("empty batch");
        assert!(active.is_empty(), "empty input must return empty set");
    }
}
