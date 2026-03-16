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

//! Near-duplicate chunk removal from fused search results.
//!
//! After fusion, overlapping chunks from the same or adjacent pages may appear
//! in the result list. This module removes exact duplicates by content hash
//! and near-duplicates by comparing 64-bit SimHash fingerprints using Hamming
//! distance. Chunks whose SimHash Hamming distance is within a configurable
//! threshold (default: 3 bits out of 64) are treated as near-duplicates.
//! Among duplicates, only the highest-scoring instance is retained.

use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::SearchError;
use crate::fusion::FusedCandidate;

/// Metadata required for deduplication. Loaded from the chunk table for each
/// fused candidate.
#[derive(Debug, Clone)]
struct DedupMeta {
    /// SHA-256 content hash of the chunk text.
    content_hash: String,
    /// 64-bit SimHash fingerprint of the chunk text. Read from the database
    /// simhash column when available, computed from content as fallback for
    /// chunks that were indexed before the simhash column was added.
    simhash: u64,
}

/// FNV-1a 64-bit hash prime: 1099511628211.
///
/// The 64-bit FNV-1a hash function uses this multiplier after each XOR step.
/// The named constant plus compile-time assertion below ensures the value
/// is not silently corrupted by future edits.
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit offset basis: 14695981039346656037.
///
/// The hash state is initialized to this value before processing any bytes.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;

// Compile-time verification: if either constant is accidentally changed to an
// incorrect value, this assertion prevents a silent miscompilation. The assert
// fires at compile time (not at runtime) via const evaluation.
const _: () = assert!(
    FNV_PRIME_64 == 1_099_511_628_211,
    "FNV_PRIME_64 is incorrect — must be 1099511628211 per the FNV-1a 64-bit specification"
);

/// Sentinel fingerprint returned for empty-text input.
///
/// When the input contains no whitespace-delimited tokens, all 64 bit
/// counters remain at zero and the standard algorithm would return 0.
/// A fingerprint of 0 is problematic because it is indistinguishable from
/// any other document whose token votes happen to cancel out completely. Two
/// empty-content chunks would be seen as SimHash-identical to each other and
/// to any other chunk that coincidentally hashes to 0.
///
/// Returning a fixed non-zero sentinel for empty text makes the empty case
/// explicit and prevents such false near-duplicate collisions. The value
/// 0x5555_5555_5555_5555 (alternating bits) was chosen to be far from the
/// typical dense-text range and visually distinctive in debug output.
const SIMHASH_EMPTY_SENTINEL: u64 = 0x5555_5555_5555_5555;

/// Computes a 64-bit SimHash fingerprint for the given text. The algorithm
/// works by:
/// 1. Splitting the text into whitespace-delimited tokens.
/// 2. Hashing each token to a 64-bit value using FNV-1a.
/// 3. For each bit position, incrementing a counter if the bit is 1,
///    decrementing if 0.
/// 4. Setting each bit in the final fingerprint to 1 if its counter is positive.
///
/// This produces a fingerprint where similar texts have similar bit patterns
/// (low Hamming distance). The function is public so the indexer can
/// precompute SimHash values at chunk insertion time and store them in the
/// simhash column of the chunk table.
///
/// Empty text returns `SIMHASH_EMPTY_SENTINEL` rather than 0 to avoid
/// false near-duplicate collisions between empty-content chunks.
pub fn compute_simhash(text: &str) -> u64 {
    let mut counters = [0_i32; 64];
    let mut token_count: usize = 0;

    for token in text.split_whitespace() {
        token_count += 1;

        // FNV-1a hash for each token, producing a 64-bit fingerprint.
        let mut hash: u64 = FNV_OFFSET_BASIS_64;
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME_64);
        }

        for (bit, counter) in counters.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *counter += 1;
            } else {
                *counter -= 1;
            }
        }
    }

    // Return the empty-text sentinel rather than 0 for inputs with no tokens.
    // See SIMHASH_EMPTY_SENTINEL for the rationale.
    if token_count == 0 {
        return SIMHASH_EMPTY_SENTINEL;
    }

    let mut fingerprint: u64 = 0;
    for (bit, &count) in counters.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1_u64 << bit;
        }
    }
    fingerprint
}

/// Returns the Hamming distance between two 64-bit values (the number of
/// bit positions where the two values differ).
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Batch-loads content_hash and SimHash fingerprint for all chunk IDs in a
/// single `WHERE id IN (...)` query. When the simhash column contains a
/// stored value, it is used directly. When simhash is NULL (chunks indexed
/// before the column was added), the content is loaded as a fallback and
/// SimHash is computed on the fly. The CASE expression avoids transferring
/// full chunk content strings to Rust when a stored simhash is available.
fn batch_load_dedup_meta(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<HashMap<i64, DedupMeta>, SearchError> {
    if chunk_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let placeholders: String = std::iter::repeat_n("?", chunk_ids.len())
        .collect::<Vec<_>>()
        .join(", ");

    // When simhash is stored (NOT NULL), return it directly and substitute an
    // empty string for content to avoid transferring the full text. When
    // simhash is NULL, return the content so we can compute SimHash in Rust.
    let sql = format!(
        "SELECT id, content_hash, simhash, \
                CASE WHEN simhash IS NULL THEN content ELSE '' END \
         FROM chunk WHERE id IN ({placeholders}) AND is_deleted = 0"
    );

    let mut stmt = conn.prepare(&sql).map_err(|e| SearchError::ChunkLookup {
        reason: format!("batch dedup metadata preparation failed: {e}"),
    })?;

    let params: Vec<&dyn rusqlite::types::ToSql> = chunk_ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();

    let rows = stmt
        .query_map(params.as_slice(), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| SearchError::ChunkLookup {
            reason: format!("batch dedup metadata query failed: {e}"),
        })?;

    let mut metas = HashMap::with_capacity(chunk_ids.len());
    for row in rows {
        let (id, content_hash, stored_simhash, fallback_content) =
            row.map_err(|e| SearchError::ChunkLookup {
                reason: format!("batch dedup metadata row read failed: {e}"),
            })?;

        // Use the stored simhash when available (reinterpret i64 as u64).
        // Fall back to computing from content for chunks without a stored value.
        let simhash = match stored_simhash {
            Some(sh) => sh as u64,
            None => compute_simhash(&fallback_content),
        };

        metas.insert(
            id,
            DedupMeta {
                content_hash,
                simhash,
            },
        );
    }

    Ok(metas)
}

/// Removes exact duplicates (by content_hash) and near-duplicates (by SimHash
/// Hamming distance within threshold) from the fused candidate list.
///
/// Among duplicates, only the candidate with the highest RRF score is retained.
/// The returned list preserves the original descending-score ordering of the
/// remaining candidates.
///
/// # Arguments
///
/// * `candidates` - Fused candidates sorted by descending RRF score.
/// * `conn` - Database connection for loading chunk content hashes.
/// * `simhash_threshold` - Maximum Hamming distance (out of 64 bits) to
///   consider two chunks as near-duplicates. Default: 3.
///
/// # Errors
///
/// Returns `SearchError::ChunkLookup` if a chunk's metadata cannot be loaded.
pub fn deduplicate(
    candidates: &[FusedCandidate],
    conn: &Connection,
    simhash_threshold: u32,
) -> Result<Vec<FusedCandidate>, SearchError> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Batch-load content_hash and content text for all candidates in a single
    // query, instead of issuing one query per candidate (N+1 problem).
    let chunk_ids: Vec<i64> = candidates.iter().map(|c| c.chunk_id).collect();
    let metas = batch_load_dedup_meta(conn, &chunk_ids)?;

    // Track which content_hashes have already been seen (exact dedup).
    let mut seen_hashes: HashMap<String, usize> = HashMap::new();
    // Track SimHash fingerprints of accepted candidates (near-dedup).
    let mut accepted_simhashes: Vec<u64> = Vec::new();
    let mut result: Vec<FusedCandidate> = Vec::with_capacity(candidates.len());

    // Candidates are already sorted by descending RRF score, so the first
    // occurrence of each duplicate group is the highest-scoring one.
    for candidate in candidates {
        // Use .get() to avoid a panic on missing chunk IDs. A missing entry
        // indicates a concurrent re-index or a bug in the caller's metadata
        // assembly; skip rather than crash.
        let meta = match metas.get(&candidate.chunk_id) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    chunk_id = candidate.chunk_id,
                    "chunk absent from dedup metadata — possible concurrent re-index, skipping"
                );
                continue;
            }
        };

        // Check exact duplicate: same content_hash already accepted.
        if seen_hashes.contains_key(&meta.content_hash) {
            tracing::debug!(
                chunk_id = candidate.chunk_id,
                "skipping exact duplicate by content_hash"
            );
            continue;
        }

        // Check near-duplicate: SimHash Hamming distance within threshold
        // to any already-accepted candidate.
        let is_near_dup = accepted_simhashes
            .iter()
            .any(|&accepted| hamming_distance(meta.simhash, accepted) <= simhash_threshold);

        if is_near_dup {
            tracing::debug!(
                chunk_id = candidate.chunk_id,
                "skipping near-duplicate by SimHash"
            );
            continue;
        }

        // Accept this candidate.
        seen_hashes.insert(meta.content_hash.clone(), result.len());
        accepted_simhashes.push(meta.simhash);
        result.push(candidate.clone());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::FusedCandidate;
    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{ChunkInsert, bulk_insert_chunks};
    use rusqlite::Connection;
    use std::path::PathBuf;

    /// Sets up an in-memory database with the given chunk contents.
    fn setup_dedup_db(contents: &[&str], hashes: &[&str]) -> (Connection, Vec<i64>) {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        neuroncite_store::migrate(&conn).expect("migration");

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

        let chunk_inserts: Vec<ChunkInsert<'_>> = contents
            .iter()
            .zip(hashes.iter())
            .enumerate()
            .map(|(i, (content, hash))| {
                let simhash = compute_simhash(content);
                ChunkInsert {
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
                    content_hash: hash,
                    simhash: Some(simhash as i64),
                }
            })
            .collect();

        let ids = bulk_insert_chunks(&conn, &chunk_inserts).expect("bulk insert");
        (conn, ids)
    }

    /// T-SCH-008: Dedup by content_hash. Identical hashes keep only the
    /// highest-scoring candidate.
    #[test]
    fn t_sch_008_dedup_by_content_hash() {
        // Two chunks with identical content_hash but different IDs
        let contents = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
            "a completely different text about statistics",
        ];
        let hashes = ["same_hash_abc", "same_hash_abc", "different_hash_xyz"];

        let (conn, ids) = setup_dedup_db(&contents, &hashes);

        let candidates = vec![
            FusedCandidate {
                chunk_id: ids[0],
                rrf_score: 0.8,
                vector_score: 0.9,
                bm25_rank: Some(1),
            },
            FusedCandidate {
                chunk_id: ids[1],
                rrf_score: 0.6,
                vector_score: 0.7,
                bm25_rank: Some(2),
            },
            FusedCandidate {
                chunk_id: ids[2],
                rrf_score: 0.4,
                vector_score: 0.5,
                bm25_rank: Some(3),
            },
        ];

        let deduped = deduplicate(&candidates, &conn, 3).expect("dedup");

        // Only 2 unique results (ids[0] kept, ids[1] dropped as exact dup)
        assert_eq!(deduped.len(), 2, "exact duplicate must be removed");

        let result_ids: Vec<i64> = deduped.iter().map(|c| c.chunk_id).collect();
        assert!(
            result_ids.contains(&ids[0]),
            "highest-scoring duplicate must be retained"
        );
        assert!(
            !result_ids.contains(&ids[1]),
            "lower-scoring duplicate must be removed"
        );
        assert!(
            result_ids.contains(&ids[2]),
            "non-duplicate must be retained"
        );
    }

    /// T-SCH-009: SimHash near-duplicate detection. Hamming distance <= 3
    /// is treated as a near-duplicate.
    ///
    /// The test constructs two texts that differ by only one short word out of
    /// many shared tokens, ensuring the SimHash fingerprints differ by at most
    /// 3 bits. A third text shares no meaningful token overlap, producing a
    /// Hamming distance well above the threshold.
    #[test]
    fn t_sch_009_simhash_near_duplicate_detection() {
        // The first two texts share all tokens except one short word at the end.
        // With 15+ shared tokens and 1 differing token, the SimHash difference
        // is small (typically 0-3 bits). The third text is entirely different.
        let base = "the role of statistical methods in modern research \
                    design is critical for data analysis and interpretation \
                    of experimental results in science and engineering fields";
        // Near-duplicate: replace the last word "fields" with "areas"
        let near_dup = "the role of statistical methods in modern research \
                       design is critical for data analysis and interpretation \
                       of experimental results in science and engineering areas";
        // Completely different text
        let different = "quantum chromodynamics describes interactions between \
                        quarks and gluons mediated by the strong nuclear force \
                        using gauge theory and renormalization group equations";

        let contents = [base, near_dup, different];
        let hashes = ["hash_a", "hash_b", "hash_c"];

        // Verify the SimHash distances before running the full test.
        let sh1 = compute_simhash(base);
        let sh2 = compute_simhash(near_dup);
        let sh3 = compute_simhash(different);
        let dist_12 = hamming_distance(sh1, sh2);
        let dist_13 = hamming_distance(sh1, sh3);

        assert!(
            dist_12 <= 3,
            "near-duplicate texts must have SimHash Hamming distance <= 3, found {dist_12}"
        );
        assert!(
            dist_13 > 3,
            "dissimilar texts must have SimHash Hamming distance > 3, found {dist_13}"
        );

        let (conn, ids) = setup_dedup_db(&contents, &hashes);

        let candidates = vec![
            FusedCandidate {
                chunk_id: ids[0],
                rrf_score: 0.9,
                vector_score: 0.95,
                bm25_rank: Some(1),
            },
            FusedCandidate {
                chunk_id: ids[1],
                rrf_score: 0.7,
                vector_score: 0.8,
                bm25_rank: Some(2),
            },
            FusedCandidate {
                chunk_id: ids[2],
                rrf_score: 0.5,
                vector_score: 0.6,
                bm25_rank: Some(3),
            },
        ];

        let deduped = deduplicate(&candidates, &conn, 3).expect("dedup");

        // Near-duplicate (ids[1]) must be removed; ids[0] and ids[2] remain.
        assert_eq!(deduped.len(), 2, "near-duplicate must be removed");

        let result_ids: Vec<i64> = deduped.iter().map(|c| c.chunk_id).collect();
        assert!(
            result_ids.contains(&ids[0]),
            "highest-scoring near-dup must be retained"
        );
        assert!(
            !result_ids.contains(&ids[1]),
            "lower-scoring near-dup must be removed"
        );
        assert!(
            result_ids.contains(&ids[2]),
            "dissimilar chunk must be retained"
        );
    }

    /// Verifies that the Hamming distance function counts differing bits.
    #[test]
    fn hamming_distance_counts_bits() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    /// Verifies that identical texts produce the same SimHash fingerprint.
    #[test]
    fn simhash_identical_texts() {
        let text = "the quick brown fox jumps over the lazy dog";
        assert_eq!(compute_simhash(text), compute_simhash(text));
    }
}
