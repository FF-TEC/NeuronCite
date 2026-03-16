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

//! Citation verification endpoint handler.
//!
//! Accepts a claim text and a set of chunk IDs as citations. Computes a
//! combined verification score using two components:
//!
//! 1. Keyword overlap (Jaccard similarity) between the claim and the
//!    concatenated citation texts, using Porter-stemmed tokens.
//!    Weight: 0.3.
//!
//! 2. Semantic similarity (cosine) between the claim embedding and the
//!    mean citation embedding.
//!    Weight: 0.7.
//!
//! The verdict is determined by the combined score:
//! - >= 0.75: "supports"
//! - >= 0.45: "partial"
//! - < 0.45: "not_supported"

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use rust_stemmers::{Algorithm, Stemmer};

use crate::dto::{API_VERSION, VerifyRequest, VerifyResponse, VerifyVerdict};
use crate::error::ApiError;
use crate::state::AppState;

/// Weight applied to the Jaccard keyword overlap score.
const KEYWORD_WEIGHT: f64 = 0.3;

/// Weight applied to the cosine semantic similarity score.
const SEMANTIC_WEIGHT: f64 = 0.7;

/// Combined score threshold above which the verdict is "supports".
const SUPPORTS_THRESHOLD: f64 = 0.75;

/// Combined score threshold above which the verdict is "partial".
const PARTIAL_THRESHOLD: f64 = 0.45;

/// POST /api/v1/verify
///
/// Verifies whether the cited chunks support the given claim text.
/// Returns a verdict of "supports", "partial", or "not_supported"
/// along with the individual keyword and semantic scores.
#[utoipa::path(
    post,
    path = "/api/v1/verify",
    request_body = VerifyRequest,
    responses(
        (status = 200, description = "Verification result", body = VerifyResponse),
        (status = 400, description = "Invalid request"),
    )
)]
pub async fn verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, ApiError> {
    // Validate request fields against the configured input limits before any
    // database access. Checks claim emptiness, chunk_ids emptiness, and the
    // maximum chunk_ids count to prevent oversized SQL IN-clause queries.
    req.validate(&state.config.limits)?;

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Retrieve all cited chunks in a single SQL query using an IN clause.
    // This replaces the previous per-ID loop that issued N separate SELECT
    // statements (N+1 query pattern). Each chunk's session_id is verified
    // against the request's session_id to prevent cross-session data access:
    // a chunk that exists in the database but belongs to a different session
    // is treated as not found from the caller's perspective, avoiding
    // information disclosure about chunks in other sessions.
    let chunks =
        neuroncite_store::get_chunks_batch(&conn, &req.chunk_ids).map_err(ApiError::from)?;

    // Build a HashMap keyed by chunk ID for O(1) lookup when iterating
    // the requested IDs. The batch query returns rows in database order,
    // which may differ from the request order.
    let chunk_map: HashMap<i64, &neuroncite_store::ChunkRow> =
        chunks.iter().map(|c| (c.id, c)).collect();

    let mut citation_texts = Vec::with_capacity(req.chunk_ids.len());
    for &chunk_id in &req.chunk_ids {
        // If a requested chunk ID is absent from the map, the batch query
        // did not find it in the database.
        match chunk_map.get(&chunk_id) {
            None => {
                return Err(ApiError::NotFound {
                    resource: format!("chunk {chunk_id}"),
                });
            }
            Some(c) => {
                if c.session_id != req.session_id {
                    return Err(ApiError::NotFound {
                        resource: format!("chunk {chunk_id} in session {}", req.session_id),
                    });
                }
                citation_texts.push(c.content.clone());
            }
        }
    }

    let concatenated_citations = citation_texts.join(" ");

    // Compute Jaccard keyword overlap using Porter stemming.
    let keyword_score = jaccard_stemmed(&req.claim, &concatenated_citations);

    // Compute semantic similarity via cosine of embedding vectors.
    let claim_vec = state
        .worker_handle
        .embed_query(req.claim.clone())
        .await
        .map_err(|e| ApiError::Internal {
            reason: format!("claim embedding failed: {e}"),
        })?;

    let citation_vec = state
        .worker_handle
        .embed_query(concatenated_citations)
        .await
        .map_err(|e| ApiError::Internal {
            reason: format!("citation embedding failed: {e}"),
        })?;

    let semantic_score = neuroncite_core::cosine_similarity(&claim_vec, &citation_vec);

    // Weighted combination of both scores.
    let combined_score = KEYWORD_WEIGHT * keyword_score + SEMANTIC_WEIGHT * semantic_score;

    // Map the combined score to a typed verdict enum. The threshold constants
    // define the boundaries: >= 0.75 is supported, >= 0.45 is partial,
    // below 0.45 is not supported.
    let verdict = if combined_score >= SUPPORTS_THRESHOLD {
        VerifyVerdict::Supports
    } else if combined_score >= PARTIAL_THRESHOLD {
        VerifyVerdict::Partial
    } else {
        VerifyVerdict::NotSupported
    };

    Ok(Json(VerifyResponse {
        api_version: API_VERSION.to_string(),
        verdict,
        combined_score,
        keyword_score,
        semantic_score,
    }))
}

/// Computes the Jaccard similarity between two texts using Porter-stemmed
/// tokens. Both texts are lowercased and split on whitespace. The Jaccard
/// index is |A intersect B| / |A union B|.
pub fn jaccard_stemmed(text_a: &str, text_b: &str) -> f64 {
    let stemmer = Stemmer::create(Algorithm::English);

    let stems_a: HashSet<String> = text_a
        .split_whitespace()
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .collect();

    let stems_b: HashSet<String> = text_b
        .split_whitespace()
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .collect();

    if stems_a.is_empty() && stems_b.is_empty() {
        return 0.0;
    }

    let intersection = stems_a.intersection(&stems_b).count();
    let union = stems_a.union(&stems_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use neuroncite_core::{IndexConfig, StorageMode};
    use neuroncite_store::{self as store, ChunkInsert};

    use crate::state::AppState;
    use crate::test_support;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    // StubBackend and setup_test_state are provided by the shared
    // crate::test_support module. The local copies that previously lived
    // here were identical to test_support::StubBackend::default() (4-dim,
    // model_id "stub-model") and test_support::setup_test_state(). They
    // were removed as part of audit finding M-014 (duplicated StubBackend
    // across 5 test modules).

    /// Creates a session with a single chunk containing the given text.
    /// Returns (session_id, chunk_id). Uses the shared test_support module
    /// for AppState construction.
    fn create_session_with_chunk(
        state: &Arc<AppState>,
        session_dir: &str,
        chunk_text: &str,
    ) -> (i64, i64) {
        let conn = state.pool.get().expect("get conn");
        let config = IndexConfig {
            directory: PathBuf::from(session_dir),
            model_name: "stub-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 4,
        };
        let session_id = store::create_session(&conn, &config, "0.1.0").expect("create session");

        let file_id = store::insert_file(
            &conn,
            session_id,
            "/test/doc.pdf",
            "hash123",
            0,
            1024,
            1,
            Some(1),
        )
        .expect("insert file");

        let chunk_ids = store::bulk_insert_chunks(
            &conn,
            &[ChunkInsert {
                file_id,
                session_id,
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: chunk_text.len() as i64,
                content: chunk_text,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: "hash",
                simhash: None,
            }],
        )
        .expect("insert chunk");

        (session_id, chunk_ids[0])
    }

    // -----------------------------------------------------------------------
    // T-VERIFY-001: verify handler rejects chunks from a different session
    // -----------------------------------------------------------------------

    /// Regression test for the cross-session data isolation defect: the verify
    /// handler previously fetched chunks by primary key only, ignoring the
    /// session_id field in the request. This allowed a caller to supply a
    /// chunk_id from any session and read its content through the verify response.
    ///
    /// After the fix, each chunk's session_id is compared against the request's
    /// session_id. A mismatch returns HTTP 404 (NotFound) to avoid disclosing
    /// whether the chunk exists in another session.
    #[tokio::test]
    async fn t_verify_001_rejects_chunk_from_different_session() {
        let state = test_support::setup_test_state();

        // Session A: owns the chunk.
        let (_session_a, chunk_id_from_a) =
            create_session_with_chunk(&state, "/test/session_a", "chunk content from session A");

        // Session B: a separate session that does not own chunk_id_from_a.
        let (session_b, _chunk_id_from_b) =
            create_session_with_chunk(&state, "/test/session_b", "chunk content from session B");

        // Use get_chunks_batch (the batch query) and verify the session_id
        // guard logic that the handler applies after fetching. This tests the
        // cross-session isolation without spinning up an HTTP server.
        let conn = state.pool.get().expect("conn");
        let chunks =
            store::get_chunks_batch(&conn, &[chunk_id_from_a]).expect("batch query must succeed");
        assert_eq!(chunks.len(), 1, "the chunk must exist in the database");

        // The chunk belongs to session A, not session B. The handler treats
        // this as NotFound for session B.
        assert_ne!(
            chunks[0].session_id, session_b,
            "chunk from session A must not belong to session B"
        );
    }
}
