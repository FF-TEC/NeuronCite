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

// End-to-end indexing pipeline integration tests.
//
// These tests exercise the multi-crate indexing pipeline: PDF discovery
// (neuroncite-pdf), text chunking (neuroncite-chunk), and persistent
// storage (neuroncite-store). The stub embedding backend from the common
// module is used in place of real ML models to keep tests fast and
// deterministic.
//
// Each test creates a temporary directory with test PDF files and an
// in-memory SQLite database, then verifies that the pipeline stages
// produce the expected records and relationships in the database.

use std::path::PathBuf;

use neuroncite_core::{Chunk, IndexConfig, StorageMode};
use neuroncite_store::{self as store, ChunkInsert};

use crate::common;

// ---------------------------------------------------------------------------
// Helper: creates a default IndexConfig for pipeline tests
// ---------------------------------------------------------------------------

/// Returns an IndexConfig pointing at the given directory with the word-window
/// chunking strategy, 4-dimensional vectors (matching the StubEmbedder), and
/// SQLite blob storage mode.
fn test_config(directory: &std::path::Path) -> IndexConfig {
    IndexConfig {
        directory: directory.to_path_buf(),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(10),
        chunk_overlap: Some(2),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    }
}

// ---------------------------------------------------------------------------
// T-INT-001: Full pipeline discover -> extract -> chunk -> store
// ---------------------------------------------------------------------------

/// T-INT-001: Verifies the complete indexing pipeline from PDF discovery
/// through text chunking to SQLite chunk storage.
///
/// Creates two test PDF files in a temporary directory, discovers them via
/// neuroncite_pdf::discover_pdfs, creates a session and file records, applies
/// the word-window chunking strategy, and bulk-inserts the resulting chunks.
/// Asserts that the session, files, and chunks exist in the database with
/// correct relationships.
#[test]
fn t_int_001_full_pipeline_discover_chunk_store() {
    let tmp = tempfile::TempDir::new().expect("failed to create temp dir");
    let root = tmp.path();

    // Create test PDF files in the temporary directory.
    common::create_test_pdf(root, "paper_a.pdf", "Statistical methods for data analysis");
    common::create_test_pdf(root, "paper_b.pdf", "Machine learning algorithms overview");

    // Step 1: Discover PDF files.
    let discovered = neuroncite_pdf::discover_pdfs(root).expect("PDF discovery failed");
    assert_eq!(
        discovered.len(),
        2,
        "two PDF files were created, discovery must find exactly two"
    );

    // Step 2: Set up the database and create a session.
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");
    let config = test_config(root);
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");

    // Step 3: Simulate extraction and chunking for each discovered file.
    // Instead of calling the real pdf-extract (which may struggle with our
    // minimal PDF), we use synthetic page text to verify pipeline structure.
    let synthetic_pages = [
        "Statistical methods for data analysis and interpretation in modern research",
        "Machine learning algorithms overview covering supervised and unsupervised approaches",
    ];

    let strategy = neuroncite_chunk::create_strategy("word", Some(10), Some(2), None, None)
        .expect("strategy creation failed");

    for (idx, (pdf_path, page_text)) in discovered.iter().zip(synthetic_pages.iter()).enumerate() {
        let file_id = store::insert_file(
            &conn,
            session_id,
            &pdf_path.to_string_lossy(),
            &Chunk::compute_content_hash(page_text),
            1_700_000_000,
            4096,
            1,
            None,
        )
        .expect("file insert failed");

        // Insert a single page record for the file.
        let pages_data = vec![(1_i64, *page_text, "pdf-extract")];
        store::bulk_insert_pages(&conn, file_id, &pages_data).expect("page insert failed");

        // Build PageText structs for the chunker.
        let page_texts = vec![neuroncite_core::PageText {
            source_file: pdf_path.clone(),
            page_number: 1,
            content: page_text.to_string(),
            backend: neuroncite_core::ExtractionBackend::PdfExtract,
        }];

        // Chunk the page text.
        let chunks = strategy.chunk(&page_texts).expect("chunking failed");

        // Convert chunks to ChunkInsert records and bulk-insert them.
        let chunk_inserts: Vec<ChunkInsert<'_>> = chunks
            .iter()
            .map(|c| ChunkInsert {
                file_id,
                session_id,
                page_start: c.page_start as i64,
                page_end: c.page_end as i64,
                chunk_index: c.chunk_index as i64,
                doc_text_offset_start: c.doc_text_offset_start as i64,
                doc_text_offset_end: c.doc_text_offset_end as i64,
                content: &c.content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &c.content_hash,
                simhash: None,
            })
            .collect();

        let ids = store::bulk_insert_chunks(&conn, &chunk_inserts).expect("chunk insert failed");
        assert!(
            !ids.is_empty(),
            "file {} (index {idx}) must produce at least one chunk",
            pdf_path.display()
        );
    }

    // Verify: session exists.
    let session = store::get_session(&conn, session_id).expect("session must exist after pipeline");
    assert_eq!(session.chunk_strategy, "word");

    // Verify: two file records exist for this session.
    let files = store::list_files_by_session(&conn, session_id).expect("file listing failed");
    assert_eq!(
        files.len(),
        2,
        "session must contain exactly two indexed file records"
    );

    // Verify: chunks exist for this session.
    let all_chunks =
        store::list_chunks_by_session(&conn, session_id).expect("chunk listing failed");
    assert!(
        all_chunks.len() >= 2,
        "session must contain at least two chunks (one per file minimum)"
    );
}

// ---------------------------------------------------------------------------
// T-INT-002: Entity creation order (session -> file -> page -> chunk)
// ---------------------------------------------------------------------------

/// T-INT-002: Verifies that the indexing pipeline creates entities in the
/// correct hierarchical order: session first, then files referencing the
/// session, then pages referencing files, then chunks referencing both
/// files and the session.
///
/// Validates foreign key relationships by checking that each chunk's
/// session_id and file_id match the parent entities.
#[test]
fn t_int_002_entity_creation_order() {
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");

    // 1. Create session
    let config = IndexConfig {
        directory: PathBuf::from("/test/docs"),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");
    assert!(session_id > 0, "session ID must be positive");

    // 2. Create file referencing the session
    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/docs/paper.pdf",
        "hash_abc",
        1_700_000_000,
        8192,
        3,
        None,
    )
    .expect("file insert failed");
    assert!(file_id > 0, "file ID must be positive");

    // 3. Create pages referencing the file
    let pages_data = vec![
        (1_i64, "First page content about statistics", "pdf-extract"),
        (2, "Second page content about regression", "pdf-extract"),
        (3, "Third page with conclusions", "pdf-extract"),
    ];
    store::bulk_insert_pages(&conn, file_id, &pages_data).expect("page insert failed");

    // 4. Create chunks referencing both file and session
    let chunks = vec![
        ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 40,
            content: "First page content about statistics",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_chunk_0",
            simhash: None,
        },
        ChunkInsert {
            file_id,
            session_id,
            page_start: 2,
            page_end: 2,
            chunk_index: 1,
            doc_text_offset_start: 40,
            doc_text_offset_end: 80,
            content: "Second page content about regression",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "hash_chunk_1",
            simhash: None,
        },
    ];
    let chunk_ids = store::bulk_insert_chunks(&conn, &chunks).expect("chunk insert failed");
    assert_eq!(chunk_ids.len(), 2, "two chunks must be inserted");

    // Verify foreign key relationships are consistent.
    let stored_chunks =
        store::list_chunks_by_session(&conn, session_id).expect("chunk listing failed");
    for chunk in &stored_chunks {
        assert_eq!(
            chunk.session_id, session_id,
            "chunk session_id must reference the parent session"
        );
        assert_eq!(
            chunk.file_id, file_id,
            "chunk file_id must reference the parent file"
        );
    }

    // Verify pages belong to the correct file.
    let stored_pages = store::get_pages_by_file(&conn, file_id).expect("page listing failed");
    assert_eq!(
        stored_pages.len(),
        3,
        "three pages must be stored for the file"
    );
    for page in &stored_pages {
        assert_eq!(
            page.file_id, file_id,
            "page file_id must reference the parent file"
        );
    }
}

// ---------------------------------------------------------------------------
// T-INT-003: Re-indexing detects unchanged files
// ---------------------------------------------------------------------------

/// T-INT-003: Verifies that the two-stage file change detection correctly
/// identifies files whose mtime and size have not changed since the last
/// indexing run, classifying them as `Unchanged`.
///
/// Simulates a re-indexing scenario by inserting a file record, then
/// calling check_file_changed with identical metadata. The file is
/// reported as unchanged, avoiding unnecessary re-extraction.
#[test]
fn t_int_003_reindex_detects_unchanged_files() {
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");

    let config = IndexConfig {
        directory: PathBuf::from("/test/docs"),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");

    let mtime = 1_700_000_000_i64;
    let size = 4096_i64;
    let file_hash = "abcdef1234567890abcdef1234567890";

    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/docs/paper.pdf",
        file_hash,
        mtime,
        size,
        5,
        None,
    )
    .expect("file insert failed");

    // Simulate re-indexing: same mtime and size as the stored record.
    let status = store::check_file_changed(&conn, file_id, mtime, size, None)
        .expect("change detection failed");
    assert_eq!(
        status,
        store::ChangeStatus::Unchanged,
        "file with identical mtime and size must be classified as unchanged"
    );

    // Simulate re-indexing: different mtime but same hash (metadata-only change).
    let status_meta = store::check_file_changed(&conn, file_id, mtime + 100, size, Some(file_hash))
        .expect("change detection failed");
    assert_eq!(
        status_meta,
        store::ChangeStatus::MetadataOnly,
        "file with changed mtime but same hash must be metadata-only"
    );

    // Simulate re-indexing: different mtime and different hash (content changed).
    let status_content = store::check_file_changed(
        &conn,
        file_id,
        mtime + 200,
        size + 512,
        Some("different_hash_value"),
    )
    .expect("change detection failed");
    assert_eq!(
        status_content,
        store::ChangeStatus::ContentChanged,
        "file with changed hash must be classified as content-changed"
    );
}

// ---------------------------------------------------------------------------
// T-INT-004: Mixed quality PDFs (some fail extraction) do not abort pipeline
// ---------------------------------------------------------------------------

/// T-INT-004: Verifies that the indexing pipeline continues processing
/// remaining files when extraction fails for some PDFs.
///
/// Simulates a batch of three files where the second file's extraction
/// "fails" (represented by an empty page). The pipeline stores chunks
/// for the successful files and the overall operation does not abort.
/// Files with zero extractable text still produce a file record but
/// contribute zero chunks to the session.
#[test]
fn t_int_004_mixed_quality_pdfs_dont_abort_pipeline() {
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");

    let config = IndexConfig {
        directory: PathBuf::from("/test/mixed"),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(5),
        chunk_overlap: Some(1),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");

    // Simulate three files: two with content, one with extraction failure
    // (represented as empty text content).
    let files_data = [
        (
            "good_paper_1.pdf",
            "Regression analysis is a statistical tool for data",
        ),
        ("bad_paper.pdf", ""), // Extraction "failed" -- no text recovered
        (
            "good_paper_2.pdf",
            "Bayesian inference provides a probabilistic framework for learning",
        ),
    ];

    let strategy = neuroncite_chunk::create_strategy("word", Some(5), Some(1), None, None)
        .expect("strategy creation failed");

    let mut total_chunks_created = 0_usize;

    for (file_name, page_text) in &files_data {
        let file_id = store::insert_file(
            &conn,
            session_id,
            &format!("/test/mixed/{file_name}"),
            &Chunk::compute_content_hash(page_text),
            1_700_000_000,
            2048,
            1,
            None,
        )
        .expect("file insert failed");

        let pages_data = vec![(1_i64, *page_text, "pdf-extract")];
        store::bulk_insert_pages(&conn, file_id, &pages_data).expect("page insert failed");

        // Skip chunking for files with empty text (simulates extraction failure).
        if page_text.is_empty() {
            continue;
        }

        let page_texts = vec![neuroncite_core::PageText {
            source_file: PathBuf::from(format!("/test/mixed/{file_name}")),
            page_number: 1,
            content: page_text.to_string(),
            backend: neuroncite_core::ExtractionBackend::PdfExtract,
        }];

        let chunks = strategy.chunk(&page_texts).expect("chunking failed");
        let chunk_inserts: Vec<ChunkInsert<'_>> = chunks
            .iter()
            .map(|c| ChunkInsert {
                file_id,
                session_id,
                page_start: c.page_start as i64,
                page_end: c.page_end as i64,
                chunk_index: c.chunk_index as i64,
                doc_text_offset_start: c.doc_text_offset_start as i64,
                doc_text_offset_end: c.doc_text_offset_end as i64,
                content: &c.content,
                embedding: None,
                ext_offset: None,
                ext_length: None,
                content_hash: &c.content_hash,
                simhash: None,
            })
            .collect();

        let ids = store::bulk_insert_chunks(&conn, &chunk_inserts).expect("chunk insert failed");
        total_chunks_created += ids.len();
    }

    // All three files must have file records in the database.
    let files = store::list_files_by_session(&conn, session_id).expect("file listing failed");
    assert_eq!(
        files.len(),
        3,
        "all three files (including the failed extraction) must have file records"
    );

    // Chunks must exist only for the two files with content.
    assert!(
        total_chunks_created > 0,
        "chunks must be created from files with extractable text"
    );

    let all_chunks =
        store::list_chunks_by_session(&conn, session_id).expect("chunk listing failed");
    assert_eq!(
        all_chunks.len(),
        total_chunks_created,
        "total stored chunks must match the count of chunks created from successful extractions"
    );
}

// ---------------------------------------------------------------------------
// T-INT-005: Chunk deduplication via content hash
// ---------------------------------------------------------------------------

/// T-INT-005: Verifies that chunks with identical content_hash values
/// represent duplicates that the deduplication layer can identify.
///
/// Inserts multiple chunks where two share the same content hash, then
/// queries the chunk table by content_hash to confirm that the index
/// on content_hash enables lookup of duplicates. The deduplication
/// logic itself is tested at the unit level in neuroncite-search; this
/// integration test verifies that the database schema supports the
/// hash-based duplicate detection across the store and chunk crates.
#[test]
fn t_int_005_chunk_deduplication_via_content_hash() {
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");

    let config = IndexConfig {
        directory: PathBuf::from("/test/dedup"),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(10),
        chunk_overlap: Some(2),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");

    // Insert two files, both containing the same text chunk.
    let shared_text = "identical chunk text appearing in multiple files for dedup testing";
    let shared_hash = Chunk::compute_content_hash(shared_text);

    let file_id_a = store::insert_file(
        &conn,
        session_id,
        "/test/dedup/file_a.pdf",
        "file_hash_a",
        1_700_000_000,
        2048,
        1,
        None,
    )
    .expect("file_a insert failed");

    let file_id_b = store::insert_file(
        &conn,
        session_id,
        "/test/dedup/file_b.pdf",
        "file_hash_b",
        1_700_000_000,
        2048,
        1,
        None,
    )
    .expect("file_b insert failed");

    let chunks = vec![
        ChunkInsert {
            file_id: file_id_a,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: shared_text.len() as i64,
            content: shared_text,
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: &shared_hash,
            simhash: None,
        },
        ChunkInsert {
            file_id: file_id_b,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: shared_text.len() as i64,
            content: shared_text,
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: &shared_hash,
            simhash: None,
        },
    ];

    let ids = store::bulk_insert_chunks(&conn, &chunks).expect("chunk insert failed");
    assert_eq!(ids.len(), 2, "both chunks must be inserted");

    // Query the database for chunks with the shared content hash.
    let dup_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk WHERE content_hash = ?1",
            rusqlite::params![shared_hash],
            |row| row.get(0),
        )
        .expect("duplicate count query failed");

    assert_eq!(
        dup_count, 2,
        "two chunks with the same content_hash must be found via the hash index"
    );
}

// ---------------------------------------------------------------------------
// T-INT-H07: ArcSwap allows concurrent reads during a hot-swap
// ---------------------------------------------------------------------------

/// T-INT-H07: Verifies that ArcSwap allows concurrent read access to the
/// current snapshot while the main thread performs hot-swaps.
///
/// Spawns 4 reader threads that each call `load()` 50 times while the
/// main thread performs 5 sequential `store()` calls. The guards returned
/// by `load()` are accessed to confirm no use-after-free occurs. No data
/// race, panic, or deadlock is expected because ArcSwap provides wait-free
/// reads and lock-free stores through an epoch-based reference counting
/// mechanism.
#[test]
fn t_int_h07_hnsw_arcswap_concurrent_reads_and_writes() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arc_swap::ArcSwap;

    // The swap cell holds a HashMap<i64, u32> to represent an HNSW node-id
    // to chunk-id mapping, which is the actual data structure used for
    // hot-reloading the HNSW index at runtime.
    let swap: Arc<ArcSwap<HashMap<i64, u32>>> = Arc::new(ArcSwap::from_pointee(HashMap::new()));

    let mut reader_handles = Vec::with_capacity(4);

    for _ in 0..4 {
        let s = swap.clone();
        reader_handles.push(std::thread::spawn(move || {
            for _ in 0..50 {
                // load() returns a Guard that keeps the Arc alive for the
                // duration of the read. Accessing len() confirms the guard
                // is valid and the pointed-to data is accessible.
                let guard = s.load();
                let _ = guard.len();
            }
        }));
    }

    // Perform 5 hot-swaps on the main thread while readers are active.
    for i in 0..5 {
        let mut new_map = HashMap::new();
        new_map.insert(i as i64, i as u32);
        swap.store(Arc::new(new_map));
    }

    for h in reader_handles {
        h.join().expect("reader thread must not panic");
    }
}

// ---------------------------------------------------------------------------
// T-INT-H09: Job state persists across pool reconnect (file-backed SQLite)
// ---------------------------------------------------------------------------

/// T-INT-H09: Verifies that a job inserted into a file-backed SQLite database
/// is still readable after the r2d2 pool is dropped and the database is
/// re-opened with a new pool.
///
/// This simulates the store-level persistence that the job executor relies on
/// across server restarts: jobs enqueued before shutdown must still be
/// visible after the process restarts and opens the same database file.
///
/// The test creates a jobs table directly (instead of using the full schema
/// migration) to keep the test self-contained and independent of schema
/// version changes.
#[test]
fn t_int_h09_job_persists_across_pool_reconnect() {
    use r2d2::Pool;
    use r2d2_sqlite::SqliteConnectionManager;

    let dir = tempfile::tempdir().expect("temp dir creation must succeed");
    let db_path = dir.path().join("test_jobs.db");

    // First pool: create schema and insert a job.
    {
        let manager = SqliteConnectionManager::file(&db_path);
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .expect("file-backed pool must build");
        let conn = pool.get().expect("connection must be available");

        // Create a minimal jobs table for the persistence test.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );",
        )
        .expect("schema creation must succeed");

        conn.execute(
            "INSERT INTO jobs (id, state, created_at) VALUES (1, 'queued', 0)",
            [],
        )
        .expect("job insert must succeed");
    } // pool is dropped here; all connections are closed and the WAL is checkpointed

    // Second pool: re-open the same database file and verify the job survived.
    {
        let manager = SqliteConnectionManager::file(&db_path);
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .expect("file-backed pool must build after reconnect");
        let conn = pool
            .get()
            .expect("connection must be available after reconnect");

        let state: String = conn
            .query_row("SELECT state FROM jobs WHERE id = 1", [], |row| row.get(0))
            .expect("job must still exist after pool reconnect");

        assert_eq!(
            state, "queued",
            "job state must be 'queued' after reconnect"
        );
    }
}

// ---------------------------------------------------------------------------
// T-INT-006: FTS5 index is populated after bulk chunk insert
// ---------------------------------------------------------------------------

/// T-INT-006: Verifies that the FTS5 full-text search index is automatically
/// populated when chunks are bulk-inserted into the chunk table.
///
/// The FTS5 synchronization triggers on the chunk table copy chunk content
/// into the chunk_fts virtual table on INSERT. This test inserts chunks
/// with distinctive terms and then queries the FTS5 index to confirm that
/// the terms are searchable.
#[test]
fn t_int_006_fts5_populated_after_bulk_insert() {
    let pool = common::create_default_test_pool();
    let conn = pool.get().expect("failed to obtain connection");

    let config = IndexConfig {
        directory: PathBuf::from("/test/fts"),
        model_name: "stub-model".to_string(),
        chunk_strategy: "word".to_string(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".to_string(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    let session_id =
        store::create_session(&conn, &config, "0.1.0").expect("session creation failed");

    let file_id = store::insert_file(
        &conn,
        session_id,
        "/test/fts/paper.pdf",
        "hash_fts_test",
        1_700_000_000,
        4096,
        1,
        None,
    )
    .expect("file insert failed");

    let pages_data = vec![(1_i64, "placeholder content", "pdf-extract")];
    store::bulk_insert_pages(&conn, file_id, &pages_data).expect("page insert failed");

    // Insert chunks with distinctive, uncommon terms for reliable FTS matching.
    let chunks = vec![
        ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 60,
            content: "thermoelectrochemical properties of gallium arsenide semiconductors",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "fts_hash_0",
            simhash: None,
        },
        ChunkInsert {
            file_id,
            session_id,
            page_start: 1,
            page_end: 1,
            chunk_index: 1,
            doc_text_offset_start: 60,
            doc_text_offset_end: 120,
            content: "magnetohydrodynamic turbulence in stellar plasma environments",
            embedding: None,
            ext_offset: None,
            ext_length: None,
            content_hash: "fts_hash_1",
            simhash: None,
        },
    ];

    let chunk_ids = store::bulk_insert_chunks(&conn, &chunks).expect("chunk insert failed");
    assert_eq!(chunk_ids.len(), 2, "two chunks must be inserted");

    // Query FTS5 for the distinctive term from the first chunk.
    let fts_count_gallium: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'gallium'",
            [],
            |row| row.get(0),
        )
        .expect("FTS5 query for 'gallium' failed");
    assert_eq!(
        fts_count_gallium, 1,
        "FTS5 index must contain exactly one entry matching 'gallium'"
    );

    // Query FTS5 for the distinctive term from the second chunk.
    let fts_count_magnetohydrodynamic: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'magnetohydrodynamic'",
            [],
            |row| row.get(0),
        )
        .expect("FTS5 query for 'magnetohydrodynamic' failed");
    assert_eq!(
        fts_count_magnetohydrodynamic, 1,
        "FTS5 index must contain exactly one entry matching 'magnetohydrodynamic'"
    );

    // Verify that the FTS5 rowids correspond to the correct chunk IDs.
    let fts_rowid: i64 = conn
        .query_row(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'gallium'",
            [],
            |row| row.get(0),
        )
        .expect("FTS5 rowid query failed");
    assert_eq!(
        fts_rowid, chunk_ids[0],
        "FTS5 rowid must match the first chunk's database ID"
    );
}
