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

// Regression tests for edge cases, boundary conditions, and failure modes.
//
// Each test exercises a specific problematic input pattern that could cause
// panics, data corruption, or incorrect behavior in the chunking, storage,
// search, or normalization subsystems. These tests use real crate functions
// and validate graceful error handling or correct output for degenerate inputs.

use std::path::PathBuf;

use neuroncite_core::types::{Chunk, ExtractionBackend, IndexConfig, PageText, StorageMode};

// -- Database test helpers ---------------------------------------------------

/// Creates an in-memory SQLite database with the full schema applied and
/// foreign keys enabled. Returns the connection ready for CRUD operations.
fn setup_db() -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_in_memory().expect("failed to open in-memory database");
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .expect("failed to enable foreign keys");
    neuroncite_store::migrate(&conn).expect("schema migration failed");
    conn
}

/// Creates a default IndexConfig for page-based chunking. All nullable
/// chunking parameters are None (matching the page strategy).
fn page_config() -> IndexConfig {
    IndexConfig {
        directory: PathBuf::from("/test/docs"),
        model_name: "test-model".into(),
        chunk_strategy: "page".into(),
        chunk_size: None,
        chunk_overlap: None,
        max_words: None,
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    }
}

/// Creates a default IndexConfig for word-based chunking with specific
/// chunk_size and overlap values.
fn word_config() -> IndexConfig {
    IndexConfig {
        directory: PathBuf::from("/test/docs"),
        model_name: "test-model".into(),
        chunk_strategy: "word".into(),
        chunk_size: Some(300),
        chunk_overlap: Some(50),
        max_words: None,
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    }
}

/// Creates a session, file, and returns (session_id, file_id) for inserting
/// chunks into the database.
fn setup_session_and_file(conn: &rusqlite::Connection, config: &IndexConfig) -> (i64, i64) {
    let session_id =
        neuroncite_store::create_session(conn, config, "0.1.0").expect("session creation failed");

    let file_id = neuroncite_store::insert_file(
        conn,
        session_id,
        "/test/docs/sample.pdf",
        "abc123",
        1_700_000_000,
        4096,
        1,
        None,
    )
    .expect("file insert failed");

    (session_id, file_id)
}

/// Constructs a PageText from a content string with a fixed source file path.
fn make_page(page_number: usize, content: &str) -> PageText {
    PageText {
        source_file: PathBuf::from("/test/docs/sample.pdf"),
        page_number,
        content: content.to_string(),
        backend: ExtractionBackend::PdfExtract,
    }
}

// -- T-REG-001 ----------------------------------------------------------------

/// T-REG-001: Empty document (0 pages) returns an error, not a panic.
///
/// All chunking strategies require at least one page. Passing an empty slice
/// must produce `Err(NeuronCiteError::Chunk(...))` without panicking.
#[test]
fn t_reg_001_empty_document_returns_error() {
    let empty_pages: Vec<PageText> = Vec::new();

    // Page strategy
    let page_strategy = neuroncite_chunk::create_strategy("page", None, None, None, None)
        .expect("page strategy creation failed");
    let result = page_strategy.chunk(&empty_pages);
    assert!(
        result.is_err(),
        "page strategy must return Err for empty page list"
    );

    // Word strategy
    let word_strategy = neuroncite_chunk::create_strategy("word", Some(10), Some(2), None, None)
        .expect("word strategy creation failed");
    let result = word_strategy.chunk(&empty_pages);
    assert!(
        result.is_err(),
        "word strategy must return Err for empty page list"
    );

    // Sentence strategy
    let sentence_strategy =
        neuroncite_chunk::create_strategy("sentence", None, None, Some(50), None)
            .expect("sentence strategy creation failed");
    let result = sentence_strategy.chunk(&empty_pages);
    assert!(
        result.is_err(),
        "sentence strategy must return Err for empty page list"
    );
}

// -- T-REG-002 ----------------------------------------------------------------

/// T-REG-002: Document with only whitespace pages produces no chunks.
///
/// Pages containing only whitespace characters (spaces, tabs, newlines) have
/// no extractable words. The word-window strategy must return an empty chunk
/// list rather than panicking or producing chunks with empty content.
#[test]
fn t_reg_002_whitespace_only_pages_produce_no_chunks() {
    let whitespace_pages = vec![
        make_page(1, "   "),
        make_page(2, "\t\t\t"),
        make_page(3, "\n\n\n"),
        make_page(4, " \t \n "),
    ];

    let strategy = neuroncite_chunk::create_strategy("word", Some(5), Some(1), None, None)
        .expect("word strategy creation failed");

    let chunks = strategy
        .chunk(&whitespace_pages)
        .expect("chunking whitespace-only pages must not fail");

    assert_eq!(
        chunks.len(),
        0,
        "whitespace-only pages must produce zero word-window chunks"
    );
}

// -- T-REG-003 ----------------------------------------------------------------

/// T-REG-003: Very long single word (10000 chars) handled without panic.
///
/// A page containing a single word of 10000 characters must be processed
/// by the word-window strategy without panicking, stack overflow, or
/// producing invalid chunks. The single word forms exactly one chunk.
#[test]
fn t_reg_003_very_long_single_word_handled() {
    let long_word: String = "a".repeat(10_000);
    let pages = vec![make_page(1, &long_word)];

    let strategy = neuroncite_chunk::create_strategy("word", Some(5), Some(1), None, None)
        .expect("word strategy creation failed");

    let chunks = strategy
        .chunk(&pages)
        .expect("chunking a 10000-char single word must not fail");

    // A single word produces exactly one chunk (the word itself).
    assert_eq!(
        chunks.len(),
        1,
        "a single 10000-char word must produce exactly one chunk"
    );

    // The chunk content must equal the original word (possibly after normalization).
    assert!(
        chunks[0].content.len() >= 10_000,
        "chunk content length ({}) must be at least 10000",
        chunks[0].content.len()
    );

    // The chunk must have valid offsets.
    assert!(
        chunks[0].doc_text_offset_end > chunks[0].doc_text_offset_start,
        "chunk byte range must be non-empty"
    );
}

// -- T-REG-004 ----------------------------------------------------------------

/// T-REG-004: Unicode BOM characters handled gracefully during chunking.
///
/// The Unicode Byte Order Mark (U+FEFF) appears at the beginning of some
/// PDF-extracted text. The normalization pipeline (whitespace collapsing,
/// hyphenation repair, NFC normalization) does not explicitly strip BOM
/// characters. U+FEFF is not in the set of characters collapsed by
/// `collapse_whitespace` (which handles only space, tab, and form feed).
///
/// This test verifies that text containing BOM characters is chunked without
/// panicking, produces valid chunks with correct content hashes, and the
/// content hash is deterministically reproducible.
#[test]
fn t_reg_004_unicode_bom_handled_gracefully() {
    // U+FEFF is the BOM character. Place it at the start of the text.
    let text_with_bom = "\u{FEFF}The quick brown fox jumps over the lazy dog";
    let pages = vec![make_page(1, text_with_bom)];

    let strategy = neuroncite_chunk::create_strategy("word", Some(5), Some(1), None, None)
        .expect("word strategy creation failed");

    let chunks = strategy
        .chunk(&pages)
        .expect("chunking text with BOM must not fail");

    // Verify that chunks are produced from the BOM-prefixed input.
    assert!(
        !chunks.is_empty(),
        "BOM-prefixed text must produce at least one chunk"
    );

    // Verify that all chunk content hashes are valid 64-character hex strings
    // and are deterministically reproducible.
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.content_hash.len(),
            64,
            "chunk {i} content_hash must be 64 hex characters"
        );

        // Recompute the hash and verify it matches the stored value.
        let recomputed = Chunk::compute_content_hash(&chunk.content);
        assert_eq!(
            chunk.content_hash, recomputed,
            "chunk {i} content_hash must match recomputed SHA-256"
        );
    }

    // Verify that chunk_index values form a contiguous 0-based sequence.
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.chunk_index, i,
            "chunk_index must be {i}, got {}",
            chunk.chunk_index
        );
    }
}

// -- T-REG-005 ----------------------------------------------------------------

/// T-REG-005: NULL bytes in text content handled gracefully.
///
/// Some PDF extraction backends may emit NULL bytes (U+0000) in the extracted
/// text. The chunking pipeline must handle them without panicking. NULL bytes
/// are valid UTF-8 characters and should pass through or be stripped by
/// normalization.
#[test]
fn t_reg_005_null_bytes_handled() {
    let text_with_nulls = "Hello\0world\0this\0is\0a\0test";
    let pages = vec![make_page(1, text_with_nulls)];

    let strategy = neuroncite_chunk::create_strategy("word", Some(3), Some(0), None, None)
        .expect("word strategy creation failed");

    // The chunking pipeline must not panic on NULL bytes.
    let result = strategy.chunk(&pages);
    assert!(
        result.is_ok(),
        "chunking text with NULL bytes must not fail: {:?}",
        result.err()
    );

    let chunks = result.expect("already verified Ok");

    // Verify that chunks were produced (NULL bytes do not prevent splitting).
    assert!(
        !chunks.is_empty(),
        "text with NULL bytes must produce at least one chunk"
    );

    // Verify that all chunk content hashes are valid SHA-256 hex strings.
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.content_hash.len(),
            64,
            "chunk {i} content_hash must be 64 hex characters, got {}",
            chunk.content_hash.len()
        );
    }
}

// -- T-REG-006 ----------------------------------------------------------------

/// T-REG-006: Session with COALESCE(-1) handles NULL directory correctly.
///
/// The COALESCE-based unique index on index_session substitutes -1 for NULL
/// values in chunk_size, chunk_overlap, and max_words. This test verifies
/// that two sessions with identical configurations where all nullable fields
/// are NULL (page-based strategy) are correctly rejected by the unique
/// constraint, and that finding a session by config returns the correct ID.
#[test]
fn t_reg_006_coalesce_null_directory_unique_constraint() {
    let conn = setup_db();

    let config = page_config();

    // First session creation must succeed.
    let session_id = neuroncite_store::create_session(&conn, &config, "0.1.0")
        .expect("first session creation must succeed");
    assert!(session_id > 0, "session ID must be positive");

    // Second session creation with identical config must fail due to the
    // COALESCE-based unique index treating NULL as -1.
    let duplicate_result = neuroncite_store::create_session(&conn, &config, "0.1.0");
    assert!(
        duplicate_result.is_err(),
        "duplicate session with all-NULL chunking params must be rejected"
    );

    // Finding the session by config must return the original session ID.
    let found =
        neuroncite_store::find_session(&conn, &config).expect("find_session query must not fail");
    assert_eq!(
        found,
        Some(session_id),
        "find_session must return the existing session ID"
    );
}

// -- T-REG-007 ----------------------------------------------------------------

/// T-REG-007: Chunk with 0-length content hash is computed correctly.
///
/// An empty string has a well-defined SHA-256 hash. The content hash function
/// must produce the correct 64-character hex digest for the empty string.
#[test]
fn t_reg_007_zero_length_content_hash() {
    let empty_hash = Chunk::compute_content_hash("");

    // SHA-256 of the empty string is a well-known constant.
    let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    assert_eq!(
        empty_hash, expected,
        "SHA-256 of empty string must match the known constant"
    );

    // Verify length is 64 hex characters.
    assert_eq!(
        empty_hash.len(),
        64,
        "content hash must be 64 hex characters"
    );

    // Verify that the hash is lowercase hex.
    assert!(
        empty_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "content hash must contain only hex digits"
    );
    assert!(
        empty_hash.chars().all(|c| !c.is_ascii_uppercase()),
        "content hash must be lowercase"
    );
}

// -- T-REG-008 ----------------------------------------------------------------

/// T-REG-008: HNSW index with a single vector can be searched.
///
/// The HNSW index must support building and searching with just one vector.
/// This is the minimum viable index size. The single vector must be its own
/// nearest neighbor with distance 0 (or very close to 0 for cosine distance).
#[test]
fn t_reg_008_hnsw_single_vector_search() {
    let dimension = 128;

    // Create a single normalized vector.
    let mut vector = vec![0.0_f32; dimension];
    vector[0] = 1.0; // unit vector along the first axis

    let vectors: Vec<(i64, &[f32])> = vec![(1, vector.as_slice())];

    let index = neuroncite_store::build_hnsw(&vectors, dimension).expect("build_hnsw");
    assert_eq!(index.len(), 1, "index must contain exactly one vector");

    // Search for the single vector. It must be its own nearest neighbor.
    let results = index.search(&vector, 1, 100);
    assert_eq!(
        results.len(),
        1,
        "search on single-vector index must return one result"
    );

    let (chunk_id, distance) = results[0];
    assert_eq!(
        chunk_id, 1,
        "the single vector must be its own nearest neighbor"
    );

    // Cosine distance for identical vectors is 0 (or very close to 0).
    assert!(
        distance < 0.01,
        "cosine distance of a vector to itself must be near 0, got {distance}"
    );
}

// -- T-REG-009 ----------------------------------------------------------------

/// T-REG-009: FTS5 query with special characters (AND, OR, NOT) is escaped.
///
/// FTS5 treats AND, OR, NOT, and NEAR as query operators. If user search text
/// contains these words literally, they must either be escaped or the query
/// must be wrapped in quotes to prevent FTS5 syntax errors. This test verifies
/// that inserting and querying chunks containing these reserved words does not
/// cause database errors.
#[test]
fn t_reg_009_fts5_special_characters_handled() {
    let conn = setup_db();
    let config = word_config();
    let (session_id, file_id) = setup_session_and_file(&conn, &config);

    // Insert chunks containing FTS5 reserved words as content.
    let contents = [
        "the AND operator combines boolean conditions in logic",
        "the OR operator provides alternatives in queries",
        "the NOT operator negates a condition in expressions",
    ];

    let hashes: Vec<String> = (0..contents.len())
        .map(|i| format!("fts_special_hash_{i}"))
        .collect();

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
            content_hash: &hashes[i],
            simhash: None,
        })
        .collect();

    neuroncite_store::bulk_insert_chunks(&conn, &chunk_inserts)
        .expect("bulk insert of chunks with reserved words failed");

    // Query for the word "operator" (present in all three chunks) using a
    // plain term that does not trigger FTS5 operator parsing.
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH ?1",
            rusqlite::params!["operator"],
            |row| row.get(0),
        )
        .expect("FTS5 query for 'operator' failed");

    assert_eq!(count, 3, "FTS5 query for 'operator' must find all 3 chunks");

    // Query using a quoted phrase to search for the literal word "AND"
    // (which is an FTS5 operator when unquoted).
    let quoted_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH '\"AND\"'",
            [],
            |row| row.get(0),
        )
        .expect("FTS5 quoted query for '\"AND\"' failed");

    assert!(
        quoted_count >= 1,
        "FTS5 quoted query for '\"AND\"' must find at least 1 chunk, found {quoted_count}"
    );
}

// -- T-REG-010 ----------------------------------------------------------------

/// T-REG-010: Database migration is idempotent (running twice is safe).
///
/// Calling `migrate()` on an already-migrated database must succeed without
/// error and must not create duplicate tables, indexes, or triggers.
#[test]
fn t_reg_010_migration_idempotent() {
    let conn = setup_db(); // This already calls migrate() once.

    // Call migrate() a second time.
    neuroncite_store::migrate(&conn).expect("second migration call must succeed (idempotent)");

    // Call migrate() a third time for good measure.
    neuroncite_store::migrate(&conn).expect("third migration call must succeed (idempotent)");

    // Verify that the schema contains exactly one of each critical table.
    let tables = [
        "index_session",
        "indexed_file",
        "page",
        "chunk",
        "job",
        "idempotency",
    ];

    for table_name in &tables {
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                rusqlite::params![table_name],
                |row| row.get(0),
            )
            .unwrap_or_else(|e| panic!("failed to count table '{table_name}': {e}"));
        assert_eq!(
            count, 1,
            "table '{table_name}' must exist exactly once after repeated migrations"
        );
    }

    // Verify the schema version is correct.
    let version: u32 = conn
        .query_row("PRAGMA user_version;", [], |row| row.get(0))
        .expect("failed to query user_version");
    assert_eq!(
        version,
        neuroncite_store::schema_version(),
        "user_version must match the schema version constant"
    );
}

// -- T-REG-011 ----------------------------------------------------------------

/// T-REG-011: Connection pool handles concurrent access from 8 threads.
///
/// The r2d2 connection pool is configured with max_size=8. This test verifies
/// that 8 threads can simultaneously acquire connections and perform read
/// operations without deadlock or timeout errors.
#[test]
fn t_reg_011_connection_pool_concurrent_access() {
    let tmp = tempfile::TempDir::new().expect("failed to create temp dir");
    let db_path = tmp.path().join("concurrent_test.db");

    let pool = neuroncite_store::create_pool(&db_path).expect("connection pool creation failed");

    // Apply the schema using a pooled connection.
    {
        let conn = pool.get().expect("failed to get connection from pool");
        neuroncite_store::migrate(&conn).expect("migration failed");
    }

    // Spawn 8 threads, each acquiring a connection and performing a read query.
    let pool_arc = std::sync::Arc::new(pool);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));

    let handles: Vec<std::thread::JoinHandle<()>> = (0..8)
        .map(|thread_idx| {
            let pool_clone = std::sync::Arc::clone(&pool_arc);
            let barrier_clone = std::sync::Arc::clone(&barrier);

            std::thread::spawn(move || {
                // All threads wait at the barrier to maximize concurrency.
                barrier_clone.wait();

                let conn = pool_clone.get().unwrap_or_else(|e| {
                    panic!("thread {thread_idx}: failed to get connection: {e}")
                });

                // Perform a simple read query to verify the connection works.
                let version: u32 = conn
                    .query_row("PRAGMA user_version;", [], |row| row.get(0))
                    .unwrap_or_else(|e| panic!("thread {thread_idx}: query failed: {e}"));

                assert_eq!(
                    version,
                    neuroncite_store::schema_version(),
                    "thread {thread_idx}: schema version mismatch"
                );
            })
        })
        .collect();

    // Wait for all threads to complete. Panic if any thread panicked.
    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|_| panic!("thread {i} panicked during concurrent pool access"));
    }
}

// -- T-REG-012 ----------------------------------------------------------------

/// T-REG-012: Session label update with NULL clears the label, and update with
/// a string sets the label. Verifies BUG #5 fix: session_update must correctly
/// distinguish between JSON null (clear label) and a string value (set label).
#[test]
fn t_reg_012_session_label_update_null_and_string() {
    let conn = setup_db();
    let config = page_config();

    let session_id =
        neuroncite_store::create_session(&conn, &config, "0.1.0").expect("session creation");

    // Initial label is NULL.
    let session = neuroncite_store::get_session(&conn, session_id).expect("get session");
    assert!(session.label.is_none(), "initial label must be NULL");

    // Set the label to a string value.
    let rows_updated = neuroncite_store::update_session_label(&conn, session_id, Some("My Papers"))
        .expect("set label");
    assert_eq!(rows_updated, 1, "update must affect exactly one row");

    let session = neuroncite_store::get_session(&conn, session_id).expect("get session after set");
    assert_eq!(
        session.label.as_deref(),
        Some("My Papers"),
        "label must be set to 'My Papers'"
    );

    // Clear the label by passing None (maps to SQL NULL).
    let rows_updated =
        neuroncite_store::update_session_label(&conn, session_id, None).expect("clear label");
    assert_eq!(rows_updated, 1, "update must affect exactly one row");

    let session =
        neuroncite_store::get_session(&conn, session_id).expect("get session after clear");
    assert!(session.label.is_none(), "label must be NULL after clearing");

    // Set the label to the literal string "null" (not JSON null) -- must be stored as-is.
    let rows_updated = neuroncite_store::update_session_label(&conn, session_id, Some("null"))
        .expect("set label to literal 'null' string");
    assert_eq!(rows_updated, 1, "update must affect exactly one row");

    let session =
        neuroncite_store::get_session(&conn, session_id).expect("get session after literal null");
    assert_eq!(
        session.label.as_deref(),
        Some("null"),
        "label must be the literal string 'null', not SQL NULL"
    );
}

// -- T-REG-013 ----------------------------------------------------------------

/// T-REG-013: Session with max_words stores and retrieves the value correctly.
/// Verifies DESIGN #5: the max_words field must be readable from SessionRow for
/// display in session listings.
#[test]
fn t_reg_013_session_max_words_stored() {
    let conn = setup_db();

    let config = IndexConfig {
        directory: PathBuf::from("/test/docs"),
        model_name: "test-model".into(),
        chunk_strategy: "sentence".into(),
        chunk_size: None,
        chunk_overlap: None,
        max_words: Some(150),
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };

    let session_id =
        neuroncite_store::create_session(&conn, &config, "0.1.0").expect("session creation");

    let session = neuroncite_store::get_session(&conn, session_id).expect("get session");
    assert_eq!(
        session.max_words,
        Some(150),
        "max_words must be 150 for sentence strategy"
    );
    assert_eq!(
        session.chunk_strategy, "sentence",
        "chunk_strategy must be 'sentence'"
    );
    assert!(
        session.chunk_size.is_none(),
        "chunk_size must be NULL for sentence strategy"
    );
    assert!(
        session.chunk_overlap.is_none(),
        "chunk_overlap must be NULL for sentence strategy"
    );
}

// -- T-REG-014 ----------------------------------------------------------------

/// T-REG-014: Schema v2 migration creates the pdf_page_count column on
/// databases that were originally created with the v1 schema. The column
/// must be nullable (accepting NULL for old files) and accept integer values
/// for new files. Verifies BUG #1 fix.
#[test]
fn t_reg_014_v2_migration_pdf_page_count_column() {
    let conn = setup_db(); // Applies all migrations including v2.

    let config = page_config();
    let session_id =
        neuroncite_store::create_session(&conn, &config, "0.1.0").expect("session creation");

    // Insert a file without pdf_page_count (NULL).
    let file_id_null = neuroncite_store::insert_file(
        &conn,
        session_id,
        "/test/docs/old_file.pdf",
        "hash_old",
        1_700_000_000,
        4096,
        10,
        None,
    )
    .expect("insert file without pdf_page_count");

    let file_null = neuroncite_store::get_file(&conn, file_id_null).expect("get file null");
    assert!(
        file_null.pdf_page_count.is_none(),
        "pdf_page_count must be NULL when not provided"
    );

    // Insert a file with pdf_page_count set.
    let file_id_set = neuroncite_store::insert_file(
        &conn,
        session_id,
        "/test/docs/new_file.pdf",
        "hash_new",
        1_700_000_000,
        8192,
        15,
        Some(20),
    )
    .expect("insert file with pdf_page_count");

    let file_set = neuroncite_store::get_file(&conn, file_id_set).expect("get file set");
    assert_eq!(
        file_set.pdf_page_count,
        Some(20),
        "pdf_page_count must be 20"
    );
    assert_eq!(file_set.page_count, 15, "page_count must be 15");
}

// -- T-REG-015 ----------------------------------------------------------------

/// T-REG-015: strip_extended_length_prefix correctly handles Windows `\\?\`
/// prefixed paths, plain Windows paths, and Unix paths. Verifies DESIGN #3 fix
/// at the utility function level.
#[test]
fn t_reg_015_strip_extended_length_prefix_variants() {
    use neuroncite_core::paths::strip_extended_length_prefix;

    // Windows extended-length prefix is stripped.
    assert_eq!(
        strip_extended_length_prefix(r"\\?\C:\Users\test\Papers"),
        r"C:\Users\test\Papers"
    );

    // Windows path without prefix is returned unchanged.
    assert_eq!(
        strip_extended_length_prefix(r"C:\Users\test\Papers"),
        r"C:\Users\test\Papers"
    );

    // Unix path is returned unchanged.
    assert_eq!(
        strip_extended_length_prefix("/home/user/papers"),
        "/home/user/papers"
    );

    // Empty string is returned unchanged.
    assert_eq!(strip_extended_length_prefix(""), "");

    // UNC path with \\?\ prefix is stripped.
    assert_eq!(
        strip_extended_length_prefix(r"\\?\UNC\server\share"),
        r"UNC\server\share"
    );
}

// -- T-REG-016 ----------------------------------------------------------------

/// T-REG-016: relevance_label returns correct labels for boundary values.
/// Verifies BUG #6 related logic where relevance labels are computed from
/// vector_score values at the threshold boundaries.
#[test]
fn t_reg_016_relevance_label_thresholds() {
    use neuroncite_search::relevance_label;

    // Boundary values for the four relevance tiers:
    // >= 0.82 -> "high", >= 0.72 -> "medium", >= 0.60 -> "low", < 0.60 -> "marginal"

    assert_eq!(relevance_label(1.0), "high");
    assert_eq!(relevance_label(0.82), "high");
    assert_eq!(relevance_label(0.85), "high");

    assert_eq!(relevance_label(0.81), "medium");
    assert_eq!(relevance_label(0.72), "medium");
    assert_eq!(relevance_label(0.75), "medium");

    assert_eq!(relevance_label(0.71), "low");
    assert_eq!(relevance_label(0.60), "low");
    assert_eq!(relevance_label(0.65), "low");

    assert_eq!(relevance_label(0.59), "marginal");
    assert_eq!(relevance_label(0.0), "marginal");
    assert_eq!(relevance_label(-0.1), "marginal");
}

// -- T-REG-017 ----------------------------------------------------------------

/// T-REG-017: Session listing includes max_words, chunk_size, chunk_overlap
/// for different chunk strategies. Verifies DESIGN #5: all chunking parameters
/// must be readable from list_sessions results.
#[test]
fn t_reg_017_session_listing_chunk_params() {
    let conn = setup_db();

    // Session 1: page strategy (all chunk params NULL).
    let page_cfg = page_config();
    neuroncite_store::create_session(&conn, &page_cfg, "0.1.0").expect("page session");

    // Session 2: word strategy (chunk_size + chunk_overlap set, max_words NULL).
    let word_cfg = word_config();
    neuroncite_store::create_session(&conn, &word_cfg, "0.1.0").expect("word session");

    // Session 3: sentence strategy (max_words set, others NULL).
    let sentence_cfg = IndexConfig {
        directory: PathBuf::from("/test/sentence"),
        model_name: "test-model".into(),
        chunk_strategy: "sentence".into(),
        chunk_size: None,
        chunk_overlap: None,
        max_words: Some(200),
        ocr_language: "eng".into(),
        embedding_storage_mode: StorageMode::SqliteBlob,
        vector_dimension: 4,
    };
    neuroncite_store::create_session(&conn, &sentence_cfg, "0.1.0").expect("sentence session");

    let sessions = neuroncite_store::list_sessions(&conn).expect("list sessions");
    assert_eq!(sessions.len(), 3, "three sessions must be listed");

    // Find the sentence session and verify max_words.
    let sentence_session = sessions
        .iter()
        .find(|s| s.chunk_strategy == "sentence")
        .expect("sentence session must exist in listing");
    assert_eq!(
        sentence_session.max_words,
        Some(200),
        "sentence session max_words must be 200"
    );
    assert!(
        sentence_session.chunk_size.is_none(),
        "sentence session chunk_size must be NULL"
    );

    // Verify the word session has chunk_size and chunk_overlap.
    let word_session = sessions
        .iter()
        .find(|s| s.chunk_strategy == "word")
        .expect("word session must exist in listing");
    assert_eq!(
        word_session.chunk_size,
        Some(300),
        "word session chunk_size must be 300"
    );
    assert_eq!(
        word_session.chunk_overlap,
        Some(50),
        "word session chunk_overlap must be 50"
    );
    assert!(
        word_session.max_words.is_none(),
        "word session max_words must be NULL"
    );
}

// -- T-REG-018 ----------------------------------------------------------------

/// T-REG-018: Finalize-progress SQL sets progress_done = progress_total.
///
/// Regression test for NEU-001: Annotation jobs where no input matched a PDF
/// completed with progress_done=0 and progress_total=N, creating an inconsistent
/// terminal state. The fix adds a `finalize_progress` call before transitioning
/// to the Completed state. This test verifies the SQL logic at the database level:
/// after setting progress_total to a value greater than progress_done, the
/// finalize query must equalize them.
#[test]
fn t_reg_018_finalize_progress_equalizes_done_and_total() {
    let conn = setup_db();

    // Create an annotation job with initial progress 0/0.
    let job_id = "test-finalize-progress";
    neuroncite_store::create_job_with_params(&conn, job_id, "annotate", None, None)
        .expect("job creation");

    // Simulate the MCP handler setting initial progress: 0 / 5 (5 input quotes).
    neuroncite_store::update_job_progress(&conn, job_id, 0, 5).expect("set initial progress");

    // Verify the inconsistent pre-fix state: progress_done=0, progress_total=5.
    let job_before = neuroncite_store::get_job(&conn, job_id).expect("get job before finalize");
    assert_eq!(
        job_before.progress_done, 0,
        "progress_done must be 0 before finalize"
    );
    assert_eq!(
        job_before.progress_total, 5,
        "progress_total must be 5 before finalize"
    );

    // Execute the finalize_progress SQL (same as the executor's finalize_progress function).
    conn.execute(
        "UPDATE job SET progress_done = progress_total WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .expect("finalize progress SQL");

    // Verify progress_done now equals progress_total.
    let job_after = neuroncite_store::get_job(&conn, job_id).expect("get job after finalize");
    assert_eq!(
        job_after.progress_done, job_after.progress_total,
        "progress_done ({}) must equal progress_total ({}) after finalize",
        job_after.progress_done, job_after.progress_total
    );
    assert_eq!(
        job_after.progress_done, 5,
        "progress_done must be 5 after finalize"
    );
}

// -- T-REG-019 ----------------------------------------------------------------

/// T-REG-019: Finalize-progress is a no-op when progress_done already equals
/// progress_total (normal completion path).
///
/// Complementary test to T-REG-021: when a job completes normally with all
/// items processed (progress_done = progress_total), the finalize step must
/// not corrupt the counters.
#[test]
fn t_reg_019_finalize_progress_noop_when_already_equal() {
    let conn = setup_db();

    let job_id = "test-finalize-noop";
    neuroncite_store::create_job_with_params(&conn, job_id, "annotate", None, None)
        .expect("job creation");

    // Simulate normal completion: progress_done = progress_total = 3.
    neuroncite_store::update_job_progress(&conn, job_id, 3, 3).expect("set progress");

    // Finalize.
    conn.execute(
        "UPDATE job SET progress_done = progress_total WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .expect("finalize progress SQL");

    let job = neuroncite_store::get_job(&conn, job_id).expect("get job");
    assert_eq!(job.progress_done, 3, "progress_done must remain 3");
    assert_eq!(job.progress_total, 3, "progress_total must remain 3");
}

// -- T-REG-020 ----------------------------------------------------------------

/// T-REG-020: Finalize-progress works when progress_done is partially advanced.
///
/// Simulates an annotation job where 2 of 5 PDFs were matched and processed,
/// but 3 inputs had no match. The pipeline reports progress_done=2 (matched PDFs
/// processed) while progress_total=5 (total input quotes). After finalize,
/// progress_done must equal progress_total.
#[test]
fn t_reg_020_finalize_progress_partial_completion() {
    let conn = setup_db();

    let job_id = "test-finalize-partial";
    neuroncite_store::create_job_with_params(&conn, job_id, "annotate", None, None)
        .expect("job creation");

    // Handler sets initial progress: 0 / 5.
    neuroncite_store::update_job_progress(&conn, job_id, 0, 5).expect("initial progress");

    // Pipeline callback updates to 2 / 2 (2 matched PDFs, overwriting total to matched count).
    neuroncite_store::update_job_progress(&conn, job_id, 2, 2).expect("pipeline progress");

    let job_before = neuroncite_store::get_job(&conn, job_id).expect("get job before");
    assert_eq!(job_before.progress_done, 2);
    assert_eq!(job_before.progress_total, 2);

    // Finalize.
    conn.execute(
        "UPDATE job SET progress_done = progress_total WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .expect("finalize progress SQL");

    let job_after = neuroncite_store::get_job(&conn, job_id).expect("get job after");
    assert_eq!(
        job_after.progress_done, job_after.progress_total,
        "progress_done must equal progress_total after finalize"
    );
}

// -- T-REG-021 ----------------------------------------------------------------

/// T-REG-021: A source_file field containing HTML special characters in the
/// filename round-trips through JSON serialization without modification.
///
/// serde_json's default serializer preserves angle brackets as literal
/// characters; it does not escape `<` or `>` to `\u003c`/`\u003e` unless
/// a custom HTML-safe serializer is configured. XSS prevention for
/// `source_file` values is the responsibility of the rendering layer: the
/// frontend must assign the value to `textContent` (or equivalent) rather
/// than inserting it into `innerHTML`.
///
/// This test confirms that the JSON value round-trips correctly and that
/// the literal filename content is present in the serialized output.
#[test]
fn t_reg_021_xss_filename_is_literal_string() {
    let filename = r#"<script>alert('xss')</script>.pdf"#;

    // Serialize a JSON object with the XSS-carrying filename as the
    // source_file field value and deserialize it back.
    let dto = serde_json::json!({ "source_file": filename });
    let serialized = dto.to_string();

    // The deserialized value must equal the original filename without any
    // modification. serde_json preserves angle brackets verbatim.
    let roundtripped: serde_json::Value =
        serde_json::from_str(&serialized).expect("serialized JSON must parse");
    let recovered = roundtripped["source_file"]
        .as_str()
        .expect("source_file must be a string");
    assert_eq!(
        recovered, filename,
        "source_file value must round-trip through JSON without modification"
    );

    // Confirm the JSON string does contain the filename content (possibly
    // with angle brackets literal or escaped -- either is acceptable here;
    // the rendering layer handles XSS prevention).
    assert!(
        serialized.contains("alert"),
        "serialized JSON must contain the filename content; got: {serialized}"
    );
}
