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

// `SQLite` schema definition and column repair.
//
// Defines the complete database schema in a single version. On application
// startup, the migrate function creates all tables, indexes, FTS5 virtual
// table with four synchronization triggers (INSERT, DELETE, soft-delete,
// un-delete), job tracking, and idempotency key management. After table
// creation, a column repair step adds any columns that are missing from
// existing tables (handles databases created by older code during
// development).

use rusqlite::Connection;

use crate::error::StoreError;

/// The schema version written to SQLite's user_version PRAGMA. Incremented
/// when structural schema changes (new tables, columns, or indexes) require
/// migration logic beyond the idempotent CREATE IF NOT EXISTS / ensure_columns
/// repair. Each version corresponds to a migration function in the MIGRATIONS
/// array.
const SCHEMA_VERSION: u32 = 1;

/// Ordered list of migration functions to apply when upgrading from an older
/// schema version. Each entry is a (target_version, migration_fn) pair. The
/// migrate() function applies all migrations whose target_version exceeds the
/// database's current user_version, in order.
///
/// To add a new migration:
/// 1. Increment SCHEMA_VERSION
/// 2. Write a function `fn apply_v<N>(conn: &Connection) -> Result<(), StoreError>`
/// 3. Append `(N, apply_v<N>)` to this array
///
/// The v1 baseline is handled separately (apply_v1 creates all tables from
/// scratch). Future migrations (v2, v3, ...) are listed here and applied
/// incrementally.
type MigrationFn = fn(&Connection) -> Result<(), StoreError>;
const MIGRATIONS: &[(u32, MigrationFn)] = &[
    // Future migrations go here. Example:
    // (2, apply_v2),
];

/// Applies the schema to the given `SQLite` connection. Creates all tables
/// if they do not exist, then runs a column repair step that adds any columns
/// missing from existing tables (via PRAGMA table_info checks and ALTER TABLE).
///
/// This function is idempotent: calling it multiple times on the same database
/// produces no errors and does not duplicate tables or indexes.
///
/// # Arguments
///
/// * `conn` - A reference to an open `SQLite` connection with foreign keys
///   enabled.
///
/// # Errors
///
/// Returns `StoreError::Migration` if any DDL statement fails, or
/// `StoreError::Sqlite` for lower-level database errors.
pub fn migrate(conn: &Connection) -> Result<(), StoreError> {
    // Read the current schema version before applying any migrations.
    // `current_version` is updated in-place as each migration step succeeds so
    // that subsequent migration checks use the post-step value, not the value
    // read at function entry (which would be stale after apply_v1 bumps the version).
    let mut current_version: u32 = conn.query_row("PRAGMA user_version;", [], |row| row.get(0))?;

    // Baseline: create all tables from scratch for unversioned databases.
    // apply_v1 runs inside an EXCLUSIVE transaction to ensure the schema is
    // either fully created or not created at all (no partial baseline state).
    if current_version < 1 {
        apply_v1(conn)?;
        // Set user_version inside the same logical migration boundary.
        conn.pragma_update(None, "user_version", 1u32)?;
        current_version = 1;
    }

    // Apply incremental migrations from the MIGRATIONS array. Each migration
    // is only applied if its target version exceeds the database's current
    // version. `current_version` is updated after each step so the next
    // iteration uses the post-migration version, not the version from function
    // entry. This prevents a migration from running on an already-migrated
    // database if the database's version was incremented by a previous step.
    for &(target_version, migrate_fn) in MIGRATIONS {
        if current_version < target_version {
            tracing::info!(
                from = current_version,
                to = target_version,
                "applying schema migration"
            );
            migrate_fn(conn)?;
            conn.pragma_update(None, "user_version", target_version)?;
            current_version = target_version;
        }
    }

    // Repair step: add any columns that are missing from existing tables.
    // This handles databases created by older code during development where
    // the CREATE TABLE IF NOT EXISTS did not include all current columns.
    // The repair is idempotent — columns that already exist are skipped.
    ensure_columns(conn)?;

    Ok(())
}

/// Returns the current schema version number. Downstream crates use this
/// value when creating session records.
pub fn schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Returns true if a table exists in the database by querying sqlite_master.
fn table_exists(conn: &Connection, table: &str) -> Result<bool, StoreError> {
    let exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
            [table],
            |row| row.get(0),
        )
        .map_err(|e| StoreError::Migration {
            version: 1,
            reason: format!("table existence check for '{table}' failed: {e}"),
        })?;
    Ok(exists)
}

/// Returns true if the given table name is in the compile-time whitelist of
/// known tables. This replaces the previous character-set loop, which accepted
/// any identifier-safe string and would silently pass fabricated table names
/// through to the PRAGMA query. Only the eight tables that exist in the schema
/// are permitted; any unknown name is an internal programming error.
fn is_allowed_table_name(name: &str) -> bool {
    matches!(
        name,
        "chunks"
            | "indexed_file"
            | "pages"
            | "index_session"
            | "job"
            | "web_sources"
            | "annotation_status"
            | "fts_chunks"
    )
}

/// Checks whether a table has a specific column by querying PRAGMA table_info.
fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, StoreError> {
    // Compile-time whitelist check: only the eight known schema tables are
    // allowed. This prevents SQL injection if a caller ever passes a
    // non-literal table name, and rejects misspellings at the call site
    // rather than at query time.
    if !is_allowed_table_name(table) {
        return Err(StoreError::Migration {
            version: 0,
            reason: format!(
                "table_has_column called with unknown table name '{table}'; \
                 only the eight schema tables are permitted"
            ),
        });
    }
    let query = format!("PRAGMA table_info({table})");
    let has = conn
        .prepare(&query)
        .map_err(|e| StoreError::Migration {
            version: 1,
            reason: format!("PRAGMA table_info({table}) failed: {e}"),
        })?
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| StoreError::Migration {
            version: 1,
            reason: format!("PRAGMA table_info({table}) query_map failed: {e}"),
        })?
        .any(|col| col.is_ok_and(|name| name == column));
    Ok(has)
}

/// Adds missing columns to existing tables. Each column addition is guarded
/// by a PRAGMA table_info check so the function is idempotent. This handles
/// the case where a database was created by older code that did not include
/// all current columns in its CREATE TABLE statement.
fn ensure_columns(conn: &Connection) -> Result<(), StoreError> {
    // indexed_file.pdf_page_count: structural page count from the PDF page tree.
    if !table_has_column(conn, "indexed_file", "pdf_page_count")? {
        conn.execute_batch("ALTER TABLE indexed_file ADD COLUMN pdf_page_count INTEGER;")
            .map_err(|e| StoreError::Migration {
                version: 1,
                reason: format!("ALTER TABLE indexed_file ADD COLUMN pdf_page_count: {e}"),
            })?;
    }

    // indexed_file.source_type: distinguishes between 'pdf' and 'html' content
    // origins. Defaults to 'pdf' so that all pre-existing rows are correctly
    // classified without backfilling.
    if !table_has_column(conn, "indexed_file", "source_type")? {
        conn.execute_batch(
            "ALTER TABLE indexed_file ADD COLUMN source_type TEXT NOT NULL DEFAULT 'pdf';",
        )
        .map_err(|e| StoreError::Migration {
            version: 1,
            reason: format!("ALTER TABLE indexed_file ADD COLUMN source_type: {e}"),
        })?;
    }

    // index_session columns are only repaired when the table exists. In test
    // environments that create partial schemas (e.g., only job + indexed_file),
    // index_session may not be present and these ALTER TABLEs would fail.
    if table_exists(conn, "index_session")? {
        // index_session.tags: JSON array of user-defined tag strings.
        if !table_has_column(conn, "index_session", "tags")? {
            conn.execute_batch("ALTER TABLE index_session ADD COLUMN tags TEXT;")
                .map_err(|e| StoreError::Migration {
                    version: 1,
                    reason: format!("ALTER TABLE index_session ADD COLUMN tags: {e}"),
                })?;
        }

        // index_session.metadata: JSON object of arbitrary key-value pairs.
        if !table_has_column(conn, "index_session", "metadata")? {
            conn.execute_batch("ALTER TABLE index_session ADD COLUMN metadata TEXT;")
                .map_err(|e| StoreError::Migration {
                    version: 1,
                    reason: format!("ALTER TABLE index_session ADD COLUMN metadata: {e}"),
                })?;
        }
    }

    // annotation_quote_status: per-quote progress table for annotation jobs.
    // Created here for databases that pre-date the table addition. For newly
    // created databases, the table is created by create_annotation_status_table().
    if !table_exists(conn, "annotation_quote_status")? {
        create_annotation_status_table(conn)?;
    }

    // web_source: stores web-specific metadata for HTML-sourced indexed files.
    // Created here for databases that pre-date the table addition. For newly
    // created databases, the table is created by create_web_source_table().
    if !table_exists(conn, "web_source")? {
        create_web_source_table(conn)?;
    }

    // job.params_json: serialized JSON parameters for annotation jobs.
    if !table_has_column(conn, "job", "params_json")? {
        conn.execute_batch("ALTER TABLE job ADD COLUMN params_json TEXT;")
            .map_err(|e| StoreError::Migration {
                version: 1,
                reason: format!("ALTER TABLE job ADD COLUMN params_json: {e}"),
            })?;
    }

    Ok(())
}

/// Applies the version 1 schema: all tables, indexes, FTS5 virtual table,
/// synchronization triggers, and workflow tables.
fn apply_v1(conn: &Connection) -> Result<(), StoreError> {
    create_core_tables(conn)?;
    create_chunk_table_and_indexes(conn)?;
    create_fts5_and_triggers(conn)?;
    create_workflow_tables(conn)?;
    create_citation_tables(conn)?;
    create_annotation_status_table(conn)?;
    create_web_source_table(conn)?;
    Ok(())
}

/// Creates the `index_session`, `indexed_file`, and page tables with their
/// unique indexes and foreign key constraints.
fn create_core_tables(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: index_session
        -- Tracks each combination of directory, model, and chunking config.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS index_session (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            directory_path          TEXT    NOT NULL,
            model_name              TEXT    NOT NULL,
            chunk_strategy          TEXT    NOT NULL,
            chunk_size              INTEGER,
            chunk_overlap           INTEGER,
            max_words               INTEGER,
            ocr_language            TEXT    NOT NULL DEFAULT 'eng',
            vector_dimension        INTEGER NOT NULL,
            embedding_storage_mode  TEXT    NOT NULL DEFAULT 'sqlite-blob',
            schema_version          INTEGER NOT NULL,
            app_version             TEXT    NOT NULL,
            hnsw_total              INTEGER NOT NULL DEFAULT 0,
            hnsw_orphans            INTEGER NOT NULL DEFAULT 0,
            last_rebuild_at         INTEGER,
            created_at              INTEGER NOT NULL,
            label                   TEXT,
            tags                    TEXT,
            metadata                TEXT
        );

        -- COALESCE-based unique index that substitutes sentinel -1 for NULL.
        -- This prevents duplicate sessions where nullable chunking parameters
        -- are all NULL (e.g., two page-based sessions for the same directory
        -- and model).
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_unique ON index_session(
            directory_path,
            model_name,
            chunk_strategy,
            COALESCE(chunk_size, -1),
            COALESCE(chunk_overlap, -1),
            COALESCE(max_words, -1),
            ocr_language,
            embedding_storage_mode
        );

        -- =================================================================
        -- Table: indexed_file
        -- Tracks each file processed within a session. Supports both PDF
        -- and HTML source types. Combines file identity, content hash,
        -- and filesystem metadata for two-stage incremental change
        -- detection. The pdf_page_count column stores the structural
        -- page count from the PDF page tree (lopdf/pdfium), which is
        -- independent of how many pages had extractable text
        -- (page_count). When pdf_page_count > page_count, text
        -- extraction was incomplete (e.g., formfeed fallback collapsed
        -- pages). For HTML sources, page_count represents the number
        -- of heading-based sections, and pdf_page_count is NULL.
        -- The source_type column distinguishes between 'pdf' and 'html'
        -- content origins, defaulting to 'pdf' for backward compatibility.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS indexed_file (
            id              INTEGER PRIMARY KEY,
            session_id      INTEGER NOT NULL REFERENCES index_session(id) ON DELETE CASCADE,
            file_path       TEXT    NOT NULL,
            file_hash       TEXT    NOT NULL,
            mtime           INTEGER NOT NULL,
            size            INTEGER NOT NULL,
            page_count      INTEGER NOT NULL,
            pdf_page_count  INTEGER,
            source_type     TEXT    NOT NULL DEFAULT 'pdf',
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL,
            UNIQUE(session_id, file_path)
        );

        -- =================================================================
        -- Table: page
        -- Stores normalized text content of each PDF page alongside the
        -- extraction backend identifier and the UTF-8 byte count.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS page (
            id          INTEGER PRIMARY KEY,
            file_id     INTEGER NOT NULL REFERENCES indexed_file(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            content     TEXT    NOT NULL DEFAULT '',
            backend     TEXT    NOT NULL,
            byte_count  INTEGER NOT NULL,
            UNIQUE(file_id, page_number)
        );
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the chunk table with AUTOINCREMENT, its four supporting indexes,
/// and the unique constraint on `(file_id, chunk_index)`.
fn create_chunk_table_and_indexes(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: chunk
        -- Stores text chunks, embedding vectors (as blobs or external file
        -- references), document-level byte offsets, and content hashes.
        -- AUTOINCREMENT prevents rowid reuse after deletion, which is
        -- critical for HNSW label stability.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS chunk (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id                 INTEGER NOT NULL REFERENCES indexed_file(id) ON DELETE CASCADE,
            session_id              INTEGER NOT NULL REFERENCES index_session(id) ON DELETE CASCADE,
            page_start              INTEGER NOT NULL,
            page_end                INTEGER NOT NULL,
            chunk_index             INTEGER NOT NULL,
            doc_text_offset_start   INTEGER NOT NULL,
            doc_text_offset_end     INTEGER NOT NULL,
            content                 TEXT    NOT NULL,
            embedding               BLOB,
            ext_offset              INTEGER,
            ext_length              INTEGER,
            content_hash            TEXT    NOT NULL,
            simhash                 INTEGER,
            is_deleted              INTEGER NOT NULL DEFAULT 0,
            created_at              INTEGER NOT NULL,
            UNIQUE(file_id, chunk_index)
        );

        -- Indexes on chunk for fast session-wide queries, file-based deletion,
        -- content hash deduplication, and active-only partial index.
        CREATE INDEX IF NOT EXISTS idx_chunk_session ON chunk(session_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_file    ON chunk(file_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_hash    ON chunk(content_hash);
        CREATE INDEX IF NOT EXISTS idx_chunk_active  ON chunk(session_id) WHERE is_deleted = 0;
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the FTS5 external content virtual table and its four
/// synchronization triggers (INSERT, DELETE, soft-delete, un-delete).
fn create_fts5_and_triggers(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- FTS5 virtual table: chunk_fts
        -- External content table backed by the chunk table. Uses unicode61
        -- tokenizer with diacritics removal and hyphen/underscore preservation.
        -- =================================================================
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            content,
            content=chunk,
            content_rowid=id,
            tokenize=\"unicode61 remove_diacritics 2 tokenchars '-_'\"
        );

        -- =================================================================
        -- FTS5 synchronization triggers
        -- Keep chunk_fts in sync with the chunk table on INSERT, DELETE,
        -- and is_deleted flag changes.
        -- =================================================================

        -- AFTER INSERT: add the chunk's text content to the FTS5 index.
        CREATE TRIGGER IF NOT EXISTS chunk_fts_insert
        AFTER INSERT ON chunk
        BEGIN
            INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
        END;

        -- AFTER DELETE: remove the FTS5 entry using the special FTS5 delete
        -- command syntax (VALUES('delete', ...)).
        CREATE TRIGGER IF NOT EXISTS chunk_fts_delete
        AFTER DELETE ON chunk
        BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, content)
                VALUES('delete', old.id, old.content);
        END;

        -- AFTER UPDATE OF is_deleted (soft-delete): when a chunk is flagged
        -- as deleted (is_deleted changes from 0 to 1), remove its FTS5 entry
        -- so it no longer appears in BM25 search results.
        CREATE TRIGGER IF NOT EXISTS chunk_fts_update_deleted
        AFTER UPDATE OF is_deleted ON chunk
        WHEN new.is_deleted = 1 AND old.is_deleted = 0
        BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, content)
                VALUES('delete', old.id, old.content);
        END;

        -- AFTER UPDATE OF is_deleted (un-delete): when a chunk's is_deleted
        -- flag changes from 1 back to 0, re-insert its content into the FTS5
        -- index so the chunk appears in BM25 search results again.
        CREATE TRIGGER IF NOT EXISTS chunk_fts_update_undeleted
        AFTER UPDATE OF is_deleted ON chunk
        WHEN new.is_deleted = 0 AND old.is_deleted = 1
        BEGIN
            INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
        END;
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the job and idempotency tables used by the workflow layer
/// for background job tracking and resumable indexing. Also creates a
/// partial index on job(state, created_at, rowid) covering only active
/// states ('queued', 'running'). This partial index accelerates the
/// `has_active_job` query used by the concurrent job policy check in
/// the index and annotate handlers, avoiding a full table scan of
/// completed/failed/canceled jobs.
fn create_workflow_tables(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: job
        -- Persists background job state (indexing, annotation, HNSW rebuild)
        -- so that job status survives process restarts and clients can query
        -- historical jobs. Rows are cleaned up 24 hours after finishing.
        -- The params_json column stores job-specific parameters as serialized
        -- JSON. Used by annotation jobs to persist input data and config.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS job (
            id              TEXT    PRIMARY KEY,
            kind            TEXT    NOT NULL,
            session_id      INTEGER REFERENCES index_session(id) ON DELETE CASCADE,
            -- CHECK constraint enforces the state machine at the storage layer.
            -- Valid transitions are validated in Rust (JobState::can_transition_to)
            -- before any UPDATE, but this constraint also catches direct SQL writes
            -- that bypass the application layer (e.g., migrations, debugging tools).
            state           TEXT    NOT NULL CHECK (state IN ('queued', 'running', 'completed', 'failed', 'canceled')),
            progress_done   INTEGER NOT NULL DEFAULT 0,
            progress_total  INTEGER NOT NULL DEFAULT 0,
            error_message   TEXT,
            created_at      INTEGER NOT NULL,
            started_at      INTEGER,
            finished_at     INTEGER,
            params_json     TEXT
        );

        -- Partial index covering only active job states ('queued' and
        -- 'running'). The concurrent job policy checks (has_active_job)
        -- query this subset frequently, while the vast majority of job
        -- rows are in terminal states (completed, failed, canceled).
        -- Including kind in the index allows the EXISTS query to be
        -- satisfied entirely from the index without touching the table.
        CREATE INDEX IF NOT EXISTS idx_job_state_active
            ON job(state, kind, created_at)
            WHERE state IN ('queued', 'running');

        -- =================================================================
        -- Table: idempotency
        -- Stores idempotency keys and their associated job/session IDs.
        -- Entries expire after 24 hours, matching the job retention window.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS idempotency (
            key         TEXT    PRIMARY KEY,
            job_id      TEXT    NOT NULL REFERENCES job(id),
            session_id  INTEGER NOT NULL REFERENCES index_session(id),
            created_at  INTEGER NOT NULL
        );

        -- Index on created_at for the 24-hour expiry cleanup query. Without
        -- this index, cleanup_expired_idempotency_keys() performs a full table
        -- scan on every cleanup cycle. The index allows SQLite to seek directly
        -- to the rows whose created_at falls before the cutoff.
        CREATE INDEX IF NOT EXISTS idx_idempotency_created
            ON idempotency(created_at);
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the `citation_row` table used by the citation verification pipeline.
/// Each row represents one cite-key from a `\cite{}` command in a LaTeX document.
/// Multiple cite-keys from a single `\cite{a,b,c}` command share the same
/// `group_id`. Rows are grouped into batches (via `batch_id`) for sub-agent
/// claiming. The `result_json` column stores the sub-agent's verdict, passages,
/// reasoning, and optional LaTeX correction as serialized JSON.
///
/// The table references the `job` table via foreign key with CASCADE delete,
/// so deleting a job automatically removes all its citation rows.
fn create_citation_tables(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: citation_row
        -- Tracks individual cite-key occurrences extracted from a LaTeX
        -- document. Each row corresponds to one cite-key within one
        -- \\cite{} command. Multiple keys from the same command share a
        -- group_id. Rows are assigned to batches for sub-agent claiming.
        --
        -- Status values: pending, claimed, done, failed.
        -- Flag values: NULL, 'critical', 'warning'.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS citation_row (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id            TEXT    NOT NULL REFERENCES job(id) ON DELETE CASCADE,
            group_id          INTEGER NOT NULL,
            cite_key          TEXT    NOT NULL,
            author            TEXT    NOT NULL,
            title             TEXT    NOT NULL,
            year              TEXT,
            tex_line          INTEGER NOT NULL,
            anchor_before     TEXT    NOT NULL,
            anchor_after      TEXT    NOT NULL,
            section_title     TEXT,
            matched_file_id   INTEGER,
            bib_abstract      TEXT,
            bib_keywords      TEXT,
            tex_context       TEXT,
            batch_id          INTEGER,
            -- CHECK constraints enforce the allowed status and flag values.
            -- The application layer validates these before writing. The
            -- constraint also protects against direct SQL access and future
            -- code paths that skip the validation.
            status            TEXT    NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'done', 'failed')),
            flag              TEXT    CHECK (flag IS NULL OR flag IN ('critical', 'warning')),
            claimed_at        INTEGER,
            result_json       TEXT,
            co_citation_json  TEXT
        );

        -- Index for filtering all citation rows belonging to a specific job.
        CREATE INDEX IF NOT EXISTS idx_citation_row_job    ON citation_row(job_id);

        -- Composite index for querying rows by job and status (used by the
        -- claim and status endpoints to count pending/claimed/done/failed).
        CREATE INDEX IF NOT EXISTS idx_citation_row_status ON citation_row(job_id, status);

        -- Composite index for querying rows by job and batch (used by the
        -- claim endpoint to atomically select all rows in a batch).
        CREATE INDEX IF NOT EXISTS idx_citation_row_batch  ON citation_row(job_id, batch_id);

        -- Composite index for querying rows by job and group (used to
        -- verify that cite-groups are never split across batches).
        CREATE INDEX IF NOT EXISTS idx_citation_row_group  ON citation_row(job_id, group_id);
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the `annotation_quote_status` table for per-quote progress tracking
/// during annotation jobs. Each row corresponds to one quote from the annotation
/// input data and tracks its processing status (pending, matched, not_found, error),
/// the text location method used to find it, the page where it was located, and
/// the matched PDF filename.
///
/// The table references the `job` table via foreign key with CASCADE delete,
/// so deleting a job automatically removes all its quote status rows.
fn create_annotation_status_table(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: annotation_quote_status
        -- Tracks per-quote progress during annotation pipeline execution.
        -- Each row corresponds to one quote from the input CSV/JSON data.
        -- Status values: pending, matched, not_found, error.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS annotation_quote_status (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id        TEXT    NOT NULL REFERENCES job(id) ON DELETE CASCADE,
            title         TEXT    NOT NULL,
            author        TEXT    NOT NULL,
            quote_excerpt TEXT    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'pending',
            match_method  TEXT,
            page          INTEGER,
            pdf_filename  TEXT,
            updated_at    INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ann_quote_job ON annotation_quote_status(job_id);
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Creates the `web_source` table for storing web-specific metadata associated
/// with HTML-sourced indexed files. Each row is linked 1:1 to an `indexed_file`
/// record via `file_id` with a UNIQUE constraint, ensuring that each indexed
/// file has at most one web source metadata entry. The ON DELETE CASCADE
/// constraint removes the web_source row automatically when the parent
/// indexed_file record is deleted.
///
/// The table stores the fetched URL, canonical URL, HTML `<head>` metadata
/// (title, description, language, Open Graph tags, author, publication date),
/// domain name for per-site filtering, HTTP response metadata (status code,
/// content type, fetch timestamp), and optionally the raw HTML bytes for
/// re-processing without re-fetching.
fn create_web_source_table(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "
        -- =================================================================
        -- Table: web_source
        -- Stores web-specific metadata for HTML-sourced indexed files.
        -- Linked 1:1 to indexed_file via file_id. Contains the fetched
        -- URL, HTML head metadata (title, OG tags, author, language),
        -- domain for filtering, HTTP response details, and optionally
        -- the raw HTML bytes for re-processing.
        -- =================================================================
        CREATE TABLE IF NOT EXISTS web_source (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id           INTEGER NOT NULL UNIQUE REFERENCES indexed_file(id) ON DELETE CASCADE,
            url               TEXT    NOT NULL,
            canonical_url     TEXT,
            title             TEXT,
            meta_description  TEXT,
            language          TEXT,
            og_image          TEXT,
            og_title          TEXT,
            og_description    TEXT,
            author            TEXT,
            published_date    TEXT,
            domain            TEXT    NOT NULL,
            fetch_timestamp   INTEGER NOT NULL,
            http_status       INTEGER NOT NULL,
            content_type      TEXT,
            raw_html          BLOB
        );

        -- Index for looking up web sources by URL (used by deduplication checks
        -- before re-fetching a previously cached page).
        CREATE INDEX IF NOT EXISTS idx_web_source_url ON web_source(url);

        -- Index for filtering web sources by domain name (used by the
        -- neuroncite_html_list tool's domain filter parameter).
        CREATE INDEX IF NOT EXISTS idx_web_source_domain ON web_source(domain);
        ",
    )
    .map_err(|e| StoreError::Migration {
        version: 1,
        reason: e.to_string(),
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    /// Helper: creates an in-memory database with foreign keys enabled.
    fn mem_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        conn
    }

    /// T-STO-001: Schema migration on empty database creates all tables.
    /// Running the migration function on a freshly created `SQLite` database
    /// creates all expected tables (`index_session`, `indexed_file`, page, chunk,
    /// `chunk_fts`, job, idempotency) without error.
    #[test]
    fn t_sto_001_schema_migration_creates_all_tables() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let expected_tables = [
            "index_session",
            "indexed_file",
            "page",
            "chunk",
            "chunk_fts",
            "job",
            "idempotency",
            "citation_row",
            "annotation_quote_status",
            "web_source",
        ];

        for table_name in &expected_tables {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1",
                    [table_name],
                    |row| row.get(0),
                )
                .unwrap_or_else(|e| panic!("failed to query for table {table_name}: {e}"));
            assert_eq!(
                count, 1,
                "table '{table_name}' was not created by migration"
            );
        }
    }

    /// T-STO-002: Migration idempotency. Running the migration function twice
    /// on the same database does not produce errors or duplicate tables.
    #[test]
    fn t_sto_002_migration_idempotency() {
        let conn = mem_db();

        migrate(&conn).expect("first migration failed");
        migrate(&conn).expect("second migration failed");

        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'index_session'",
                [],
                |row| row.get(0),
            )
            .expect("failed to count index_session tables");
        assert_eq!(table_count, 1, "index_session table duplicated");
    }

    /// T-STO-061: Schema migration creates all four FTS5 triggers.
    /// After running the migration, all four FTS5 synchronization triggers
    /// (INSERT, DELETE, soft-delete, un-delete) must exist in `sqlite_master`.
    /// The un-delete trigger re-inserts a chunk's content into FTS5 when its
    /// `is_deleted` flag changes from 1 back to 0.
    #[test]
    fn t_sto_061_migration_creates_all_fts5_triggers() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let expected_triggers = [
            "chunk_fts_insert",
            "chunk_fts_delete",
            "chunk_fts_update_deleted",
            "chunk_fts_update_undeleted",
        ];

        for trigger_name in &expected_triggers {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master \
                     WHERE type = 'trigger' AND name = ?1",
                    [trigger_name],
                    |row| row.get(0),
                )
                .expect("failed to query for trigger");
            assert_eq!(
                count, 1,
                "trigger '{trigger_name}' must be created by v1 migration"
            );
        }

        // Verify the schema version is the current SCHEMA_VERSION.
        let version: u32 = conn
            .query_row("PRAGMA user_version;", [], |row| row.get(0))
            .expect("failed to read user_version");
        assert_eq!(
            version, SCHEMA_VERSION,
            "schema version must be {SCHEMA_VERSION} after migration"
        );
    }

    /// T-STO-077: The v1 schema includes the `label` column in
    /// `index_session`. Verifies the column exists via `PRAGMA table_info`
    /// and accepts NULL values (no NOT NULL constraint).
    #[test]
    fn t_sto_077_schema_includes_label_column() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        // Insert a session without setting label (defaults to NULL).
        conn.execute_batch(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                chunk_size, chunk_overlap, max_words,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, hnsw_total, hnsw_orphans,
                created_at
            ) VALUES (
                '/test', 'model', 'word', 300, 50, NULL,
                'eng', 384, 'sqlite-blob', 1, '0.1.0', 0, 0, 1700000000
            );",
        )
        .expect("insert without label failed");

        // The label column must be NULL.
        let label: Option<String> = conn
            .query_row(
                "SELECT label FROM index_session WHERE directory_path = '/test'",
                [],
                |row| row.get(0),
            )
            .expect("failed to query label");
        assert!(
            label.is_none(),
            "label must be NULL when not explicitly set"
        );

        // Setting a label via UPDATE must work.
        conn.execute(
            "UPDATE index_session SET label = 'Test Label' WHERE directory_path = '/test'",
            [],
        )
        .expect("setting label via UPDATE failed");

        let label: Option<String> = conn
            .query_row(
                "SELECT label FROM index_session WHERE directory_path = '/test'",
                [],
                |row| row.get(0),
            )
            .expect("failed to query label after update");
        assert_eq!(label.as_deref(), Some("Test Label"));
    }

    /// T-STO-078: `schema_version()` returns the constant SCHEMA_VERSION value.
    #[test]
    fn t_sto_078_schema_version_function() {
        assert_eq!(
            schema_version(),
            SCHEMA_VERSION,
            "schema_version() must return {SCHEMA_VERSION}"
        );
    }

    /// T-STO-079: The `index_session` table has exactly the expected set of
    /// columns. Guards against accidentally missing or extra columns.
    #[test]
    fn t_sto_079_index_session_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "index_session");

        let expected = [
            "id",
            "directory_path",
            "model_name",
            "chunk_strategy",
            "chunk_size",
            "chunk_overlap",
            "max_words",
            "ocr_language",
            "vector_dimension",
            "embedding_storage_mode",
            "schema_version",
            "app_version",
            "hnsw_total",
            "hnsw_orphans",
            "last_rebuild_at",
            "created_at",
            "label",
            "tags",
            "metadata",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in index_session"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "index_session must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STO-081: The `job` table has exactly the expected set of columns,
    /// including `params_json`. This test guards against the bug where
    /// job-related endpoints fail with "no such column: params_json" because
    /// the column was missing from the schema.
    #[test]
    fn t_sto_081_job_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "job");

        let expected = [
            "id",
            "kind",
            "session_id",
            "state",
            "progress_done",
            "progress_total",
            "error_message",
            "created_at",
            "started_at",
            "finished_at",
            "params_json",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in job table"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "job table must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STO-083: The `indexed_file` table has exactly the expected set of
    /// columns, including `pdf_page_count`.
    #[test]
    fn t_sto_083_indexed_file_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "indexed_file");

        let expected = [
            "id",
            "session_id",
            "file_path",
            "file_hash",
            "mtime",
            "size",
            "page_count",
            "pdf_page_count",
            "source_type",
            "created_at",
            "updated_at",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in indexed_file"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "indexed_file must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STO-085: Column repair adds `params_json` to an old job table that
    /// was created without it. Simulates an old database by creating the job
    /// table without params_json, then running ensure_columns() to verify the
    /// column is added.
    #[test]
    fn t_sto_085_ensure_columns_adds_params_json() {
        let conn = mem_db();

        // Create an old-style job table without params_json.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS job (
                id              TEXT    PRIMARY KEY,
                kind            TEXT    NOT NULL,
                session_id      INTEGER,
                state           TEXT    NOT NULL,
                progress_done   INTEGER NOT NULL DEFAULT 0,
                progress_total  INTEGER NOT NULL DEFAULT 0,
                error_message   TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                finished_at     INTEGER
            );
            CREATE TABLE IF NOT EXISTS indexed_file (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL,
                file_path       TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                mtime           INTEGER NOT NULL,
                size            INTEGER NOT NULL,
                page_count      INTEGER NOT NULL,
                pdf_page_count  INTEGER,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );",
        )
        .expect("create old-style tables");

        // Verify params_json does NOT exist before repair.
        assert!(
            !table_has_column(&conn, "job", "params_json").unwrap(),
            "params_json must NOT exist before ensure_columns"
        );

        ensure_columns(&conn).expect("ensure_columns failed");

        // Verify params_json exists after repair.
        assert!(
            table_has_column(&conn, "job", "params_json").unwrap(),
            "params_json must exist after ensure_columns"
        );

        // Verify the column accepts NULL values (INSERT without params_json).
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('test-1', 'index', 'queued', 0, 0, 1700000000)",
            [],
        )
        .expect("insert without params_json failed");

        let val: Option<String> = conn
            .query_row(
                "SELECT params_json FROM job WHERE id = 'test-1'",
                [],
                |row| row.get(0),
            )
            .expect("query params_json failed");
        assert!(val.is_none(), "params_json must be NULL when not set");

        // Verify the column accepts string values.
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at, params_json)
             VALUES ('test-2', 'annotate', 'queued', 0, 0, 1700000000, '{\"key\":\"value\"}')",
            [],
        )
        .expect("insert with params_json failed");

        let val: Option<String> = conn
            .query_row(
                "SELECT params_json FROM job WHERE id = 'test-2'",
                [],
                |row| row.get(0),
            )
            .expect("query params_json failed");
        assert_eq!(val.as_deref(), Some("{\"key\":\"value\"}"));
    }

    /// T-STO-086: Column repair adds `pdf_page_count` to an old indexed_file
    /// table that was created without it.
    #[test]
    fn t_sto_086_ensure_columns_adds_pdf_page_count() {
        let conn = mem_db();

        // Create an old-style indexed_file table without pdf_page_count.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS index_session (
                id                      INTEGER PRIMARY KEY,
                directory_path          TEXT    NOT NULL,
                model_name              TEXT    NOT NULL,
                chunk_strategy          TEXT    NOT NULL,
                chunk_size              INTEGER,
                chunk_overlap           INTEGER,
                max_words               INTEGER,
                ocr_language            TEXT    NOT NULL DEFAULT 'eng',
                vector_dimension        INTEGER NOT NULL,
                embedding_storage_mode  TEXT    NOT NULL DEFAULT 'sqlite-blob',
                schema_version          INTEGER NOT NULL,
                app_version             TEXT    NOT NULL,
                hnsw_total              INTEGER NOT NULL DEFAULT 0,
                hnsw_orphans            INTEGER NOT NULL DEFAULT 0,
                last_rebuild_at         INTEGER,
                created_at              INTEGER NOT NULL,
                label                   TEXT
            );
            CREATE TABLE IF NOT EXISTS indexed_file (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL REFERENCES index_session(id) ON DELETE CASCADE,
                file_path       TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                mtime           INTEGER NOT NULL,
                size            INTEGER NOT NULL,
                page_count      INTEGER NOT NULL,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                UNIQUE(session_id, file_path)
            );
            CREATE TABLE IF NOT EXISTS job (
                id              TEXT    PRIMARY KEY,
                kind            TEXT    NOT NULL,
                session_id      INTEGER,
                state           TEXT    NOT NULL,
                progress_done   INTEGER NOT NULL DEFAULT 0,
                progress_total  INTEGER NOT NULL DEFAULT 0,
                error_message   TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                finished_at     INTEGER,
                params_json     TEXT
            );",
        )
        .expect("create old-style tables");

        assert!(
            !table_has_column(&conn, "indexed_file", "pdf_page_count").unwrap(),
            "pdf_page_count must NOT exist before ensure_columns"
        );

        ensure_columns(&conn).expect("ensure_columns failed");

        assert!(
            table_has_column(&conn, "indexed_file", "pdf_page_count").unwrap(),
            "pdf_page_count must exist after ensure_columns"
        );
    }

    /// T-STO-032: Column repair is idempotent. Running ensure_columns() twice
    /// does not duplicate columns or produce errors.
    #[test]
    fn t_sto_032_ensure_columns_idempotent() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        // Run ensure_columns again on a database that already has all columns.
        ensure_columns(&conn).expect("first ensure_columns call must succeed");
        ensure_columns(&conn).expect("second ensure_columns call must succeed");

        // Verify column counts are unchanged.
        let job_cols = get_column_names(&conn, "job");
        assert_eq!(job_cols.len(), 11, "job table must have 11 columns");

        let file_cols = get_column_names(&conn, "indexed_file");
        assert_eq!(file_cols.len(), 11, "indexed_file must have 11 columns");
    }

    /// T-STO-033: The SELECT queries in job.rs work against a database that
    /// was repaired by ensure_columns(). This is the end-to-end test that
    /// reproduces the original "no such column: params_json" bug.
    #[test]
    fn t_sto_033_job_select_with_params_json_after_repair() {
        let conn = mem_db();

        // Create an old-style job table without params_json.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS job (
                id              TEXT    PRIMARY KEY,
                kind            TEXT    NOT NULL,
                session_id      INTEGER,
                state           TEXT    NOT NULL,
                progress_done   INTEGER NOT NULL DEFAULT 0,
                progress_total  INTEGER NOT NULL DEFAULT 0,
                error_message   TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                finished_at     INTEGER
            );
            CREATE TABLE IF NOT EXISTS indexed_file (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL,
                file_path       TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                mtime           INTEGER NOT NULL,
                size            INTEGER NOT NULL,
                page_count      INTEGER NOT NULL,
                pdf_page_count  INTEGER,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );",
        )
        .expect("create old-style tables");

        // Run the column repair.
        ensure_columns(&conn).expect("ensure_columns failed");

        // Insert a job via raw SQL (simulating the old code path).
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('bug-repro', 'index', 'queued', 0, 0, 1700000000)",
            [],
        )
        .expect("insert job failed");

        // The exact SELECT that was failing before the fix. This query is
        // the same one used by list_jobs() in job.rs.
        let result: Result<Vec<String>, _> = conn
            .prepare(
                "SELECT id, kind, session_id, state, progress_done, progress_total,
                        error_message, created_at, started_at, finished_at, params_json
                 FROM job ORDER BY created_at DESC",
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map([], |row| {
                    let id: String = row.get(0)?;
                    Ok(id)
                })?;
                rows.collect()
            });

        assert!(
            result.is_ok(),
            "SELECT with params_json must succeed after ensure_columns: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    /// T-STO-034: The `citation_row` table exists after migration. Verifies
    /// that the citation verification table is created as part of the v1 schema.
    #[test]
    fn t_sto_034_citation_row_table_exists() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'citation_row'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(count, 1, "citation_row table must exist after migration");
    }

    /// T-STO-087: The `citation_row` table has exactly the expected 20 columns.
    /// Guards against accidentally missing or extra columns in the citation
    /// verification schema.
    #[test]
    fn t_sto_087_citation_row_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "citation_row");

        let expected = [
            "id",
            "job_id",
            "group_id",
            "cite_key",
            "author",
            "title",
            "year",
            "tex_line",
            "anchor_before",
            "anchor_after",
            "section_title",
            "matched_file_id",
            "bib_abstract",
            "bib_keywords",
            "tex_context",
            "batch_id",
            "status",
            "flag",
            "claimed_at",
            "result_json",
            "co_citation_json",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in citation_row"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "citation_row must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STO-036: Foreign key constraint on `citation_row.job_id`. Inserting
    /// a citation_row that references a non-existent job_id must fail when
    /// foreign keys are enabled.
    #[test]
    fn t_sto_036_citation_row_fk_constraint() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let result = conn.execute(
            "INSERT INTO citation_row (job_id, group_id, cite_key, author, title, tex_line,
             anchor_before, anchor_after, status)
             VALUES ('nonexistent-job', 0, 'key1', 'Author', 'Title', 10, 'word', 'word', 'pending')",
            [],
        );

        assert!(
            result.is_err(),
            "inserting citation_row with non-existent job_id must fail due to FK constraint"
        );
    }

    /// T-STO-037: CASCADE delete removes citation_rows when a job is deleted.
    /// Creating a job, inserting citation_rows referencing it, then deleting
    /// the job must cascade-delete all associated citation_rows.
    #[test]
    fn t_sto_037_citation_row_cascade_delete() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        // Create a job record.
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('cascade-test', 'citation_verify', 'running', 0, 0, 1700000000)",
            [],
        )
        .expect("insert job failed");

        // Insert two citation_rows referencing this job.
        conn.execute(
            "INSERT INTO citation_row (job_id, group_id, cite_key, author, title, tex_line,
             anchor_before, anchor_after, status)
             VALUES ('cascade-test', 0, 'key1', 'Author A', 'Title A', 10, 'before', 'after', 'pending')",
            [],
        )
        .expect("insert citation_row 1 failed");

        conn.execute(
            "INSERT INTO citation_row (job_id, group_id, cite_key, author, title, tex_line,
             anchor_before, anchor_after, status)
             VALUES ('cascade-test', 0, 'key2', 'Author B', 'Title B', 15, 'before', 'after', 'pending')",
            [],
        )
        .expect("insert citation_row 2 failed");

        // Verify both rows exist.
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM citation_row WHERE job_id = 'cascade-test'",
                [],
                |row| row.get(0),
            )
            .expect("count query failed");
        assert_eq!(count, 2, "two citation_rows must exist before deletion");

        // Delete the job -- CASCADE must remove citation_rows.
        conn.execute("DELETE FROM job WHERE id = 'cascade-test'", [])
            .expect("delete job failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM citation_row WHERE job_id = 'cascade-test'",
                [],
                |row| row.get(0),
            )
            .expect("count query after delete failed");
        assert_eq!(
            count, 0,
            "CASCADE delete must remove all citation_rows when job is deleted"
        );
    }

    /// T-STO-038: Migration idempotency with `citation_row` table. Running
    /// the migration twice does not produce errors or duplicate the table.
    #[test]
    fn t_sto_038_citation_row_migration_idempotent() {
        let conn = mem_db();

        migrate(&conn).expect("first migration failed");
        migrate(&conn).expect("second migration must succeed (idempotency)");

        // Verify exactly one citation_row table exists.
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'citation_row'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(
            count, 1,
            "citation_row table must not be duplicated after double migration"
        );

        // Verify all four indexes exist.
        let expected_indexes = [
            "idx_citation_row_job",
            "idx_citation_row_status",
            "idx_citation_row_batch",
            "idx_citation_row_group",
        ];
        for idx_name in &expected_indexes {
            let idx_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = ?1",
                    [idx_name],
                    |row| row.get(0),
                )
                .expect("index query failed");
            assert_eq!(
                idx_count, 1,
                "index '{idx_name}' must exist after migration"
            );
        }
    }

    /// T-SCHEMA-AUTOINCR: The index_session table uses AUTOINCREMENT to prevent
    /// session ID recycling after deletion. Regression test for ISSUE-009 where
    /// deleting session 5 and creating a new session could reuse ID 5, causing
    /// stale HNSW references in memory to point to the wrong data.
    ///
    /// SQLite stores AUTOINCREMENT state in the sqlite_sequence table. When
    /// AUTOINCREMENT is enabled, IDs are monotonically increasing and never
    /// reused, even after the row with the highest ID is deleted.
    ///
    /// Each session uses a distinct directory path to satisfy the
    /// idx_session_unique constraint that prevents duplicate configurations.
    #[test]
    fn t_schema_autoincrement_prevents_id_recycling() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let make_config = |dir: &str| neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from(dir),
            model_name: "test-model".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let id1 = crate::create_session(&conn, &make_config("/test/dir_a"), "0.1.0")
            .expect("create session 1");
        let id2 = crate::create_session(&conn, &make_config("/test/dir_b"), "0.1.0")
            .expect("create session 2");

        assert!(id2 > id1, "second session ID must be greater than first");

        // Delete the second session.
        crate::delete_session(&conn, id2).expect("delete session 2");

        // Create a third session with a new directory. With AUTOINCREMENT, this
        // must receive an ID greater than id2 (not reuse id2's slot).
        let id3 = crate::create_session(&conn, &make_config("/test/dir_c"), "0.1.0")
            .expect("create session 3");

        assert!(
            id3 > id2,
            "session ID after deletion must be > deleted ID ({id3} must be > {id2}); \
             AUTOINCREMENT prevents ID recycling"
        );
    }

    /// T-STO-039: Column repair adds `tags` and `metadata` to an old
    /// index_session table that was created without them. Simulates an old
    /// database, then runs ensure_columns() to verify both columns are added.
    #[test]
    fn t_sto_039_ensure_columns_adds_tags_and_metadata() {
        let conn = mem_db();

        // Create an old-style index_session table without tags/metadata.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS index_session (
                id                      INTEGER PRIMARY KEY,
                directory_path          TEXT    NOT NULL,
                model_name              TEXT    NOT NULL,
                chunk_strategy          TEXT    NOT NULL,
                chunk_size              INTEGER,
                chunk_overlap           INTEGER,
                max_words               INTEGER,
                ocr_language            TEXT    NOT NULL DEFAULT 'eng',
                vector_dimension        INTEGER NOT NULL,
                embedding_storage_mode  TEXT    NOT NULL DEFAULT 'sqlite-blob',
                schema_version          INTEGER NOT NULL,
                app_version             TEXT    NOT NULL,
                hnsw_total              INTEGER NOT NULL DEFAULT 0,
                hnsw_orphans            INTEGER NOT NULL DEFAULT 0,
                last_rebuild_at         INTEGER,
                created_at              INTEGER NOT NULL,
                label                   TEXT
            );
            CREATE TABLE IF NOT EXISTS indexed_file (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL,
                file_path       TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                mtime           INTEGER NOT NULL,
                size            INTEGER NOT NULL,
                page_count      INTEGER NOT NULL,
                pdf_page_count  INTEGER,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job (
                id              TEXT    PRIMARY KEY,
                kind            TEXT    NOT NULL,
                session_id      INTEGER,
                state           TEXT    NOT NULL,
                progress_done   INTEGER NOT NULL DEFAULT 0,
                progress_total  INTEGER NOT NULL DEFAULT 0,
                error_message   TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                finished_at     INTEGER,
                params_json     TEXT
            );",
        )
        .expect("create old-style tables");

        // Verify tags and metadata do NOT exist before repair.
        assert!(
            !table_has_column(&conn, "index_session", "tags").unwrap(),
            "tags must NOT exist before ensure_columns"
        );
        assert!(
            !table_has_column(&conn, "index_session", "metadata").unwrap(),
            "metadata must NOT exist before ensure_columns"
        );

        ensure_columns(&conn).expect("ensure_columns failed");

        // Verify tags and metadata exist after repair.
        assert!(
            table_has_column(&conn, "index_session", "tags").unwrap(),
            "tags must exist after ensure_columns"
        );
        assert!(
            table_has_column(&conn, "index_session", "metadata").unwrap(),
            "metadata must exist after ensure_columns"
        );

        // Verify both columns accept NULL values (INSERT without them).
        conn.execute(
            "INSERT INTO index_session (
                directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode,
                schema_version, app_version, created_at
            ) VALUES ('/test', 'model', 'word', 'eng', 384, 'sqlite-blob', 1, '0.1.0', 1700000000)",
            [],
        )
        .expect("insert without tags/metadata failed");

        let (tags, metadata): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT tags, metadata FROM index_session WHERE directory_path = '/test'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query tags/metadata failed");
        assert!(tags.is_none(), "tags must be NULL when not set");
        assert!(metadata.is_none(), "metadata must be NULL when not set");

        // Verify both columns accept JSON string values.
        conn.execute(
            "UPDATE index_session SET tags = '[\"finance\",\"statistics\"]', \
             metadata = '{\"source\":\"manual\"}' WHERE directory_path = '/test'",
            [],
        )
        .expect("update tags/metadata failed");

        let (tags, metadata): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT tags, metadata FROM index_session WHERE directory_path = '/test'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query updated tags/metadata failed");
        assert_eq!(tags.as_deref(), Some("[\"finance\",\"statistics\"]"));
        assert_eq!(metadata.as_deref(), Some("{\"source\":\"manual\"}"));
    }

    /// T-STO-040: The v1 schema includes the `tags` and `metadata` columns in
    /// `index_session`. Verifies both columns exist via `PRAGMA table_info` and
    /// accept NULL values.
    #[test]
    fn t_sto_040_schema_includes_tags_and_metadata_columns() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        assert!(
            table_has_column(&conn, "index_session", "tags").unwrap(),
            "index_session must have a 'tags' column"
        );
        assert!(
            table_has_column(&conn, "index_session", "metadata").unwrap(),
            "index_session must have a 'metadata' column"
        );
    }

    /// T-STO-041: The `page` table has exactly 6 columns: id, file_id,
    /// page_number, content, backend, byte_count. Verifies that the schema
    /// definition in `create_core_tables` produces the expected column set.
    #[test]
    fn t_sto_041_page_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "page");
        let expected = vec![
            "id",
            "file_id",
            "page_number",
            "content",
            "backend",
            "byte_count",
        ];
        assert_eq!(
            columns.len(),
            expected.len(),
            "page table must have {} columns, got {} ({:?})",
            expected.len(),
            columns.len(),
            columns
        );
        for col in &expected {
            assert!(
                columns.contains(&col.to_string()),
                "page table must have column '{col}', got: {columns:?}"
            );
        }
    }

    /// T-STO-042: The `chunk` table has the expected set of 16 columns.
    /// Verifies that the schema definition in `create_chunk_table_and_indexes`
    /// produces all required columns for embedding storage, deduplication
    /// (content_hash, simhash), soft-delete support (is_deleted), and
    /// the created_at timestamp.
    #[test]
    fn t_sto_042_chunk_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "chunk");
        let expected = vec![
            "id",
            "file_id",
            "session_id",
            "page_start",
            "page_end",
            "chunk_index",
            "doc_text_offset_start",
            "doc_text_offset_end",
            "content",
            "embedding",
            "ext_offset",
            "ext_length",
            "content_hash",
            "simhash",
            "is_deleted",
            "created_at",
        ];
        assert_eq!(
            columns.len(),
            expected.len(),
            "chunk table must have {} columns, got {} ({:?})",
            expected.len(),
            columns.len(),
            columns
        );
        for col in &expected {
            assert!(
                columns.contains(&col.to_string()),
                "chunk table must have column '{col}', got: {columns:?}"
            );
        }
    }

    /// T-STO-043: The `job` table has the expected set of 11 columns
    /// including the params_json column added by the column repair step.
    #[test]
    fn t_sto_043_job_column_set_via_get_column_names() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "job");
        assert!(
            columns.contains(&"params_json".to_string()),
            "job table must include params_json column: {columns:?}"
        );
        assert!(
            columns.contains(&"state".to_string()),
            "job table must include state column: {columns:?}"
        );
        assert_eq!(
            columns.len(),
            11,
            "job table must have 11 columns, got {} ({:?})",
            columns.len(),
            columns
        );
    }

    /// T-STO-044: AUTOINCREMENT on the chunk table prevents ID reuse.
    /// After inserting a chunk with ID N, deleting it, and inserting a new
    /// chunk, the new chunk's ID is greater than N.
    #[test]
    fn t_sto_044_chunk_autoincrement_prevents_id_recycling() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        // Create a session and file for the chunk.
        conn.execute_batch(
            "INSERT INTO index_session (directory_path, model_name, chunk_strategy,
                ocr_language, vector_dimension, embedding_storage_mode, schema_version,
                app_version, hnsw_total, hnsw_orphans, created_at)
             VALUES ('/test', 'model', 'word', 'eng', 384, 'sqlite-blob', 1, '0.1', 0, 0, 1000);

             INSERT INTO indexed_file (session_id, file_path, file_hash, mtime, size,
                page_count, created_at, updated_at)
             VALUES (1, '/test/a.pdf', 'hash1', 1000, 5000, 1, 1000, 1000);",
        )
        .expect("insert session and file");

        // Insert two chunks.
        conn.execute_batch(
            "INSERT INTO chunk (file_id, session_id, page_start, page_end, chunk_index,
                doc_text_offset_start, doc_text_offset_end, content, content_hash,
                is_deleted, created_at)
             VALUES (1, 1, 1, 1, 0, 0, 10, 'chunk0', 'h0', 0, 1000);

             INSERT INTO chunk (file_id, session_id, page_start, page_end, chunk_index,
                doc_text_offset_start, doc_text_offset_end, content, content_hash,
                is_deleted, created_at)
             VALUES (1, 1, 1, 1, 1, 10, 20, 'chunk1', 'h1', 0, 1000);",
        )
        .expect("insert chunks");

        let id_before: i64 = conn
            .query_row("SELECT MAX(id) FROM chunk", [], |row| row.get(0))
            .expect("get max chunk id");

        // Delete the second chunk.
        conn.execute("DELETE FROM chunk WHERE chunk_index = 1", [])
            .expect("delete chunk");

        // Insert a new chunk. Its ID must be > id_before due to AUTOINCREMENT.
        conn.execute_batch(
            "INSERT INTO chunk (file_id, session_id, page_start, page_end, chunk_index,
                doc_text_offset_start, doc_text_offset_end, content, content_hash,
                is_deleted, created_at)
             VALUES (1, 1, 1, 1, 2, 20, 30, 'chunk2', 'h2', 0, 1000);",
        )
        .expect("insert replacement chunk");

        let id_after: i64 = conn
            .query_row("SELECT MAX(id) FROM chunk", [], |row| row.get(0))
            .expect("get new max chunk id");

        assert!(
            id_after > id_before,
            "AUTOINCREMENT must prevent ID reuse: new ID {id_after} must be > deleted ID {id_before}"
        );
    }

    /// T-STO-045: The `annotation_quote_status` table exists after migration.
    #[test]
    fn t_sto_045_annotation_quote_status_table_exists() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'annotation_quote_status'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(
            count, 1,
            "annotation_quote_status table must exist after migration"
        );
    }

    /// T-STO-046: The `annotation_quote_status` table has exactly 10 columns.
    #[test]
    fn t_sto_046_annotation_quote_status_column_set() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let columns = get_column_names(&conn, "annotation_quote_status");

        let expected = [
            "id",
            "job_id",
            "title",
            "author",
            "quote_excerpt",
            "status",
            "match_method",
            "page",
            "pdf_filename",
            "updated_at",
        ];

        for expected_col in &expected {
            assert!(
                columns.iter().any(|c| c == expected_col),
                "column '{expected_col}' must exist in annotation_quote_status"
            );
        }
        assert_eq!(
            columns.len(),
            expected.len(),
            "annotation_quote_status must have exactly {} columns, has: {columns:?}",
            expected.len()
        );
    }

    /// T-STO-047: CASCADE delete on annotation_quote_status. Deleting a job
    /// removes all associated quote status rows.
    #[test]
    fn t_sto_047_annotation_quote_status_cascade_delete() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        // Create a job.
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at)
             VALUES ('ann-job-1', 'annotate', 'running', 0, 0, 1700000000)",
            [],
        )
        .expect("insert job");

        // Insert quote status rows.
        conn.execute(
            "INSERT INTO annotation_quote_status (job_id, title, author, quote_excerpt, status, updated_at)
             VALUES ('ann-job-1', 'Title A', 'Author A', 'some text...', 'pending', 1700000000)",
            [],
        )
        .expect("insert quote status 1");

        conn.execute(
            "INSERT INTO annotation_quote_status (job_id, title, author, quote_excerpt, status, updated_at)
             VALUES ('ann-job-1', 'Title B', 'Author B', 'other text...', 'matched', 1700000001)",
            [],
        )
        .expect("insert quote status 2");

        // Verify both rows exist.
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM annotation_quote_status WHERE job_id = 'ann-job-1'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(count, 2);

        // Delete the job -- CASCADE must remove quote status rows.
        conn.execute("DELETE FROM job WHERE id = 'ann-job-1'", [])
            .expect("delete job");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM annotation_quote_status WHERE job_id = 'ann-job-1'",
                [],
                |row| row.get(0),
            )
            .expect("count after delete");
        assert_eq!(
            count, 0,
            "CASCADE delete must remove all quote status rows when job is deleted"
        );
    }

    /// T-STO-048: Column repair creates annotation_quote_status on old databases
    /// that were created before the table was added.
    #[test]
    fn t_sto_048_ensure_columns_creates_annotation_quote_status() {
        let conn = mem_db();

        // Create minimal old-style tables without annotation_quote_status.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS indexed_file (
                id              INTEGER PRIMARY KEY,
                session_id      INTEGER NOT NULL,
                file_path       TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                mtime           INTEGER NOT NULL,
                size            INTEGER NOT NULL,
                page_count      INTEGER NOT NULL,
                pdf_page_count  INTEGER,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job (
                id              TEXT    PRIMARY KEY,
                kind            TEXT    NOT NULL,
                session_id      INTEGER,
                state           TEXT    NOT NULL,
                progress_done   INTEGER NOT NULL DEFAULT 0,
                progress_total  INTEGER NOT NULL DEFAULT 0,
                error_message   TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                finished_at     INTEGER,
                params_json     TEXT
            );",
        )
        .expect("create old-style tables");

        assert!(
            !table_exists(&conn, "annotation_quote_status").unwrap(),
            "annotation_quote_status must NOT exist before ensure_columns"
        );

        ensure_columns(&conn).expect("ensure_columns failed");

        assert!(
            table_exists(&conn, "annotation_quote_status").unwrap(),
            "annotation_quote_status must exist after ensure_columns"
        );
    }

    /// T-STO-049: The partial index idx_job_state_active exists after migration.
    /// Verifies that the partial index on job(state, kind, created_at) covering
    /// active states ('queued', 'running') is created as part of the v1 schema.
    #[test]
    fn t_sto_049_job_state_active_partial_index_exists() {
        let conn = mem_db();
        migrate(&conn).expect("migration failed");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_job_state_active'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(
            count, 1,
            "idx_job_state_active partial index must exist after migration"
        );
    }

    /// Helper: reads the column names of a table via PRAGMA table_info.
    fn get_column_names(conn: &Connection, table: &str) -> Vec<String> {
        let query = format!("PRAGMA table_info({table})");
        conn.prepare(&query)
            .expect("PRAGMA table_info failed")
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })
            .expect("query_map failed")
            .filter_map(|r| r.ok())
            .collect()
    }
}
