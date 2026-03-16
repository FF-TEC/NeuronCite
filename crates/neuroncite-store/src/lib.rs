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

//! neuroncite-store: `SQLite` schema management, connection pooling, and persistent
//! storage for the `NeuronCite` workspace.
//!
//! This crate is organized into three internal layers:
//!
//! - **Repository** (`repo`) -- CRUD operations for sessions, files, pages,
//!   and chunks. Each entity has its own submodule with functions that accept
//!   a borrowed `rusqlite::Connection`.
//! - **Index** (`index`) -- HNSW approximate nearest neighbor index management,
//!   FTS5 full-text search index management, and external embedding storage via
//!   memory-mapped files.
//! - **Workflow** (`workflow`) -- Job tracking and idempotency key management for
//!   the indexing pipeline, ensuring that interrupted indexing operations can be
//!   resumed without re-processing already-completed work.
//!
//! The crate does not use `#![forbid(unsafe_code)]` because the HNSW and
//! memory-mapped index modules may require unsafe blocks for performance-critical
//! pointer operations (e.g., zero-copy deserialization of memory-mapped files).

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.
// The cast_* lints are suppressed globally because the store crate performs
// frequent i64 <-> usize conversions at the SQLite boundary (SQLite stores all
// integers as i64, Rust indexes are usize). Critical conversions in hnsw.rs and
// external.rs use usize::try_from() for safety. The remaining casts are on
// trusted DB values where overflow is structurally impossible.
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

pub mod error;
mod index;
mod manifest;
mod pool;
pub mod repo;
mod schema;
mod workflow;

// Re-export the primary error type at crate root for ergonomic imports.
pub use error::StoreError;

// Re-export pool creation for application startup.
pub use pool::{create_pool, create_pool_with_size};

// Re-export schema migration for application startup.
pub use schema::{migrate, schema_version};

// Re-export manifest I/O for model management.
pub use manifest::{read_manifest, write_manifest};

// Re-export annotation quote status types and functions for the annotate_status handler.
pub use repo::annotation_status::{
    AnnotationQuoteStatusRow, count_quote_statuses_by_status, insert_pending_quotes,
    list_quote_statuses, update_quote_status,
};

// Re-export aggregate query types and functions for statistics endpoints.
pub use repo::aggregate::{
    ChunkBrowseRow, FileChunkStats, FilePageStats, FileQualityRow, PageQualityDetail,
    SessionAggregates, all_session_aggregates, browse_chunks, file_chunk_stats_by_session,
    file_page_stats_by_session, find_files_by_path_pattern, page_quality_details,
    session_quality_data, single_file_chunk_stats,
};

// Re-export repository types and functions for CRUD operations.
pub use repo::chunk::{
    ChunkInsert, ChunkRow, bulk_insert_chunks, delete_chunks_by_file, get_chunk, get_chunks_batch,
    list_chunks_by_file, list_chunks_by_session, load_embeddings_for_hnsw, set_chunk_deleted,
};
pub use repo::diff::{FileDiff, SessionDiff, diff_sessions};
pub use repo::file::{
    ChangeStatus, FileRow, check_file_changed, delete_file, delete_file_by_session_path,
    find_file_by_session_path, get_file, insert_file, insert_file_with_source_type,
    list_files_by_session, update_file_hash,
};
pub use repo::page::{
    PageRow, PageTextMatch, TextMatch, bulk_insert_pages, delete_pages_by_file, get_page,
    get_pages_by_file, get_pages_range, search_page_text,
};
pub use repo::session::{
    SessionRow, create_session, delete_session, delete_sessions_by_directory, find_session,
    find_sessions_by_directory, get_session, list_sessions, update_session_label,
    update_session_metadata, update_session_tags,
};
pub use repo::web_source::{
    WebSourceInsert, WebSourceRow, delete_web_source, get_web_source_by_file,
    get_web_source_by_url, insert_web_source, list_web_sources, list_web_sources_all,
};

// Re-export index layer functions.
pub use index::external::{
    ExternalReader, ExternalWriter, append_embedding, create_external_file, read_embedding,
    verify_external_file_integrity,
};
pub use index::fts::{integrity_check_fts, optimize_fts};
pub use index::hnsw::{HnswIndex, build_hnsw, deserialize_hnsw, serialize_hnsw};

// Re-export workflow types and functions.
pub use workflow::idempotency::{
    CreateJobResult, IdempotencyRow, cleanup_expired_keys, create_job_with_key, lookup_key,
    store_key,
};
pub use workflow::job::{
    JobRow, JobState, cleanup_expired_jobs, create_job, create_job_with_params, get_job,
    has_active_job, list_jobs, update_job_progress, update_job_state,
};
