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

//! Request and response data transfer objects for all API endpoints.
//!
//! Each struct in this module represents either a JSON request body or a JSON
//! response body for one of the API endpoints. All types derive `serde::Serialize`
//! and/or `serde::Deserialize` for JSON conversion, and `utoipa::ToSchema` for
//! automatic OpenAPI specification generation.
//!
//! The `api_version` field on response DTOs is marked with `#[serde(skip)]`
//! because the API version is conveyed via the `X-API-Version` response header
//! (injected by the `ApiVersionLayer` middleware). The field remains in the
//! struct for backward compatibility with internal code that constructs
//! responses, but is excluded from the serialized JSON body.

use std::collections::HashMap;

use neuroncite_core::InputLimits;
use neuroncite_core::config::RequestDefaults;
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

use crate::error::ApiError;

/// The API version string used across all response DTOs. Centralizes the
/// version identifier so that a version bump requires changing only this
/// constant instead of every handler that constructs a response.
pub const API_VERSION: &str = "v1";

// ---------------------------------------------------------------------------
// Typed verdict and state enums
// ---------------------------------------------------------------------------

/// Typed verdict returned by POST /api/v1/verify.
///
/// Serializes and deserializes as a lowercase snake_case string, matching the
/// original string-based API contract. Using a typed enum prevents callers
/// from constructing invalid verdict strings and makes exhaustive matching
/// possible in the handler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VerifyVerdict {
    /// The cited passages collectively support the claim.
    Supports,
    /// The cited passages partially support the claim.
    Partial,
    /// The cited passages do not support the claim.
    NotSupported,
}

/// Typed job state used in all job response DTOs.
///
/// Serializes and deserializes as a lowercase snake_case string, matching the
/// string representations stored in the database and previously returned as
/// plain strings. The `From<neuroncite_store::JobState>` impl converts from
/// the store layer's enum without any string allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobStateDto {
    /// Job is waiting for an executor slot.
    Queued,
    /// Job is actively being processed by the executor.
    Running,
    /// Job finished without errors.
    Completed,
    /// Job terminated with an error.
    Failed,
    /// Job was canceled by a client request.
    Canceled,
}

impl From<neuroncite_store::JobState> for JobStateDto {
    fn from(s: neuroncite_store::JobState) -> Self {
        match s {
            neuroncite_store::JobState::Queued => Self::Queued,
            neuroncite_store::JobState::Running => Self::Running,
            neuroncite_store::JobState::Completed => Self::Completed,
            neuroncite_store::JobState::Failed => Self::Failed,
            neuroncite_store::JobState::Canceled => Self::Canceled,
        }
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Response body for GET /api/v1/health.
/// Reports the application's version, available features, and backend status.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Application version string from Cargo.toml.
    pub version: String,
    /// List of compile-time feature flags that are active.
    pub build_features: Vec<String>,
    /// Whether a GPU device is available for embedding inference.
    pub gpu_available: bool,
    /// Name of the active embedding backend (e.g., "ort").
    pub active_backend: String,
    /// Whether a cross-encoder reranker model is configured. When false,
    /// the `rerank` search parameter will return an error.
    pub reranker_available: bool,
    /// Whether the pdfium extraction backend is compiled in.
    pub pdfium_available: bool,
    /// Whether the Tesseract OCR backend is compiled in.
    pub tesseract_available: bool,
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/index.
/// Specifies the directory to index and indexing parameters.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexRequest {
    /// Absolute path to the directory containing PDF files.
    pub directory: String,
    /// Client-provided idempotency key for deduplication.
    pub idempotency_key: Option<String>,
    /// Embedding model identifier (uses config default if absent).
    pub model_name: Option<String>,
    /// Chunking strategy name (uses config default if absent).
    pub chunk_strategy: Option<String>,
    /// Chunk size in tokens or words depending on the strategy (uses config
    /// default if absent).
    pub chunk_size: Option<usize>,
    /// Chunk overlap in tokens or words depending on the strategy (uses config
    /// default if absent).
    pub chunk_overlap: Option<usize>,
}

/// Response body for POST /api/v1/index.
/// Returns the created job and session identifiers.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the created indexing job.
    pub job_id: String,
    /// Numeric ID of the indexing session.
    pub session_id: i64,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/search and /api/v1/search/hybrid.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SearchRequest {
    /// The natural-language search query text.
    pub query: String,
    /// Session ID to search within.
    pub session_id: i64,
    /// Number of results to return (default: 10).
    pub top_k: Option<usize>,
    /// Whether to use FTS5 keyword search in addition to vector search.
    pub use_fts: Option<bool>,
    /// Whether to apply cross-encoder reranking to results.
    pub rerank: Option<bool>,
    /// Whether to apply sub-chunk refinement after the initial search.
    /// Refinement splits each result chunk into overlapping sub-chunks at
    /// multiple scales, embeds them, and replaces the content with the most
    /// relevant sub-section when it scores higher than the original chunk.
    /// (default: true)
    pub refine: Option<bool>,
    /// Comma-separated list of divisor values controlling the sub-chunk
    /// split granularity. For each divisor d, the chunk is split into
    /// windows of size T/d tokens with 25% overlap. Multiple values enable
    /// multi-scale refinement. (default: "4,8,16")
    pub refine_divisors: Option<String>,
}

/// A single search result within the response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SearchResultDto {
    /// Final relevance score (higher is more relevant).
    pub score: f64,
    /// Chunk text content.
    pub content: String,
    /// Formatted citation string.
    pub citation: String,
    /// Cosine similarity from vector search.
    pub vector_score: f64,
    /// BM25 rank from keyword search (null if vector-only).
    pub bm25_rank: Option<usize>,
    /// Cross-encoder reranker score (null if reranking was skipped).
    pub reranker_score: Option<f64>,
    /// Database file ID for the source document. Pass to the page endpoint
    /// to retrieve the full text of any page.
    pub file_id: i64,
    /// Source PDF file path.
    pub source_file: String,
    /// Start page number.
    pub page_start: usize,
    /// End page number (inclusive).
    pub page_end: usize,
    /// 0-indexed position of this chunk within the source file's chunk
    /// sequence. Used by the frontend to fetch neighboring chunks via the
    /// chunks browsing endpoint (offset = chunk_index - N).
    pub chunk_index: usize,
    /// Session ID that this result originated from. Only populated in
    /// multi-search responses where results are merged across sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<i64>,
    /// Unix timestamp (seconds since epoch) when the source document was
    /// first indexed. Sourced from the `indexed_file.created_at` column.
    /// None when the file record cannot be resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_created_at: Option<i64>,
    /// Unix timestamp (seconds since epoch) when the source document's
    /// index record was last updated. Sourced from the
    /// `indexed_file.updated_at` column. None when the file record
    /// cannot be resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_modified_at: Option<i64>,
    /// Author metadata from the source document. The `indexed_file` table
    /// does not store an author column, so this field is always None in
    /// the current schema. Reserved for future extraction backends that
    /// can read PDF metadata (XMP, Info dict).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_author: Option<String>,
}

/// Response body for search endpoints.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SearchResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// List of ranked search results.
    pub results: Vec<SearchResultDto>,
}

/// Request body for POST /api/v1/search/multi.
/// Searches across multiple sessions simultaneously. The query is embedded
/// once, then the SearchPipeline runs for each session. Results are merged
/// into a single ranked list sorted by score, with each result tagged by
/// its source session_id.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MultiSearchRequest {
    /// The natural-language search query text.
    pub query: String,
    /// Session IDs to search across (1-10 sessions). All sessions must
    /// have the same vector dimension as the currently loaded model.
    pub session_ids: Vec<i64>,
    /// Number of results to return (default: 10, max: 50).
    pub top_k: Option<usize>,
    /// Whether to use FTS5 keyword search in addition to vector search.
    pub use_fts: Option<bool>,
    /// Whether to apply cross-encoder reranking to the merged results.
    pub rerank: Option<bool>,
    /// Whether to apply sub-chunk refinement after the initial search.
    pub refine: Option<bool>,
    /// Comma-separated list of divisor values controlling the sub-chunk
    /// split granularity. (default: "4,8,16")
    pub refine_divisors: Option<String>,
}

/// Status of a per-session search operation within a multi-search request.
/// Indicates whether the session's HNSW index was available and the search
/// completed, or whether an error prevented results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionSearchStatus {
    /// The search completed without errors.
    Ok,
    /// The session has no HNSW index loaded in memory, so vector search
    /// could not be executed.
    NoHnswIndex,
    /// An unexpected error occurred during the search pipeline execution
    /// for this session.
    Error,
}

/// Per-session status in a multi-search response. Reports how many results
/// each session contributed (or an error if that session's search failed).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MultiSearchSessionStat {
    /// Session ID.
    pub session_id: i64,
    /// Number of results this session contributed before merging and trimming.
    pub result_count: usize,
    /// Outcome of the search operation for this session.
    pub status: SessionSearchStatus,
    /// Error description when status is Error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response body for POST /api/v1/search/multi.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MultiSearchResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The session IDs that were searched.
    pub session_ids: Vec<i64>,
    /// Per-session search statistics.
    pub session_stats: Vec<MultiSearchSessionStat>,
    /// Merged and ranked search results from all sessions.
    pub results: Vec<SearchResultDto>,
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/verify.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct VerifyRequest {
    /// The claim text to verify against cited passages.
    pub claim: String,
    /// Session ID containing the cited chunks.
    pub session_id: i64,
    /// List of chunk IDs referenced as citations.
    pub chunk_ids: Vec<i64>,
}

/// Response body for POST /api/v1/verify.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct VerifyResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Typed verification verdict. Serializes as "supports", "partial", or
    /// "not_supported" in JSON to preserve the original API contract.
    pub verdict: VerifyVerdict,
    /// Combined score (weighted: 0.3 keyword + 0.7 semantic).
    pub combined_score: f64,
    /// Jaccard keyword overlap score.
    pub keyword_score: f64,
    /// Cosine semantic similarity score.
    pub semantic_score: f64,
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// Response body for GET /api/v1/jobs/{id} and items in job list.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct JobResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Job UUID.
    pub id: String,
    /// Job kind (e.g., "index", "rebuild").
    pub kind: String,
    /// Associated session ID (if applicable).
    pub session_id: Option<i64>,
    /// Typed job state. Serializes as "queued", "running", "completed",
    /// "failed", or "canceled" in JSON to preserve the original API contract.
    pub state: JobStateDto,
    /// Number of items processed.
    pub progress_done: i64,
    /// Total number of items to process.
    pub progress_total: i64,
    /// Error message (only present for failed jobs).
    pub error_message: Option<String>,
    /// Unix timestamp when the job was created.
    pub created_at: i64,
    /// Unix timestamp when execution started (null if still queued).
    pub started_at: Option<i64>,
    /// Unix timestamp when the job finished (null if still running).
    pub finished_at: Option<i64>,
}

/// Response body for GET /api/v1/jobs (list all jobs).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct JobListResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// List of job records.
    pub jobs: Vec<JobResponse>,
}

/// Response body for POST /api/v1/jobs/{id}/cancel.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct JobCancelResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the canceled job.
    pub job_id: String,
    /// State after cancellation. Always `JobStateDto::Canceled`.
    pub state: JobStateDto,
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// A session record in list and detail responses.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SessionDto {
    /// Session numeric ID.
    pub id: i64,
    /// Root directory path of the indexed document collection.
    pub directory_path: String,
    /// Embedding model identifier.
    pub model_name: String,
    /// Chunking strategy name (e.g. "word", "token", "sentence").
    pub chunk_strategy: String,
    /// Chunk size in tokens or words (used by "token" and "word" strategies).
    /// Null for the "sentence" strategy, which uses max_words instead.
    pub chunk_size: Option<i64>,
    /// Token/word overlap between consecutive chunks (used by "token" and
    /// "word" strategies). Null for "sentence" and "page" strategies.
    pub chunk_overlap: Option<i64>,
    /// Maximum words per sentence chunk (used by the "sentence" strategy).
    /// Null for "token" and "word" strategies.
    pub max_words: Option<i64>,
    /// Embedding vector dimensionality.
    pub vector_dimension: i64,
    /// Unix timestamp when the session was created.
    pub created_at: i64,
    /// Number of indexed files in this session.
    pub file_count: i64,
    /// Total extracted pages across all files.
    pub total_pages: i64,
    /// Total active (non-deleted) chunks.
    pub total_chunks: i64,
    /// Total content bytes across all pages.
    pub total_content_bytes: i64,
}

/// Response body for GET /api/v1/sessions.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SessionListResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// List of session records.
    pub sessions: Vec<SessionDto>,
}

/// Response body for DELETE /api/v1/sessions/{id}.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SessionDeleteResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Whether the session was deleted (true) or was already absent (false).
    pub deleted: bool,
}

/// Request body for POST /api/v1/sessions/delete-by-directory.
/// Deletes all sessions whose directory_path matches the specified directory.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SessionDeleteByDirectoryRequest {
    /// Absolute path to the PDF directory whose sessions should be deleted.
    /// The path is canonicalized before comparison.
    pub directory: String,
}

/// Response body for POST /api/v1/sessions/delete-by-directory.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SessionDeleteByDirectoryResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// IDs of all sessions that were deleted.
    pub deleted_session_ids: Vec<i64>,
    /// The canonicalized directory path that was matched.
    pub directory: String,
    /// Whether at least one session was found and deleted for this directory.
    /// False indicates the directory was valid but no sessions referenced it.
    pub matched_directory: bool,
}

/// Response body for POST /api/v1/sessions/{id}/optimize.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OptimizeResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Status message confirming the FTS5 optimize was triggered.
    pub status: String,
}

/// Response body for POST /api/v1/sessions/{id}/rebuild.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RebuildResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the rebuild job.
    pub job_id: String,
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

/// Response body for GET /api/v1/documents/{id}/pages/{n}.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct PageResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The page number (1-indexed).
    pub page_number: i64,
    /// The text content of the page.
    pub content: String,
    /// The extraction backend that produced this page's text.
    pub backend: String,
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// Information about a single embedding backend.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BackendDto {
    /// Backend identifier (e.g., "ort").
    pub name: String,
    /// Whether GPU inference is supported by this backend.
    pub gpu_supported: bool,
    /// Number of models available through this backend.
    pub model_count: usize,
}

/// Response body for GET /api/v1/backends.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BackendListResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// List of available embedding backends.
    pub backends: Vec<BackendDto>,
}

// ---------------------------------------------------------------------------
// Annotate
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/annotate.
/// Specifies the source PDFs, output directory, and annotation instructions.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnnotateRequest {
    /// Absolute path to the directory containing source PDF files.
    pub source_directory: String,
    /// Absolute path where annotated PDFs and the report will be saved.
    pub output_directory: String,
    /// CSV or JSON string with annotation instructions. Required columns:
    /// title, author, quote. Optional: color (#RRGGBB hex), comment.
    pub input_data: String,
    /// Default highlight color in hex (#RRGGBB) for rows without a color field.
    /// Defaults to #FFFF00 (yellow) when absent.
    pub default_color: Option<String>,
}

/// Response body for POST /api/v1/annotate.
/// Returns the created job identifier and quote count.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnnotateResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the created annotation job.
    pub job_id: String,
    /// Number of annotation rows parsed from the input data.
    pub total_quotes: usize,
}

/// Request body for POST /api/v1/annotate/from-file.
/// Loads annotation instructions from a file on disk (CSV or JSON) rather
/// than requiring the raw data to be embedded in the request body. The file
/// is typically the `annotation_pipeline_input.csv` produced by the citation
/// export pipeline.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnnotateFromFileRequest {
    /// Absolute path to the CSV or JSON file containing annotation instructions.
    /// The file format is auto-detected: CSV uses the 6-column format
    /// (title, author, quote, color, comment, page); JSON uses an array of
    /// objects with the same fields.
    pub input_file: String,
    /// Absolute path to the directory containing source PDF files.
    pub source_directory: String,
    /// Absolute path where annotated PDFs will be saved.
    pub output_directory: String,
    /// Default highlight color in hex (#RRGGBB) for rows without a color field.
    /// Defaults to #FFFF00 (yellow) when absent.
    pub default_color: Option<String>,
}

/// Response body for POST /api/v1/annotate/from-file.
/// Returns the created job identifier and quote count.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnnotateFromFileResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the created annotation job.
    pub job_id: String,
    /// Number of annotation rows parsed from the file.
    pub total_quotes: usize,
}

// ---------------------------------------------------------------------------
// Citation Verification
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/citation/create.
/// Specifies the LaTeX and BibTeX files to parse, the index session(s) to
/// match against, and the batch size for sub-agent claiming.
///
/// Supports both single-session (`session_id`) and multi-session
/// (`session_ids`) operation. When `session_ids` is provided and non-empty,
/// it overrides `session_id`. Files from all listed sessions are aggregated
/// for cite-key matching, and the agent searches across all sessions during
/// verification.
///
/// Supports dry-run mode for previewing PDF matches before committing,
/// and file_overrides for correcting automatic PDF matching.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationCreateRequest {
    /// Absolute path to the .tex file containing citation commands.
    pub tex_path: String,
    /// Absolute path to the .bib file with BibTeX entries.
    pub bib_path: String,
    /// Primary index session ID. Used when `session_ids` is absent or empty
    /// (backward compatibility with single-session clients).
    pub session_id: i64,
    /// Multiple session IDs to search against. When present and non-empty,
    /// overrides `session_id`. Files from all sessions are aggregated for
    /// cite-key matching, and the verification agent searches across all
    /// listed sessions.
    #[serde(default)]
    pub session_ids: Option<Vec<i64>>,
    /// Target number of citations per batch (default: 5).
    pub batch_size: Option<usize>,
    /// When true, runs the parse/match pipeline without creating a job or
    /// inserting rows. Returns a match preview with per-cite-key data
    /// (author, title, year, matched_file_id, overlap_score). (default: false)
    pub dry_run: Option<bool>,
    /// Map of cite_key -> file_id to override automatic PDF matching.
    /// Applied after automatic matching. Use the dry_run response to
    /// identify incorrect matches, then pass corrections here.
    pub file_overrides: Option<HashMap<String, i64>>,
}

/// Response body for POST /api/v1/citation/create.
/// Returns the created job ID and statistics about the parsed citations.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationCreateResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the created citation verification job.
    pub job_id: String,
    /// Session ID the job is linked to.
    pub session_id: i64,
    /// Total citation rows inserted.
    pub total_citations: usize,
    /// Number of distinct batches assigned.
    pub total_batches: usize,
    /// Number of distinct cite-keys across all citations.
    pub unique_cite_keys: usize,
    /// Number of unique cite-keys that were matched to indexed PDF files via
    /// token-overlap filename matching. This counts distinct cite-keys, not
    /// individual citation rows (a cite-key can appear in multiple \cite
    /// commands, producing multiple citation rows that share one match).
    pub cite_keys_matched: usize,
    /// Number of unique cite-keys without a matching indexed PDF file.
    pub cite_keys_unmatched: usize,
    /// Cite-keys not found in the BibTeX file.
    pub unresolved_cite_keys: Vec<String>,
}

/// Request body for POST /api/v1/citation/claim.
/// Specifies the job and optionally a specific batch to claim.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationClaimRequest {
    /// The citation verification job ID.
    pub job_id: String,
    /// Specific batch to claim. When absent, the lowest pending batch is
    /// claimed in FIFO order. Specifying a batch_id allows retrying failed
    /// batches and out-of-order processing.
    pub batch_id: Option<i64>,
}

/// Response body for POST /api/v1/citation/claim.
/// Returns the claimed batch data or null if no batches are pending.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationClaimResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The claimed batch ID, or null if no pending batches.
    pub batch_id: Option<i64>,
    /// Absolute path to the .tex file (present when batch_id is not null).
    pub tex_path: Option<String>,
    /// Absolute path to the .bib file (present when batch_id is not null).
    pub bib_path: Option<String>,
    /// Index session ID (present when batch_id is not null).
    pub session_id: Option<i64>,
    /// Citation rows in the claimed batch (empty when batch_id is null).
    pub rows: Vec<serde_json::Value>,
    /// Remaining pending batches (present when batch_id is null).
    pub remaining_batches: Option<i64>,
    /// Descriptive message when no batch was claimed. When a specific batch_id
    /// was requested but could not be claimed, this message explains whether
    /// the batch was not found or is in a non-claimable state, rather than
    /// returning a generic "no pending batches" message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// A single verification result for one citation row, submitted by a sub-agent.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationSubmitEntryDto {
    /// The citation_row.id this result corresponds to.
    pub row_id: i64,
    /// Verdict: supported, partial, unsupported, not_found, wrong_source, unverifiable.
    pub verdict: String,
    /// The claim as stated in the LaTeX document.
    pub claim_original: String,
    /// The sub-agent's English formulation of the core claim.
    pub claim_english: String,
    /// Whether the claim was found in the cited PDF.
    pub source_match: bool,
    /// Whether the claim was found in other indexed PDFs.
    pub other_sources: bool,
    /// Passages found during verification.
    pub passages: Vec<serde_json::Value>,
    /// Justification for the verdict.
    pub reasoning: String,
    /// Confidence level (0.0 to 1.0).
    pub confidence: f64,
    /// Number of search iterations performed.
    pub search_rounds: i64,
    /// Sub-agent flag for escalation: "critical", "warning", or null.
    pub flag: Option<String>,
    /// Title or cite-key of a better source (for wrong_source verdicts).
    pub better_source: Option<String>,
    /// Suggested LaTeX correction. Null if no correction is needed.
    pub latex_correction: Option<serde_json::Value>,
}

/// Request body for POST /api/v1/citation/submit.
/// Contains an array of verification results for a claimed batch.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationSubmitRequest {
    /// The citation verification job ID.
    pub job_id: String,
    /// Array of result entries, one per row_id in the batch.
    pub results: Vec<serde_json::Value>,
}

/// Response body for POST /api/v1/citation/submit.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationSubmitResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Acceptance status ("accepted").
    pub status: String,
    /// Number of rows submitted.
    pub rows_submitted: usize,
    /// Total citation rows in the job.
    pub total: i64,
    /// Whether all batches are done.
    pub is_complete: bool,
}

/// Response body for GET /api/v1/citation/{job_id}/status.
/// Contains counts by row status, batch status, verdict aggregates, and alerts.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationStatusResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The citation verification job ID.
    pub job_id: String,
    /// Typed job state for the citation verification job.
    pub job_state: JobStateDto,
    /// Total citation rows.
    pub total: i64,
    /// Rows with status "pending".
    pub pending: i64,
    /// Rows with status "claimed".
    pub claimed: i64,
    /// Rows with status "done".
    pub done: i64,
    /// Rows with status "failed".
    pub failed: i64,
    /// Total distinct batches.
    pub total_batches: i64,
    /// Batches where all rows are done or failed.
    pub batches_done: i64,
    /// Batches where all rows are pending.
    pub batches_pending: i64,
    /// Batches where at least one row is claimed.
    pub batches_claimed: i64,
    /// Verdict counts keyed by verdict name (e.g. "supported", "partial").
    pub verdicts: HashMap<String, i64>,
    /// Flagged citation rows requiring attention.
    pub alerts: Vec<serde_json::Value>,
    /// Elapsed seconds since job started.
    pub elapsed_seconds: i64,
    /// Whether all rows are done or failed.
    pub is_complete: bool,
}

/// Request body for POST /api/v1/citation/{job_id}/export.
/// Specifies the output and source directories for export file generation.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationExportRequest {
    /// Absolute path where output files will be written.
    pub output_directory: String,
    /// Absolute path where the source PDFs are located.
    pub source_directory: String,
    /// Controls whether the annotation pipeline runs automatically after
    /// export. When true, an annotation job is created using the generated
    /// annotation_pipeline_input.csv and annotated PDFs are saved to the
    /// `annotated_pdfs/` subfolder inside the output directory. When false,
    /// only the export files (CSV, Excel, JSON) are produced. Defaults to
    /// true for backward compatibility with MCP clients that omit this field.
    #[serde(default = "default_annotate_true")]
    pub annotate: bool,
}

/// Returns true. Used as serde default for CitationExportRequest.annotate
/// so that existing callers (MCP tools, older frontends) that omit the field
/// continue to receive automatic annotation after export.
fn default_annotate_true() -> bool {
    true
}

/// Response body for POST /api/v1/citation/{job_id}/export.
/// Returns paths to the generated files and a summary of results.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationExportResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// UUID of the annotation job triggered by the export (null if no annotations).
    pub annotation_job_id: Option<String>,
    /// Path to the annotation pipeline CSV (5-column format for neuroncite-annotate).
    pub annotation_csv_path: String,
    /// Path to the full 38-column data CSV sorted by tex_line ascending.
    pub data_csv_path: String,
    /// Path to the formatted Excel workbook (.xlsx) with verdict-colored rows.
    pub excel_path: String,
    /// Path to the generated summary report JSON file.
    pub report_path: String,
    /// Path to the generated corrections JSON file.
    pub corrections_path: String,
    /// Path to the full-detail JSON file containing all citation rows with
    /// complete verification results.
    pub full_detail_path: String,
    /// Summary of verdict counts and correction statistics.
    pub summary: serde_json::Value,
}

/// Query parameters for GET /api/v1/citation/{job_id}/rows.
/// Supports optional status filtering and pagination.
#[derive(Debug, Serialize, Deserialize, IntoParams, ToSchema)]
pub struct CitationRowsQuery {
    /// Filter rows by status: "pending", "claimed", "done", "failed".
    /// When absent, returns all rows regardless of status.
    pub status: Option<String>,
    /// Number of rows to skip for pagination (default: 0).
    pub offset: Option<i64>,
    /// Maximum number of rows to return (default: 100, max: 500).
    pub limit: Option<i64>,
}

/// Response body for GET /api/v1/citation/{job_id}/rows.
/// Returns paginated citation rows with total count for cursor-based paging.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CitationRowsResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The citation verification job ID.
    pub job_id: String,
    /// Citation rows matching the query filters.
    pub rows: Vec<serde_json::Value>,
    /// Total number of rows matching the filter (before pagination).
    pub total: i64,
    /// The offset used for this page.
    pub offset: i64,
    /// The limit used for this page.
    pub limit: i64,
}

// ---------------------------------------------------------------------------
// Chunks
// ---------------------------------------------------------------------------

/// Query parameters for GET /api/v1/sessions/{session_id}/files/{file_id}/chunks.
#[derive(Debug, Serialize, Deserialize, IntoParams, ToSchema)]
pub struct ChunksQuery {
    /// Filter to chunks spanning this page number (1-indexed).
    pub page_number: Option<i64>,
    /// Number of chunks to skip for pagination (default: 0).
    pub offset: Option<i64>,
    /// Maximum number of chunks to return (default: 20, max: 100).
    pub limit: Option<i64>,
}

/// A single chunk record returned in the chunks list endpoint.
///
/// Contains the raw chunk content together with its positional metadata.
/// `word_count` is computed from `content.split_whitespace().count()` at
/// query time; `byte_count` is `content.len()` (UTF-8 bytes).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChunkDto {
    /// Database row ID of the chunk.
    pub chunk_id: i64,
    /// Zero-based position of the chunk within its file.
    pub chunk_index: i64,
    /// First PDF page this chunk spans (1-indexed).
    pub page_start: i64,
    /// Last PDF page this chunk spans (1-indexed).
    pub page_end: i64,
    /// Approximate word count (whitespace-delimited tokens).
    pub word_count: usize,
    /// UTF-8 byte length of the chunk content string.
    pub byte_count: usize,
    /// Full text content of the chunk.
    pub content: String,
}

/// Response body for GET /api/v1/sessions/{session_id}/files/{file_id}/chunks.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChunksResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Session ID containing the file.
    pub session_id: i64,
    /// File ID whose chunks are returned.
    pub file_id: i64,
    /// Total number of chunks matching the filter.
    pub total_chunks: i64,
    /// Offset used for pagination.
    pub offset: i64,
    /// Limit used for pagination.
    pub limit: i64,
    /// Number of chunks returned in this response.
    pub returned: usize,
    /// Chunk records with content, page range, and computed word/byte counts.
    pub chunks: Vec<ChunkDto>,
}

// ---------------------------------------------------------------------------
// Quality Report
// ---------------------------------------------------------------------------

/// Aggregate extraction method distribution across all files in a session.
///
/// Categorizes each file as native-text-only, OCR-only, or mixed, and
/// reports page-level totals and average content density.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractionSummaryDto {
    /// Total number of indexed files in the session.
    pub total_files: usize,
    /// Files where all pages used native text extraction (ocr_pages == 0).
    pub native_text_count: i64,
    /// Files where all pages required OCR (native_pages == 0).
    pub ocr_required_count: i64,
    /// Files where both native and OCR pages are present.
    pub mixed_count: i64,
    /// Total extracted pages across all files.
    pub total_pages: i64,
    /// Total pages with no extracted text content.
    pub total_empty_pages: i64,
    /// Average content bytes per page across all files.
    pub avg_bytes_per_page: i64,
}

/// Per-page-count details for a file with quality flags.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QualityFlagDetailsDto {
    /// Number of pages successfully extracted from this file.
    pub page_count: i64,
    /// Total pages as reported in the PDF metadata (null if unavailable).
    pub pdf_page_count: Option<i64>,
    /// Pages extracted using native text layer.
    pub native_pages: i64,
    /// Pages extracted using OCR.
    pub ocr_pages: i64,
    /// Pages with no extracted content.
    pub empty_pages: i64,
    /// Total UTF-8 bytes across all extracted page content.
    pub total_bytes: i64,
}

/// Quality flags for one file that has at least one detected issue.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QualityFlagDto {
    /// Database row ID of the file.
    pub file_id: i64,
    /// Base file name (without directory path).
    pub file_name: String,
    /// List of issue flag strings detected for this file.
    /// Possible values: "incomplete_extraction", "page_count_mismatch",
    /// "ocr_heavy", "low_text_density", "many_empty_pages".
    pub flags: Vec<String>,
    /// Raw per-page-count metrics used to determine the flags.
    pub details: QualityFlagDetailsDto,
}

/// Response body for GET /api/v1/sessions/{session_id}/quality.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QualityReportResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Session ID the report covers.
    pub session_id: i64,
    /// Aggregate extraction method distribution across all files.
    pub extraction_summary: ExtractionSummaryDto,
    /// Per-file quality flags for files with detected issues.
    pub quality_flags: Vec<QualityFlagDto>,
    /// Number of files with at least one quality flag.
    pub files_with_issues: usize,
    /// Number of files with no quality issues.
    pub files_clean: i64,
}

// ---------------------------------------------------------------------------
// File Compare
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/files/compare.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FileCompareRequest {
    /// Exact file path to search for across sessions.
    pub file_path: Option<String>,
    /// SQL LIKE pattern to match file paths (% = any, _ = single char).
    pub file_name_pattern: Option<String>,
}

/// Per-session data for one indexed instance of a compared file.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FileComparisonSessionDto {
    /// Session the file was indexed under.
    pub session_id: i64,
    /// Database row ID of this file record.
    pub file_id: i64,
    /// Optional human-readable session label.
    pub label: Option<String>,
    /// Embedding model name used in this session.
    pub model_name: String,
    /// Chunk strategy name used in this session.
    pub chunk_strategy: String,
    /// Number of PDF pages successfully extracted.
    pub pages_extracted: i64,
    /// Total pages as reported in the PDF metadata (null if unavailable).
    pub pdf_page_count: Option<i64>,
    /// Number of chunks created for this file in this session.
    pub chunks: i64,
    /// Average chunk size in UTF-8 bytes (floating-point from SQL AVG).
    pub avg_chunk_bytes: f64,
}

/// Cross-session comparison result for one file name.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FileComparisonDto {
    /// Base file name shared across all instances.
    pub file_name: String,
    /// Number of indexed instances of this file across all sessions.
    pub instances: usize,
    /// Per-session data for each indexed instance.
    pub sessions: Vec<FileComparisonSessionDto>,
    /// Whether all instances have identical content hashes.
    pub same_content: bool,
    /// `[min, max]` range of chunk counts across all instances.
    pub chunk_count_range: [i64; 2],
}

/// Response body for POST /api/v1/files/compare.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FileCompareResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The LIKE pattern used to match files.
    pub pattern: String,
    /// Number of distinct file names matched.
    pub matched_files: usize,
    /// Per-file comparison data across sessions.
    pub comparisons: Vec<FileComparisonDto>,
}

// ---------------------------------------------------------------------------
// Directory Discovery
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/discover.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DiscoverRequest {
    /// Absolute path to the directory to discover.
    pub directory: String,
}

/// Filesystem scan results for the discover endpoint.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FilesystemSummaryDto {
    /// Per-format file counts keyed by canonical type identifier (e.g. "pdf", "html").
    /// Scales to any number of formats without adding struct fields.
    pub type_counts: std::collections::HashMap<String, usize>,
    /// Total size in bytes of all discovered indexable files.
    pub total_size_bytes: u64,
}

/// Per-session index data with aggregate statistics for the discover endpoint.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DiscoverSessionDto {
    /// Numeric session ID.
    pub session_id: i64,
    /// Optional human-readable session label.
    pub label: Option<String>,
    /// Embedding model name used when this session was indexed.
    pub model_name: String,
    /// Chunk strategy name used when this session was indexed.
    pub chunk_strategy: String,
    /// Configured chunk size (tokens or words depending on strategy).
    pub chunk_size: Option<i64>,
    /// Number of files indexed in this session.
    pub file_count: i64,
    /// Total number of chunks across all files in this session.
    pub total_chunks: i64,
    /// Total extracted pages across all files in this session.
    pub total_pages: i64,
    /// Total UTF-8 bytes of extracted content across all chunks.
    pub total_content_bytes: i64,
    /// Rough word count estimate computed as total_content_bytes / 6.
    pub total_words: i64,
    /// Unix timestamp when the session was created.
    pub created_at: i64,
}

/// Response body for POST /api/v1/discover.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DiscoverResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// The canonicalized directory path.
    pub directory: String,
    /// Whether the directory exists on disk.
    pub directory_exists: bool,
    /// Filesystem scan results: per-type counts and total size.
    pub filesystem: FilesystemSummaryDto,
    /// Per-session index data with aggregate statistics.
    pub sessions: Vec<DiscoverSessionDto>,
    /// Indexable files on disk that are not indexed in any session.
    pub unindexed_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// Source Fetching
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/citation/fetch-sources.
/// Specifies the BibTeX file, target session, and output directory for
/// downloading cited source documents via the DOI resolution chain.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FetchSourcesRequest {
    /// Absolute path to the .bib file containing BibTeX entries with URL/DOI fields.
    pub bib_path: String,
    /// Absolute path where downloaded PDF files are stored. Created if absent.
    pub output_directory: String,
    /// Delay in milliseconds between consecutive HTTP requests (default: 1000).
    /// Higher values reduce the risk of rate limiting by publishers.
    pub delay_ms: Option<u64>,
    /// Email address for Unpaywall API access. When provided, DOI resolution
    /// queries Unpaywall first for direct open-access PDF URLs before falling
    /// back to Semantic Scholar, OpenAlex, and doi.org.
    pub email: Option<String>,
}

/// Response body for POST /api/v1/citation/fetch-sources.
/// Contains aggregate statistics and per-entry results from the BibTeX
/// source fetching pipeline.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct FetchSourcesResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Total BibTeX entries parsed from the .bib file.
    pub total_entries: usize,
    /// Entries that have a URL or DOI field (eligible for fetching).
    pub entries_with_url: usize,
    /// PDF files downloaded to the output directory.
    pub pdfs_downloaded: usize,
    /// PDF download attempts that failed (HTTP errors, timeouts).
    pub pdfs_failed: usize,
    /// PDF files skipped because they already exist in the output directory.
    pub pdfs_skipped: usize,
    /// HTML pages fetched and saved to the output directory.
    pub html_fetched: usize,
    /// HTML fetch attempts that failed.
    pub html_failed: usize,
    /// HTML pages classified as bot-detection stubs (Cloudflare, access denied).
    pub html_blocked: usize,
    /// HTML pages skipped because they already exist in the output directory.
    pub html_skipped: usize,
    /// Per-entry results with cite_key, URL, type, status, and metadata.
    pub results: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Parse BibTeX (lightweight preview without downloading)
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/citation/parse-bib.
/// Parses a .bib file and returns structured entry metadata for live preview
/// in the Sources tab without performing any network operations. When
/// output_directory is provided, each entry is checked against existing files
/// in that directory to determine download status.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ParseBibRequest {
    /// Absolute path to the .bib file.
    pub bib_path: String,
    /// When provided, each entry is checked against files in this directory
    /// to determine whether the source has already been downloaded. The check
    /// matches filenames generated by build_source_filename() and also looks
    /// for cite_key-based filenames.
    pub output_directory: Option<String>,
}

/// Single BibTeX entry in the parse-bib response. Contains all metadata
/// fields from the .bib file for live preview in the Sources tab. The
/// extra_fields map carries every BibTeX field not captured in named struct
/// fields (e.g., journal, volume, pages, booktitle, publisher).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BibEntryPreview {
    /// BibTeX cite-key (e.g., "fama1970").
    pub cite_key: String,
    /// BibTeX entry type (e.g., "article", "book", "inproceedings").
    pub entry_type: String,
    /// Author field from the BibTeX entry.
    pub author: String,
    /// Title field from the BibTeX entry.
    pub title: String,
    /// Year field (optional, absent in some entries).
    pub year: Option<String>,
    /// Whether the entry has a direct URL field.
    pub has_url: bool,
    /// Whether the entry has a DOI field (resolvable via the DOI chain).
    pub has_doi: bool,
    /// The URL value when present.
    pub url: Option<String>,
    /// The DOI value when present.
    pub doi: Option<String>,
    /// Abstract text from the BibTeX entry. None if absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bib_abstract: Option<String>,
    /// Keywords from the BibTeX entry. None if absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<String>,
    /// Whether a downloaded file for this entry already exists in the
    /// output_directory. Only populated when output_directory was provided
    /// in the request; false by default.
    #[serde(default)]
    pub file_exists: bool,
    /// Filename of the existing file in the output directory when file_exists
    /// is true. Contains just the filename (not the full path).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub existing_file: Option<String>,
    /// The canonical filename that the download pipeline generates for this
    /// entry via build_source_filename(). Displayed in the Sources tab so
    /// users can see what filename the system produces when fetching sources
    /// and can rename their manually downloaded PDFs accordingly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_filename: Option<String>,
    /// When two or more files in the output directory match this BibTeX entry
    /// (detected via token-overlap), all relative file paths are listed here.
    /// The UI shows a "duplicate" badge and the detail view lists every path
    /// so the user can identify and consolidate duplicate copies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate_files: Option<Vec<String>>,
    /// Additional BibTeX fields not captured in the named fields above
    /// (e.g., journal, volume, pages, booktitle, publisher, editor, month,
    /// note). Keyed by the lowercase field name.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub extra_fields: std::collections::HashMap<String, String>,
}

/// Response body for POST /api/v1/citation/parse-bib.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ParseBibResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Parsed BibTeX entries with metadata for preview.
    pub entries: Vec<BibEntryPreview>,
}

// ---------------------------------------------------------------------------
// BibTeX Report
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/citation/bib-report.
/// Parses a .bib file, checks file existence in the output directory, and
/// generates CSV and XLSX report files listing all entries with their link
/// type and download status. Overwrites existing report files.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BibReportRequest {
    /// Absolute path to the .bib file.
    pub bib_path: String,
    /// Absolute path to the output directory where report files are written
    /// and where existing source files are checked for the status column.
    pub output_directory: String,
}

/// Response body for POST /api/v1/citation/bib-report.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BibReportResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Absolute path to the generated CSV report file.
    pub csv_path: String,
    /// Absolute path to the generated XLSX report file.
    pub xlsx_path: String,
    /// Total number of BibTeX entries in the report.
    pub total_entries: usize,
    /// Number of entries where a downloaded file exists in the output directory.
    /// Includes entries with status "duplicate" (file present in multiple locations).
    pub existing_count: usize,
    /// Number of entries where two or more files match the same BibTeX entry
    /// (same source file found in different subdirectories).
    pub duplicate_count: usize,
    /// Number of entries that have a URL or DOI but no downloaded file.
    pub missing_count: usize,
    /// Number of entries without URL or DOI.
    pub no_link_count: usize,
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

/// Request body for POST /api/v1/shutdown.
/// The caller must supply the nonce that was printed to stdout at server startup.
/// The nonce provides protection against unauthenticated shutdown in environments
/// where no bearer token is configured (local desktop use). Constant-time comparison
/// in the handler prevents timing side-channel attacks.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ShutdownRequest {
    /// Cryptographically random 32-byte hex nonce generated at server startup.
    /// Must match the value printed to stdout as "shutdown_nonce: <value>" on startup.
    pub nonce: String,
}

/// Response body for POST /api/v1/shutdown.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ShutdownResponse {
    /// API version identifier. Excluded from JSON serialization because the
    /// version is conveyed via the X-API-Version response header.
    #[serde(skip)]
    #[allow(dead_code)]
    pub api_version: String,
    /// Status message confirming shutdown was initiated.
    pub status: String,
}

// ---------------------------------------------------------------------------
// Request defaults
//
// The `with_defaults()` method fills in absent optional fields from the
// `RequestDefaults` struct in the application config. Handlers call this
// method after deserialization to apply server-configured defaults instead
// of hardcoding fallback values inline.
// ---------------------------------------------------------------------------

impl SearchRequest {
    /// Fills in absent optional fields from the server-configured defaults.
    /// Called by the search handler before processing so that the default
    /// top_k value is read from the config file rather than being hardcoded
    /// in the handler.
    pub fn with_defaults(&mut self, defaults: &RequestDefaults) {
        if self.top_k.is_none() {
            self.top_k = Some(defaults.top_k);
        }
    }
}

impl MultiSearchRequest {
    /// Fills in absent optional fields from the server-configured defaults.
    /// Called by the multi-search handler before processing so that the
    /// default top_k value is read from the config file rather than being
    /// hardcoded in the handler.
    pub fn with_defaults(&mut self, defaults: &RequestDefaults) {
        if self.top_k.is_none() {
            self.top_k = Some(defaults.top_k);
        }
    }
}

// ---------------------------------------------------------------------------
// Request validation
//
// Each impl block adds a `validate()` method that checks the request fields
// against the configured `InputLimits`. Handlers call `req.validate(&state.config.limits)?`
// as the first statement to reject invalid requests before any expensive work
// (connection acquisition, embedding, SQL) begins.
// ---------------------------------------------------------------------------

impl SearchRequest {
    /// Validates search request fields against the configured input limits.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `query` is empty (embedding of an empty string produces a degenerate
    ///   zero-vector that matches everything equally).
    /// - `query` exceeds `limits.max_query_chars` (prevents tokenizer memory
    ///   exhaustion from arbitrarily long strings).
    /// - `refine_divisors` contains more entries than `limits.max_refine_divisors`
    ///   (each divisor causes an additional embedding pass during refinement).
    pub fn validate(&self, limits: &InputLimits) -> Result<(), ApiError> {
        if self.query.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "query must not be empty".to_string(),
            });
        }
        if self.query.len() > limits.max_query_chars {
            return Err(ApiError::BadRequest {
                reason: format!(
                    "query length {} exceeds the maximum of {} characters",
                    self.query.len(),
                    limits.max_query_chars
                ),
            });
        }
        if let Some(ref divisors_str) = self.refine_divisors {
            let divisor_count = divisors_str
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .count();
            if divisor_count > limits.max_refine_divisors {
                return Err(ApiError::BadRequest {
                    reason: format!(
                        "refine_divisors contains {} entries, which exceeds the maximum of {}",
                        divisor_count, limits.max_refine_divisors
                    ),
                });
            }
        }
        Ok(())
    }
}

impl MultiSearchRequest {
    /// Validates multi-search request fields against the configured input limits.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `query` is empty or exceeds `limits.max_query_chars`.
    /// - `session_ids` is empty (at least one session must be specified).
    /// - `session_ids` contains more entries than `limits.max_session_ids`
    ///   (each session requires a separate HNSW search pass).
    pub fn validate(&self, limits: &InputLimits) -> Result<(), ApiError> {
        if self.query.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "query must not be empty".to_string(),
            });
        }
        if self.query.len() > limits.max_query_chars {
            return Err(ApiError::BadRequest {
                reason: format!(
                    "query length {} exceeds the maximum of {} characters",
                    self.query.len(),
                    limits.max_query_chars
                ),
            });
        }
        if self.session_ids.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "session_ids must contain at least 1 session ID".to_string(),
            });
        }
        if self.session_ids.len() > limits.max_session_ids {
            return Err(ApiError::BadRequest {
                reason: format!(
                    "session_ids contains {} entries, which exceeds the maximum of {}",
                    self.session_ids.len(),
                    limits.max_session_ids
                ),
            });
        }
        Ok(())
    }
}

impl VerifyRequest {
    /// Validates verify request fields against the configured input limits.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `claim` is empty.
    /// - `chunk_ids` is empty (at least one cited chunk must be specified).
    /// - `chunk_ids` contains more entries than `limits.max_chunk_ids`
    ///   (enforced to prevent the bulk SQL IN-clause from becoming excessively
    ///   large, which degrades SQLite query planning performance).
    pub fn validate(&self, limits: &InputLimits) -> Result<(), ApiError> {
        if self.claim.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "claim must not be empty".to_string(),
            });
        }
        if self.chunk_ids.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "chunk_ids must not be empty".to_string(),
            });
        }
        if self.chunk_ids.len() > limits.max_chunk_ids {
            return Err(ApiError::BadRequest {
                reason: format!(
                    "chunk_ids contains {} entries, which exceeds the maximum of {}",
                    self.chunk_ids.len(),
                    limits.max_chunk_ids
                ),
            });
        }
        Ok(())
    }
}

impl IndexRequest {
    /// Validates index request fields against the configured input limits.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `directory` is empty.
    /// - `chunk_size`, when provided, is zero (zero-size chunks produce no content)
    ///   or exceeds `limits.max_chunk_size_tokens` (values larger than the model's
    ///   context window cause silent truncation during tokenization).
    /// - `chunk_overlap`, when provided, is not strictly less than `chunk_size`
    ///   (overlap >= chunk_size produces infinite loops in the chunking pipeline).
    pub fn validate(&self, limits: &InputLimits) -> Result<(), ApiError> {
        if self.directory.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "directory must not be empty".to_string(),
            });
        }
        if let Some(size) = self.chunk_size {
            if size == 0 {
                return Err(ApiError::BadRequest {
                    reason: "chunk_size must be greater than 0".to_string(),
                });
            }
            if size > limits.max_chunk_size_tokens {
                return Err(ApiError::BadRequest {
                    reason: format!(
                        "chunk_size {} exceeds the maximum of {}",
                        size, limits.max_chunk_size_tokens
                    ),
                });
            }
        }
        // When both chunk_size and chunk_overlap are provided, verify that the
        // overlap is strictly less than the chunk size. Overlap >= chunk_size
        // causes the chunking pipeline to loop indefinitely or produce empty chunks.
        if let (Some(size), Some(overlap)) = (self.chunk_size, self.chunk_overlap)
            && overlap >= size
        {
            return Err(ApiError::BadRequest {
                reason: format!("chunk_overlap ({overlap}) must be less than chunk_size ({size})"),
            });
        }
        Ok(())
    }
}

impl CitationCreateRequest {
    /// Validates citation create request fields.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `tex_path` is empty (no LaTeX file to parse).
    /// - `bib_path` is empty (no BibTeX file to resolve cite-keys from).
    /// - `batch_size`, when provided, is zero (zero-size batches produce no
    ///   work units for the sub-agent to claim).
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.tex_path.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "tex_path must not be empty".to_string(),
            });
        }
        if self.bib_path.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "bib_path must not be empty".to_string(),
            });
        }
        if let Some(bs) = self.batch_size
            && bs == 0
        {
            return Err(ApiError::BadRequest {
                reason: "batch_size must be greater than 0".to_string(),
            });
        }
        Ok(())
    }
}

impl AnnotateRequest {
    /// Validates annotation request fields.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `source_directory` is empty (no source PDFs to annotate).
    /// - `output_directory` is empty (no destination for annotated output).
    /// - `input_data` is empty (no annotation instructions provided).
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.source_directory.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "source_directory must not be empty".to_string(),
            });
        }
        if self.output_directory.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "output_directory must not be empty".to_string(),
            });
        }
        if self.input_data.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "input_data must not be empty".to_string(),
            });
        }
        Ok(())
    }
}

impl AnnotateFromFileRequest {
    /// Validates annotate-from-file request fields.
    ///
    /// Returns `ApiError::BadRequest` when:
    /// - `input_file` is empty (no file path to read annotations from).
    /// - `source_directory` is empty (no source PDFs to annotate).
    /// - `output_directory` is empty (no destination for annotated output).
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.input_file.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "input_file must not be empty".to_string(),
            });
        }
        if self.source_directory.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "source_directory must not be empty".to_string(),
            });
        }
        if self.output_directory.is_empty() {
            return Err(ApiError::BadRequest {
                reason: "output_directory must not be empty".to_string(),
            });
        }
        Ok(())
    }
}
