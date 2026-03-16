/**
 * TypeScript interfaces matching the Rust DTO types from neuroncite-api.
 * Each interface corresponds to a Serialize/Deserialize struct defined in
 * crates/neuroncite-api/src/dto.rs and crates/neuroncite-web/src/handlers/.
 */

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/** Response from GET /api/v1/health. */
export interface HealthResponse {
  api_version: string;
  version: string;
  build_features: string[];
  gpu_available: boolean;
  active_backend: string;
  reranker_available: boolean;
  pdfium_available: boolean;
  tesseract_available: boolean;
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/index. */
export interface IndexRequest {
  directory: string;
  idempotency_key?: string;
  model_name?: string;
  chunk_strategy?: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

/** Response from POST /api/v1/index. */
export interface IndexResponse {
  api_version: string;
  job_id: string;
  session_id: number;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/search. */
export interface SearchRequest {
  query: string;
  session_id: number;
  top_k?: number;
  use_fts?: boolean;
  rerank?: boolean;
  refine?: boolean;
  refine_divisors?: string;
}

/** Single result within a SearchResponse. */
export interface SearchResultDto {
  score: number;
  content: string;
  citation: string;
  vector_score: number;
  bm25_rank: number | null;
  reranker_score: number | null;
  file_id: number;
  source_file: string;
  page_start: number;
  page_end: number;
  /** 0-indexed position of this chunk within the source file's chunk
   *  sequence. Used to fetch neighboring chunks for the context viewer. */
  chunk_index: number;
  /** Session ID that this result originated from. Only populated in
   *  multi-search responses where results are merged across sessions. */
  session_id?: number;
}

/** Response from POST /api/v1/search. */
export interface SearchResponse {
  api_version: string;
  results: SearchResultDto[];
}

/** Request body for POST /api/v1/search/multi. */
export interface MultiSearchRequest {
  query: string;
  session_ids: number[];
  top_k?: number;
  use_fts?: boolean;
  rerank?: boolean;
  refine?: boolean;
  refine_divisors?: string;
}

/** Per-session status in a multi-search response. */
export interface MultiSearchSessionStat {
  session_id: number;
  result_count: number;
  status: string;
  error?: string;
}

/** Response from POST /api/v1/search/multi. */
export interface MultiSearchResponse {
  api_version: string;
  session_ids: number[];
  session_stats: MultiSearchSessionStat[];
  results: SearchResultDto[];
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/verify. */
export interface VerifyRequest {
  claim: string;
  session_id: number;
  chunk_ids: number[];
}

/** Response from POST /api/v1/verify. */
export interface VerifyResponse {
  api_version: string;
  verdict: string;
  combined_score: number;
  keyword_score: number;
  semantic_score: number;
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/** Single job record used in both list and detail responses. */
export interface JobResponse {
  api_version: string;
  id: string;
  kind: string;
  session_id: number | null;
  state: string;
  progress_done: number;
  progress_total: number;
  error_message: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

/** Response from GET /api/v1/jobs. */
export interface JobListResponse {
  api_version: string;
  jobs: JobResponse[];
}

/** Response from POST /api/v1/jobs/{id}/cancel. */
export interface JobCancelResponse {
  api_version: string;
  job_id: string;
  state: string;
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/** Session record with aggregate statistics. */
export interface SessionDto {
  id: number;
  directory_path: string;
  model_name: string;
  chunk_strategy: string;
  /** Chunk size in tokens or words. Null for "sentence" strategy. */
  chunk_size: number | null;
  /** Token/word overlap between consecutive chunks. Null for "sentence"
   *  and "page" strategies. */
  chunk_overlap: number | null;
  /** Max words per sentence chunk. Null for non-sentence strategies. */
  max_words: number | null;
  vector_dimension: number;
  created_at: number;
  file_count: number;
  total_pages: number;
  total_chunks: number;
  total_content_bytes: number;
}

/** Response from GET /api/v1/sessions. */
export interface SessionListResponse {
  api_version: string;
  sessions: SessionDto[];
}

/** Response from DELETE /api/v1/sessions/{id}. */
export interface SessionDeleteResponse {
  api_version: string;
  deleted: boolean;
}

/** Response from POST /api/v1/sessions/{id}/optimize. */
export interface OptimizeResponse {
  api_version: string;
  status: string;
}

/** Response from POST /api/v1/sessions/{id}/rebuild. */
export interface RebuildResponse {
  api_version: string;
  job_id: string;
}

// ---------------------------------------------------------------------------
// Documents / Chunks
// ---------------------------------------------------------------------------

/** Response from GET /api/v1/documents/{id}/pages/{n}. */
export interface PageResponse {
  api_version: string;
  page_number: number;
  content: string;
  backend: string;
}

/** Single chunk record returned by the chunks browsing endpoint. Fields
 *  match the JSON object constructed in the Rust chunks handler. */
export interface ChunkDto {
  chunk_id: number;
  chunk_index: number;
  page_start: number;
  page_end: number;
  word_count: number;
  byte_count: number;
  content: string;
}

/** Response from GET /api/v1/sessions/{sid}/files/{fid}/chunks. */
export interface ChunksResponse {
  api_version: string;
  session_id: number;
  file_id: number;
  total_chunks: number;
  offset: number;
  limit: number;
  returned: number;
  chunks: ChunkDto[];
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/** Single embedding backend descriptor. */
export interface BackendDto {
  name: string;
  gpu_supported: boolean;
  model_count: number;
}

/** Response from GET /api/v1/backends. */
export interface BackendListResponse {
  api_version: string;
  backends: BackendDto[];
}

// ---------------------------------------------------------------------------
// Discover
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/discover. */
export interface DiscoverRequest {
  directory: string;
}

/** Filesystem scan summary returned within a DiscoverResponse. Contains
 *  aggregate statistics about the scanned directory. The Rust backend
 *  serializes this as a serde_json::Value with these known fields. */
export interface DiscoverFilesystemInfo {
  type_counts: Record<string, number>;
  total_size_bytes: number;
  [key: string]: string | number | boolean | Record<string, number>;
}

/** Per-session summary within a DiscoverResponse. Contains session metadata
 *  and aggregate statistics. The Rust backend serializes each entry as
 *  a serde_json::Value with these known fields. */
export interface DiscoverSessionInfo {
  session_id: number;
  model_name: string;
  chunk_strategy: string;
  total_chunks: number;
  file_count: number;
  [key: string]: string | number | boolean;
}

/** Response from POST /api/v1/discover. */
export interface DiscoverResponse {
  api_version: string;
  directory: string;
  directory_exists: boolean;
  filesystem: DiscoverFilesystemInfo;
  sessions: DiscoverSessionInfo[];
  unindexed_files: string[];
}

// ---------------------------------------------------------------------------
// Web-specific endpoints (neuroncite-web handlers)
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/web/browse. */
export interface BrowseRequest {
  path: string;
}

/** Single entry in a directory listing. */
export interface DirEntry {
  name: string;
  kind: "directory" | "file";
  size?: number;
}

/** Response from POST /api/v1/web/browse. */
export interface BrowseResponse {
  path: string;
  parent: string;
  entries: DirEntry[];
  drives: string[];
}

/** Request body for POST /api/v1/web/scan-documents. */
export interface ScanDocumentsRequest {
  path: string;
  /** When provided, the backend cross-references discovered documents against
   *  the indexed_file table for this session to determine per-file status. */
  session_id?: number;
}

/** Single indexable document found during directory scan. */
export interface DocumentEntry {
  path: string;
  name: string;
  subfolder: string;
  size: number;
  /** File modification time as UNIX timestamp (seconds since epoch). */
  mtime: number;
  /** Canonical file type identifier (e.g. "pdf", "html"). */
  file_type: string;
  /** Indexing status relative to the active session: "indexed" when the file
   *  path exists in the indexed_file table and metadata matches, "outdated"
   *  when metadata differs, "pending" when the file is not in the database. */
  status: string;
  /** Database file ID from the indexed_file table. Only present when the file
   *  is indexed or outdated in the active session. */
  file_id?: number;
  /** Number of extracted text pages stored in the database. */
  page_count?: number;
  /** Number of active chunks produced from this file. */
  chunk_count?: number;
}

/** Response from POST /api/v1/web/scan-documents. */
export interface ScanDocumentsResponse {
  files: DocumentEntry[];
}

/** Response from GET /api/v1/web/config. */
export interface ConfigResponse {
  default_model: string;
  default_strategy: string;
  default_chunk_size: number;
  default_overlap: number;
  bind_address: string;
  port: number;
  /** HuggingFace model identifier of the embedding model currently loaded
   *  in the GPU worker. May differ from default_model when started with --model. */
  loaded_model_id: string;
  /** Output vector dimensionality of the currently loaded embedding model. */
  loaded_model_dimension: number;
}

// ---------------------------------------------------------------------------
// SSE event payloads
// ---------------------------------------------------------------------------

/** Payload for SSE "index_progress" events. */
export interface IndexProgressEvent {
  files_total: number;
  files_done: number;
  chunks_created: number;
  complete: boolean;
}

/** Payload for SSE "job_update" events. */
export interface JobUpdateEvent {
  job_id: string;
  state: string;
  progress_done: number;
  progress_total: number;
  error_message: string | null;
}

/** Payload for SSE "model_download_progress" events. */
export interface ModelDownloadProgressEvent {
  model_id: string;
  bytes_downloaded: number;
  bytes_total: number;
}

/** Payload for SSE "model_switched" events. */
export interface ModelSwitchedEvent {
  model_id: string;
  vector_dimension: number;
}

// ---------------------------------------------------------------------------
// Ollama (local LLM via /api/v1/web/ollama/*)
// ---------------------------------------------------------------------------

/** Single model installed on the Ollama server (from GET /api/tags). */
export interface OllamaModelDto {
  name: string;
  size: number;
  parameter_size: string | null;
  family: string | null;
  /** Quantization format identifier (e.g., "Q4_K_M", "Q8_0"). */
  quantization_level: string | null;
}

/** Model currently loaded in Ollama GPU/CPU RAM (from GET /api/ps). */
export interface OllamaRunningModelDto {
  name: string;
  /** Memory footprint in bytes (RAM + VRAM combined). */
  size: number;
  /** VRAM consumption in bytes. Null when the model runs entirely on CPU. */
  size_vram: number | null;
  /** ISO 8601 timestamp when the model will be automatically unloaded. */
  expires_at: string | null;
  parameter_size: string | null;
  family: string | null;
}

/** Entry from the curated Ollama model catalog (static list). */
export interface OllamaCatalogEntry {
  /** Ollama model tag for pulling (e.g., "llama3.2:3b"). */
  name: string;
  /** Human-readable display name for the UI table. */
  display_name: string;
  /** Model architecture family (e.g., "llama", "qwen2"). */
  family: string;
  /** Human-readable parameter count (e.g., "3B", "7B"). */
  parameter_size: string;
  /** Approximate disk size after download in megabytes. */
  size_mb: number;
  /** Short description of the model's strengths and use case. */
  description: string;
}

/** Response from GET /api/v1/web/ollama/status. */
export interface OllamaStatusResponse {
  connected: boolean;
  url: string;
}

/** Response from GET /api/v1/web/ollama/models. */
export interface OllamaModelsResponse {
  models: OllamaModelDto[];
  url: string;
}

/** Response from GET /api/v1/web/ollama/running. */
export interface OllamaRunningResponse {
  models: OllamaRunningModelDto[];
  url: string;
}

/** Response from GET /api/v1/web/ollama/catalog. */
export interface OllamaCatalogResponse {
  models: OllamaCatalogEntry[];
}

/** Response from POST /api/v1/web/ollama/pull (immediate acknowledgement). */
export interface OllamaPullResponse {
  status: string;
  model: string;
}

/** Response from POST /api/v1/web/ollama/delete. */
export interface OllamaDeleteResponse {
  status: string;
  model: string;
}

/** Response from POST /api/v1/web/ollama/show. */
export interface OllamaShowResponse {
  model: string;
  family: string | null;
  parameter_size: string | null;
  quantization_level: string | null;
  template: string | null;
  license: string | null;
  url: string;
}

// ---------------------------------------------------------------------------
// Citation auto-verify (POST /api/v1/citation/{job_id}/auto-verify)
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/citation/{job_id}/auto-verify.
 *  Includes verification tuning parameters that control search depth,
 *  cross-corpus coverage, reranking, and retry behavior. */
export interface AutoVerifyRequest {
  ollama_url?: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  /** Directory where cited source PDFs and scraped HTML pages are stored.
   *  When fetch_sources is true, the agent downloads missing sources here. */
  source_directory?: string;
  /** When true, the agent downloads source PDFs/HTML from BibTeX URL/DOI
   *  fields before starting verification. */
  fetch_sources?: boolean;
  /** Email address for Unpaywall API access in the DOI resolution chain.
   *  When provided, DOI resolution tries Unpaywall first for direct PDF URLs.
   *  When absent, resolution starts with Semantic Scholar. */
  unpaywall_email?: string;
  /** Number of search results per individual query (default: 5). */
  top_k?: number;
  /** Maximum cross-corpus search queries per citation (default: 2, range 0-4). */
  cross_corpus_queries?: number;
  /** Maximum retry attempts for LLM verdict parsing (default: 3, range 1-5). */
  max_retry_attempts?: number;
  /** Minimum vector similarity score threshold (default: undefined = no filter). */
  min_score?: number;
  /** When true, applies cross-encoder reranking (requires loaded reranker model). */
  rerank?: boolean;
}

/** Response from POST /api/v1/citation/{job_id}/auto-verify. */
export interface AutoVerifyResponse {
  api_version: string;
  status: string;
  job_id: string;
  message: string;
}

/** Request body for POST /api/v1/citation/create.
 *  Supports both single-session (session_id) and multi-session
 *  (session_ids) operation. When session_ids is provided and non-empty,
 *  it overrides session_id. */
export interface CitationCreateRequest {
  tex_path: string;
  bib_path: string;
  session_id: number;
  session_ids?: number[];
  dry_run?: boolean;
  batch_size?: number;
}

/** Response from POST /api/v1/citation/create. */
export interface CitationCreateResponse {
  api_version: string;
  job_id: string;
  total_citations: number;
  total_batches: number;
  matched_pdfs: number;
  unmatched_pdfs: number;
}

/** Single alert from the citation status endpoint, representing a
 *  flagged citation row that requires user attention. The Rust backend
 *  serializes this as a serde_json::Value with these known fields. */
export interface CitationAlert {
  row_id: number;
  cite_key: string;
  flag: string;
  message: string;
  [key: string]: string | number | boolean;
}

/** Response from GET /api/v1/citation/{job_id}/status.
 *  Field names match the Rust CitationStatusResponse DTO directly. */
export interface CitationStatusResponse {
  api_version: string;
  job_id: string;
  job_state: string;
  total: number;
  pending: number;
  claimed: number;
  done: number;
  failed: number;
  total_batches: number;
  batches_done: number;
  batches_pending: number;
  batches_claimed: number;
  verdicts: Record<string, number>;
  alerts: CitationAlert[];
  elapsed_seconds: number;
  is_complete: boolean;
}

/** Single citation row from GET /api/v1/citation/{job_id}/rows.
 *  Matches the serialized Rust CitationRow struct. The result_json field
 *  contains the full verification result (verdict, confidence, reasoning)
 *  as a serialized JSON string for completed rows. */
export interface CitationRowDto {
  id: number;
  job_id: string;
  group_id: number;
  cite_key: string;
  author: string;
  title: string;
  year: string | null;
  tex_line: number;
  anchor_before: string;
  anchor_after: string;
  section_title: string | null;
  matched_file_id: number | null;
  bib_abstract: string | null;
  bib_keywords: string | null;
  tex_context: string | null;
  batch_id: number | null;
  status: string;
  flag: string | null;
  claimed_at: number | null;
  /** Serialized JSON string containing the verification result (SubmitEntry).
   *  Null for pending/claimed rows that have not been verified yet. */
  result_json: string | null;
  co_citation_json: string | null;
}

/** Response from GET /api/v1/citation/{job_id}/rows. */
export interface CitationRowsResponse {
  api_version: string;
  job_id: string;
  rows: CitationRowDto[];
  total: number;
  offset: number;
  limit: number;
}

/** Request body for POST /api/v1/citation/{job_id}/export. */
export interface CitationExportRequest {
  output_directory: string;
  source_directory: string;
  /** Controls whether the annotation pipeline runs automatically after export.
   *  When true, the backend creates an annotation job that highlights matching
   *  quotes in the source PDFs and saves them to annotated_pdfs/ subfolder.
   *  Defaults to true on the backend when omitted. */
  annotate?: boolean;
}

/** Request body for POST /api/v1/annotate/from-file. Reads annotation
 *  instructions from a previously exported CSV or JSON file on disk and
 *  creates an annotation job without re-running the citation verification. */
export interface AnnotateFromFileRequest {
  input_file: string;
  source_directory: string;
  output_directory: string;
  default_color?: string;
}

/** Response from POST /api/v1/annotate/from-file. */
export interface AnnotateFromFileResponse {
  api_version: string;
  job_id: string;
  total_quotes: number;
}

// ---------------------------------------------------------------------------
// Source Fetching (POST /api/v1/citation/fetch-sources)
// ---------------------------------------------------------------------------

/** Request body for POST /api/v1/citation/parse-bib. */
export interface ParseBibRequest {
  bib_path: string;
  /** When provided, each entry is checked against files in this directory
   *  to determine whether the source has already been downloaded. */
  output_directory?: string;
}

/** Single BibTeX entry from the parse-bib endpoint with metadata for
 *  live preview in the Sources tab before downloading starts. Contains
 *  all fields from the .bib file for the expandable detail view. */
export interface BibEntryPreview {
  cite_key: string;
  /** BibTeX entry type (e.g., "article", "book", "inproceedings"). */
  entry_type: string;
  author: string;
  title: string;
  year: string | null;
  has_url: boolean;
  has_doi: boolean;
  url: string | null;
  doi: string | null;
  /** Abstract text from the BibTeX entry. */
  bib_abstract?: string | null;
  /** Keywords from the BibTeX entry. */
  keywords?: string | null;
  /** Whether a downloaded file for this entry already exists in the
   *  output_directory. Only populated when output_directory was provided. */
  file_exists?: boolean;
  /** Filename of the existing file when file_exists is true. */
  existing_file?: string;
  /** The canonical filename that the download pipeline generates for this
   *  entry. Displayed so users can see what filename the system expects. */
  expected_filename?: string | null;
  /** When two or more files match the same BibTeX entry, all relative
   *  file paths are listed here. The UI shows a "duplicate" status badge
   *  and the detail view lists every path for the user to consolidate. */
  duplicate_files?: string[] | null;
  /** Additional BibTeX fields not captured in named fields (e.g., journal,
   *  volume, pages, booktitle, publisher). Keyed by lowercase field name. */
  extra_fields?: Record<string, string>;
}

/** Response from POST /api/v1/citation/parse-bib. */
export interface ParseBibResponse {
  api_version: string;
  entries: BibEntryPreview[];
}

/** Request body for POST /api/v1/citation/bib-report. */
export interface BibReportRequest {
  bib_path: string;
  output_directory: string;
}

/** Response from POST /api/v1/citation/bib-report. */
export interface BibReportResponse {
  api_version: string;
  csv_path: string;
  xlsx_path: string;
  total_entries: number;
  existing_count: number;
  /** Number of entries where two or more files match the same BibTeX entry. */
  duplicate_count: number;
  missing_count: number;
  no_link_count: number;
}

/** Request body for POST /api/v1/citation/fetch-sources. */
export interface FetchSourcesRequest {
  bib_path: string;
  output_directory: string;
  /** Delay in milliseconds between consecutive HTTP requests. Default: 1000. */
  delay_ms?: number;
  /** Email address for Unpaywall API access in the DOI resolution chain.
   *  When provided, DOI resolution queries Unpaywall first for direct PDF URLs. */
  email?: string;
}

/** Single source entry result from the fetch-sources endpoint. Each entry
 *  represents one BibTeX cite-key and the outcome of URL classification,
 *  download, and indexing. */
export interface FetchSourceResult {
  cite_key: string;
  url: string;
  type: string;
  status: string;
  file_path?: string;
  cache_path?: string;
  title?: string;
  error?: string;
  reason?: string;
  doi_resolved_via?: string;
}

/** Response from POST /api/v1/citation/fetch-sources. Contains aggregate
 *  download statistics and per-entry results from the BibTeX source fetching
 *  pipeline. Does not perform indexing; the user indexes downloaded files
 *  separately via the Indexing tab. */
export interface FetchSourcesResponse {
  api_version: string;
  total_entries: number;
  entries_with_url: number;
  pdfs_downloaded: number;
  pdfs_failed: number;
  pdfs_skipped: number;
  html_fetched: number;
  html_failed: number;
  html_blocked: number;
  html_skipped: number;
  results: FetchSourceResult[];
}

// ---------------------------------------------------------------------------
// Citation SSE event payloads
// ---------------------------------------------------------------------------

/** Payload for SSE "citation_row_update" events from the autonomous agent. */
export interface CitationRowUpdateEvent {
  job_id: string;
  row_id: number;
  cite_key: string;
  phase: string;
  verdict?: string;
  confidence?: number;
  reasoning?: string;
  search_queries?: string[];
  error_message?: string;
}

/** Payload for SSE "citation_reasoning_token" events. */
export interface CitationReasoningTokenEvent {
  job_id: string;
  row_id: number;
  token: string;
}

/** Payload for SSE "citation_job_progress" events. */
export interface CitationJobProgressEvent {
  job_id: string;
  rows_done: number;
  rows_total: number;
  verdicts: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Model Doctor (GET /api/v1/web/doctor/models)
// ---------------------------------------------------------------------------

/** Single model diagnostic result from the Model Doctor endpoint. Contains
 *  file-level detail about cache state and a derived health status string. */
export interface ModelHealthEntry {
  model_id: string;
  directory_exists: boolean;
  files_present: string[];
  files_missing: string[];
  checksums_valid: boolean | null;
  total_size_bytes: number;
  repairable: boolean;
  health: "healthy" | "incomplete" | "corrupt" | "missing";
}

/** Response from GET /api/v1/web/doctor/models. */
export interface ModelDoctorResponse {
  models: ModelHealthEntry[];
}

// ---------------------------------------------------------------------------
// Setup status (GET /api/v1/web/setup/status)
// ---------------------------------------------------------------------------

/** Response from GET /api/v1/web/setup/status. Indicates whether this is
 *  the first launch (no .setup_complete marker file exists) and the
 *  absolute path of the data directory for display in the welcome dialog. */
export interface SetupStatusResponse {
  is_first_run: boolean;
  data_dir: string;
}

// ---------------------------------------------------------------------------
// Model catalog (GET /api/v1/web/models/catalog)
// ---------------------------------------------------------------------------

/** Embedding model entry from the models/catalog endpoint. Contains model
 *  metadata, quality ratings, resource requirements, and cache/active status. */
export interface EmbedModel {
  model_id: string;
  display_name: string;
  vector_dimension: number;
  max_seq_len: number;
  quality_rating: string;
  language_scope: string;
  de_en_retrieval: string;
  gpu_recommendation: string;
  ram_requirement: string;
  typical_use_case: string;
  model_size_mb: number;
  cached: boolean;
  active: boolean;
}

/** Reranker model entry from the models/catalog endpoint. Fields mirror the
 *  embedding model columns (language_scope, gpu_recommendation, ram_requirement)
 *  so both tables render with identical structure in the ModelsPanel. */
export interface RerankModel {
  model_id: string;
  display_name: string;
  quality_rating: string;
  layer_count: number;
  param_count_m: number;
  language_scope: string;
  model_size_mb: number;
  gpu_recommendation: string;
  ram_requirement: string;
  cached: boolean;
  loaded: boolean;
}

/** Response from GET /api/v1/web/models/catalog. Contains GPU detection
 *  results, the full embedding model catalog, and the reranker model catalog. */
export interface ModelCatalogResponse {
  gpu_name: string;
  cuda_available: boolean;
  embedding_models: EmbedModel[];
  reranker_models: RerankModel[];
  /** Model ID of the reranker that is currently loaded. Populated when a
   *  reranker was loaded before the catalog request (e.g., from a previous
   *  session). Each entry in reranker_models has its own loaded flag. */
  reranker_loaded_id?: string;
}

/** Request body for POST /api/v1/web/models/download. */
export interface ModelDownloadRequest {
  model_id: string;
}

/** Response from POST /api/v1/web/models/download. */
export interface ModelDownloadResponse {
  status: string;
  model_id: string;
  error?: string;
}

/** Request body for POST /api/v1/web/models/activate. */
export interface ModelActivateRequest {
  model_id: string;
}

/** Response from POST /api/v1/web/models/activate. */
export interface ModelActivateResponse {
  status: string;
  model_id: string;
  vector_dimension?: number;
  error?: string;
}

/** Request body for POST /api/v1/web/models/load-reranker. */
export interface LoadRerankerRequest {
  model_id: string;
}

/** Response from POST /api/v1/web/models/load-reranker. */
export interface LoadRerankerResponse {
  status: string;
  model_id: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// Dependency probes (GET /api/v1/web/doctor/probes)
// Shared between DoctorPanel and WelcomeDialog.
// ---------------------------------------------------------------------------

/** Single dependency probe result from the server. Describes the availability,
 *  installability, and version of a runtime dependency (pdfium, tesseract,
 *  ONNX Runtime, Ollama, poppler). */
export interface DependencyProbe {
  name: string;
  /** Identifier accepted by POST /api/v1/web/doctor/install (e.g., "pdfium",
   *  "tesseract", "onnxruntime"). Empty string for non-installable deps. */
  install_id: string;
  /** One-line description of what this dependency does. */
  purpose: string;
  available: boolean;
  installable: boolean;
  hint: string;
  /** URL for manual installation instructions. Empty when not applicable. */
  link: string;
  /** Short version string (e.g., "1.23.2"). Empty when version is unknown. */
  version: string;
}

/** Request body for POST /api/v1/web/doctor/install. */
export interface DoctorInstallRequest {
  dependency: string;
}

/** Response from POST /api/v1/web/doctor/install. */
export interface DoctorInstallResponse {
  status: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// MCP Server (GET /api/v1/web/mcp/status)
// ---------------------------------------------------------------------------

/** MCP registration status response from GET /api/v1/web/mcp/status.
 *  Reports whether the NeuronCite MCP server is registered in the
 *  Claude Code configuration file (~/.claude.json mcpServers key).
 *  Field names match the Rust McpStatusResponse struct in
 *  crates/neuroncite-web/src/handlers/mcp.rs exactly. */
export interface McpStatusResponse {
  /** True when mcpServers.neuroncite is present in ~/.claude.json. */
  registered: boolean;
  /** The binary's own CARGO_PKG_VERSION string. */
  server_version: string;
}

// ---------------------------------------------------------------------------
// Setup complete (POST /api/v1/web/setup/complete)
// ---------------------------------------------------------------------------

/** Response from POST /api/v1/web/setup/complete. Writes the .setup_complete
 *  marker file to suppress the welcome dialog on subsequent launches. */
export interface SetupCompleteResponse {
  status: string;
}

// ---------------------------------------------------------------------------
// Native file/folder browse responses
// ---------------------------------------------------------------------------

/** Response from POST /api/v1/web/browse/native and
 *  POST /api/v1/web/browse/native-file. Both endpoints return the
 *  same shape: the selected path and whether the user confirmed. */
export interface NativeBrowseResponse {
  path: string;
  selected: boolean;
}
