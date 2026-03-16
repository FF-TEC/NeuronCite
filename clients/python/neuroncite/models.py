"""Typed dataclasses for all NeuronCite REST API (v1) response structures.

Each dataclass mirrors one JSON response object returned by the NeuronCite
server. Field names match the wire-format JSON keys exactly. The ``api_version``
field present in every Rust response DTO is stripped by the client layer and
does not appear in these models -- callers receive only the payload fields.

Optional fields that the server may omit are typed as ``X | None`` and default
to ``None``. Numeric identifiers use ``int``; scores use ``float``; timestamps
are Unix epoch integers; free-form text uses ``str``. Fields typed as ``dict``
or ``list[dict]`` correspond to ``serde_json::Value`` in Rust where the inner
schema is variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Health & system
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HealthResponse:
    """Response from ``GET /api/v1/health``.

    Contains runtime status: application version, compiled feature flags, GPU
    availability, active embedding backend, reranker readiness, and the
    availability of optional extraction backends (pdfium, Tesseract).
    """

    api_version: str
    version: str
    build_features: list[str]
    gpu_available: bool
    active_backend: str
    reranker_available: bool
    pdfium_available: bool
    tesseract_available: bool


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Single entry in the backend list from ``GET /api/v1/backends``.

    Describes one compiled-in embedding backend with its identifier, GPU
    support flag, and the count of models available through it.
    """

    name: str
    gpu_supported: bool
    model_count: int


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IndexResponse:
    """Response from ``POST /api/v1/index``.

    Contains the UUID of the background indexing job and the numeric session
    ID (either newly created or reused from an existing session with matching
    parameters).
    """

    api_version: str
    job_id: str
    session_id: int


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single ranked result from ``POST /api/v1/search``.

    Contains the final relevance score, chunk text, a formatted citation
    string, diagnostic sub-scores from vector and keyword retrieval, source
    file metadata, chunk positional index, and the ``file_id`` for subsequent
    page retrieval via ``GET /api/v1/documents/{id}/pages/{n}``.
    """

    score: float
    content: str
    citation: str
    vector_score: float
    file_id: int
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    bm25_rank: int | None = None
    reranker_score: float | None = None
    session_id: int | None = None
    doc_created_at: int | None = None
    doc_modified_at: int | None = None
    doc_author: str | None = None


@dataclass(frozen=True, slots=True)
class MultiSearchSessionStat:
    """Per-session status in a multi-search response.

    Reports the number of results each session contributed and the outcome
    of the search operation (ok, no_hnsw_index, or error).
    """

    session_id: int
    result_count: int
    status: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MultiSearchResponse:
    """Response from ``POST /api/v1/search/multi``.

    Contains the session IDs that were searched, per-session statistics,
    and the merged ranked results from all sessions.
    """

    session_ids: list[int]
    session_stats: list[MultiSearchSessionStat]
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VerifyResponse:
    """Response from ``POST /api/v1/verify``.

    Contains a weighted verification verdict derived from keyword overlap
    (Jaccard similarity, 30% weight) and semantic similarity (cosine, 70%
    weight). The ``verdict`` field is one of: ``supports``, ``partial``,
    ``not_supported``.
    """

    api_version: str
    verdict: str
    combined_score: float
    keyword_score: float
    semantic_score: float


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class JobStatus:
    """Response from ``GET /api/v1/jobs/{id}`` and entries in the job list.

    Contains the job UUID, kind (e.g., ``index``, ``rebuild``), associated
    session, execution state, flat progress counters, and timestamps as Unix
    epoch integers. The ``state`` field is one of: ``queued``, ``running``,
    ``completed``, ``failed``, ``canceled``.
    """

    id: str
    kind: str
    session_id: int | None
    state: str
    progress_done: int
    progress_total: int
    error_message: str | None
    created_at: int
    started_at: int | None
    finished_at: int | None


@dataclass(frozen=True, slots=True)
class JobCancelResponse:
    """Response from ``POST /api/v1/jobs/{id}/cancel``.

    Confirms the job was transitioned to the ``canceled`` state.
    """

    job_id: str
    state: str


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Single entry from ``GET /api/v1/sessions``.

    Contains session metadata (directory, model, chunking strategy, vector
    dimensionality), aggregate statistics (file count, pages, chunks, content
    bytes), and optional chunking parameters (chunk_size, chunk_overlap,
    max_words) that depend on the chosen chunk strategy.
    """

    id: int
    directory_path: str
    model_name: str
    chunk_strategy: str
    vector_dimension: int
    created_at: int
    file_count: int
    total_pages: int
    total_chunks: int
    total_content_bytes: int
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    max_words: int | None = None


@dataclass(frozen=True, slots=True)
class SessionDeleteResponse:
    """Response from ``DELETE /api/v1/sessions/{id}``.

    ``deleted`` is ``True`` if the session existed and was removed, ``False``
    if the session was already absent.
    """

    deleted: bool


@dataclass(frozen=True, slots=True)
class SessionDeleteByDirectoryResponse:
    """Response from ``POST /api/v1/sessions/delete-by-directory``.

    Contains the list of session IDs that were deleted, the canonicalized
    directory path that was matched, and whether the directory matched any
    existing sessions.
    """

    deleted_session_ids: list[int]
    directory: str
    matched_directory: bool


@dataclass(frozen=True, slots=True)
class OptimizeResponse:
    """Response from ``POST /api/v1/sessions/{id}/optimize``.

    Confirms the FTS5 optimize operation was triggered.
    """

    status: str


@dataclass(frozen=True, slots=True)
class RebuildResponse:
    """Response from ``POST /api/v1/sessions/{id}/rebuild``.

    Contains the UUID of the rebuild job.
    """

    job_id: str


# ---------------------------------------------------------------------------
# Documents & chunks
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PageResponse:
    """Response from ``GET /api/v1/documents/{id}/pages/{n}``.

    Contains the text content of a single page, its 1-indexed page number,
    and the extraction backend that produced the text (e.g., ``pdfium``,
    ``tesseract``).
    """

    page_number: int
    content: str
    backend: str


@dataclass(frozen=True, slots=True)
class ChunkDto:
    """Single chunk record from the chunks list endpoint.

    Contains the chunk text content together with positional metadata (page
    range, index within file) and computed size metrics (word count, byte count).
    """

    chunk_id: int
    chunk_index: int
    page_start: int
    page_end: int
    word_count: int
    byte_count: int
    content: str


@dataclass(frozen=True, slots=True)
class ChunksResponse:
    """Response from ``GET /api/v1/sessions/{session_id}/files/{file_id}/chunks``.

    Contains paginated chunk records with their content, page ranges, and
    word counts. The ``chunks`` list contains dict objects whose fields
    match the chunk schema returned by the server.
    """

    session_id: int
    file_id: int
    total_chunks: int
    offset: int
    limit: int
    returned: int
    chunks: list[dict]


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExtractionSummaryDto:
    """Aggregate extraction method distribution across all files in a session.

    Categorizes each file as native-text-only, OCR-only, or mixed, and reports
    page-level totals and average content density.
    """

    total_files: int
    native_text_count: int
    ocr_required_count: int
    mixed_count: int
    total_pages: int
    total_empty_pages: int
    avg_bytes_per_page: int


@dataclass(frozen=True, slots=True)
class QualityFlagDetailsDto:
    """Per-page-count details for a file with quality flags.

    Contains the raw extraction metrics used to determine which quality flags
    apply to a given file.
    """

    page_count: int
    native_pages: int
    ocr_pages: int
    empty_pages: int
    total_bytes: int
    pdf_page_count: int | None = None


@dataclass(frozen=True, slots=True)
class QualityFlagDto:
    """Quality flags for one file that has at least one detected issue.

    Contains the file identifier, detected flag strings, and the underlying
    extraction metrics that triggered the flags.
    """

    file_id: int
    file_name: str
    flags: list[str]
    details: QualityFlagDetailsDto


@dataclass(frozen=True, slots=True)
class QualityReportResponse:
    """Response from ``GET /api/v1/sessions/{id}/quality``.

    Contains an extraction method distribution summary, per-file quality
    flags for files with detected issues, and aggregate clean/issue counts.
    """

    session_id: int
    extraction_summary: dict
    quality_flags: list[dict]
    files_with_issues: int
    files_clean: int


# ---------------------------------------------------------------------------
# File comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FileComparisonSessionDto:
    """Per-session data for one indexed instance of a compared file.

    Contains the session and file identifiers, indexing configuration, and
    extraction/chunk statistics for one instance of a file across sessions.
    """

    session_id: int
    file_id: int
    model_name: str
    chunk_strategy: str
    pages_extracted: int
    chunks: int
    avg_chunk_bytes: float
    label: str | None = None
    pdf_page_count: int | None = None


@dataclass(frozen=True, slots=True)
class FileComparisonDto:
    """Cross-session comparison result for one file name.

    Contains the base file name, number of indexed instances across all
    sessions, per-session data, content hash consistency flag, and chunk
    count range.
    """

    file_name: str
    instances: int
    sessions: list[FileComparisonSessionDto]
    same_content: bool
    chunk_count_range: list[int]


@dataclass(frozen=True, slots=True)
class FileCompareResponse:
    """Response from ``POST /api/v1/files/compare``.

    Contains the matched file count and per-file comparison data across
    sessions with different chunking configurations.
    """

    pattern: str
    matched_files: int
    comparisons: list[dict]


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FilesystemSummaryDto:
    """Filesystem scan results for the discover endpoint.

    Contains per-format file counts and the total size in bytes of all
    discovered indexable files.
    """

    type_counts: dict
    total_size_bytes: int


@dataclass(frozen=True, slots=True)
class DiscoverSessionDto:
    """Per-session index data with aggregate statistics for the discover endpoint.

    Contains session metadata, indexing configuration, and aggregate statistics
    (file count, chunks, pages, content bytes, word estimate).
    """

    session_id: int
    model_name: str
    chunk_strategy: str
    file_count: int
    total_chunks: int
    total_pages: int
    total_content_bytes: int
    total_words: int
    created_at: int
    label: str | None = None
    chunk_size: int | None = None


@dataclass(frozen=True, slots=True)
class DiscoverResponse:
    """Response from ``POST /api/v1/discover``.

    Reports filesystem scan results (per-format file counts, total size),
    per-session index data with aggregate statistics, and a list of files
    on disk that are not indexed in any session.
    """

    directory: str
    directory_exists: bool
    filesystem: dict
    sessions: list[dict]
    unindexed_files: list[str]


# ---------------------------------------------------------------------------
# Backends (list response)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BackendListResponse:
    """Response wrapper from ``GET /api/v1/backends``.

    The REST endpoint returns ``{"api_version": "v1", "backends": [...]}``.
    This wrapper holds the list of ``BackendInfo`` entries extracted from the
    ``backends`` array.
    """

    backends: list[BackendInfo]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AnnotateResponse:
    """Response from ``POST /api/v1/annotate``.

    Contains the UUID of the background annotation job and the total number
    of annotation rows parsed from the input data.
    """

    job_id: str
    total_quotes: int


@dataclass(frozen=True, slots=True)
class AnnotateFromFileResponse:
    """Response from ``POST /api/v1/annotate/from-file``.

    Contains the UUID of the background annotation job and the total number
    of annotation rows parsed from the input file on disk.
    """

    job_id: str
    total_quotes: int


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CitationCreateResponse:
    """Response from ``POST /api/v1/citation/create``.

    Contains the created job UUID, linked session ID, total citation row count,
    batch count, cite-key match statistics, and any unresolved cite-keys not
    found in the BibTeX file.
    """

    job_id: str
    session_id: int
    total_citations: int
    total_batches: int
    unique_cite_keys: int
    cite_keys_matched: int
    cite_keys_unmatched: int
    unresolved_cite_keys: list[str]


@dataclass(frozen=True, slots=True)
class CitationClaimResponse:
    """Response from ``POST /api/v1/citation/claim``.

    Contains the claimed batch data. When no pending batch is available,
    ``batch_id`` is ``None`` and ``remaining_batches`` indicates how many
    batches are still outstanding. ``rows`` contains the citation row dicts
    for the claimed batch.
    """

    batch_id: int | None
    tex_path: str | None
    bib_path: str | None
    session_id: int | None
    rows: list[dict]
    remaining_batches: int | None
    message: str | None


@dataclass(frozen=True, slots=True)
class CitationSubmitResponse:
    """Response from ``POST /api/v1/citation/submit``.

    Confirms the number of rows submitted and whether all batches in the
    job are now complete.
    """

    status: str
    rows_submitted: int
    total: int
    is_complete: bool


@dataclass(frozen=True, slots=True)
class CitationStatusResponse:
    """Response from ``GET /api/v1/citation/{job_id}/status``.

    Contains row counts by status, batch counts, verdict distribution, flagged
    alerts, elapsed time, and completion status for a citation verification job.
    """

    job_id: str
    job_state: str
    total: int
    pending: int
    claimed: int
    done: int
    failed: int
    total_batches: int
    batches_done: int
    batches_pending: int
    batches_claimed: int
    verdicts: dict
    alerts: list[dict]
    elapsed_seconds: int
    is_complete: bool


@dataclass(frozen=True, slots=True)
class CitationRowsResponse:
    """Response from ``GET /api/v1/citation/{job_id}/rows``.

    Contains paginated citation rows with their verification result data
    and total count for cursor-based paging.
    """

    job_id: str
    rows: list[dict]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True, slots=True)
class CitationExportResponse:
    """Response from ``POST /api/v1/citation/{job_id}/export``.

    Contains paths to all generated export files (annotation CSV, data CSV,
    Excel workbook, report JSON, corrections JSON, full-detail JSON) and an
    optional annotation job UUID.
    """

    annotation_job_id: str | None
    annotation_csv_path: str
    data_csv_path: str
    excel_path: str
    report_path: str
    corrections_path: str
    full_detail_path: str
    summary: dict


@dataclass(frozen=True, slots=True)
class AutoVerifyResponse:
    """Response from ``POST /api/v1/citation/{job_id}/auto-verify``.

    Confirms that the automatic citation verification agent was started for
    the specified job. The ``status`` field is ``started`` on success.
    """

    status: str
    job_id: str
    message: str


@dataclass(frozen=True, slots=True)
class FetchSourcesResponse:
    """Response from ``POST /api/v1/citation/fetch-sources``.

    Contains aggregate statistics from the BibTeX source fetching pipeline:
    counts of parsed entries, downloaded PDFs, fetched HTML pages, and
    per-entry result details.
    """

    total_entries: int
    entries_with_url: int
    pdfs_downloaded: int
    pdfs_failed: int
    pdfs_skipped: int
    html_fetched: int
    html_failed: int
    html_blocked: int
    html_skipped: int
    results: list[dict]


@dataclass(frozen=True, slots=True)
class BibEntryPreview:
    """Single BibTeX entry in the parse-bib response.

    Contains all metadata fields from the .bib file for live preview in the
    Sources tab. Fields like ``bib_abstract``, ``keywords``, ``existing_file``,
    ``expected_filename``, and ``duplicate_files`` are optional because they
    depend on the BibTeX content and whether an output_directory was provided.
    """

    cite_key: str
    entry_type: str
    author: str
    title: str
    has_url: bool
    has_doi: bool
    file_exists: bool
    year: str | None = None
    url: str | None = None
    doi: str | None = None
    bib_abstract: str | None = None
    keywords: str | None = None
    existing_file: str | None = None
    expected_filename: str | None = None
    duplicate_files: list[str] | None = None
    extra_fields: dict | None = None


@dataclass(frozen=True, slots=True)
class ParseBibResponse:
    """Response from ``POST /api/v1/citation/parse-bib``.

    Contains the parsed BibTeX entries with metadata for live preview.
    """

    entries: list[BibEntryPreview]


@dataclass(frozen=True, slots=True)
class BibReportResponse:
    """Response from ``POST /api/v1/citation/bib-report``.

    Contains paths to the generated CSV and XLSX report files, total entry
    count, and per-status breakdown (existing, duplicate, missing, no link).
    """

    csv_path: str
    xlsx_path: str
    total_entries: int
    existing_count: int
    duplicate_count: int
    missing_count: int
    no_link_count: int


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ShutdownResponse:
    """Response from ``POST /api/v1/shutdown``.

    Confirms that graceful shutdown was initiated.
    """

    status: str
