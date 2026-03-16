"""HTTP client wrapper for the NeuronCite REST API (v1).

``NeuronCiteClient`` translates typed Python method calls into HTTP requests
against a running NeuronCite server instance. Every public method corresponds
to exactly one REST API endpoint defined in the Axum router
(``crates/neuroncite-api/src/router.rs``). Return values are typed dataclasses
defined in ``neuroncite.models``.

The client strips the ``api_version`` field present in every Rust response
DTO so that callers receive only the payload fields.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from neuroncite.models import (
    AnnotateFromFileResponse,
    AnnotateResponse,
    AutoVerifyResponse,
    BackendInfo,
    BibEntryPreview,
    BibReportResponse,
    ChunksResponse,
    CitationClaimResponse,
    CitationCreateResponse,
    CitationExportResponse,
    CitationRowsResponse,
    CitationStatusResponse,
    CitationSubmitResponse,
    DiscoverResponse,
    FetchSourcesResponse,
    FileCompareResponse,
    HealthResponse,
    IndexResponse,
    JobCancelResponse,
    JobStatus,
    MultiSearchResponse,
    MultiSearchSessionStat,
    OptimizeResponse,
    PageResponse,
    ParseBibResponse,
    QualityReportResponse,
    RebuildResponse,
    SearchResult,
    SessionDeleteByDirectoryResponse,
    SessionDeleteResponse,
    SessionInfo,
    ShutdownResponse,
    VerifyResponse,
)


class NeuronCiteError(Exception):
    """Raised when the NeuronCite API returns a non-success HTTP status.

    Attributes:
        status_code: The HTTP status code from the server response.
        code:        The machine-readable error identifier from the JSON body,
                     or ``None`` if the response body was not valid JSON.
        message:     The human-readable error message from the JSON body.
    """

    def __init__(self, status_code: int, code: str | None, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class NeuronCiteClient:
    """Typed wrapper around the NeuronCite REST API (v1).

    All methods raise ``NeuronCiteError`` when the server returns a non-2xx
    status code, and ``requests.ConnectionError`` when the server is
    unreachable.

    Args:
        base_url: Root URL of the NeuronCite server (without trailing slash).
        token:    Bearer token for LAN-mode authentication. ``None`` when the
                  server binds to localhost and does not require authentication.
        timeout:  Default timeout in seconds for all HTTP requests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:3030",
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._session = requests.Session()
        if token is not None:
            self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request and return the decoded JSON body.

        Args:
            method:  HTTP method (GET, POST, DELETE).
            path:    URL path relative to the API root (e.g. "/api/v1/health").
            json:    Optional JSON body for POST requests.
            params:  Optional query string parameters for GET requests.
            headers: Optional extra headers merged into the request.

        Returns:
            The parsed JSON response body as a dictionary.

        Raises:
            NeuronCiteError: If the server returns a non-2xx status code.
        """
        url = f"{self._base_url}{path}"
        response = self._session.request(
            method,
            url,
            json=json,
            params=params,
            headers=headers,
            timeout=self._timeout,
        )
        if not response.ok:
            try:
                body = response.json()
                code = body.get("code")
                message = body.get("error", response.text)
            except (ValueError, KeyError):
                code = None
                message = response.text
            raise NeuronCiteError(response.status_code, code, message)

        # Endpoints that return 202 Accepted or 204 No Content with no body.
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    @staticmethod
    def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *d* with all ``None``-valued keys removed.

        The NeuronCite API treats absent fields as "use server default", so
        ``None`` values must not appear in the serialized JSON.
        """
        return {k: v for k, v in d.items() if v is not None}

    # ------------------------------------------------------------------
    # Health and discovery
    # ------------------------------------------------------------------

    def health(self) -> HealthResponse:
        """``GET /api/v1/health`` -- Retrieve server health and configuration.

        Returns:
            A ``HealthResponse`` with version info, feature flags, GPU status,
            backend info, reranker availability, and extraction backend status.
        """
        data = self._request("GET", "/api/v1/health")
        return HealthResponse(
            api_version=data["api_version"],
            version=data["version"],
            build_features=data["build_features"],
            gpu_available=data["gpu_available"],
            active_backend=data["active_backend"],
            reranker_available=data["reranker_available"],
            pdfium_available=data["pdfium_available"],
            tesseract_available=data["tesseract_available"],
        )

    def backends(self) -> list[BackendInfo]:
        """``GET /api/v1/backends`` -- List compiled-in embedding backends.

        Returns:
            A list of ``BackendInfo`` objects, one per backend.
        """
        data = self._request("GET", "/api/v1/backends")
        return [
            BackendInfo(
                name=b["name"],
                gpu_supported=b["gpu_supported"],
                model_count=b["model_count"],
            )
            for b in data["backends"]
        ]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        directory: str,
        model_name: str | None = None,
        chunk_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        idempotency_key: str | None = None,
    ) -> IndexResponse:
        """``POST /api/v1/index`` -- Start a background indexing job.

        Args:
            directory:       Absolute path to the directory containing PDFs.
            model_name:      Embedding model identifier (e.g.
                             ``BAAI/bge-small-en-v1.5``). ``None`` uses the
                             server-configured default.
            chunk_strategy:  Chunking strategy (``word``, ``token``,
                             ``sentence``, ``page``). ``None`` uses the
                             server-configured default.
            chunk_size:      Window size in words or tokens. ``None`` for
                             strategies that do not use a fixed window.
            chunk_overlap:   Overlap in words or tokens. ``None`` when not
                             applicable.
            idempotency_key: Client-chosen string (typically a UUID) for
                             deduplication. Prevents duplicate jobs on retried
                             requests.

        Returns:
            An ``IndexResponse`` with the ``job_id`` (UUID) and ``session_id``.
        """
        body = self._strip_none(
            {
                "directory": directory,
                "model_name": model_name,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "idempotency_key": idempotency_key,
            }
        )
        data = self._request("POST", "/api/v1/index", json=body)
        return IndexResponse(
            api_version=data["api_version"],
            job_id=data["job_id"],
            session_id=data["session_id"],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(items: list[dict[str, Any]]) -> list[SearchResult]:
        """Convert a list of raw JSON result dicts into ``SearchResult`` objects.

        Each dict corresponds to one ``SearchResultDto`` from the Rust API. The
        ``citation`` field is a flat formatted string, ``vector_score`` is always
        present, and ``file_id`` identifies the source document.
        """
        results: list[SearchResult] = []
        for item in items:
            results.append(
                SearchResult(
                    score=item["score"],
                    content=item["content"],
                    citation=item["citation"],
                    vector_score=item["vector_score"],
                    file_id=item["file_id"],
                    source_file=item["source_file"],
                    page_start=item["page_start"],
                    page_end=item["page_end"],
                    chunk_index=item["chunk_index"],
                    bm25_rank=item.get("bm25_rank"),
                    reranker_score=item.get("reranker_score"),
                    session_id=item.get("session_id"),
                    doc_created_at=item.get("doc_created_at"),
                    doc_modified_at=item.get("doc_modified_at"),
                    doc_author=item.get("doc_author"),
                )
            )
        return results

    def search(
        self,
        session_id: int,
        query: str,
        top_k: int = 10,
        use_fts: bool = True,
        rerank: bool = False,
        refine: bool = True,
        refine_divisors: str | None = None,
    ) -> list[SearchResult]:
        """``POST /api/v1/search`` -- Hybrid vector + keyword search.

        Performs a combined search using vector similarity and FTS5 keyword
        matching with Reciprocal Rank Fusion (RRF). When ``use_fts`` is
        ``False``, keyword results are omitted and fusion degrades to
        vector-only ranking. When ``rerank`` is ``True``, cross-encoder
        reranking is applied.

        Args:
            session_id:      Target index session identifier.
            query:           Natural-language search query.
            top_k:           Maximum number of results to return (1-200).
            use_fts:         Enable BM25 keyword matching via FTS5.
            rerank:          Enable cross-encoder reranking of fused results.
            refine:          Enable sub-chunk refinement for narrowing results
                             to the most relevant passage within each chunk.
            refine_divisors: Comma-separated divisor values controlling the
                             sub-chunk split granularity (e.g. ``4,8,16``).

        Returns:
            A ranked list of ``SearchResult`` objects.
        """
        body = self._strip_none(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "use_fts": use_fts,
                "rerank": rerank,
                "refine": refine,
                "refine_divisors": refine_divisors,
            }
        )
        data = self._request("POST", "/api/v1/search", json=body)
        return self._parse_results(data["results"])

    def hybrid_search(
        self,
        session_id: int,
        query: str,
        top_k: int = 10,
        use_fts: bool = True,
        rerank: bool = False,
        refine: bool = True,
        refine_divisors: str | None = None,
    ) -> list[SearchResult]:
        """``POST /api/v1/search/hybrid`` -- Alias for hybrid search.

        Identical to ``search()`` but uses the ``/search/hybrid`` endpoint
        path. Both endpoints share the same handler in the server. This method
        exists for backward compatibility with code that explicitly targets
        the hybrid search URL.

        See ``search()`` for full parameter documentation.
        """
        body = self._strip_none(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "use_fts": use_fts,
                "rerank": rerank,
                "refine": refine,
                "refine_divisors": refine_divisors,
            }
        )
        data = self._request("POST", "/api/v1/search/hybrid", json=body)
        return self._parse_results(data["results"])

    def multi_search(
        self,
        session_ids: list[int],
        query: str,
        top_k: int = 10,
        use_fts: bool = True,
        rerank: bool = False,
        refine: bool = True,
        refine_divisors: str | None = None,
    ) -> MultiSearchResponse:
        """``POST /api/v1/search/multi`` -- Search across multiple sessions.

        Performs a combined search across multiple index sessions simultaneously.
        The query is embedded once, then the search pipeline runs for each session.
        Results are merged into a single ranked list sorted by score, with each
        result tagged by its source session_id.

        Args:
            session_ids:     List of session IDs to search across (1-10 sessions).
            query:           Natural-language search query.
            top_k:           Maximum number of results to return (1-50).
            use_fts:         Enable BM25 keyword matching via FTS5.
            rerank:          Enable cross-encoder reranking of merged results.
            refine:          Enable sub-chunk refinement after the initial search.
            refine_divisors: Comma-separated divisor values controlling the
                             sub-chunk split granularity (e.g. ``4,8,16``).

        Returns:
            A ``MultiSearchResponse`` with session stats and merged results.
        """
        body = self._strip_none(
            {
                "query": query,
                "session_ids": session_ids,
                "top_k": top_k,
                "use_fts": use_fts,
                "rerank": rerank,
                "refine": refine,
                "refine_divisors": refine_divisors,
            }
        )
        data = self._request("POST", "/api/v1/search/multi", json=body)
        return MultiSearchResponse(
            session_ids=data["session_ids"],
            session_stats=[
                MultiSearchSessionStat(
                    session_id=s["session_id"],
                    result_count=s["result_count"],
                    status=s["status"],
                    error=s.get("error"),
                )
                for s in data["session_stats"]
            ],
            results=self._parse_results(data["results"]),
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: str,
        session_id: int,
        chunk_ids: list[int],
    ) -> VerifyResponse:
        """``POST /api/v1/verify`` -- Heuristic claim verification.

        Scores the claim against the cited chunks using a weighted combination
        of keyword overlap (Jaccard, 30%) and semantic similarity (cosine, 70%).

        Args:
            claim:      The assertion to verify against the cited chunks.
            session_id: Session ID containing the cited chunks. Chunks
                        belonging to a different session are rejected.
            chunk_ids:  Database chunk IDs whose stored text and embeddings
                        are used for scoring.

        Returns:
            A ``VerifyResponse`` with the verdict (``supports``, ``partial``,
            ``not_supported``), combined score, and individual component scores.
        """
        body = {
            "claim": claim,
            "session_id": session_id,
            "chunk_ids": chunk_ids,
        }
        data = self._request("POST", "/api/v1/verify", json=body)
        return VerifyResponse(
            api_version=data["api_version"],
            verdict=data["verdict"],
            combined_score=data["combined_score"],
            keyword_score=data["keyword_score"],
            semantic_score=data["semantic_score"],
        )

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_job(data: dict[str, Any]) -> JobStatus:
        """Convert a raw JSON job dict into a ``JobStatus`` object.

        The Rust ``JobResponse`` uses flat progress fields (``progress_done``,
        ``progress_total``) and Unix epoch timestamps as integers.
        """
        return JobStatus(
            id=data["id"],
            kind=data["kind"],
            session_id=data.get("session_id"),
            state=data["state"],
            progress_done=data["progress_done"],
            progress_total=data["progress_total"],
            error_message=data.get("error_message"),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )

    def get_job(self, job_id: str) -> JobStatus:
        """``GET /api/v1/jobs/{id}`` -- Poll a single background job.

        Args:
            job_id: UUID of the job (returned by ``index()``, ``annotate()``,
                    or ``rebuild_index()``).

        Returns:
            The current ``JobStatus`` of the requested job.
        """
        data = self._request("GET", f"/api/v1/jobs/{job_id}")
        return self._parse_job(data)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        timeout: float = 3600,
    ) -> JobStatus:
        """Poll a background job until it reaches a terminal state.

        Repeatedly calls ``get_job()`` and sleeps between polls. Returns as
        soon as the job state is ``completed``, ``failed``, or ``canceled``.

        Args:
            job_id:        UUID of the job to wait for.
            poll_interval: Seconds between consecutive poll requests.
            timeout:       Maximum total seconds to wait before raising
                           ``TimeoutError``.

        Returns:
            The terminal ``JobStatus``.

        Raises:
            TimeoutError: If the job does not finish within *timeout* seconds.
        """
        terminal_states = {"completed", "failed", "canceled"}
        deadline = time.monotonic() + timeout
        while True:
            status = self.get_job(job_id)
            if status.state in terminal_states:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {job_id} did not finish within {timeout} seconds "
                    f"(last state: {status.state})"
                )
            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> JobCancelResponse:
        """``POST /api/v1/jobs/{id}/cancel`` -- Request cooperative cancellation.

        The current processing batch completes before cancellation takes effect.

        Args:
            job_id: UUID of the job to cancel.

        Returns:
            A ``JobCancelResponse`` confirming the state transition.
        """
        data = self._request("POST", f"/api/v1/jobs/{job_id}/cancel")
        return JobCancelResponse(
            job_id=data["job_id"],
            state=data["state"],
        )

    def list_jobs(self) -> list[JobStatus]:
        """``GET /api/v1/jobs`` -- List all tracked jobs.

        Returns active and recently completed/failed jobs within the 24-hour
        retention window, ordered by creation time descending.

        Returns:
            A list of ``JobStatus`` objects.
        """
        data = self._request("GET", "/api/v1/jobs")
        return [self._parse_job(j) for j in data["jobs"]]

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[SessionInfo]:
        """``GET /api/v1/sessions`` -- List all index sessions.

        Returns:
            A list of ``SessionInfo`` objects with configuration metadata and
            aggregate statistics.
        """
        data = self._request("GET", "/api/v1/sessions")
        return [
            SessionInfo(
                id=s["id"],
                directory_path=s["directory_path"],
                model_name=s["model_name"],
                chunk_strategy=s["chunk_strategy"],
                vector_dimension=s["vector_dimension"],
                created_at=s["created_at"],
                file_count=s["file_count"],
                total_pages=s["total_pages"],
                total_chunks=s["total_chunks"],
                total_content_bytes=s["total_content_bytes"],
                chunk_size=s.get("chunk_size"),
                chunk_overlap=s.get("chunk_overlap"),
                max_words=s.get("max_words"),
            )
            for s in data["sessions"]
        ]

    def delete_session(self, session_id: int) -> SessionDeleteResponse:
        """``DELETE /api/v1/sessions/{id}`` -- Delete a session and all its data.

        Removes the session row and cascades to all associated chunks, FTS5
        entries, and the HNSW index file.

        Args:
            session_id: Numeric identifier of the session to delete.

        Returns:
            A ``SessionDeleteResponse`` indicating whether the session existed.
        """
        data = self._request("DELETE", f"/api/v1/sessions/{session_id}")
        return SessionDeleteResponse(deleted=data["deleted"])

    def delete_sessions_by_directory(self, directory: str) -> SessionDeleteByDirectoryResponse:
        """``POST /api/v1/sessions/delete-by-directory`` -- Delete all sessions for a directory.

        The directory path is canonicalized server-side before comparison.

        Args:
            directory: Absolute path to the PDF directory whose sessions should
                       be deleted.

        Returns:
            A ``SessionDeleteByDirectoryResponse`` with the deleted session IDs
            and the canonicalized directory.
        """
        data = self._request(
            "POST",
            "/api/v1/sessions/delete-by-directory",
            json={"directory": directory},
        )
        return SessionDeleteByDirectoryResponse(
            deleted_session_ids=data["deleted_session_ids"],
            directory=data["directory"],
            matched_directory=data["matched_directory"],
        )

    def optimize_session(self, session_id: int) -> OptimizeResponse:
        """``POST /api/v1/sessions/{id}/optimize`` -- Trigger FTS5 optimization.

        Runs the SQLite FTS5 ``optimize`` command to merge the inverted index
        segments for the session's chunks. This reduces disk usage and improves
        keyword search performance.

        Args:
            session_id: Numeric identifier of the session to optimize.

        Returns:
            An ``OptimizeResponse`` confirming the operation was triggered.
        """
        data = self._request("POST", f"/api/v1/sessions/{session_id}/optimize")
        return OptimizeResponse(status=data["status"])

    def rebuild_index(self, session_id: int) -> RebuildResponse:
        """``POST /api/v1/sessions/{id}/rebuild`` -- Trigger HNSW index rebuild.

        Creates a background job that rebuilds the HNSW vector index for the
        session from scratch using the stored embeddings.

        Args:
            session_id: Numeric identifier of the session to rebuild.

        Returns:
            A ``RebuildResponse`` with the UUID of the rebuild job.
        """
        data = self._request("POST", f"/api/v1/sessions/{session_id}/rebuild")
        return RebuildResponse(job_id=data["job_id"])

    # ------------------------------------------------------------------
    # Document and chunk retrieval
    # ------------------------------------------------------------------

    def get_page(self, file_id: int, page_number: int) -> PageResponse:
        """``GET /api/v1/documents/{id}/pages/{n}`` -- Retrieve page text.

        Returns the full text content of a single page from an indexed
        document.

        Args:
            file_id:     Database file ID of the document (from search results).
            page_number: 1-indexed page number to retrieve.

        Returns:
            A ``PageResponse`` with the page text, number, and extraction
            backend.
        """
        data = self._request("GET", f"/api/v1/documents/{file_id}/pages/{page_number}")
        return PageResponse(
            page_number=data["page_number"],
            content=data["content"],
            backend=data["backend"],
        )

    def list_chunks(
        self,
        session_id: int,
        file_id: int,
        page_number: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> ChunksResponse:
        """``GET /api/v1/sessions/{session_id}/files/{file_id}/chunks`` -- Browse chunks.

        Returns paginated chunk records for a file with their content, page
        ranges, and word counts.

        Args:
            session_id:  Session ID containing the file.
            file_id:     File ID to browse chunks for.
            page_number: Filter to chunks spanning this page (1-indexed).
                         ``None`` returns all chunks.
            offset:      Number of chunks to skip for pagination.
            limit:       Maximum chunks to return (max 100).

        Returns:
            A ``ChunksResponse`` with paginated chunk data.
        """
        params = self._strip_none(
            {
                "page_number": page_number,
                "offset": offset,
                "limit": limit,
            }
        )
        data = self._request(
            "GET",
            f"/api/v1/sessions/{session_id}/files/{file_id}/chunks",
            params=params,
        )
        return ChunksResponse(
            session_id=data["session_id"],
            file_id=data["file_id"],
            total_chunks=data["total_chunks"],
            offset=data["offset"],
            limit=data["limit"],
            returned=data["returned"],
            chunks=data["chunks"],
        )

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def quality_report(self, session_id: int) -> QualityReportResponse:
        """``GET /api/v1/sessions/{id}/quality`` -- Text extraction quality report.

        Reports extraction method distribution (native text vs OCR), page
        count mismatches, empty pages, and per-file quality flags.

        Args:
            session_id: Session ID to generate the quality report for.

        Returns:
            A ``QualityReportResponse`` with extraction statistics and flags.
        """
        data = self._request("GET", f"/api/v1/sessions/{session_id}/quality")
        return QualityReportResponse(
            session_id=data["session_id"],
            extraction_summary=data["extraction_summary"],
            quality_flags=data["quality_flags"],
            files_with_issues=data["files_with_issues"],
            files_clean=data["files_clean"],
        )

    # ------------------------------------------------------------------
    # File comparison
    # ------------------------------------------------------------------

    def compare_files(
        self,
        file_path: str | None = None,
        file_name_pattern: str | None = None,
    ) -> FileCompareResponse:
        """``POST /api/v1/files/compare`` -- Compare a file across sessions.

        Finds all indexed instances of a PDF by path or name pattern and reports
        per-session statistics (pages extracted, chunks produced, strategy).

        Args:
            file_path:         Exact file path to search for across sessions.
            file_name_pattern: SQL LIKE pattern with wildcards (``%`` for any,
                               ``_`` for single char).

        Returns:
            A ``FileCompareResponse`` with comparison data.
        """
        body = self._strip_none(
            {
                "file_path": file_path,
                "file_name_pattern": file_name_pattern,
            }
        )
        data = self._request("POST", "/api/v1/files/compare", json=body)
        return FileCompareResponse(
            pattern=data["pattern"],
            matched_files=data["matched_files"],
            comparisons=data["comparisons"],
        )

    # ------------------------------------------------------------------
    # Directory discovery
    # ------------------------------------------------------------------

    def discover(self, directory: str) -> DiscoverResponse:
        """``POST /api/v1/discover`` -- Discover indexed and unindexed files.

        Scans the filesystem for indexable files, queries all existing sessions
        for the directory, and reports aggregate statistics per session plus any
        files that exist on disk but are not indexed in any session.

        Args:
            directory: Absolute path to the directory to discover.

        Returns:
            A ``DiscoverResponse`` with filesystem scan results, session data,
            and unindexed file list.
        """
        data = self._request("POST", "/api/v1/discover", json={"directory": directory})
        return DiscoverResponse(
            directory=data["directory"],
            directory_exists=data["directory_exists"],
            filesystem=data["filesystem"],
            sessions=data["sessions"],
            unindexed_files=data["unindexed_files"],
        )

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        source_directory: str,
        output_directory: str,
        input_data: str,
        default_color: str | None = None,
    ) -> AnnotateResponse:
        """``POST /api/v1/annotate`` -- Highlight text passages in PDFs.

        Starts a background job that searches for quoted text in source PDFs,
        creates highlight annotations with configurable colors, and writes
        annotated copies to the output directory.

        Args:
            source_directory: Absolute path to the directory containing source
                              PDF files.
            output_directory: Absolute path where annotated PDFs and the report
                              will be saved.
            input_data:       CSV or JSON string with annotation instructions.
                              Required columns: ``title``, ``author``, ``quote``.
                              Optional: ``color`` (#RRGGBB hex), ``comment``.
            default_color:    Default highlight color in hex (#RRGGBB) for rows
                              without a color field. Defaults to ``#FFFF00``
                              (yellow) when absent.

        Returns:
            An ``AnnotateResponse`` with the annotation job UUID and quote count.
        """
        body = self._strip_none(
            {
                "source_directory": source_directory,
                "output_directory": output_directory,
                "input_data": input_data,
                "default_color": default_color,
            }
        )
        data = self._request("POST", "/api/v1/annotate", json=body)
        return AnnotateResponse(
            job_id=data["job_id"],
            total_quotes=data["total_quotes"],
        )

    def annotate_from_file(
        self,
        input_file: str,
        source_directory: str,
        output_directory: str,
        default_color: str | None = None,
    ) -> AnnotateFromFileResponse:
        """``POST /api/v1/annotate/from-file`` -- Annotate PDFs from a file.

        Starts a background job that reads annotation instructions from a CSV
        or JSON file on disk (rather than embedding the data in the request body),
        searches for quoted text in source PDFs, creates highlight annotations,
        and writes annotated copies to the output directory.

        Args:
            input_file:       Absolute path to the CSV or JSON file containing
                              annotation instructions.
            source_directory: Absolute path to the directory containing source
                              PDF files.
            output_directory: Absolute path where annotated PDFs and the report
                              will be saved.
            default_color:    Default highlight color in hex (#RRGGBB) for rows
                              without a color field.

        Returns:
            An ``AnnotateFromFileResponse`` with the annotation job UUID and
            quote count.
        """
        body = self._strip_none(
            {
                "input_file": input_file,
                "source_directory": source_directory,
                "output_directory": output_directory,
                "default_color": default_color,
            }
        )
        data = self._request("POST", "/api/v1/annotate/from-file", json=body)
        return AnnotateFromFileResponse(
            job_id=data["job_id"],
            total_quotes=data["total_quotes"],
        )

    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    def citation_create(
        self,
        tex_path: str,
        bib_path: str,
        session_id: int,
        batch_size: int | None = None,
        dry_run: bool = False,
        file_overrides: dict[str, int] | None = None,
    ) -> CitationCreateResponse:
        """``POST /api/v1/citation/create`` -- Create a citation verification job.

        Parses citation commands from the LaTeX file, resolves cite-keys against
        the BibTeX file, matches cited works to indexed PDFs via filename
        similarity, and groups citations into batches for sub-agent processing.

        Args:
            tex_path:       Absolute path to the ``.tex`` file containing
                            citation commands.
            bib_path:       Absolute path to the ``.bib`` file with BibTeX entries.
            session_id:     NeuronCite index session ID to search against.
            batch_size:     Target number of citations per batch (default: 5).
            dry_run:        When ``True``, runs the parse/match pipeline without
                            creating a job. Returns a match preview.
            file_overrides: Map of cite_key -> file_id to override automatic
                            PDF matching.

        Returns:
            A ``CitationCreateResponse`` with job ID, citation counts, and
            match statistics.
        """
        body = self._strip_none(
            {
                "tex_path": tex_path,
                "bib_path": bib_path,
                "session_id": session_id,
                "batch_size": batch_size,
                "dry_run": dry_run if dry_run else None,
                "file_overrides": file_overrides,
            }
        )
        data = self._request("POST", "/api/v1/citation/create", json=body)
        return CitationCreateResponse(
            job_id=data["job_id"],
            session_id=data["session_id"],
            total_citations=data["total_citations"],
            total_batches=data["total_batches"],
            unique_cite_keys=data["unique_cite_keys"],
            cite_keys_matched=data["cite_keys_matched"],
            cite_keys_unmatched=data["cite_keys_unmatched"],
            unresolved_cite_keys=data["unresolved_cite_keys"],
        )

    def citation_claim(
        self,
        job_id: str,
        batch_id: int | None = None,
    ) -> CitationClaimResponse:
        """``POST /api/v1/citation/claim`` -- Claim a batch for verification.

        Claims the next pending batch (FIFO order) or a specific batch by ID.
        Stale batches (claimed > 5 minutes ago) are automatically reclaimed.

        Args:
            job_id:   Citation verification job ID.
            batch_id: Specific batch to claim. ``None`` claims the next
                      pending batch in FIFO order.

        Returns:
            A ``CitationClaimResponse`` with the claimed batch data, or
            ``batch_id=None`` if no pending batches remain.
        """
        body = self._strip_none(
            {
                "job_id": job_id,
                "batch_id": batch_id,
            }
        )
        data = self._request("POST", "/api/v1/citation/claim", json=body)
        return CitationClaimResponse(
            batch_id=data.get("batch_id"),
            tex_path=data.get("tex_path"),
            bib_path=data.get("bib_path"),
            session_id=data.get("session_id"),
            rows=data.get("rows", []),
            remaining_batches=data.get("remaining_batches"),
            message=data.get("message"),
        )

    def citation_submit(
        self,
        job_id: str,
        results: list[dict],
    ) -> CitationSubmitResponse:
        """``POST /api/v1/citation/submit`` -- Submit verification results.

        Submits verification results for a claimed batch of citation rows.
        When all rows are done, the job is automatically completed.

        Args:
            job_id:  Citation verification job ID.
            results: Array of verification result dicts, one per ``row_id``
                     in the claimed batch.

        Returns:
            A ``CitationSubmitResponse`` with submission counts and completion
            status.
        """
        body = {
            "job_id": job_id,
            "results": results,
        }
        data = self._request("POST", "/api/v1/citation/submit", json=body)
        return CitationSubmitResponse(
            status=data["status"],
            rows_submitted=data["rows_submitted"],
            total=data["total"],
            is_complete=data["is_complete"],
        )

    def citation_status(self, job_id: str) -> CitationStatusResponse:
        """``GET /api/v1/citation/{job_id}/status`` -- Get citation job status.

        Returns row counts by status, batch counts, verdict distribution,
        flagged alerts, elapsed time, and completion status.

        Args:
            job_id: Citation verification job ID.

        Returns:
            A ``CitationStatusResponse`` with full job statistics.
        """
        data = self._request("GET", f"/api/v1/citation/{job_id}/status")
        return CitationStatusResponse(
            job_id=data["job_id"],
            job_state=data["job_state"],
            total=data["total"],
            pending=data["pending"],
            claimed=data["claimed"],
            done=data["done"],
            failed=data["failed"],
            total_batches=data["total_batches"],
            batches_done=data["batches_done"],
            batches_pending=data["batches_pending"],
            batches_claimed=data["batches_claimed"],
            verdicts=data["verdicts"],
            alerts=data["alerts"],
            elapsed_seconds=data["elapsed_seconds"],
            is_complete=data["is_complete"],
        )

    def citation_rows(
        self,
        job_id: str,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> CitationRowsResponse:
        """``GET /api/v1/citation/{job_id}/rows`` -- Get citation rows.

        Returns paginated citation rows with optional status filtering.

        Args:
            job_id: Citation verification job ID.
            status: Filter by row status (``pending``, ``claimed``, ``done``,
                    ``failed``). ``None`` returns all rows.
            offset: Number of rows to skip for pagination.
            limit:  Maximum rows to return (max 500).

        Returns:
            A ``CitationRowsResponse`` with paginated row data.
        """
        params = self._strip_none(
            {
                "status": status,
                "offset": offset,
                "limit": limit,
            }
        )
        data = self._request("GET", f"/api/v1/citation/{job_id}/rows", params=params)
        return CitationRowsResponse(
            job_id=data["job_id"],
            rows=data["rows"],
            total=data["total"],
            offset=data["offset"],
            limit=data["limit"],
        )

    def citation_export(
        self,
        job_id: str,
        output_directory: str,
        source_directory: str,
    ) -> CitationExportResponse:
        """``POST /api/v1/citation/{job_id}/export`` -- Export citation results.

        Generates six output files (annotation CSV, data CSV, Excel workbook,
        corrections JSON, report JSON, full-detail JSON) and triggers the
        annotation pipeline to create highlighted PDFs.

        Args:
            job_id:           Citation verification job ID.
            output_directory: Absolute path where output files and annotated
                              PDFs are saved.
            source_directory: Absolute path to the source PDF directory for
                              annotation.

        Returns:
            A ``CitationExportResponse`` with paths to all generated files.
        """
        body = {
            "output_directory": output_directory,
            "source_directory": source_directory,
        }
        data = self._request("POST", f"/api/v1/citation/{job_id}/export", json=body)
        return CitationExportResponse(
            annotation_job_id=data.get("annotation_job_id"),
            annotation_csv_path=data["annotation_csv_path"],
            data_csv_path=data["data_csv_path"],
            excel_path=data["excel_path"],
            report_path=data["report_path"],
            corrections_path=data["corrections_path"],
            full_detail_path=data["full_detail_path"],
            summary=data["summary"],
        )

    def citation_auto_verify(
        self,
        job_id: str,
        model: str,
        ollama_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        source_directory: str | None = None,
        fetch_sources: bool | None = None,
        unpaywall_email: str | None = None,
        top_k: int | None = None,
        cross_corpus_queries: int | None = None,
        max_retry_attempts: int | None = None,
    ) -> AutoVerifyResponse:
        """``POST /api/v1/citation/{job_id}/auto-verify`` -- Start automatic verification.

        Launches the citation verification agent that uses an Ollama-hosted LLM
        to verify each citation in the job. The agent claims batches, runs search
        queries, composes prompts, parses verdicts, and submits results.

        Args:
            job_id:               Citation verification job ID.
            model:                Ollama model identifier (e.g. ``qwen2.5:14b``).
            ollama_url:           Ollama server URL (default: ``http://localhost:11434``).
            temperature:          Sampling temperature (default: 0.1).
            max_tokens:           Maximum tokens per completion (default: 4096).
            source_directory:     Directory containing cited source PDFs/HTML.
            fetch_sources:        When true, download missing sources before verification.
            unpaywall_email:      Email for Unpaywall API access in DOI resolution.
            top_k:                Search results per query (default: 5).
            cross_corpus_queries: Cross-corpus queries per citation (default: 2).
            max_retry_attempts:   LLM retry attempts before failure (default: 3).

        Returns:
            An ``AutoVerifyResponse`` confirming the agent was started.
        """
        body = self._strip_none(
            {
                "model": model,
                "ollama_url": ollama_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "source_directory": source_directory,
                "fetch_sources": fetch_sources,
                "unpaywall_email": unpaywall_email,
                "top_k": top_k,
                "cross_corpus_queries": cross_corpus_queries,
                "max_retry_attempts": max_retry_attempts,
            }
        )
        data = self._request("POST", f"/api/v1/citation/{job_id}/auto-verify", json=body)
        return AutoVerifyResponse(
            status=data["status"],
            job_id=data["job_id"],
            message=data["message"],
        )

    def citation_fetch_sources(
        self,
        bib_path: str,
        output_directory: str,
        delay_ms: int | None = None,
        email: str | None = None,
    ) -> FetchSourcesResponse:
        """``POST /api/v1/citation/fetch-sources`` -- Download cited sources.

        Parses a BibTeX file, resolves URL/DOI fields via the DOI resolution chain
        (Unpaywall, Semantic Scholar, OpenAlex, doi.org), downloads PDF and HTML
        sources, and saves them to the output directory.

        Args:
            bib_path:         Absolute path to the .bib file.
            output_directory: Absolute path where downloaded files are saved.
            delay_ms:         Delay in milliseconds between HTTP requests
                              (default: 1000).
            email:            Email address for Unpaywall API access.

        Returns:
            A ``FetchSourcesResponse`` with download statistics and per-entry results.
        """
        body = self._strip_none(
            {
                "bib_path": bib_path,
                "output_directory": output_directory,
                "delay_ms": delay_ms,
                "email": email,
            }
        )
        data = self._request("POST", "/api/v1/citation/fetch-sources", json=body)
        return FetchSourcesResponse(
            total_entries=data["total_entries"],
            entries_with_url=data["entries_with_url"],
            pdfs_downloaded=data["pdfs_downloaded"],
            pdfs_failed=data["pdfs_failed"],
            pdfs_skipped=data["pdfs_skipped"],
            html_fetched=data["html_fetched"],
            html_failed=data["html_failed"],
            html_blocked=data["html_blocked"],
            html_skipped=data["html_skipped"],
            results=data["results"],
        )

    def citation_parse_bib(
        self,
        bib_path: str,
        output_directory: str | None = None,
    ) -> ParseBibResponse:
        """``POST /api/v1/citation/parse-bib`` -- Parse BibTeX for preview.

        Parses a .bib file and returns structured entry metadata for live preview
        without performing any network operations. When output_directory is
        provided, each entry is checked against existing files in that directory
        to determine download status.

        Args:
            bib_path:         Absolute path to the .bib file.
            output_directory: Directory to check for existing downloaded files.

        Returns:
            A ``ParseBibResponse`` with parsed BibTeX entries.
        """
        body = self._strip_none(
            {
                "bib_path": bib_path,
                "output_directory": output_directory,
            }
        )
        data = self._request("POST", "/api/v1/citation/parse-bib", json=body)
        return ParseBibResponse(
            entries=[
                BibEntryPreview(
                    cite_key=e["cite_key"],
                    entry_type=e["entry_type"],
                    author=e["author"],
                    title=e["title"],
                    has_url=e["has_url"],
                    has_doi=e["has_doi"],
                    file_exists=e["file_exists"],
                    year=e.get("year"),
                    url=e.get("url"),
                    doi=e.get("doi"),
                    bib_abstract=e.get("bib_abstract"),
                    keywords=e.get("keywords"),
                    existing_file=e.get("existing_file"),
                    expected_filename=e.get("expected_filename"),
                    duplicate_files=e.get("duplicate_files"),
                    extra_fields=e.get("extra_fields"),
                )
                for e in data["entries"]
            ],
        )

    def citation_bib_report(
        self,
        bib_path: str,
        output_directory: str,
    ) -> BibReportResponse:
        """``POST /api/v1/citation/bib-report`` -- Generate BibTeX report.

        Parses a .bib file, checks file existence in the output directory, and
        generates CSV and XLSX report files listing all entries with their link
        type and download status.

        Args:
            bib_path:         Absolute path to the .bib file.
            output_directory: Absolute path where report files are written and
                              where existing source files are checked.

        Returns:
            A ``BibReportResponse`` with paths to generated files and entry counts.
        """
        body = {
            "bib_path": bib_path,
            "output_directory": output_directory,
        }
        data = self._request("POST", "/api/v1/citation/bib-report", json=body)
        return BibReportResponse(
            csv_path=data["csv_path"],
            xlsx_path=data["xlsx_path"],
            total_entries=data["total_entries"],
            existing_count=data["existing_count"],
            duplicate_count=data["duplicate_count"],
            missing_count=data["missing_count"],
            no_link_count=data["no_link_count"],
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> ShutdownResponse:
        """``POST /api/v1/shutdown`` -- Initiate graceful server shutdown.

        Available only when the server runs in headless mode
        (``neuroncite serve``). The server returns 202 Accepted immediately;
        the actual shutdown proceeds asynchronously.

        Returns:
            A ``ShutdownResponse`` confirming shutdown was initiated.
        """
        data = self._request("POST", "/api/v1/shutdown")
        return ShutdownResponse(status=data["status"])
