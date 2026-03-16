"""Tests for ``neuroncite.client.NeuronCiteClient``.

Each test method registers a mock HTTP endpoint using the ``responses``
library, calls the corresponding client method, and verifies:

1. The returned dataclass fields match the expected values from the mock.
2. The request body or query parameters sent by the client match expectations.

All JSON payloads mirror the exact structure of the Rust DTOs defined in
``crates/neuroncite-api/src/dto.rs``.
"""

from __future__ import annotations

import pytest
import requests
import responses

from neuroncite import (
    NeuronCiteClient,
    NeuronCiteError,
)

BASE_URL = "http://127.0.0.1:3030"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> NeuronCiteClient:
    """A client instance pointing at the default localhost URL."""
    return NeuronCiteClient(base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for ``GET /api/v1/health``."""

    @responses.activate
    def test_health_parses_response(self, client: NeuronCiteClient) -> None:
        """All fields from the Rust ``HealthResponse`` are parsed correctly."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={
                "api_version": "v1",
                "version": "0.5.0",
                "build_features": ["backend-ort", "pdfium"],
                "gpu_available": True,
                "active_backend": "ort",
                "reranker_available": False,
                "pdfium_available": True,
                "tesseract_available": False,
            },
        )
        resp = client.health()
        assert resp.api_version == "v1"
        assert resp.version == "0.5.0"
        assert resp.build_features == ["backend-ort", "pdfium"]
        assert resp.gpu_available is True
        assert resp.active_backend == "ort"
        assert resp.reranker_available is False
        assert resp.pdfium_available is True
        assert resp.tesseract_available is False

    @responses.activate
    def test_health_reranker_available_true(self, client: NeuronCiteClient) -> None:
        """The ``reranker_available`` field is parsed when ``True``."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={
                "api_version": "v1",
                "version": "0.5.0",
                "build_features": [],
                "gpu_available": False,
                "active_backend": "ort",
                "reranker_available": True,
                "pdfium_available": False,
                "tesseract_available": False,
            },
        )
        resp = client.health()
        assert resp.reranker_available is True


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TestBackends:
    """Tests for ``GET /api/v1/backends``."""

    @responses.activate
    def test_backends_parses_response(self, client: NeuronCiteClient) -> None:
        """Backend entries are parsed from the ``backends`` wrapper array."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/backends",
            json={
                "api_version": "v1",
                "backends": [
                    {"name": "ort", "gpu_supported": True, "model_count": 3},
                ],
            },
        )
        backends = client.backends()
        assert len(backends) == 1
        assert backends[0].name == "ort"
        assert backends[0].gpu_supported is True
        assert backends[0].model_count == 3

    @responses.activate
    def test_backends_empty_list(self, client: NeuronCiteClient) -> None:
        """An empty backend list is handled correctly."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/backends",
            json={"api_version": "v1", "backends": []},
        )
        assert client.backends() == []


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class TestIndex:
    """Tests for ``POST /api/v1/index``."""

    @responses.activate
    def test_index_minimal(self, client: NeuronCiteClient) -> None:
        """Indexing with only the required ``directory`` parameter."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/index",
            json={"api_version": "v1", "job_id": "abc-123", "session_id": 7},
        )
        resp = client.index(directory="/data/papers")
        assert resp.job_id == "abc-123"
        assert resp.session_id == 7
        # Verify the request body contains directory and omits None fields
        body = responses.calls[0].request.body
        import json
        sent = json.loads(body)
        assert sent["directory"] == "/data/papers"
        assert "model_name" not in sent
        assert "chunk_strategy" not in sent

    @responses.activate
    def test_index_all_params(self, client: NeuronCiteClient) -> None:
        """Indexing with all optional parameters populated."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/index",
            json={"api_version": "v1", "job_id": "xyz-789", "session_id": 12},
        )
        resp = client.index(
            directory="/data/papers",
            model_name="BAAI/bge-small-en-v1.5",
            chunk_strategy="sentence",
            chunk_size=256,
            chunk_overlap=32,
            idempotency_key="my-key",
        )
        assert resp.job_id == "xyz-789"
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["model_name"] == "BAAI/bge-small-en-v1.5"
        assert sent["chunk_strategy"] == "sentence"
        assert sent["chunk_size"] == 256
        assert sent["chunk_overlap"] == 32
        assert sent["idempotency_key"] == "my-key"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for ``POST /api/v1/search`` and ``POST /api/v1/search/hybrid``."""

    SEARCH_RESULT = {
        "score": 0.85,
        "content": "The capital asset pricing model assumes...",
        "citation": "paper.pdf, pp. 5-7",
        "vector_score": 0.92,
        "bm25_rank": 3,
        "reranker_score": None,
        "file_id": 42,
        "source_file": "paper.pdf",
        "page_start": 5,
        "page_end": 7,
        "chunk_index": 2,
    }

    @responses.activate
    def test_search_basic(self, client: NeuronCiteClient) -> None:
        """A single search result is parsed with all fields."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search",
            json={"api_version": "v1", "results": [self.SEARCH_RESULT]},
        )
        results = client.search(session_id=1, query="CAPM")
        assert len(results) == 1
        r = results[0]
        assert r.score == 0.85
        assert r.citation == "paper.pdf, pp. 5-7"
        assert r.vector_score == 0.92
        assert r.bm25_rank == 3
        assert r.reranker_score is None
        assert r.file_id == 42
        assert r.source_file == "paper.pdf"
        assert r.page_start == 5
        assert r.page_end == 7

    @responses.activate
    def test_search_sends_refine_params(self, client: NeuronCiteClient) -> None:
        """The ``refine`` and ``refine_divisors`` parameters are sent in the body."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search",
            json={"api_version": "v1", "results": []},
        )
        client.search(
            session_id=1,
            query="test",
            refine=False,
            refine_divisors="2,4",
        )
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["refine"] is False
        assert sent["refine_divisors"] == "2,4"

    @responses.activate
    def test_search_empty_results(self, client: NeuronCiteClient) -> None:
        """An empty result list is returned as an empty Python list."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search",
            json={"api_version": "v1", "results": []},
        )
        assert client.search(session_id=1, query="nonexistent") == []

    @responses.activate
    def test_hybrid_search_uses_hybrid_path(self, client: NeuronCiteClient) -> None:
        """``hybrid_search()`` hits the ``/search/hybrid`` endpoint path."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search/hybrid",
            json={"api_version": "v1", "results": [self.SEARCH_RESULT]},
        )
        results = client.hybrid_search(session_id=1, query="test")
        assert len(results) == 1
        assert results[0].file_id == 42


# ---------------------------------------------------------------------------
# Multi-session search
# ---------------------------------------------------------------------------

class TestMultiSearch:
    """Tests for ``POST /api/v1/search/multi``."""

    @responses.activate
    def test_multi_search_merged_results(self, client: NeuronCiteClient) -> None:
        """Multi-search returns per-session stats and a merged result list."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search/multi",
            json={
                "api_version": "v1",
                "session_ids": [1, 2],
                "session_stats": [
                    {
                        "session_id": 1,
                        "result_count": 3,
                        "status": "ok",
                        "error": None,
                    },
                    {
                        "session_id": 2,
                        "result_count": 0,
                        "status": "no_hnsw_index",
                        "error": None,
                    },
                ],
                "results": [
                    {
                        "score": 0.91,
                        "content": "Cross-session result text",
                        "citation": "paper.pdf, pp. 1-3",
                        "vector_score": 0.88,
                        "file_id": 10,
                        "source_file": "paper.pdf",
                        "page_start": 1,
                        "page_end": 3,
                        "chunk_index": 0,
                        "session_id": 1,
                    },
                ],
            },
        )
        resp = client.multi_search(session_ids=[1, 2], query="test query")
        assert resp.session_ids == [1, 2]
        assert len(resp.session_stats) == 2
        assert resp.session_stats[0].session_id == 1
        assert resp.session_stats[0].result_count == 3
        assert resp.session_stats[0].status == "ok"
        assert resp.session_stats[1].status == "no_hnsw_index"
        assert len(resp.results) == 1
        assert resp.results[0].score == 0.91
        assert resp.results[0].session_id == 1

    @responses.activate
    def test_multi_search_request_body(self, client: NeuronCiteClient) -> None:
        """The request body contains ``session_ids`` as an integer list."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/search/multi",
            json={
                "api_version": "v1",
                "session_ids": [3, 7],
                "session_stats": [],
                "results": [],
            },
        )
        client.multi_search(session_ids=[3, 7], query="q", top_k=5, rerank=True)
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["session_ids"] == [3, 7]
        assert sent["query"] == "q"
        assert sent["top_k"] == 5
        assert sent["rerank"] is True


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestVerify:
    """Tests for ``POST /api/v1/verify``."""

    @responses.activate
    def test_verify_response_schema(self, client: NeuronCiteClient) -> None:
        """The response contains ``combined_score``, ``keyword_score``, ``semantic_score``."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/verify",
            json={
                "api_version": "v1",
                "verdict": "supports",
                "combined_score": 0.82,
                "keyword_score": 0.65,
                "semantic_score": 0.89,
            },
        )
        resp = client.verify(claim="Beta is the sole risk factor", session_id=1, chunk_ids=[10, 11])
        assert resp.verdict == "supports"
        assert resp.combined_score == 0.82
        assert resp.keyword_score == 0.65
        assert resp.semantic_score == 0.89

    @responses.activate
    def test_verify_sends_session_id(self, client: NeuronCiteClient) -> None:
        """The ``session_id`` and ``chunk_ids`` are sent in the request body."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/verify",
            json={
                "api_version": "v1",
                "verdict": "partial",
                "combined_score": 0.5,
                "keyword_score": 0.3,
                "semantic_score": 0.6,
            },
        )
        client.verify(claim="test claim", session_id=5, chunk_ids=[1, 2, 3])
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["session_id"] == 5
        assert sent["chunk_ids"] == [1, 2, 3]
        assert sent["claim"] == "test claim"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class TestJobs:
    """Tests for job management endpoints."""

    JOB_DATA = {
        "api_version": "v1",
        "id": "job-uuid-1",
        "kind": "index",
        "session_id": 3,
        "state": "running",
        "progress_done": 5,
        "progress_total": 10,
        "error_message": None,
        "created_at": 1700000000,
        "started_at": 1700000001,
        "finished_at": None,
    }

    @responses.activate
    def test_get_job_flat_progress(self, client: NeuronCiteClient) -> None:
        """Job status uses flat ``progress_done``/``progress_total`` fields."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/jobs/job-uuid-1",
            json=self.JOB_DATA,
        )
        job = client.get_job("job-uuid-1")
        assert job.id == "job-uuid-1"
        assert job.kind == "index"
        assert job.session_id == 3
        assert job.state == "running"
        assert job.progress_done == 5
        assert job.progress_total == 10
        assert job.error_message is None
        assert job.created_at == 1700000000
        assert job.started_at == 1700000001
        assert job.finished_at is None

    @responses.activate
    def test_list_jobs_wrapper_format(self, client: NeuronCiteClient) -> None:
        """Job list is parsed from the ``{"jobs": [...]}`` wrapper."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/jobs",
            json={"api_version": "v1", "jobs": [self.JOB_DATA]},
        )
        jobs = client.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "job-uuid-1"

    @responses.activate
    def test_cancel_job_response(self, client: NeuronCiteClient) -> None:
        """Cancel returns a ``JobCancelResponse`` with ``job_id`` and ``state``."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/jobs/job-uuid-1/cancel",
            json={"api_version": "v1", "job_id": "job-uuid-1", "state": "canceled"},
        )
        resp = client.cancel_job("job-uuid-1")
        assert resp.job_id == "job-uuid-1"
        assert resp.state == "canceled"

    @responses.activate
    def test_wait_for_job_returns_on_completed(self, client: NeuronCiteClient) -> None:
        """``wait_for_job()`` returns immediately when the job is already completed."""
        completed = dict(self.JOB_DATA, state="completed", finished_at=1700000010)
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/jobs/job-uuid-1",
            json=completed,
        )
        job = client.wait_for_job("job-uuid-1", poll_interval=0.01, timeout=1)
        assert job.state == "completed"

    @responses.activate
    def test_wait_for_job_timeout(self, client: NeuronCiteClient) -> None:
        """``wait_for_job()`` raises ``TimeoutError`` when the job stays running."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/jobs/job-uuid-1",
            json=self.JOB_DATA,
        )
        with pytest.raises(TimeoutError, match="did not finish"):
            client.wait_for_job("job-uuid-1", poll_interval=0.01, timeout=0.05)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    """Tests for session management endpoints."""

    SESSION_DATA = {
        "id": 1,
        "directory_path": "/data/papers",
        "model_name": "BAAI/bge-small-en-v1.5",
        "chunk_strategy": "sentence",
        "vector_dimension": 384,
        "created_at": 1700000000,
        "file_count": 5,
        "total_pages": 120,
        "total_chunks": 500,
        "total_content_bytes": 250000,
        "max_words": 128,
    }

    @responses.activate
    def test_list_sessions_new_fields(self, client: NeuronCiteClient) -> None:
        """Session list contains ``vector_dimension``, aggregate stats, and chunking params."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/sessions",
            json={"api_version": "v1", "sessions": [self.SESSION_DATA]},
        )
        sessions = client.list_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s.id == 1
        assert s.vector_dimension == 384
        assert s.file_count == 5
        assert s.total_pages == 120
        assert s.total_chunks == 500
        assert s.total_content_bytes == 250000
        assert s.max_words == 128

    @responses.activate
    def test_delete_session_returns_response(self, client: NeuronCiteClient) -> None:
        """``delete_session()`` returns a ``SessionDeleteResponse``."""
        responses.add(
            responses.DELETE,
            f"{BASE_URL}/api/v1/sessions/1",
            json={"api_version": "v1", "deleted": True},
        )
        resp = client.delete_session(1)
        assert resp.deleted is True

    @responses.activate
    def test_delete_sessions_by_directory(self, client: NeuronCiteClient) -> None:
        """Delete by directory returns the deleted session IDs and canonicalized path."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/sessions/delete-by-directory",
            json={
                "api_version": "v1",
                "deleted_session_ids": [1, 2],
                "directory": "/data/papers",
                "matched_directory": "/data/papers",
            },
        )
        resp = client.delete_sessions_by_directory("/data/papers")
        assert resp.deleted_session_ids == [1, 2]
        assert resp.directory == "/data/papers"
        assert resp.matched_directory == "/data/papers"

    @responses.activate
    def test_optimize_session(self, client: NeuronCiteClient) -> None:
        """``optimize_session()`` returns an ``OptimizeResponse``."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/sessions/5/optimize",
            json={"api_version": "v1", "status": "FTS5 optimize triggered"},
        )
        resp = client.optimize_session(5)
        assert resp.status == "FTS5 optimize triggered"

    @responses.activate
    def test_rebuild_index(self, client: NeuronCiteClient) -> None:
        """``rebuild_index()`` returns a ``RebuildResponse`` with the job UUID."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/sessions/5/rebuild",
            json={"api_version": "v1", "job_id": "rebuild-uuid"},
        )
        resp = client.rebuild_index(5)
        assert resp.job_id == "rebuild-uuid"


# ---------------------------------------------------------------------------
# Documents & chunks
# ---------------------------------------------------------------------------

class TestDocuments:
    """Tests for ``GET /api/v1/documents/{id}/pages/{n}``."""

    @responses.activate
    def test_get_page(self, client: NeuronCiteClient) -> None:
        """Page retrieval returns page number, content, and extraction backend."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/documents/42/pages/3",
            json={
                "api_version": "v1",
                "page_number": 3,
                "content": "This is the page text.",
                "backend": "pdfium",
            },
        )
        resp = client.get_page(file_id=42, page_number=3)
        assert resp.page_number == 3
        assert resp.content == "This is the page text."
        assert resp.backend == "pdfium"


class TestChunks:
    """Tests for ``GET /api/v1/sessions/{sid}/files/{fid}/chunks``."""

    @responses.activate
    def test_list_chunks(self, client: NeuronCiteClient) -> None:
        """Chunk listing returns paginated chunk data."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/sessions/1/files/42/chunks",
            json={
                "api_version": "v1",
                "session_id": 1,
                "file_id": 42,
                "total_chunks": 50,
                "offset": 0,
                "limit": 20,
                "returned": 20,
                "chunks": [{"id": 1, "content": "chunk text"}],
            },
        )
        resp = client.list_chunks(session_id=1, file_id=42)
        assert resp.session_id == 1
        assert resp.file_id == 42
        assert resp.total_chunks == 50
        assert resp.returned == 20
        assert len(resp.chunks) == 1

    @responses.activate
    def test_list_chunks_with_page_filter(self, client: NeuronCiteClient) -> None:
        """The ``page_number`` query parameter is sent when specified."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/sessions/1/files/42/chunks",
            json={
                "api_version": "v1",
                "session_id": 1,
                "file_id": 42,
                "total_chunks": 3,
                "offset": 0,
                "limit": 20,
                "returned": 3,
                "chunks": [],
            },
        )
        client.list_chunks(session_id=1, file_id=42, page_number=5)
        # Verify the query parameters contain page_number
        request_url = responses.calls[0].request.url
        assert "page_number=5" in request_url


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

class TestQuality:
    """Tests for ``GET /api/v1/sessions/{id}/quality``."""

    @responses.activate
    def test_quality_report(self, client: NeuronCiteClient) -> None:
        """Quality report contains extraction summary and quality flags."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/sessions/1/quality",
            json={
                "api_version": "v1",
                "session_id": 1,
                "extraction_summary": {"native_text": 100, "ocr": 5},
                "quality_flags": [{"file_id": 3, "flags": ["ocr_heavy"]}],
                "files_with_issues": 1,
                "files_clean": 4,
            },
        )
        resp = client.quality_report(1)
        assert resp.session_id == 1
        assert resp.extraction_summary["native_text"] == 100
        assert len(resp.quality_flags) == 1
        assert resp.files_with_issues == 1
        assert resp.files_clean == 4


# ---------------------------------------------------------------------------
# File comparison
# ---------------------------------------------------------------------------

class TestCompare:
    """Tests for ``POST /api/v1/files/compare``."""

    @responses.activate
    def test_compare_files_by_path(self, client: NeuronCiteClient) -> None:
        """Comparison by exact file path."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/files/compare",
            json={
                "api_version": "v1",
                "pattern": "/data/paper.pdf",
                "matched_files": 1,
                "comparisons": [{"sessions": [1, 2]}],
            },
        )
        resp = client.compare_files(file_path="/data/paper.pdf")
        assert resp.matched_files == 1

    @responses.activate
    def test_compare_files_by_pattern(self, client: NeuronCiteClient) -> None:
        """Comparison by SQL LIKE pattern."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/files/compare",
            json={
                "api_version": "v1",
                "pattern": "%Fama%",
                "matched_files": 2,
                "comparisons": [],
            },
        )
        resp = client.compare_files(file_name_pattern="%Fama%")
        assert resp.matched_files == 2
        assert resp.pattern == "%Fama%"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscover:
    """Tests for ``POST /api/v1/discover``."""

    @responses.activate
    def test_discover(self, client: NeuronCiteClient) -> None:
        """Discovery returns filesystem data, session info, and unindexed PDFs."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/discover",
            json={
                "api_version": "v1",
                "directory": "/data/papers",
                "directory_exists": True,
                "filesystem": {"pdf_count": 10, "total_bytes": 50000000},
                "sessions": [{"id": 1, "file_count": 8}],
                "unindexed_files": ["new_paper.pdf", "draft.pdf"],
            },
        )
        resp = client.discover("/data/papers")
        assert resp.directory == "/data/papers"
        assert resp.directory_exists is True
        assert resp.filesystem["pdf_count"] == 10
        assert len(resp.sessions) == 1
        assert resp.unindexed_files == ["new_paper.pdf", "draft.pdf"]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestAnnotate:
    """Tests for ``POST /api/v1/annotate``."""

    @responses.activate
    def test_annotate(self, client: NeuronCiteClient) -> None:
        """Annotation response contains job UUID and quote count."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/annotate",
            json={
                "api_version": "v1",
                "job_id": "annot-uuid",
                "total_quotes": 15,
            },
        )
        resp = client.annotate(
            source_directory="/data/papers",
            output_directory="/data/output",
            input_data='title,author,quote\n"Paper","Author","some text"',
        )
        assert resp.job_id == "annot-uuid"
        assert resp.total_quotes == 15

    @responses.activate
    def test_annotate_with_color(self, client: NeuronCiteClient) -> None:
        """The ``default_color`` parameter is sent when specified."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/annotate",
            json={"api_version": "v1", "job_id": "annot-2", "total_quotes": 1},
        )
        client.annotate(
            source_directory="/src",
            output_directory="/out",
            input_data="[]",
            default_color="#FF0000",
        )
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["default_color"] == "#FF0000"

    @responses.activate
    def test_annotate_from_file(self, client: NeuronCiteClient) -> None:
        """Annotation from a file path returns job UUID and quote count."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/annotate/from-file",
            json={
                "api_version": "v1",
                "job_id": "from-file-uuid",
                "total_quotes": 8,
            },
        )
        resp = client.annotate_from_file(
            input_file="/data/annotations.csv",
            source_directory="/data/papers",
            output_directory="/data/output",
        )
        assert resp.job_id == "from-file-uuid"
        assert resp.total_quotes == 8
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["input_file"] == "/data/annotations.csv"
        assert sent["source_directory"] == "/data/papers"
        assert sent["output_directory"] == "/data/output"


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

class TestCitation:
    """Tests for the citation verification endpoints."""

    @responses.activate
    def test_citation_create(self, client: NeuronCiteClient) -> None:
        """Citation creation returns job ID and match statistics."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/create",
            json={
                "api_version": "v1",
                "job_id": "cit-uuid",
                "session_id": 1,
                "total_citations": 20,
                "total_batches": 4,
                "unique_cite_keys": 15,
                "cite_keys_matched": 12,
                "cite_keys_unmatched": 3,
                "unresolved_cite_keys": ["missing_ref"],
            },
        )
        resp = client.citation_create(
            tex_path="/doc/paper.tex",
            bib_path="/doc/refs.bib",
            session_id=1,
        )
        assert resp.job_id == "cit-uuid"
        assert resp.total_citations == 20
        assert resp.total_batches == 4
        assert resp.cite_keys_matched == 12
        assert resp.cite_keys_unmatched == 3
        assert resp.unresolved_cite_keys == ["missing_ref"]

    @responses.activate
    def test_citation_create_with_overrides(self, client: NeuronCiteClient) -> None:
        """The ``file_overrides`` parameter is sent in the request body."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/create",
            json={
                "api_version": "v1",
                "job_id": "cit-2",
                "session_id": 1,
                "total_citations": 5,
                "total_batches": 1,
                "unique_cite_keys": 5,
                "cite_keys_matched": 5,
                "cite_keys_unmatched": 0,
                "unresolved_cite_keys": [],
            },
        )
        client.citation_create(
            tex_path="/paper.tex",
            bib_path="/refs.bib",
            session_id=1,
            file_overrides={"smith2020": 42},
        )
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["file_overrides"] == {"smith2020": 42}

    @responses.activate
    def test_citation_claim_with_batch(self, client: NeuronCiteClient) -> None:
        """Claiming a batch returns the batch data and citation rows."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/claim",
            json={
                "api_version": "v1",
                "batch_id": 3,
                "tex_path": "/paper.tex",
                "bib_path": "/refs.bib",
                "session_id": 1,
                "rows": [{"row_id": 10, "cite_key": "smith2020"}],
                "remaining_batches": None,
                "message": None,
            },
        )
        resp = client.citation_claim(job_id="cit-uuid")
        assert resp.batch_id == 3
        assert resp.tex_path == "/paper.tex"
        assert len(resp.rows) == 1

    @responses.activate
    def test_citation_claim_no_batch(self, client: NeuronCiteClient) -> None:
        """When no pending batch exists, ``batch_id`` is ``None``."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/claim",
            json={
                "api_version": "v1",
                "batch_id": None,
                "tex_path": None,
                "bib_path": None,
                "session_id": None,
                "rows": [],
                "remaining_batches": 0,
                "message": "no pending batches",
            },
        )
        resp = client.citation_claim(job_id="cit-uuid")
        assert resp.batch_id is None
        assert resp.message == "no pending batches"

    @responses.activate
    def test_citation_submit(self, client: NeuronCiteClient) -> None:
        """Submit returns acceptance status and completion flag."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/submit",
            json={
                "api_version": "v1",
                "status": "accepted",
                "rows_submitted": 5,
                "total": 20,
                "is_complete": False,
            },
        )
        resp = client.citation_submit(
            job_id="cit-uuid",
            results=[{"row_id": 1, "verdict": "supported"}],
        )
        assert resp.status == "accepted"
        assert resp.rows_submitted == 5
        assert resp.is_complete is False

    @responses.activate
    def test_citation_status(self, client: NeuronCiteClient) -> None:
        """Citation status returns full job statistics."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/citation/cit-uuid/status",
            json={
                "api_version": "v1",
                "job_id": "cit-uuid",
                "job_state": "running",
                "total": 20,
                "pending": 10,
                "claimed": 5,
                "done": 5,
                "failed": 0,
                "total_batches": 4,
                "batches_done": 1,
                "batches_pending": 2,
                "batches_claimed": 1,
                "verdicts": {"supported": 3, "partial": 2},
                "alerts": [],
                "elapsed_seconds": 120,
                "is_complete": False,
            },
        )
        resp = client.citation_status("cit-uuid")
        assert resp.job_state == "running"
        assert resp.total == 20
        assert resp.pending == 10
        assert resp.verdicts == {"supported": 3, "partial": 2}

    @responses.activate
    def test_citation_rows(self, client: NeuronCiteClient) -> None:
        """Citation rows are returned with pagination metadata."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/citation/cit-uuid/rows",
            json={
                "api_version": "v1",
                "job_id": "cit-uuid",
                "rows": [{"row_id": 1, "status": "done"}],
                "total": 20,
                "offset": 0,
                "limit": 100,
            },
        )
        resp = client.citation_rows("cit-uuid")
        assert len(resp.rows) == 1
        assert resp.total == 20

    @responses.activate
    def test_citation_rows_with_filters(self, client: NeuronCiteClient) -> None:
        """The ``status``, ``offset``, and ``limit`` query parameters are sent."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/citation/cit-uuid/rows",
            json={
                "api_version": "v1",
                "job_id": "cit-uuid",
                "rows": [],
                "total": 5,
                "offset": 10,
                "limit": 50,
            },
        )
        client.citation_rows("cit-uuid", status="done", offset=10, limit=50)
        request_url = responses.calls[0].request.url
        assert "status=done" in request_url
        assert "offset=10" in request_url
        assert "limit=50" in request_url

    @responses.activate
    def test_citation_export(self, client: NeuronCiteClient) -> None:
        """Citation export returns paths to all generated files."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/cit-uuid/export",
            json={
                "api_version": "v1",
                "annotation_job_id": "annot-123",
                "annotation_csv_path": "/out/annotations.csv",
                "data_csv_path": "/out/data.csv",
                "excel_path": "/out/data.xlsx",
                "report_path": "/out/report.json",
                "corrections_path": "/out/corrections.json",
                "full_detail_path": "/out/full_detail.json",
                "summary": {"supported": 10, "partial": 5},
            },
        )
        resp = client.citation_export(
            job_id="cit-uuid",
            output_directory="/out",
            source_directory="/data/papers",
        )
        assert resp.annotation_job_id == "annot-123"
        assert resp.data_csv_path == "/out/data.csv"
        assert resp.excel_path == "/out/data.xlsx"
        assert resp.summary["supported"] == 10

    @responses.activate
    def test_citation_auto_verify(self, client: NeuronCiteClient) -> None:
        """Auto-verify returns confirmation that the agent was started."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/cit-uuid/auto-verify",
            json={
                "api_version": "v1",
                "status": "started",
                "job_id": "cit-uuid",
                "message": "Auto-verify agent started for 20 rows",
            },
        )
        resp = client.citation_auto_verify(job_id="cit-uuid", model="qwen2.5:14b")
        assert resp.status == "started"
        assert resp.job_id == "cit-uuid"
        assert resp.message == "Auto-verify agent started for 20 rows"
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["model"] == "qwen2.5:14b"
        # Optional parameters are stripped when None
        assert "ollama_url" not in sent
        assert "temperature" not in sent

    @responses.activate
    def test_citation_fetch_sources(self, client: NeuronCiteClient) -> None:
        """Fetch-sources returns download statistics and per-entry results."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/fetch-sources",
            json={
                "api_version": "v1",
                "total_entries": 25,
                "entries_with_url": 20,
                "pdfs_downloaded": 15,
                "pdfs_failed": 2,
                "pdfs_skipped": 3,
                "html_fetched": 5,
                "html_failed": 1,
                "html_blocked": 0,
                "html_skipped": 14,
                "results": [{"cite_key": "smith2020", "status": "downloaded"}],
            },
        )
        resp = client.citation_fetch_sources(
            bib_path="/doc/refs.bib",
            output_directory="/doc/sources",
            delay_ms=500,
            email="user@example.com",
        )
        assert resp.total_entries == 25
        assert resp.pdfs_downloaded == 15
        assert resp.html_fetched == 5
        assert resp.html_blocked == 0
        assert len(resp.results) == 1
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["bib_path"] == "/doc/refs.bib"
        assert sent["delay_ms"] == 500
        assert sent["email"] == "user@example.com"

    @responses.activate
    def test_citation_parse_bib(self, client: NeuronCiteClient) -> None:
        """Parse-bib returns a list of ``BibEntryPreview`` dataclasses."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/parse-bib",
            json={
                "api_version": "v1",
                "entries": [
                    {
                        "cite_key": "smith2020",
                        "entry_type": "article",
                        "author": "Smith, J.",
                        "title": "A Study",
                        "has_url": True,
                        "has_doi": True,
                        "file_exists": True,
                        "year": "2020",
                        "url": "https://example.com/paper.pdf",
                        "doi": "10.1234/example",
                        "bib_abstract": "Abstract text.",
                        "keywords": "keyword1, keyword2",
                        "existing_file": "smith2020.pdf",
                        "expected_filename": "smith2020.pdf",
                        "duplicate_files": [],
                        "extra_fields": {"journal": "Nature"},
                    },
                ],
            },
        )
        resp = client.citation_parse_bib(
            bib_path="/doc/refs.bib",
            output_directory="/doc/sources",
        )
        assert len(resp.entries) == 1
        entry = resp.entries[0]
        assert entry.cite_key == "smith2020"
        assert entry.entry_type == "article"
        assert entry.has_url is True
        assert entry.has_doi is True
        assert entry.file_exists is True
        assert entry.year == "2020"
        assert entry.doi == "10.1234/example"
        assert entry.existing_file == "smith2020.pdf"
        assert entry.extra_fields == {"journal": "Nature"}

    @responses.activate
    def test_citation_bib_report(self, client: NeuronCiteClient) -> None:
        """Bib-report returns paths to generated files and entry counts."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/citation/bib-report",
            json={
                "api_version": "v1",
                "csv_path": "/out/report.csv",
                "xlsx_path": "/out/report.xlsx",
                "total_entries": 50,
                "existing_count": 30,
                "duplicate_count": 5,
                "missing_count": 10,
                "no_link_count": 5,
            },
        )
        resp = client.citation_bib_report(
            bib_path="/doc/refs.bib",
            output_directory="/out",
        )
        assert resp.csv_path == "/out/report.csv"
        assert resp.xlsx_path == "/out/report.xlsx"
        assert resp.total_entries == 50
        assert resp.existing_count == 30
        assert resp.duplicate_count == 5
        assert resp.missing_count == 10
        assert resp.no_link_count == 5
        import json
        sent = json.loads(responses.calls[0].request.body)
        assert sent["bib_path"] == "/doc/refs.bib"
        assert sent["output_directory"] == "/out"


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    """Tests for ``POST /api/v1/shutdown``."""

    @responses.activate
    def test_shutdown_returns_response(self, client: NeuronCiteClient) -> None:
        """Shutdown returns a ``ShutdownResponse`` with status message."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/shutdown",
            json={"api_version": "v1", "status": "shutting down"},
        )
        resp = client.shutdown()
        assert resp.status == "shutting down"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for HTTP error handling."""

    @responses.activate
    def test_non_2xx_raises_error(self, client: NeuronCiteClient) -> None:
        """A non-2xx response raises ``NeuronCiteError`` with status code."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={"code": "INTERNAL", "error": "something broke"},
            status=500,
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.health()
        assert exc_info.value.status_code == 500
        assert exc_info.value.message == "something broke"

    @responses.activate
    def test_404_error(self, client: NeuronCiteClient) -> None:
        """A 404 response raises ``NeuronCiteError``."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/jobs/nonexistent",
            json={"code": "NOT_FOUND", "error": "job not found"},
            status=404,
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.get_job("nonexistent")
        assert exc_info.value.status_code == 404

    @responses.activate
    def test_non_json_error_body(self, client: NeuronCiteClient) -> None:
        """A non-JSON error body is captured as the message text."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            body="Internal Server Error",
            status=500,
            content_type="text/plain",
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.health()
        assert "Internal Server Error" in exc_info.value.message

    def test_connection_error_propagates(self) -> None:
        """A ``ConnectionError`` is raised when the server is unreachable."""
        # Connect to a port where nothing is listening
        client = NeuronCiteClient(base_url="http://127.0.0.1:19999")
        with pytest.raises(requests.ConnectionError):
            client.health()

    @responses.activate
    def test_400_bad_request(self, client: NeuronCiteClient) -> None:
        """A 400 response raises ``NeuronCiteError`` with the validation message."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/index",
            json={"code": "VALIDATION", "error": "directory field is required"},
            status=400,
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.index(directory="", model_name="test-model")
        assert exc_info.value.status_code == 400
        assert "directory" in exc_info.value.message

    @responses.activate
    def test_401_unauthorized(self, client: NeuronCiteClient) -> None:
        """A 401 response raises ``NeuronCiteError`` with status code 401."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={"code": "UNAUTHORIZED", "error": "invalid or missing bearer token"},
            status=401,
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.health()
        assert exc_info.value.status_code == 401

    @responses.activate
    def test_503_service_unavailable(self, client: NeuronCiteClient) -> None:
        """A 503 response raises ``NeuronCiteError`` with status code 503."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={"code": "UNAVAILABLE", "error": "model loading in progress"},
            status=503,
        )
        with pytest.raises(NeuronCiteError) as exc_info:
            client.health()
        assert exc_info.value.status_code == 503

    @responses.activate
    def test_timeout_error(self, client: NeuronCiteClient) -> None:
        """A request timeout raises ``requests.exceptions.ReadTimeout``."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            body=requests.exceptions.ReadTimeout("Read timed out"),
        )
        with pytest.raises(requests.exceptions.ReadTimeout):
            client.health()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuth:
    """Tests for bearer token authentication."""

    @responses.activate
    def test_bearer_token_in_header(self) -> None:
        """The bearer token is sent in the Authorization header."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={
                "api_version": "v1",
                "version": "0.5.0",
                "build_features": [],
                "gpu_available": False,
                "active_backend": "ort",
                "reranker_available": False,
                "pdfium_available": False,
                "tesseract_available": False,
            },
        )
        client = NeuronCiteClient(token="secret-token-123")
        client.health()
        auth_header = responses.calls[0].request.headers.get("Authorization")
        assert auth_header == "Bearer secret-token-123"

    @responses.activate
    def test_no_token_no_header(self) -> None:
        """Without a token, no Authorization header is sent."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={
                "api_version": "v1",
                "version": "0.5.0",
                "build_features": [],
                "gpu_available": False,
                "active_backend": "ort",
                "reranker_available": False,
                "pdfium_available": False,
                "tesseract_available": False,
            },
        )
        client = NeuronCiteClient()
        client.health()
        auth_header = responses.calls[0].request.headers.get("Authorization")
        assert auth_header is None

    @responses.activate
    def test_empty_token_sends_bearer_header(self) -> None:
        """An empty-string token sends ``Authorization: Bearer `` (empty bearer value)."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/health",
            json={
                "api_version": "v1",
                "version": "0.5.0",
                "build_features": [],
                "gpu_available": False,
                "active_backend": "ort",
                "reranker_available": False,
                "pdfium_available": False,
                "tesseract_available": False,
            },
        )
        client = NeuronCiteClient(token="")
        client.health()
        auth_header = responses.calls[0].request.headers.get("Authorization")
        assert auth_header == "Bearer "
