# NeuronCite Python Client

Python client library for the [NeuronCite](https://github.com/FF-TEC/NeuronCite)
semantic document search API. Provides typed access to all REST endpoints and a
subprocess manager for the server binary.

```bash
# From PyPI (once published):
pip install neuroncite

# From source:
pip install ./clients/python
```

**Requires Python 3.10+.** The only runtime dependency is
[requests](https://docs.python-requests.org/).

---

## Quickstart

```python
from neuroncite import NeuronCiteClient

client = NeuronCiteClient()                       # default: http://127.0.0.1:3030

# Index a directory of PDFs
resp = client.index(directory="/data/papers", model_name="bge-base-en-v1.5")
job = client.wait_for_job(resp.job_id, timeout=600)
print(f"Indexed session {resp.session_id}, state: {job.state}")

# Search
results = client.search(session_id=resp.session_id, query="capital asset pricing model")
for r in results:
    print(f"  [{r.score:.2f}] {r.citation}")
    print(f"         {r.content[:120]}...")
```

## Server Lifecycle

Use `NeuronCiteServer` to spawn and manage the server binary as a subprocess:

```python
from neuroncite import NeuronCiteServer, NeuronCiteClient

with NeuronCiteServer(binary_path="./neuroncite", port=3030) as server:
    client = NeuronCiteClient(base_url=server.url)
    print(client.health().version)
# Server is stopped automatically when the context manager exits.
```

## Authentication

For LAN-mode deployments that require a bearer token:

```python
client = NeuronCiteClient(token="my-secret-token")
# All requests include: Authorization: Bearer my-secret-token
```

## Error Handling

All non-2xx HTTP responses raise `NeuronCiteError`:

```python
from neuroncite import NeuronCiteClient, NeuronCiteError

try:
    client.get_job("nonexistent-id")
except NeuronCiteError as e:
    print(e.status_code)  # 404
    print(e.code)         # "NOT_FOUND"
    print(e.message)      # "job not found"
```

Network errors (`ConnectionError`, `Timeout`) propagate from the `requests` library.

---

## API Reference

Every public method on `NeuronCiteClient` maps to a single REST endpoint.
Response objects are frozen dataclasses with type annotations.

| Method | Endpoint | Return Type |
|--------|----------|-------------|
| `health()` | `GET /api/v1/health` | `HealthResponse` |
| `backends()` | `GET /api/v1/backends` | `list[BackendInfo]` |
| `index(...)` | `POST /api/v1/index` | `IndexResponse` |
| `search(...)` | `POST /api/v1/search` | `list[SearchResult]` |
| `hybrid_search(...)` | `POST /api/v1/search/hybrid` | `list[SearchResult]` |
| `multi_search(...)` | `POST /api/v1/search/multi` | `MultiSearchResponse` |
| `verify(...)` | `POST /api/v1/verify` | `VerifyResponse` |
| `get_job(job_id)` | `GET /api/v1/jobs/{id}` | `JobStatus` |
| `list_jobs()` | `GET /api/v1/jobs` | `list[JobStatus]` |
| `cancel_job(job_id)` | `POST /api/v1/jobs/{id}/cancel` | `JobCancelResponse` |
| `wait_for_job(job_id)` | polling `GET /api/v1/jobs/{id}` | `JobStatus` |
| `list_sessions()` | `GET /api/v1/sessions` | `list[SessionInfo]` |
| `delete_session(id)` | `DELETE /api/v1/sessions/{id}` | `SessionDeleteResponse` |
| `delete_sessions_by_directory(...)` | `POST /api/v1/sessions/delete-by-directory` | `SessionDeleteByDirectoryResponse` |
| `optimize_session(id)` | `POST /api/v1/sessions/{id}/optimize` | `OptimizeResponse` |
| `rebuild_index(id)` | `POST /api/v1/sessions/{id}/rebuild` | `RebuildResponse` |
| `get_page(file_id, page)` | `GET /api/v1/documents/{id}/pages/{n}` | `PageResponse` |
| `list_chunks(...)` | `GET /api/v1/sessions/{sid}/files/{fid}/chunks` | `ChunksResponse` |
| `quality_report(id)` | `GET /api/v1/sessions/{id}/quality` | `QualityReportResponse` |
| `compare_files(...)` | `POST /api/v1/files/compare` | `FileCompareResponse` |
| `discover(directory)` | `POST /api/v1/discover` | `DiscoverResponse` |
| `annotate(...)` | `POST /api/v1/annotate` | `AnnotateResponse` |
| `annotate_from_file(...)` | `POST /api/v1/annotate/from-file` | `AnnotateFromFileResponse` |
| `citation_create(...)` | `POST /api/v1/citation/create` | `CitationCreateResponse` |
| `citation_claim(job_id)` | `POST /api/v1/citation/claim` | `CitationClaimResponse` |
| `citation_submit(...)` | `POST /api/v1/citation/submit` | `CitationSubmitResponse` |
| `citation_status(job_id)` | `GET /api/v1/citation/{id}/status` | `CitationStatusResponse` |
| `citation_rows(job_id)` | `GET /api/v1/citation/{id}/rows` | `CitationRowsResponse` |
| `citation_export(...)` | `POST /api/v1/citation/{id}/export` | `CitationExportResponse` |
| `citation_auto_verify(...)` | `POST /api/v1/citation/{id}/auto-verify` | `AutoVerifyResponse` |
| `citation_fetch_sources(...)` | `POST /api/v1/citation/fetch-sources` | `FetchSourcesResponse` |
| `citation_parse_bib(...)` | `POST /api/v1/citation/parse-bib` | `ParseBibResponse` |
| `citation_bib_report(...)` | `POST /api/v1/citation/bib-report` | `BibReportResponse` |
| `shutdown()` | `POST /api/v1/shutdown` | `ShutdownResponse` |

---

## Development

```bash
cd clients/python
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite uses [responses](https://github.com/getsentry/responses) to mock
HTTP endpoints. No running server is required for testing.

## License

AGPL-3.0-only. See the repository root
[LICENSE](https://github.com/FF-TEC/NeuronCite/blob/main/LICENSE) file.
