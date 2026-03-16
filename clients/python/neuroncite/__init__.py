"""NeuronCite Python client library.

Provides typed access to the NeuronCite REST API (v1) and a subprocess manager
for the server binary. This package re-exports the two main classes
(``NeuronCiteClient``, ``NeuronCiteServer``) and all response model dataclasses
so consumers can import everything from a single namespace::

    from neuroncite import NeuronCiteServer, NeuronCiteClient, SearchResult
"""

from neuroncite.client import NeuronCiteClient, NeuronCiteError
from neuroncite.models import (
    AnnotateFromFileResponse,
    AnnotateResponse,
    AutoVerifyResponse,
    BackendInfo,
    BackendListResponse,
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
from neuroncite.server import NeuronCiteServer, ServerStartError

__all__ = [
    # Main classes
    "NeuronCiteClient",
    "NeuronCiteError",
    "NeuronCiteServer",
    "ServerStartError",
    # Health & system
    "HealthResponse",
    "BackendInfo",
    "BackendListResponse",
    # Indexing
    "IndexResponse",
    # Search
    "SearchResult",
    # Verification
    "VerifyResponse",
    # Jobs
    "JobStatus",
    "JobCancelResponse",
    # Sessions
    "SessionInfo",
    "SessionDeleteResponse",
    "SessionDeleteByDirectoryResponse",
    "OptimizeResponse",
    "RebuildResponse",
    # Documents & chunks
    "PageResponse",
    "ChunksResponse",
    # Quality
    "QualityReportResponse",
    # File comparison
    "FileCompareResponse",
    # Discovery
    "DiscoverResponse",
    # Search (multi-session)
    "MultiSearchResponse",
    "MultiSearchSessionStat",
    # Annotation
    "AnnotateResponse",
    "AnnotateFromFileResponse",
    # Citation verification
    "CitationCreateResponse",
    "CitationClaimResponse",
    "CitationSubmitResponse",
    "CitationStatusResponse",
    "CitationRowsResponse",
    "CitationExportResponse",
    "AutoVerifyResponse",
    "FetchSourcesResponse",
    "BibEntryPreview",
    "ParseBibResponse",
    "BibReportResponse",
    # Shutdown
    "ShutdownResponse",
]
