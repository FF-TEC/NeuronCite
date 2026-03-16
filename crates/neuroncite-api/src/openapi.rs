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

//! OpenAPI specification generation via utoipa.
//!
//! Aggregates all handler annotations and DTO schemas into a single OpenAPI 3.1
//! specification document. The specification is served as JSON at the
//! `/api/v1/openapi.json` endpoint.

use utoipa::OpenApi;

use crate::agent;
use crate::dto;
use crate::handlers;

/// The OpenAPI specification aggregating all API endpoints and DTO schemas.
/// Each handler function annotated with `#[utoipa::path]` is listed in the
/// `paths` attribute, and each DTO type annotated with `ToSchema` is listed
/// in the `schemas` attribute.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "NeuronCite API",
        version = "1.0.0",
        description = "REST API for the NeuronCite citation search and verification system"
    ),
    paths(
        handlers::health::health,
        handlers::index::start_index,
        handlers::search::search,
        handlers::search::hybrid_search,
        handlers::search::multi_search,
        handlers::verify::verify,
        handlers::jobs::get_job,
        handlers::jobs::list_jobs,
        handlers::jobs::cancel_job,
        handlers::sessions::list_sessions,
        handlers::sessions::delete_session,
        handlers::sessions::delete_sessions_by_directory,
        handlers::sessions::optimize_session,
        handlers::sessions::rebuild_index,
        handlers::documents::get_page,
        handlers::chunks::list_chunks,
        handlers::quality::quality_report,
        handlers::compare::compare_files,
        handlers::discover::discover,
        handlers::backends::list_backends,
        handlers::annotate::start_annotate,
        handlers::annotate::annotate_from_file,
        handlers::citation::create,
        handlers::citation::claim,
        handlers::citation::submit,
        handlers::citation::status,
        handlers::citation::rows,
        handlers::citation::export,
        handlers::citation::auto_verify,
        handlers::citation::fetch_sources,
        handlers::citation::parse_bib,
        handlers::citation::bib_report,
        handlers::shutdown::shutdown,
    ),
    components(schemas(
        dto::HealthResponse,
        dto::IndexRequest,
        dto::IndexResponse,
        dto::SearchRequest,
        dto::SearchResultDto,
        dto::SearchResponse,
        dto::MultiSearchRequest,
        dto::MultiSearchResponse,
        dto::SessionSearchStatus,
        dto::MultiSearchSessionStat,
        dto::VerifyRequest,
        dto::VerifyResponse,
        dto::JobResponse,
        dto::JobListResponse,
        dto::JobCancelResponse,
        dto::SessionDto,
        dto::SessionListResponse,
        dto::SessionDeleteResponse,
        dto::SessionDeleteByDirectoryRequest,
        dto::SessionDeleteByDirectoryResponse,
        dto::OptimizeResponse,
        dto::RebuildResponse,
        dto::PageResponse,
        dto::ChunksQuery,
        dto::ChunkDto,
        dto::ChunksResponse,
        dto::ExtractionSummaryDto,
        dto::QualityFlagDetailsDto,
        dto::QualityFlagDto,
        dto::QualityReportResponse,
        dto::FileCompareRequest,
        dto::FileComparisonSessionDto,
        dto::FileComparisonDto,
        dto::FileCompareResponse,
        dto::DiscoverRequest,
        dto::FilesystemSummaryDto,
        dto::DiscoverSessionDto,
        dto::DiscoverResponse,
        dto::BackendDto,
        dto::BackendListResponse,
        dto::AnnotateRequest,
        dto::AnnotateResponse,
        dto::AnnotateFromFileRequest,
        dto::AnnotateFromFileResponse,
        dto::CitationCreateRequest,
        dto::CitationCreateResponse,
        dto::CitationClaimRequest,
        dto::CitationClaimResponse,
        dto::CitationSubmitEntryDto,
        dto::CitationSubmitRequest,
        dto::CitationSubmitResponse,
        dto::CitationStatusResponse,
        dto::CitationExportRequest,
        dto::CitationExportResponse,
        dto::CitationRowsQuery,
        dto::CitationRowsResponse,
        dto::FetchSourcesRequest,
        dto::FetchSourcesResponse,
        dto::ParseBibRequest,
        dto::BibEntryPreview,
        dto::ParseBibResponse,
        dto::BibReportRequest,
        dto::BibReportResponse,
        agent::AutoVerifyRequest,
        agent::AutoVerifyResponse,
        dto::ShutdownResponse,
    ))
)]
pub struct ApiDoc;

/// Returns the OpenAPI specification as a JSON string.
///
/// # Errors
///
/// Returns an error string if the specification cannot be serialized to JSON.
/// This is not expected to occur with valid utoipa annotations, but returning
/// a Result allows callers to handle the error without panicking.
pub fn openapi_json() -> Result<String, String> {
    ApiDoc::openapi()
        .to_json()
        .map_err(|e| format!("OpenAPI spec serialization failed: {e}"))
}
