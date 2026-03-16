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

//! Axum router construction.
//!
//! Builds the complete axum `Router` with all endpoint routes, middleware layers
//! (CORS, optional authentication), and the OpenAPI specification endpoint.
//! The router is parameterized by `AppState` and is returned to the binary crate
//! for binding to a TCP listener.
//!
//! The OpenAPI specification endpoint (`/api/v1/openapi.json`) is placed inside
//! the authentication-protected `api_routes` block. When a bearer token is
//! configured, the spec is therefore subject to the same auth check as all other
//! endpoints. This prevents unauthenticated callers from enumerating the full
//! API surface (endpoint paths, parameter names, error codes) of a server that
//! was intentionally secured with a token.

use std::sync::Arc;

use axum::Router;
use axum::routing::{delete, get, post};

use crate::handlers;
use crate::middleware;
use crate::openapi;
use crate::state::AppState;

/// Body limit for citation creation requests. Citation payloads contain BibTeX
/// data and optional CSV text which can be large for bibliographies with hundreds
/// of entries. 10 MiB accommodates typical academic BibTeX files.
const CITATION_BODY_LIMIT: usize = 10 * 1024 * 1024;

/// Body limit for annotation requests. Annotation payloads contain BibTeX text
/// that is typically smaller than citation payloads. 2 MiB matches the global
/// default but is stated explicitly here so it remains stable if the global
/// default changes.
const ANNOTATE_BODY_LIMIT: usize = 2 * 1024 * 1024;

/// Builds the complete axum router with all /api/v1/ routes, middleware layers,
/// and the OpenAPI JSON endpoint.
///
/// # Arguments
///
/// * `state` - Shared application state wrapped in `Arc`.
///
/// # Returns
///
/// A fully configured `Router` ready to be served by an axum TCP listener.
pub fn build_router(state: Arc<AppState>) -> Router {
    let cors_layer = middleware::cors::build_cors_layer(&state.config);

    // Capture the OpenAPI JSON once at startup. The closure is moved into the
    // route handler so each request clones only the pre-built Arc<str>.
    let openapi_spec =
        openapi::openapi_json().expect("OpenAPI specification must serialize at startup");

    let api_routes = Router::new()
        // Health
        .route("/health", get(handlers::health::health))
        // Index
        .route("/index", post(handlers::index::start_index))
        // Search
        .route("/search", post(handlers::search::search))
        .route("/search/hybrid", post(handlers::search::hybrid_search))
        .route("/search/multi", post(handlers::search::multi_search))
        // Verify
        .route("/verify", post(handlers::verify::verify))
        // Jobs
        .route("/jobs", get(handlers::jobs::list_jobs))
        .route("/jobs/{id}", get(handlers::jobs::get_job))
        .route("/jobs/{id}/cancel", post(handlers::jobs::cancel_job))
        // Sessions
        .route("/sessions", get(handlers::sessions::list_sessions))
        .route("/sessions/{id}", delete(handlers::sessions::delete_session))
        .route(
            "/sessions/delete-by-directory",
            post(handlers::sessions::delete_sessions_by_directory),
        )
        .route(
            "/sessions/{id}/optimize",
            post(handlers::sessions::optimize_session),
        )
        .route(
            "/sessions/{id}/rebuild",
            post(handlers::sessions::rebuild_index),
        )
        // Documents
        .route(
            "/documents/{id}/pages/{n}",
            get(handlers::documents::get_page),
        )
        // Chunks
        .route(
            "/sessions/{session_id}/files/{file_id}/chunks",
            get(handlers::chunks::list_chunks),
        )
        // Quality
        .route(
            "/sessions/{id}/quality",
            get(handlers::quality::quality_report),
        )
        // File Compare
        .route("/files/compare", post(handlers::compare::compare_files))
        // Discover
        .route("/discover", post(handlers::discover::discover))
        // Backends
        .route("/backends", get(handlers::backends::list_backends))
        // Annotate -- per-handler body limit for BibTeX payloads.
        .route(
            "/annotate",
            post(handlers::annotate::start_annotate)
                .layer(axum::extract::DefaultBodyLimit::max(ANNOTATE_BODY_LIMIT)),
        )
        .route(
            "/annotate/from-file",
            post(handlers::annotate::annotate_from_file)
                .layer(axum::extract::DefaultBodyLimit::max(ANNOTATE_BODY_LIMIT)),
        )
        // Citation verification -- per-handler body limit for large BibTeX and
        // CSV payloads. The create endpoint accepts full BibTeX bibliographies
        // which can exceed the global 2 MiB default for large reference lists.
        .route(
            "/citation/create",
            post(handlers::citation::create)
                .layer(axum::extract::DefaultBodyLimit::max(CITATION_BODY_LIMIT)),
        )
        .route("/citation/claim", post(handlers::citation::claim))
        .route(
            "/citation/submit",
            post(handlers::citation::submit)
                .layer(axum::extract::DefaultBodyLimit::max(CITATION_BODY_LIMIT)),
        )
        .route(
            "/citation/{job_id}/status",
            get(handlers::citation::status),
        )
        .route(
            "/citation/{job_id}/rows",
            get(handlers::citation::rows),
        )
        .route(
            "/citation/{job_id}/export",
            post(handlers::citation::export),
        )
        .route(
            "/citation/{job_id}/auto-verify",
            post(handlers::citation::auto_verify),
        )
        // Citation source fetching (download PDFs/HTML from BibTeX URL/DOI fields)
        .route(
            "/citation/fetch-sources",
            post(handlers::citation::fetch_sources),
        )
        // BibTeX parsing for live preview (no network operations)
        .route(
            "/citation/parse-bib",
            post(handlers::citation::parse_bib)
                .layer(axum::extract::DefaultBodyLimit::max(CITATION_BODY_LIMIT)),
        )
        // BibTeX report generation (CSV + XLSX export of all entries)
        .route(
            "/citation/bib-report",
            post(handlers::citation::bib_report)
                .layer(axum::extract::DefaultBodyLimit::max(CITATION_BODY_LIMIT)),
        )
        // Shutdown
        .route("/shutdown", post(handlers::shutdown::shutdown))
        // OpenAPI specification -- placed inside the route group so that the
        // auth middleware (applied below) covers this endpoint when a bearer
        // token is configured. Unauthenticated callers must not be able to
        // enumerate the API surface of a token-secured server.
        .route(
            "/openapi.json",
            get(move || {
                let spec = openapi_spec.clone();
                async move {
                    (
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        spec,
                    )
                }
            }),
        );

    // Apply authentication middleware if a bearer token is configured.
    // All routes registered above -- including /openapi.json -- are covered.
    let api_routes = if state.auth.bearer_token_hash.is_some() {
        api_routes.layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::auth::auth_middleware,
        ))
    } else {
        api_routes
    };

    Router::new()
        .nest("/api/v1", api_routes)
        .layer(cors_layer)
        // Security headers are applied to every response after CORS processing
        // so that defensive policies (`X-Content-Type-Options`, `X-Frame-Options`,
        // `Content-Security-Policy`) are present on all responses.
        .layer(middleware::security_headers::SecurityHeadersLayer)
        // API version header is applied to every response so clients can detect
        // version mismatches at the HTTP header level without parsing the JSON
        // body. This replaces the per-DTO `api_version` field pattern.
        .layer(middleware::api_version::ApiVersionLayer)
        // Limit request body size to 2 MiB to prevent memory exhaustion from
        // oversized payloads. Individual endpoints that need larger bodies
        // (citation create, annotate) override this with per-handler limits
        // defined above.
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}
