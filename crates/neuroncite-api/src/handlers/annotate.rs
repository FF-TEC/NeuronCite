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

//! Annotation endpoint handler.
//!
//! Accepts a request to start annotating PDF files with highlights and comments.
//! Creates an annotation job with the input data stored as serialized JSON in the
//! `params_json` column, validates the input early (before job creation), and
//! returns the job ID for progress polling. The annotation pipeline runs
//! asynchronously through the job executor.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use neuroncite_store::{self as store};

use crate::dto::{
    API_VERSION, AnnotateFromFileRequest, AnnotateFromFileResponse, AnnotateRequest,
    AnnotateResponse,
};
use crate::error::ApiError;
use crate::state::AppState;

/// POST /api/v1/annotate
///
/// Validates the annotation request, performs early input parsing to catch
/// malformed CSV/JSON before creating the job, stores the job parameters as
/// serialized JSON, and returns 202 Accepted with the job_id and quote count.
#[utoipa::path(
    post,
    path = "/api/v1/annotate",
    request_body = AnnotateRequest,
    responses(
        (status = 202, description = "Annotation job created", body = AnnotateResponse),
        (status = 400, description = "Invalid request parameters"),
        (status = 409, description = "Concurrent annotation job in progress"),
    )
)]
pub async fn start_annotate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnnotateRequest>,
) -> Result<(StatusCode, HeaderMap, Json<AnnotateResponse>), ApiError> {
    // Validate required fields via the DTO validate() method, which returns
    // ApiError::UnprocessableEntity for structural request errors (empty
    // source_directory, empty output_directory, empty input_data).
    req.validate()?;

    // Path containment: verify source and output directories are within the
    // allowed roots when the server is configured with an allowlist. Prevents
    // network clients from reading source PDFs or writing annotated outputs
    // to arbitrary filesystem locations.
    crate::util::validate_path_access(std::path::Path::new(&req.source_directory), &state.config)?;
    crate::util::validate_path_access(std::path::Path::new(&req.output_directory), &state.config)?;

    // Early validation: parse input data to detect malformed CSV/JSON before
    // creating the job record. This provides immediate feedback to the caller.
    let rows = neuroncite_annotate::parse_input(req.input_data.as_bytes()).map_err(|e| {
        ApiError::BadRequest {
            reason: format!("input_data parsing failed: {e}"),
        }
    })?;

    let total_quotes = rows.len();
    if total_quotes == 0 {
        return Err(ApiError::BadRequest {
            reason: "input_data contains no annotation rows".to_string(),
        });
    }

    // Verify the source directory exists.
    let source_dir = std::path::Path::new(&req.source_directory);
    if !source_dir.is_dir() {
        return Err(ApiError::BadRequest {
            reason: format!("source_directory does not exist: {}", req.source_directory),
        });
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Enforce concurrent job policy: only one annotation job at a time.
    // Uses has_active_job() with the partial index idx_job_state_active instead
    // of fetching all jobs and filtering client-side.
    let has_running = store::has_active_job(&conn, "annotate").map_err(ApiError::from)?;
    if has_running {
        return Err(ApiError::Conflict {
            reason: "an annotation job is already in progress".to_string(),
        });
    }

    // Serialize job parameters for storage in the params_json column.
    let params = serde_json::json!({
        "input_data": req.input_data,
        "source_directory": req.source_directory,
        "output_directory": req.output_directory,
        "default_color": req.default_color,
    });
    let params_json = serde_json::to_string(&params).map_err(|e| ApiError::Internal {
        reason: format!("params serialization: {e}"),
    })?;

    // Create the job record with params_json. Annotation jobs have no session_id.
    let job_id = uuid::Uuid::new_v4().to_string();
    store::create_job_with_params(&conn, &job_id, "annotate", None, Some(&params_json))
        .map_err(ApiError::from)?;

    // Set the initial progress_total to the number of quotes so that callers
    // polling the job_status endpoint see a meaningful "0 / N" progress
    // immediately, rather than the default "0 / 0" which provides no
    // indication of the expected workload.
    if let Err(e) = store::update_job_progress(&conn, &job_id, 0, total_quotes as i64) {
        tracing::warn!(job_id = %job_id, error = %e, "failed to set initial annotation progress");
    }

    // Wake the executor immediately so it picks up the new job without
    // waiting for the next poll interval.
    state.job_notify.notify_one();

    let mut headers = HeaderMap::new();
    let location = format!("/api/v1/jobs/{job_id}");
    if let Ok(v) = HeaderValue::from_str(&location) {
        headers.insert(axum::http::header::LOCATION, v);
    }
    Ok((
        StatusCode::ACCEPTED,
        headers,
        Json(AnnotateResponse {
            api_version: API_VERSION.to_string(),
            job_id,
            total_quotes,
        }),
    ))
}

/// POST /api/v1/annotate/from-file
///
/// Reads annotation instructions from a file on disk (CSV or JSON produced by
/// the citation export pipeline) and creates an annotation job. This endpoint
/// is used by the Annotations tab to annotate PDFs from a previously exported
/// file without re-running the full citation verification workflow.
///
/// The file content is read server-side and stored in the job's params_json
/// column, so the executor processes it identically to inline annotation
/// requests from the `/api/v1/annotate` endpoint.
#[utoipa::path(
    post,
    path = "/api/v1/annotate/from-file",
    request_body = AnnotateFromFileRequest,
    responses(
        (status = 202, description = "Annotation job created", body = AnnotateFromFileResponse),
        (status = 400, description = "Invalid request or file not found"),
        (status = 409, description = "Concurrent annotation job in progress"),
    )
)]
pub async fn annotate_from_file(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnnotateFromFileRequest>,
) -> Result<(StatusCode, HeaderMap, Json<AnnotateFromFileResponse>), ApiError> {
    // Validate required fields via the DTO validate() method, which returns
    // ApiError::UnprocessableEntity for structural request errors (empty
    // input_file, empty source_directory, empty output_directory).
    req.validate()?;

    // Path containment validation runs BEFORE any filesystem I/O to prevent
    // reading files that are outside the allowed roots. validate_path_access
    // returns the canonicalized path so that subsequent file operations use
    // the validated canonical path, preventing TOCTOU attacks where a symlink
    // is swapped between the containment check and the file open.
    let canonical_input =
        crate::util::validate_path_access(std::path::Path::new(&req.input_file), &state.config)?;
    crate::util::validate_path_access(std::path::Path::new(&req.source_directory), &state.config)?;
    crate::util::validate_path_access(std::path::Path::new(&req.output_directory), &state.config)?;

    // Check file existence AFTER containment validation so that an attacker
    // who provides a path outside the allowed roots gets a containment error
    // rather than an existence error (defense in depth). Uses the canonical
    // path returned by validate_path_access.
    if !canonical_input.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("input_file does not exist: {}", req.input_file),
        });
    }

    // Open the file using the canonicalized path from validate_path_access.
    // This ensures the file opened is the same file that passed the
    // containment check, closing the TOCTOU window.
    let file = std::fs::File::open(&canonical_input).map_err(|e| ApiError::BadRequest {
        reason: format!("failed to open input_file: {e}"),
    })?;
    let input_data = std::io::read_to_string(file).map_err(|e| ApiError::BadRequest {
        reason: format!("failed to read input_file: {e}"),
    })?;

    // Verify the source directory exists.
    let source_dir = std::path::Path::new(&req.source_directory);
    if !source_dir.is_dir() {
        return Err(ApiError::BadRequest {
            reason: format!("source_directory does not exist: {}", req.source_directory),
        });
    }

    // Early validation: parse input data to detect malformed content before
    // creating the job record.
    let rows = neuroncite_annotate::parse_input(input_data.as_bytes()).map_err(|e| {
        ApiError::BadRequest {
            reason: format!("input_file parsing failed: {e}"),
        }
    })?;

    let total_quotes = rows.len();
    if total_quotes == 0 {
        return Err(ApiError::BadRequest {
            reason: "input_file contains no annotation rows".to_string(),
        });
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Enforce concurrent job policy: only one annotation job at a time.
    // Uses has_active_job() with the partial index idx_job_state_active instead
    // of fetching all jobs and filtering client-side.
    let has_running = store::has_active_job(&conn, "annotate").map_err(ApiError::from)?;
    if has_running {
        return Err(ApiError::Conflict {
            reason: "an annotation job is already in progress".to_string(),
        });
    }

    // Serialize job parameters. The input_data is the file content (not the
    // file path), so the executor processes it identically to inline requests.
    let params = serde_json::json!({
        "input_data": input_data,
        "source_directory": req.source_directory,
        "output_directory": req.output_directory,
        "default_color": req.default_color,
    });
    let params_json = serde_json::to_string(&params).map_err(|e| ApiError::Internal {
        reason: format!("params serialization: {e}"),
    })?;

    let job_id = uuid::Uuid::new_v4().to_string();
    store::create_job_with_params(&conn, &job_id, "annotate", None, Some(&params_json))
        .map_err(ApiError::from)?;

    if let Err(e) = store::update_job_progress(&conn, &job_id, 0, total_quotes as i64) {
        tracing::warn!(job_id = %job_id, error = %e, "failed to set initial annotation progress");
    }

    // Wake the executor immediately so it picks up the new job.
    state.job_notify.notify_one();

    let mut headers = HeaderMap::new();
    let location = format!("/api/v1/jobs/{job_id}");
    if let Ok(v) = HeaderValue::from_str(&location) {
        headers.insert(axum::http::header::LOCATION, v);
    }
    Ok((
        StatusCode::ACCEPTED,
        headers,
        Json(AnnotateFromFileResponse {
            api_version: API_VERSION.to_string(),
            job_id,
            total_quotes,
        }),
    ))
}
