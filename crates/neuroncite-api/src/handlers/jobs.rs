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

//! Job status endpoint handlers.
//!
//! Provides handlers for querying individual job status, listing all jobs,
//! and canceling a running or queued job. The GUI polls the job status
//! endpoint to display a progress bar during indexing operations.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};

use neuroncite_store::{self as store, JobState};

use crate::dto::{API_VERSION, JobCancelResponse, JobListResponse, JobResponse, JobStateDto};
use crate::error::ApiError;
use crate::state::AppState;

/// Converts a `JobRow` from the store layer into a `JobResponse` DTO.
fn job_row_to_dto(row: &store::JobRow) -> JobResponse {
    JobResponse {
        api_version: API_VERSION.to_string(),
        id: row.id.clone(),
        kind: row.kind.clone(),
        session_id: row.session_id,
        state: JobStateDto::from(row.state),
        progress_done: row.progress_done,
        progress_total: row.progress_total,
        error_message: row.error_message.clone(),
        created_at: row.created_at,
        started_at: row.started_at,
        finished_at: row.finished_at,
    }
}

/// GET /api/v1/jobs/{id}
///
/// Returns the current status of a single job by its UUID.
#[utoipa::path(
    get,
    path = "/api/v1/jobs/{id}",
    params(("id" = String, Path, description = "Job UUID")),
    responses(
        (status = 200, description = "Job status", body = JobResponse),
        (status = 404, description = "Job not found"),
    )
)]
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<JobResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    let row = store::get_job(&conn, &id).map_err(ApiError::from)?;
    Ok(Json(job_row_to_dto(&row)))
}

/// GET /api/v1/jobs
///
/// Lists all jobs ordered by creation time descending.
#[utoipa::path(
    get,
    path = "/api/v1/jobs",
    responses(
        (status = 200, description = "Job list", body = JobListResponse),
    )
)]
pub async fn list_jobs(
    State(state): State<Arc<AppState>>,
) -> Result<Json<JobListResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    let rows = store::list_jobs(&conn).map_err(ApiError::from)?;
    let jobs = rows.iter().map(job_row_to_dto).collect();

    Ok(Json(JobListResponse {
        api_version: API_VERSION.to_string(),
        jobs,
    }))
}

/// POST /api/v1/jobs/{id}/cancel
///
/// Cancels a running or queued job. Terminal states (completed, failed,
/// already canceled) return a 409 Conflict.
#[utoipa::path(
    post,
    path = "/api/v1/jobs/{id}/cancel",
    params(("id" = String, Path, description = "Job UUID")),
    responses(
        (status = 200, description = "Job canceled", body = JobCancelResponse),
        (status = 404, description = "Job not found"),
        (status = 409, description = "Job cannot be canceled"),
    )
)]
pub async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<JobCancelResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the job exists before attempting the transition.
    let row = store::get_job(&conn, &id).map_err(ApiError::from)?;

    if row.state != JobState::Queued && row.state != JobState::Running {
        return Err(ApiError::Conflict {
            reason: format!(
                "job {} is in state {} and cannot be canceled",
                id, row.state
            ),
        });
    }

    store::update_job_state(&conn, &id, JobState::Canceled, None).map_err(ApiError::from)?;

    // Wake the executor so it detects the cancellation immediately.
    state.job_notify.notify_one();

    Ok(Json(JobCancelResponse {
        api_version: API_VERSION.to_string(),
        job_id: id,
        state: JobStateDto::Canceled,
    }))
}
