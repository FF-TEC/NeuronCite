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

//! Graceful shutdown endpoint handler.
//!
//! Accepts a shutdown request body containing the one-time nonce that was
//! printed to stdout at server startup. The nonce is compared in constant
//! time to prevent timing side-channel attacks. This endpoint is only
//! available when the server runs in headless mode (no GUI).
//!
//! The utoipa annotation declares HTTP 400 for the GUI-mode rejection path.
//! This matches `ApiError::BadRequest`, which maps to `StatusCode::BAD_REQUEST`
//! in the `IntoResponse` impl. The previous annotation incorrectly declared 403.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use subtle::ConstantTimeEq;

use crate::dto::{API_VERSION, ShutdownRequest, ShutdownResponse};
use crate::error::ApiError;
use crate::state::AppState;

/// POST /api/v1/shutdown
///
/// Initiates graceful server shutdown. Only available in headless mode.
/// The request body must include the nonce printed to stdout at startup.
/// Returns 200 OK with a confirmation message before the server begins
/// its shutdown sequence.
#[utoipa::path(
    post,
    path = "/api/v1/shutdown",
    request_body = ShutdownRequest,
    responses(
        (status = 200, description = "Shutdown initiated", body = ShutdownResponse),
        (status = 400, description = "Shutdown not available in GUI mode"),
        (status = 403, description = "Invalid shutdown nonce"),
    )
)]
pub async fn shutdown(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShutdownRequest>,
) -> Result<Json<ShutdownResponse>, ApiError> {
    if !state.headless {
        return Err(ApiError::BadRequest {
            reason: "shutdown is only available in headless mode".to_string(),
        });
    }

    // Constant-time comparison prevents timing side-channel attacks that could
    // allow an attacker to infer nonce bytes by measuring response latency.
    // Length check is performed first; mismatched lengths produce an immediate
    // rejection without leaking which byte position diverges.
    let provided = req.nonce.as_bytes();
    let expected = state.auth.shutdown_nonce.as_str().as_bytes();
    let nonce_valid = provided.len() == expected.len() && provided.ct_eq(expected).unwrap_u8() == 1;

    if !nonce_valid {
        tracing::warn!("shutdown request rejected: invalid nonce");
        return Err(ApiError::Forbidden {
            reason: "invalid shutdown nonce".to_string(),
        });
    }

    tracing::info!("shutdown requested via API with valid nonce; canceling server token");
    state.cancellation_token.cancel();

    Ok(Json(ShutdownResponse {
        api_version: API_VERSION.to_string(),
        status: "shutdown initiated".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    use crate::error::ApiError;

    // ---------------------------------------------------------------------------
    // T-SHUT-001: GUI-mode rejection returns HTTP 400 (not 403)
    // ---------------------------------------------------------------------------

    /// T-SHUT-001: Verifies that the error returned when shutdown is called in
    /// GUI mode (`state.headless == false`) maps to HTTP 400 Bad Request.
    ///
    /// The utoipa annotation on the shutdown handler declares `status = 400` for
    /// this branch. This test confirms that `ApiError::BadRequest` — the variant
    /// the handler returns — produces exactly HTTP 400, so the annotation and the
    /// implementation agree.
    ///
    /// The previous annotation incorrectly declared `status = 403` (Forbidden),
    /// which would have caused API clients and the generated OpenAPI spec to
    /// expect the wrong status code.
    #[test]
    fn t_shut_001_gui_mode_rejection_returns_400() {
        let err = ApiError::BadRequest {
            reason: "shutdown is only available in headless mode".to_string(),
        };

        let response = err.into_response();

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "shutdown GUI-mode rejection must return HTTP 400, not 403 or any other code"
        );
    }

    // ---------------------------------------------------------------------------
    // T-SHUT-002: Invalid nonce returns HTTP 403 Forbidden
    // ---------------------------------------------------------------------------

    /// T-SHUT-002: Verifies that a nonce mismatch (wrong value or wrong length)
    /// maps to HTTP 403 Forbidden, not HTTP 400 or any other code.
    ///
    /// The handler uses `ApiError::Forbidden` when the provided nonce does not
    /// match the startup nonce. This test confirms that variant maps to exactly
    /// HTTP 403 so API clients receive a distinct status from the GUI-mode
    /// rejection path (which returns HTTP 400).
    #[test]
    fn t_shut_002_invalid_nonce_returns_403() {
        let err = ApiError::Forbidden {
            reason: "invalid shutdown nonce".to_string(),
        };

        let response = err.into_response();

        assert_eq!(
            response.status(),
            StatusCode::FORBIDDEN,
            "invalid shutdown nonce must return HTTP 403, not 400 or any other code"
        );
    }
}
