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

//! Error types for the neuroncite-api crate.
//!
//! `ApiError` covers failures at the API boundary: request validation, handler
//! errors, worker channel failures, and serialization issues. Each variant maps
//! to an appropriate HTTP status code for the axum response.
//!
//! The `Internal` variant receives special treatment in the `IntoResponse` impl:
//! the full error details are logged at `tracing::error!` level for server-side
//! diagnostics, but only a generic "internal server error" message is returned
//! to the client. This prevents leaking internal implementation details (database
//! schemas, file paths, stack traces) to external callers.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Represents all error conditions that can occur during API request processing.
/// Each variant maps to a specific HTTP status code via the `IntoResponse` impl.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The request body failed validation (missing fields, invalid values).
    /// Maps to HTTP 400 Bad Request.
    #[error("bad request: {reason}")]
    BadRequest {
        /// A human-readable description of the validation failure.
        reason: String,
    },

    /// The requested resource (session, job, document) was not found.
    /// Maps to HTTP 404 Not Found.
    #[error("not found: {resource}")]
    NotFound {
        /// A description of the resource that was not found.
        resource: String,
    },

    /// A conflicting operation is in progress (e.g., concurrent indexing job).
    /// Maps to HTTP 409 Conflict.
    #[error("conflict: {reason}")]
    Conflict {
        /// A human-readable description of the conflict.
        reason: String,
    },

    /// The request lacks valid authentication credentials.
    /// Maps to HTTP 401 Unauthorized.
    #[error("unauthorized: {reason}")]
    Unauthorized {
        /// A human-readable description of the authentication failure.
        reason: String,
    },

    /// The caller is identified but is not permitted to perform this action.
    /// Maps to HTTP 403 Forbidden.
    /// Used for the shutdown endpoint nonce mismatch: the caller reached a
    /// guarded endpoint but supplied an incorrect authorization token.
    #[error("forbidden: {reason}")]
    Forbidden {
        /// A human-readable description of why access is denied.
        reason: String,
    },

    /// The GPU worker channel is closed or the worker task has panicked.
    /// Maps to HTTP 503 Service Unavailable.
    #[error("worker unavailable: {reason}")]
    WorkerUnavailable {
        /// A human-readable description of the worker failure.
        reason: String,
    },

    /// The database connection pool is exhausted or timed out. The server is
    /// temporarily unable to service the request but a retry may succeed once
    /// connections are released. Maps to HTTP 503 Service Unavailable.
    #[error("service unavailable: {reason}")]
    ServiceUnavailable {
        /// A human-readable description of the unavailability reason.
        reason: String,
    },

    /// An internal error occurred in a downstream crate (store, search, embed).
    /// Maps to HTTP 500 Internal Server Error.
    #[error("internal error: {reason}")]
    Internal {
        /// A human-readable description of the internal failure.
        reason: String,
    },
}

impl IntoResponse for ApiError {
    /// Converts an `ApiError` into an axum HTTP response with the appropriate
    /// status code and a JSON body containing the error message and api_version.
    ///
    /// For the `Internal` variant, the full error details are logged at
    /// `tracing::error!` level, but the client receives only a generic
    /// "internal server error" message. This prevents leaking implementation
    /// details (database errors, file paths, stack traces) in HTTP responses.
    /// All other variants return their user-facing messages directly.
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            Self::BadRequest { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            Self::NotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),
            Self::Conflict { .. } => (StatusCode::CONFLICT, self.to_string()),
            Self::Unauthorized { .. } => (StatusCode::UNAUTHORIZED, self.to_string()),
            Self::Forbidden { .. } => (StatusCode::FORBIDDEN, self.to_string()),
            Self::WorkerUnavailable { .. } => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            Self::ServiceUnavailable { .. } => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            Self::Internal { reason } => {
                // Log the full error details server-side for diagnostics.
                // The client receives only a generic message to prevent
                // information leakage of internal implementation details.
                tracing::error!(reason = %reason, "internal API error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
            }
        };

        let body = serde_json::json!({
            "api_version": crate::dto::API_VERSION,
            "error": error_message,
        });

        (status, axum::Json(body)).into_response()
    }
}

impl From<neuroncite_store::StoreError> for ApiError {
    /// Converts a `StoreError` into the corresponding `ApiError` variant.
    /// `NotFound` maps to `ApiError::NotFound`; all others map to `Internal`.
    fn from(err: neuroncite_store::StoreError) -> Self {
        match err {
            neuroncite_store::StoreError::NotFound { entity, id } => Self::NotFound {
                resource: format!("{entity} {id}"),
            },
            other => Self::Internal {
                reason: other.to_string(),
            },
        }
    }
}

impl From<r2d2::Error> for ApiError {
    /// Converts an `r2d2::Error` into `ApiError::ServiceUnavailable`. Pool
    /// exhaustion and connection timeouts are transient conditions: the server
    /// is temporarily unable to service the request but will recover when
    /// connections are released. HTTP 503 with a client-visible message is
    /// more appropriate than HTTP 500 (which implies an unrecoverable failure).
    fn from(err: r2d2::Error) -> Self {
        Self::ServiceUnavailable {
            reason: format!("database connection pool unavailable: {err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    // -----------------------------------------------------------------------
    // #48 -- ApiError variant -> HTTP status code mapping tests
    //
    // Each test constructs a single ApiError variant, calls into_response(),
    // and asserts that the HTTP status code on the response matches the
    // expected value from the IntoResponse implementation.
    // -----------------------------------------------------------------------

    /// Validates that `ApiError::BadRequest` produces an HTTP 400 response.
    /// The BadRequest variant represents client-side validation failures such
    /// as missing required fields or malformed JSON in the request body.
    #[test]
    fn t_err_001_bad_request_maps_to_400() {
        let err = ApiError::BadRequest {
            reason: "missing field: name".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "BadRequest variant must produce HTTP 400"
        );
    }

    /// Validates that `ApiError::NotFound` produces an HTTP 404 response.
    /// The NotFound variant represents lookups for sessions, jobs, or documents
    /// that do not exist in the database.
    #[test]
    fn t_err_002_not_found_maps_to_404() {
        let err = ApiError::NotFound {
            resource: "session abc-123".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "NotFound variant must produce HTTP 404"
        );
    }

    /// Validates that `ApiError::Internal` produces an HTTP 500 response.
    /// The Internal variant wraps errors from downstream crates (store, search,
    /// embed) that do not map to a more specific HTTP status code.
    #[test]
    fn t_err_003_internal_maps_to_500() {
        let err = ApiError::Internal {
            reason: "database connection lost".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal variant must produce HTTP 500"
        );
    }

    /// Validates that `ApiError::Unauthorized` produces an HTTP 401 response.
    /// The Unauthorized variant represents missing or invalid authentication
    /// credentials (bearer token mismatch, expired token, etc.).
    #[test]
    fn t_err_004_unauthorized_maps_to_401() {
        let err = ApiError::Unauthorized {
            reason: "invalid bearer token".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Unauthorized variant must produce HTTP 401"
        );
    }

    /// Validates that `ApiError::Conflict` produces an HTTP 409 response.
    /// The Conflict variant represents concurrent operation collisions such
    /// as two indexing jobs running simultaneously on the same session.
    #[test]
    fn t_err_005_conflict_maps_to_409() {
        let err = ApiError::Conflict {
            reason: "indexing job already in progress".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::CONFLICT,
            "Conflict variant must produce HTTP 409"
        );
    }

    /// Validates that `ApiError::WorkerUnavailable` produces an HTTP 503 response.
    /// The WorkerUnavailable variant represents GPU worker channel closure or
    /// worker task panics, indicating the embedding service is temporarily down.
    #[test]
    fn t_err_006_worker_unavailable_maps_to_503() {
        let err = ApiError::WorkerUnavailable {
            reason: "worker channel closed".into(),
        };
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "WorkerUnavailable variant must produce HTTP 503"
        );
    }

    // -----------------------------------------------------------------------
    // Response body structure tests
    //
    // The JSON body returned by into_response() must contain two fields:
    // "api_version" (always "v1") and "error" (the Display string of the
    // ApiError variant, except for Internal which returns a generic message).
    // -----------------------------------------------------------------------

    /// Validates that the JSON response body for a BadRequest contains the
    /// "api_version" field set to "v1" and the "error" field matching the
    /// Display output. BadRequest returns the user-facing message directly.
    #[tokio::test]
    async fn t_err_007_response_body_contains_api_version_and_error() {
        let err = ApiError::BadRequest {
            reason: "invalid page count".into(),
        };
        let expected_message = err.to_string();
        let response = err.into_response();

        // Extract the body bytes from the response
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("reading response body must not fail");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body must be valid JSON");

        assert_eq!(
            json["api_version"], "v1",
            "api_version field must be \"v1\""
        );
        assert_eq!(
            json["error"], expected_message,
            "error field must match the Display output of the ApiError"
        );
    }

    /// Validates that the Display trait output for each variant follows the
    /// expected format with the lowercase prefix and the reason/resource string.
    /// This confirms the thiserror #[error(...)] attributes produce correct messages.
    #[test]
    fn t_err_008_display_format_all_variants() {
        let cases: Vec<(ApiError, &str)> = vec![
            (
                ApiError::BadRequest { reason: "x".into() },
                "bad request: x",
            ),
            (
                ApiError::NotFound {
                    resource: "y".into(),
                },
                "not found: y",
            ),
            (ApiError::Conflict { reason: "z".into() }, "conflict: z"),
            (
                ApiError::Unauthorized { reason: "a".into() },
                "unauthorized: a",
            ),
            (
                ApiError::WorkerUnavailable { reason: "b".into() },
                "worker unavailable: b",
            ),
            (
                ApiError::Internal { reason: "c".into() },
                "internal error: c",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(
                err.to_string(),
                expected,
                "Display output mismatch for variant {:?}",
                std::mem::discriminant(&err)
            );
        }
    }

    // -----------------------------------------------------------------------
    // #49 -- From<StoreError> conversion tests
    //
    // The From<StoreError> impl maps StoreError::NotFound to ApiError::NotFound
    // with the entity and id concatenated into the resource string. All other
    // StoreError variants map to ApiError::Internal with the Display string as
    // the reason.
    // -----------------------------------------------------------------------

    /// Validates that `StoreError::NotFound` converts to `ApiError::NotFound`
    /// with the resource field formatted as "{entity} {id}". This is the only
    /// StoreError variant that receives special treatment in the conversion.
    #[test]
    fn t_err_009_store_not_found_converts_to_api_not_found() {
        let store_err = neuroncite_store::StoreError::NotFound {
            entity: "session".into(),
            id: "abc-123".into(),
        };
        let api_err: ApiError = store_err.into();

        // The ApiError must be the NotFound variant with the concatenated resource string
        match api_err {
            ApiError::NotFound { resource } => {
                assert_eq!(
                    resource, "session abc-123",
                    "resource field must combine entity and id with a space separator"
                );
            }
            other => panic!(
                "expected ApiError::NotFound, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that the HTTP response produced from a converted StoreError::NotFound
    /// carries status code 404. This tests the full conversion chain:
    /// StoreError::NotFound -> ApiError::NotFound -> Response(404).
    #[test]
    fn t_err_010_store_not_found_produces_404_response() {
        let store_err = neuroncite_store::StoreError::NotFound {
            entity: "file".into(),
            id: "doc.pdf".into(),
        };
        let api_err: ApiError = store_err.into();
        let response = api_err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "StoreError::NotFound -> ApiError::NotFound must produce HTTP 404"
        );
    }

    /// Validates that `StoreError::Pool` converts to `ApiError::Internal`.
    /// Pool errors are infrastructure failures (connection pool exhaustion)
    /// and do not map to a client-facing 4xx status code.
    #[test]
    fn t_err_011_store_pool_converts_to_api_internal() {
        let store_err = neuroncite_store::StoreError::pool("pool exhausted");
        let api_err: ApiError = store_err.into();

        match api_err {
            ApiError::Internal { reason } => {
                assert!(
                    reason.contains("pool"),
                    "reason must contain the original pool error message, got: {reason}"
                );
            }
            other => panic!(
                "expected ApiError::Internal, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that `StoreError::Migration` converts to `ApiError::Internal`.
    /// Schema migration failures are server-side issues, so they map to HTTP 500.
    #[test]
    fn t_err_012_store_migration_converts_to_api_internal() {
        let store_err = neuroncite_store::StoreError::migration(5, "column already exists");
        let api_err: ApiError = store_err.into();

        match api_err {
            ApiError::Internal { reason } => {
                assert!(
                    reason.contains("migration"),
                    "reason must contain the migration error context, got: {reason}"
                );
            }
            other => panic!(
                "expected ApiError::Internal, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that `StoreError::HnswIndex` converts to `ApiError::Internal`.
    /// HNSW index corruption is an internal data integrity issue.
    #[test]
    fn t_err_013_store_hnsw_converts_to_api_internal() {
        let store_err = neuroncite_store::StoreError::hnsw("corrupted level graph");
        let api_err: ApiError = store_err.into();

        match api_err {
            ApiError::Internal { reason } => {
                assert!(
                    reason.contains("HNSW") || reason.contains("corrupted"),
                    "reason must reflect the HNSW error, got: {reason}"
                );
            }
            other => panic!(
                "expected ApiError::Internal, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that `StoreError::Manifest` converts to `ApiError::Internal`.
    /// Manifest parse failures are server-side configuration issues.
    #[test]
    fn t_err_014_store_manifest_converts_to_api_internal() {
        let store_err = neuroncite_store::StoreError::manifest("invalid TOML");
        let api_err: ApiError = store_err.into();

        match api_err {
            ApiError::Internal { reason } => {
                assert!(
                    reason.contains("manifest"),
                    "reason must contain the manifest error context, got: {reason}"
                );
            }
            other => panic!(
                "expected ApiError::Internal, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that `StoreError::Io` converts to `ApiError::Internal`.
    /// File system I/O errors in the store layer are server-side failures.
    #[test]
    fn t_err_015_store_io_converts_to_api_internal() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let store_err = neuroncite_store::StoreError::io("/data/index.bin", io_err);
        let api_err: ApiError = store_err.into();

        match api_err {
            ApiError::Internal { reason } => {
                assert!(
                    reason.contains("I/O") || reason.contains("access denied"),
                    "reason must reflect the I/O error, got: {reason}"
                );
            }
            other => panic!(
                "expected ApiError::Internal, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// Validates that the HTTP response produced from non-NotFound StoreError
    /// variants carries status code 500. This tests the conversion chain:
    /// StoreError::Pool -> ApiError::Internal -> Response(500).
    #[test]
    fn t_err_016_store_non_not_found_produces_500_response() {
        let store_err = neuroncite_store::StoreError::pool("timeout");
        let api_err: ApiError = store_err.into();
        let response = api_err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "non-NotFound StoreError variants must produce HTTP 500"
        );
    }

    /// Validates that the Internal variant returns a generic error message to
    /// the client rather than the detailed reason string. The detailed reason
    /// is logged server-side via tracing::error! but must not appear in the
    /// HTTP response body to prevent information leakage.
    #[tokio::test]
    async fn t_err_017_internal_error_does_not_leak_details() {
        let sensitive_reason = "database connection pool: /var/lib/neuroncite/data.db: SQLITE_BUSY";
        let err = ApiError::Internal {
            reason: sensitive_reason.into(),
        };
        let response = err.into_response();

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("reading response body must not fail");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body must be valid JSON");

        // The error field must contain only the generic message, not the
        // sensitive details from the reason field.
        assert_eq!(
            json["error"], "internal server error",
            "Internal variant must return generic error message to the client"
        );
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(
            !body_str.contains("SQLITE_BUSY"),
            "response body must not contain internal error details"
        );
        assert!(
            !body_str.contains("/var/lib"),
            "response body must not contain file system paths"
        );
    }
}
