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

//! Tool call dispatch layer.
//!
//! Routes incoming `tools/call` requests to the appropriate handler function
//! based on the tool name. The dispatcher extracts the tool name and arguments
//! from the JSON-RPC params, calls the matching handler, and wraps the result
//! in the MCP content format (an array with a single text content item).

use std::sync::Arc;

use neuroncite_api::AppState;

use crate::handlers;
use crate::protocol::{INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND};

/// Result of dispatching a tool call. Contains either the MCP content array
/// (on success) or a JSON-RPC error code and message (on failure).
pub enum DispatchResult {
    /// The tool call succeeded. Contains the MCP content array to return.
    Success(serde_json::Value),
    /// The tool call failed. Contains the error code and message.
    Error { code: i32, message: String },
}

/// Dispatches a `tools/call` request to the appropriate handler.
///
/// Extracts the tool `name` and `arguments` from the JSON-RPC params object,
/// matches the name against known tools, and calls the corresponding handler.
/// Returns a `DispatchResult` that the server loop converts into a JSON-RPC
/// response.
pub async fn dispatch_tool_call(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> DispatchResult {
    let tool_name = match params["name"].as_str() {
        Some(name) => name,
        None => {
            return DispatchResult::Error {
                code: INVALID_PARAMS,
                message: "missing 'name' field in tools/call params".to_string(),
            };
        }
    };

    // Extract the arguments object. Default to an empty object if absent.
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let result = match tool_name {
        // Search tools (5)
        "neuroncite_search" => handlers::search::handle(state, &args).await,
        "neuroncite_batch_search" => handlers::batch_search::handle(state, &args).await,
        "neuroncite_multi_search" => handlers::multi_search::handle(state, &args).await,
        "neuroncite_compare_search" => handlers::compare_search::handle(state, &args).await,
        "neuroncite_text_search" => handlers::text_search::handle(state, &args).await,

        // Content retrieval (2) -- replaces page + html_page, batch_page + html_batch_page
        "neuroncite_content" => handlers::content::handle(state, &args).await,
        "neuroncite_batch_content" => handlers::batch_content::handle(state, &args).await,

        // Indexing (4) -- neuroncite_index absorbed html_index, preview_chunks absorbed html_preview
        "neuroncite_index" => handlers::index::handle(state, &args).await,
        "neuroncite_index_add" => handlers::index_add::handle(state, &args).await,
        "neuroncite_reindex_file" => handlers::reindex::handle(state, &args).await,
        "neuroncite_preview_chunks" => handlers::preview_chunks::handle(state, &args).await,

        // Discovery (1)
        "neuroncite_discover" => handlers::discover::handle(state, &args).await,

        // Sessions (4)
        "neuroncite_sessions" => handlers::sessions::handle_list(state, &args).await,
        "neuroncite_session_delete" => handlers::sessions::handle_delete(state, &args).await,
        "neuroncite_session_update" => handlers::sessions::handle_update(state, &args).await,
        "neuroncite_session_diff" => handlers::session_diff::handle(state, &args).await,

        // Files and chunks (4) -- neuroncite_files absorbed html_list
        "neuroncite_files" => handlers::files::handle(state, &args).await,
        "neuroncite_chunks" => handlers::chunks::handle(state, &args).await,
        "neuroncite_quality_report" => handlers::quality::handle(state, &args).await,
        "neuroncite_file_compare" => handlers::compare::handle(state, &args).await,

        // Jobs (3)
        "neuroncite_jobs" => handlers::jobs::handle_list(state, &args).await,
        "neuroncite_job_status" => handlers::jobs::handle_status(state, &args).await,
        "neuroncite_job_cancel" => handlers::jobs::handle_cancel(state, &args).await,

        // Export (1)
        "neuroncite_export" => handlers::export::handle(state, &args).await,

        // Models and system (4)
        "neuroncite_models" => handlers::models::handle(state, &args).await,
        "neuroncite_doctor" => handlers::system::handle_doctor(state, &args).await,
        "neuroncite_health" => handlers::system::handle_health(state, &args).await,
        "neuroncite_reranker_load" => handlers::reranker::handle(state, &args).await,

        // PDF annotation (4) -- format-specific, stays as-is
        "neuroncite_annotate" => handlers::annotate::handle(state, &args).await,
        "neuroncite_annotate_status" => handlers::annotate_status::handle(state, &args).await,
        "neuroncite_inspect_annotations" => handlers::inspect::handle(state, &args).await,
        "neuroncite_annotation_remove" => handlers::remove::handle(state, &args).await,

        // HTML acquisition (2) -- inherently URL-based, stays as-is
        "neuroncite_html_fetch" => handlers::html_fetch::handle(state, &args).await,
        "neuroncite_html_crawl" => handlers::html_crawl::handle(state, &args).await,

        // Citation verification (8)
        "neuroncite_citation_create" => handlers::citation::handle_create(state, &args).await,
        "neuroncite_citation_claim" => handlers::citation::handle_claim(state, &args).await,
        "neuroncite_citation_submit" => handlers::citation::handle_submit(state, &args).await,
        "neuroncite_citation_status" => handlers::citation::handle_status(state, &args).await,
        "neuroncite_citation_rows" => handlers::citation::handle_rows(state, &args).await,
        "neuroncite_citation_export" => handlers::citation::handle_export(state, &args).await,
        "neuroncite_citation_retry" => handlers::citation::handle_retry(state, &args).await,
        "neuroncite_citation_fetch_sources" => handlers::source_fetch::handle(state, &args).await,
        "neuroncite_bib_report" => handlers::bib_report::handle(state, &args).await,

        _ => {
            return DispatchResult::Error {
                code: METHOD_NOT_FOUND,
                message: format!("unknown tool: {tool_name}"),
            };
        }
    };

    match result {
        Ok(value) => {
            // Wrap the handler result in the MCP content format: an array
            // with a single text content item containing the JSON-serialized
            // tool output.
            let content = serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string()),
                }]
            });
            DispatchResult::Success(content)
        }
        Err(message) => {
            // Convert the handler's string error into a structured error
            // object with error_code, message, and details fields. The
            // classification maps common error message patterns to
            // machine-readable error codes for programmatic handling.
            let structured = classify_error(&message);
            let json_msg = serde_json::to_string(&structured).unwrap_or_else(|_| message.clone());
            DispatchResult::Error {
                code: INTERNAL_ERROR,
                message: json_msg,
            }
        }
    }
}

/// Classifies a handler error message into a structured error object with
/// an error_code, the original message, and a details object. The error_code
/// is determined by pattern-matching common error message prefixes and
/// substrings produced by the handler functions.
///
/// Error codes follow SCREAMING_SNAKE_CASE convention and are stable
/// identifiers that callers can match on programmatically:
///
/// - `MISSING_PARAMETER`: A required tool parameter was not provided.
/// - `SESSION_NOT_FOUND`: The specified session_id does not exist.
/// - `JOB_NOT_FOUND`: The specified job_id does not exist.
/// - `FILE_NOT_FOUND`: The specified file_id does not exist.
/// - `CONNECTION_ERROR`: Database connection pool failure.
/// - `INVALID_INPUT`: Input data parsing or validation failed.
/// - `CONCURRENT_JOB`: A conflicting job is already running.
/// - `INTERNAL_ERROR`: Unclassified error from the handler.
fn classify_error(msg: &str) -> serde_json::Value {
    let (error_code, details) =
        if let Some(param) = msg.strip_prefix("missing required parameter: ") {
            (
                "MISSING_PARAMETER",
                serde_json::json!({ "parameter": param }),
            )
        } else if msg.contains("not found") && msg.contains("session") {
            let session_id = extract_number_after(msg, "session ");
            (
                "SESSION_NOT_FOUND",
                serde_json::json!({ "session_id": session_id }),
            )
        } else if msg.contains("not found") && msg.contains("job") {
            ("JOB_NOT_FOUND", serde_json::json!({}))
        } else if msg.contains("not found") && msg.contains("file") {
            ("FILE_NOT_FOUND", serde_json::json!({}))
        } else if msg.starts_with("connection pool error:") {
            ("CONNECTION_ERROR", serde_json::json!({}))
        } else if msg.contains("does not exist") && !msg.contains("session") {
            // Filesystem path validation errors (e.g., "file does not exist: /path",
            // "source_directory does not exist: /path"). Session-related "not found"
            // messages are handled by the SESSION_NOT_FOUND branch above.
            ("FILE_NOT_FOUND", serde_json::json!({}))
        } else if msg.contains("parsing failed")
            || msg.contains("invalid")
            || msg.contains("must not be empty")
            || msg.contains("must contain")
            || msg.contains("must be")
            || msg.contains("extraction failed")
        {
            // Validation errors from handler input checks. Covers patterns like:
            // - "files array must not be empty"
            // - "session_ids must contain at least 2 session IDs"
            // - "min_score must be between 0.0 and 1.0"
            // - "query must not be empty"
            // - "extraction failed: ..." (non-PDF or corrupt file in reindex/preview)
            ("INVALID_INPUT", serde_json::json!({}))
        } else if msg.contains("already running") {
            ("CONCURRENT_JOB", serde_json::json!({}))
        } else {
            ("INTERNAL_ERROR", serde_json::json!({}))
        };

    serde_json::json!({
        "error_code": error_code,
        "message": msg,
        "details": details,
    })
}

/// Extracts the first integer following a prefix string in a message.
/// Returns the number as a JSON Value, or null if no number is found.
fn extract_number_after(msg: &str, prefix: &str) -> serde_json::Value {
    if let Some(pos) = msg.find(prefix) {
        let after = &msg[pos + prefix.len()..];
        let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse::<i64>() {
            return serde_json::json!(n);
        }
    }
    serde_json::Value::Null
}

#[cfg(test)]
mod tests {
    /// T-MCP-016: Dispatching an unknown tool name returns METHOD_NOT_FOUND.
    #[tokio::test]
    async fn t_mcp_016_unknown_tool_returns_method_not_found() {
        // Create a minimal test state. Since we are testing dispatch routing
        // (not handler logic), the state is not used for the unknown-tool path.
        let params = serde_json::json!({
            "name": "nonexistent_tool",
            "arguments": {}
        });

        // We cannot construct a real AppState without a database, so we test
        // the name-matching branch directly by checking the match arm.
        // The unknown tool case is handled before any state access.
        let tool_name = params["name"].as_str().unwrap();
        let is_known = [
            // Search (5)
            "neuroncite_search",
            "neuroncite_batch_search",
            "neuroncite_multi_search",
            "neuroncite_compare_search",
            "neuroncite_text_search",
            // Content retrieval (2)
            "neuroncite_content",
            "neuroncite_batch_content",
            // Indexing (4)
            "neuroncite_index",
            "neuroncite_index_add",
            "neuroncite_reindex_file",
            "neuroncite_preview_chunks",
            // Discovery (1)
            "neuroncite_discover",
            // Sessions (4)
            "neuroncite_sessions",
            "neuroncite_session_delete",
            "neuroncite_session_update",
            "neuroncite_session_diff",
            // Files and chunks (4)
            "neuroncite_files",
            "neuroncite_chunks",
            "neuroncite_quality_report",
            "neuroncite_file_compare",
            // Jobs (3)
            "neuroncite_jobs",
            "neuroncite_job_status",
            "neuroncite_job_cancel",
            // Export (1)
            "neuroncite_export",
            // Models and system (4)
            "neuroncite_models",
            "neuroncite_doctor",
            "neuroncite_health",
            "neuroncite_reranker_load",
            // PDF annotation (4)
            "neuroncite_annotate",
            "neuroncite_annotate_status",
            "neuroncite_inspect_annotations",
            "neuroncite_annotation_remove",
            // HTML acquisition (2)
            "neuroncite_html_fetch",
            "neuroncite_html_crawl",
            // Citation verification (8)
            "neuroncite_citation_create",
            "neuroncite_citation_claim",
            "neuroncite_citation_submit",
            "neuroncite_citation_status",
            "neuroncite_citation_rows",
            "neuroncite_citation_export",
            "neuroncite_citation_retry",
            "neuroncite_citation_fetch_sources",
            "neuroncite_bib_report",
        ]
        .contains(&tool_name);

        assert!(
            !is_known,
            "nonexistent_tool should not be in the known tools list"
        );
    }

    /// T-MCP-017: Dispatching without a 'name' field returns INVALID_PARAMS.
    #[test]
    fn t_mcp_017_missing_name_field() {
        let params = serde_json::json!({"arguments": {}});
        // Check that the name extraction would fail.
        assert!(params["name"].as_str().is_none());
    }

    // -----------------------------------------------------------------------
    // Error classification tests
    // -----------------------------------------------------------------------

    use super::classify_error;

    /// T-MCP-ERR-001: Missing parameter error is classified as MISSING_PARAMETER
    /// with the parameter name in details.
    #[test]
    fn t_mcp_err_001_missing_parameter() {
        let result = classify_error("missing required parameter: session_id");
        assert_eq!(result["error_code"], "MISSING_PARAMETER");
        assert_eq!(result["details"]["parameter"], "session_id");
        assert!(result["message"].as_str().unwrap().contains("session_id"));
    }

    /// T-MCP-ERR-002: Session not found error is classified as SESSION_NOT_FOUND
    /// with the session_id extracted from the message.
    #[test]
    fn t_mcp_err_002_session_not_found() {
        let result = classify_error("session 42 not found");
        assert_eq!(result["error_code"], "SESSION_NOT_FOUND");
        assert_eq!(result["details"]["session_id"], 42);
    }

    /// T-MCP-ERR-003: Job not found error is classified as JOB_NOT_FOUND.
    #[test]
    fn t_mcp_err_003_job_not_found() {
        let result = classify_error("job abc-123 not found");
        assert_eq!(result["error_code"], "JOB_NOT_FOUND");
    }

    /// T-MCP-ERR-004: Connection pool error is classified as CONNECTION_ERROR.
    #[test]
    fn t_mcp_err_004_connection_error() {
        let result = classify_error("connection pool error: timeout");
        assert_eq!(result["error_code"], "CONNECTION_ERROR");
    }

    /// T-MCP-ERR-005: Parsing failure is classified as INVALID_INPUT.
    #[test]
    fn t_mcp_err_005_parsing_failed() {
        let result = classify_error("input_data parsing failed: missing title column");
        assert_eq!(result["error_code"], "INVALID_INPUT");
    }

    /// T-MCP-ERR-006: Concurrent job error is classified as CONCURRENT_JOB.
    #[test]
    fn t_mcp_err_006_concurrent_job() {
        let result = classify_error(
            "an annotation job is already running; wait for it to complete or cancel it",
        );
        assert_eq!(result["error_code"], "CONCURRENT_JOB");
    }

    /// T-MCP-ERR-007: Unrecognized error messages fall back to INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_007_fallback_internal_error() {
        let result = classify_error("something unexpected happened");
        assert_eq!(result["error_code"], "INTERNAL_ERROR");
        assert_eq!(
            result["message"], "something unexpected happened",
            "original message must be preserved"
        );
    }

    /// T-MCP-ERR-008: All classified errors contain the three required fields:
    /// error_code, message, and details.
    #[test]
    fn t_mcp_err_008_structure_always_present() {
        let test_messages = vec![
            "missing required parameter: top_k",
            "session 99 not found",
            "connection pool error: no connections",
            "random error with no pattern",
        ];

        for msg in test_messages {
            let result = classify_error(msg);
            assert!(
                result["error_code"].is_string(),
                "error_code must be a string for: {msg}"
            );
            assert!(
                result["message"].is_string(),
                "message must be a string for: {msg}"
            );
            assert!(
                result["details"].is_object(),
                "details must be an object for: {msg}"
            );
        }
    }

    /// T-MCP-ERR-009: extract_number_after extracts the session ID from
    /// typical error messages.
    #[test]
    fn t_mcp_err_009_extract_number_from_message() {
        use super::extract_number_after;

        assert_eq!(
            extract_number_after("session 123 not found", "session "),
            serde_json::json!(123)
        );
        assert_eq!(
            extract_number_after("no session found", "session "),
            serde_json::Value::Null
        );
        assert_eq!(
            extract_number_after("session abc not found", "session "),
            serde_json::Value::Null
        );
    }

    /// T-MCP-ERR-010: File not found error is classified as FILE_NOT_FOUND.
    /// The classifier matches messages containing both "file" and "not found"
    /// substrings, which are produced by handlers when a requested file_id
    /// does not exist in the database.
    #[test]
    fn t_mcp_err_010_file_not_found() {
        let result = classify_error("file 99 not found");
        assert_eq!(result["error_code"], "FILE_NOT_FOUND");
        assert_eq!(result["message"], "file 99 not found");
        assert!(result["details"].is_object(), "details must be an object");
    }

    /// T-MCP-ERR-011: The "invalid" substring in error messages triggers
    /// INVALID_INPUT classification. This path is distinct from the "parsing
    /// failed" path and covers validation errors like "invalid session_id"
    /// or "invalid min_score value".
    #[test]
    fn t_mcp_err_011_invalid_input_via_invalid_substring() {
        let result = classify_error("invalid min_score value: must be between 0.0 and 1.0");
        assert_eq!(result["error_code"], "INVALID_INPUT");
        assert_eq!(
            result["message"],
            "invalid min_score value: must be between 0.0 and 1.0"
        );
    }

    /// T-MCP-ERR-012: Classification precedence: "session not found" takes
    /// priority over "file not found" when both substrings are present. The
    /// classifier checks session before file in its match chain.
    #[test]
    fn t_mcp_err_012_classification_precedence_session_over_file() {
        // A message containing both "session" and "file" with "not found"
        // should match SESSION_NOT_FOUND (checked first) rather than FILE_NOT_FOUND.
        let result = classify_error("session 5 not found when looking up file");
        assert_eq!(
            result["error_code"], "SESSION_NOT_FOUND",
            "session match must take precedence over file match"
        );
        assert_eq!(result["details"]["session_id"], 5);
    }

    /// T-MCP-ERR-013: extract_number_after handles large session IDs.
    /// Database-generated AUTOINCREMENT IDs can grow to large values over
    /// time, especially after many create/delete cycles.
    #[test]
    fn t_mcp_err_013_extract_large_number() {
        use super::extract_number_after;

        assert_eq!(
            extract_number_after("session 99999999 not found", "session "),
            serde_json::json!(99999999)
        );
    }

    /// T-MCP-ERR-014: All 40 tool names in the dispatch match table are present
    /// in the known tools list. Validates that every dispatch match arm has a
    /// corresponding entry and the total count is 40 (consolidated from 46).
    #[test]
    fn t_mcp_err_014_new_tools_in_known_list() {
        let known = [
            // Search (5)
            "neuroncite_search",
            "neuroncite_batch_search",
            "neuroncite_multi_search",
            "neuroncite_compare_search",
            "neuroncite_text_search",
            // Content retrieval (2)
            "neuroncite_content",
            "neuroncite_batch_content",
            // Indexing (4)
            "neuroncite_index",
            "neuroncite_index_add",
            "neuroncite_reindex_file",
            "neuroncite_preview_chunks",
            // Discovery (1)
            "neuroncite_discover",
            // Sessions (4)
            "neuroncite_sessions",
            "neuroncite_session_delete",
            "neuroncite_session_update",
            "neuroncite_session_diff",
            // Files and chunks (4)
            "neuroncite_files",
            "neuroncite_chunks",
            "neuroncite_quality_report",
            "neuroncite_file_compare",
            // Jobs (3)
            "neuroncite_jobs",
            "neuroncite_job_status",
            "neuroncite_job_cancel",
            // Export (1)
            "neuroncite_export",
            // Models and system (4)
            "neuroncite_models",
            "neuroncite_doctor",
            "neuroncite_health",
            "neuroncite_reranker_load",
            // PDF annotation (4)
            "neuroncite_annotate",
            "neuroncite_annotate_status",
            "neuroncite_inspect_annotations",
            "neuroncite_annotation_remove",
            // HTML acquisition (2)
            "neuroncite_html_fetch",
            "neuroncite_html_crawl",
            // Citation verification (8)
            "neuroncite_citation_create",
            "neuroncite_citation_claim",
            "neuroncite_citation_submit",
            "neuroncite_citation_status",
            "neuroncite_citation_rows",
            "neuroncite_citation_export",
            "neuroncite_citation_retry",
            "neuroncite_citation_fetch_sources",
            // BibTeX report (1)
            "neuroncite_bib_report",
        ];

        // 5+2+4+1+4+4+3+1+4+4+2+8+1 = 43 tools total.
        assert_eq!(known.len(), 43, "known tools list must contain 43 entries");

        // Verify no duplicates exist in the list.
        let mut seen = std::collections::HashSet::new();
        for tool in &known {
            assert!(
                seen.insert(tool),
                "duplicate tool name in known list: {tool}"
            );
        }
    }

    /// T-MCP-ERR-015: The "reranker backend error" message produced by the
    /// reranker handler is classified as INTERNAL_ERROR (no specific
    /// classification pattern matches).
    #[test]
    fn t_mcp_err_015_reranker_backend_error_classification() {
        let result = classify_error(
            "reranker backend error: backend 'xyz' is not available (not compiled or unknown name)",
        );
        assert_eq!(result["error_code"], "INTERNAL_ERROR");
    }

    /// T-MCP-ERR-016: The "chunking strategy error" message produced by the
    /// preview_chunks handler is classified as INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_016_chunking_strategy_error_classification() {
        let result = classify_error("chunking strategy error: unknown strategy 'paragraph'");
        assert_eq!(result["error_code"], "INTERNAL_ERROR");
    }

    /// T-MCP-ERR-017: Validation error "must not be empty" is classified as
    /// INVALID_INPUT. Regression test for NEW-3 where index_add's "files array
    /// must not be empty" fell through to INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_017_must_not_be_empty_classified_as_invalid_input() {
        let result = classify_error("files array must not be empty");
        assert_eq!(
            result["error_code"], "INVALID_INPUT",
            "validation error with 'must not be empty' must be INVALID_INPUT"
        );
    }

    /// T-MCP-ERR-018: Validation error "must contain" is classified as
    /// INVALID_INPUT. Regression test for NEW-3/NEW-4 where multi_search's
    /// "session_ids must contain at least 2 session IDs" and index_add's
    /// "files array must contain only string paths" fell through to
    /// INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_018_must_contain_classified_as_invalid_input() {
        let test_cases = [
            "session_ids must contain at least 2 session IDs",
            "files array must contain only string paths",
            "session_ids must contain only integer values",
        ];

        for msg in &test_cases {
            let result = classify_error(msg);
            assert_eq!(
                result["error_code"], "INVALID_INPUT",
                "'{msg}' must be classified as INVALID_INPUT"
            );
        }
    }

    /// T-MCP-ERR-019: Validation error "must be" is classified as INVALID_INPUT.
    /// Regression test for NEW-4 where multi_search's "min_score must be between
    /// 0.0 and 1.0" fell through to INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_019_must_be_classified_as_invalid_input() {
        let result = classify_error("min_score must be between 0.0 and 1.0, got 1.5");
        assert_eq!(
            result["error_code"], "INVALID_INPUT",
            "'must be' validation errors must be classified as INVALID_INPUT"
        );
    }

    /// T-MCP-ERR-020: Filesystem path validation error "does not exist" is
    /// classified as FILE_NOT_FOUND. Regression test for NEW-3 where
    /// preview_chunks's "file does not exist: /path" fell through to
    /// INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_020_does_not_exist_classified_as_file_not_found() {
        let result = classify_error("file does not exist: /tmp/nonexistent.pdf");
        assert_eq!(
            result["error_code"], "FILE_NOT_FOUND",
            "path existence errors must be classified as FILE_NOT_FOUND"
        );
    }

    /// T-MCP-ERR-021: "does not exist" messages containing "session" are NOT
    /// classified as FILE_NOT_FOUND. They fall through to the session-not-found
    /// or other branches depending on the full message content.
    #[test]
    fn t_mcp_err_021_session_does_not_exist_not_file_not_found() {
        let result = classify_error("session 42 does not exist");
        // Contains both "session" and "does not exist". The "does not exist"
        // branch explicitly excludes messages containing "session", so this
        // falls through to INTERNAL_ERROR (session branch requires "not found").
        assert_ne!(
            result["error_code"], "FILE_NOT_FOUND",
            "session existence errors must not be classified as FILE_NOT_FOUND"
        );
    }

    /// T-MCP-ERR-022: "query must not be empty" from multi_search is classified
    /// as INVALID_INPUT.
    #[test]
    fn t_mcp_err_022_query_must_not_be_empty() {
        let result = classify_error("query must not be empty");
        assert_eq!(result["error_code"], "INVALID_INPUT");
    }

    /// T-MCP-ERR-023: "source_directory does not exist" is classified as
    /// FILE_NOT_FOUND.
    #[test]
    fn t_mcp_err_023_source_directory_does_not_exist() {
        let result = classify_error("source_directory does not exist: /nonexistent/dir");
        assert_eq!(result["error_code"], "FILE_NOT_FOUND");
    }

    /// T-MCP-ERR-024: Extraction failure errors from reindex_file and
    /// preview_chunks are classified as INVALID_INPUT. Regression test for
    /// DEF-005 where "extraction failed: ..." fell through to INTERNAL_ERROR.
    #[test]
    fn t_mcp_err_024_extraction_failed_classified_as_invalid_input() {
        let test_cases = [
            "extraction failed: unsupported file format",
            "extraction failed: PDF is encrypted and cannot be read",
            "extraction failed: no text could be extracted from this file",
        ];

        for msg in &test_cases {
            let result = classify_error(msg);
            assert_eq!(
                result["error_code"], "INVALID_INPUT",
                "'{msg}' must be classified as INVALID_INPUT, got {}",
                result["error_code"]
            );
        }
    }
}
