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

//! Handler for the `neuroncite_annotate_status` MCP tool.
//!
//! Returns per-quote progress information for a running or completed annotation
//! job. Provides paginated access to the `annotation_quote_status` table which
//! tracks each quote's processing status (pending, matched, not_found, error),
//! the match method used, and the page where the quote was located.
//!
//! Complements `neuroncite_job_status` which reports aggregate progress
//! (done/total). This handler exposes fine-grained per-quote details that are
//! also written into `annotate_report.json` after completion.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Returns per-quote annotation progress for a specific job.
///
/// # Parameters (from MCP tool call)
///
/// - `job_id` (required): The annotation job ID returned by `neuroncite_annotate`.
/// - `limit` (optional): Maximum number of quote rows to return (default: 100,
///   max: 500).
/// - `offset` (optional): Number of rows to skip for pagination (default: 0).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let limit = params["limit"].as_i64().unwrap_or(100).clamp(1, 500);
    let offset = params["offset"].as_i64().unwrap_or(0).max(0);

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate job exists and is an annotate job.
    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;

    if job.kind != "annotate" {
        return Err(format!(
            "job '{job_id}' is not an annotate job (kind='{}')",
            job.kind
        ));
    }

    // Aggregate status counts across all quotes in this job.
    let status_counts = neuroncite_store::count_quote_statuses_by_status(&conn, job_id)
        .map_err(|e| format!("counting quote statuses: {e}"))?;

    // Paginated list of individual quote status rows.
    let rows = neuroncite_store::list_quote_statuses(&conn, job_id, limit, offset)
        .map_err(|e| format!("listing quote statuses: {e}"))?;

    let quotes: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            serde_json::json!({
                "id": row.id,
                "title": row.title,
                "author": row.author,
                "quote_excerpt": row.quote_excerpt,
                "status": row.status,
                "match_method": row.match_method,
                "page": row.page,
                "pdf_filename": row.pdf_filename,
                "updated_at": row.updated_at,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "job_id": job_id,
        "job_state": format!("{:?}", job.state).to_lowercase(),
        "progress_done": job.progress_done,
        "progress_total": job.progress_total,
        "status_counts": {
            "pending": status_counts.get("pending").copied().unwrap_or(0),
            "matched": status_counts.get("matched").copied().unwrap_or(0),
            "not_found": status_counts.get("not_found").copied().unwrap_or(0),
            "error": status_counts.get("error").copied().unwrap_or(0),
        },
        "quotes": quotes,
        "limit": limit,
        "offset": offset,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-093: Missing job_id parameter returns None from the JSON accessor.
    #[test]
    fn t_mcp_093_annotate_status_missing_job_id() {
        let params = serde_json::json!({});
        assert!(
            params["job_id"].as_str().is_none(),
            "missing job_id must return None"
        );
    }

    /// T-MCP-094: limit defaults to 100 and offset defaults to 0 when absent.
    #[test]
    fn t_mcp_094_annotate_status_defaults() {
        let params = serde_json::json!({"job_id": "test-job"});
        let limit = params["limit"].as_i64().unwrap_or(100).clamp(1, 500);
        let offset = params["offset"].as_i64().unwrap_or(0).max(0);
        assert_eq!(limit, 100);
        assert_eq!(offset, 0);
    }

    /// T-MCP-095: limit is clamped to 500 when a larger value is provided.
    #[test]
    fn t_mcp_095_annotate_status_limit_clamped() {
        let params = serde_json::json!({"job_id": "test-job", "limit": 1000});
        let limit = params["limit"].as_i64().unwrap_or(100).clamp(1, 500);
        assert_eq!(limit, 500);
    }

    /// T-MCP-096: limit is clamped to 1 when zero or negative value is provided.
    #[test]
    fn t_mcp_096_annotate_status_limit_minimum() {
        let params = serde_json::json!({"job_id": "test-job", "limit": 0});
        let limit = params["limit"].as_i64().unwrap_or(100).clamp(1, 500);
        assert_eq!(limit, 1);
    }

    /// T-MCP-097: offset is clamped to 0 when a negative value is provided.
    #[test]
    fn t_mcp_097_annotate_status_negative_offset() {
        let params = serde_json::json!({"job_id": "test-job", "offset": -5});
        let offset = params["offset"].as_i64().unwrap_or(0).max(0);
        assert_eq!(offset, 0);
    }
}
