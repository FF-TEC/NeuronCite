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

//! Handler for the `neuroncite_batch_content` MCP tool.
//!
//! Retrieves content parts from multiple indexed documents in a single call.
//! Each request specifies a file_id and either a single part number or a
//! start/end range. Results are returned per-request with a partial failure
//! model: individual request errors are reported inline without failing the
//! entire batch.
//!
//! For HTML sources, each result is enriched with web_source metadata (URL,
//! title, domain) from the web_source table.

use std::collections::HashMap;
use std::sync::Arc;

use neuroncite_api::AppState;

use super::common::content_part_to_json;

/// Maximum number of individual content requests in a single batch call.
const MAX_BATCH_REQUESTS: usize = 10;

/// Maximum total parts across all requests in a single batch call. Prevents
/// unbounded response sizes from requests that each specify large ranges.
const MAX_TOTAL_PARTS: usize = 20;

/// Retrieves content parts from multiple documents in a single batch call.
///
/// # Parameters (from MCP tool call)
///
/// - `requests` (required): Array of request objects. Each object contains:
///   - `file_id` (required): Database file ID of the document.
///   - `part` (optional): Single part number to retrieve (1-indexed).
///   - `start` / `end` (optional): Part range to retrieve (1-indexed, inclusive).
///
/// Maximum 10 requests per batch, maximum 20 total parts across all requests.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let requests = params["requests"]
        .as_array()
        .ok_or("missing required parameter: requests (must be an array)")?;

    if requests.is_empty() {
        return Err("requests array must not be empty".to_string());
    }

    if requests.len() > MAX_BATCH_REQUESTS {
        return Err(format!(
            "batch exceeds maximum of {MAX_BATCH_REQUESTS} requests (got {})",
            requests.len()
        ));
    }

    // Pre-validate all requests and count total parts to enforce the limit
    // before executing any database queries.
    let mut total_parts: usize = 0;
    for (i, req) in requests.iter().enumerate() {
        req["file_id"]
            .as_i64()
            .ok_or_else(|| format!("request[{i}]: missing required field 'file_id'"))?;

        let part = req["part"].as_i64();
        let start = req["start"].as_i64();
        let end = req["end"].as_i64();

        match (part, start, end) {
            (Some(pn), None, None) => {
                if pn < 1 {
                    return Err(format!("request[{i}]: part must be >= 1, got {pn}"));
                }
                total_parts += 1;
            }
            (None, Some(s), Some(e)) => {
                if s < 1 {
                    return Err(format!("request[{i}]: start must be >= 1"));
                }
                if e < s {
                    return Err(format!("request[{i}]: end ({e}) must be >= start ({s})"));
                }
                total_parts += (e - s + 1) as usize;
            }
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(format!(
                    "request[{i}]: 'part' is mutually exclusive with 'start'/'end'"
                ));
            }
            (None, Some(_), None) | (None, None, Some(_)) => {
                return Err(format!(
                    "request[{i}]: both 'start' and 'end' are required for range retrieval"
                ));
            }
            (None, None, None) => {
                return Err(format!(
                    "request[{i}]: provide either 'part' or both 'start' and 'end'"
                ));
            }
        }
    }

    if total_parts > MAX_TOTAL_PARTS {
        return Err(format!(
            "total parts across all requests ({total_parts}) exceeds maximum of {MAX_TOTAL_PARTS}"
        ));
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Cache web_source metadata per file_id to avoid repeated DB lookups when
    // multiple requests target the same HTML file.
    let mut web_source_cache: HashMap<i64, Option<serde_json::Value>> = HashMap::new();

    // Process each request individually, collecting results with inline errors
    // for partial failure reporting.
    let mut results: Vec<serde_json::Value> = Vec::with_capacity(requests.len());
    let mut parts_returned: usize = 0;

    for (i, req) in requests.iter().enumerate() {
        let file_id = req["file_id"].as_i64().expect("validated above");
        let part = req["part"].as_i64();
        let start = req["start"].as_i64();
        let end = req["end"].as_i64();

        // Look up file record and web_source metadata (cached per file_id).
        let web_meta = web_source_cache
            .entry(file_id)
            .or_insert_with(|| {
                let file_row = neuroncite_store::get_file(&conn, file_id).ok()?;
                if file_row.source_type == "html" {
                    neuroncite_store::get_web_source_by_file(&conn, file_id)
                        .ok()
                        .map(|ws| {
                            serde_json::json!({
                                "url": ws.url,
                                "title": ws.title,
                                "domain": ws.domain,
                            })
                        })
                } else {
                    None
                }
            })
            .clone();

        let result = match (part, start, end) {
            (Some(pn), None, None) => match neuroncite_store::get_page(&conn, file_id, pn) {
                Ok(page) => {
                    parts_returned += 1;
                    let mut json = content_part_to_json(pn, &page.content, &page.backend);
                    json["file_id"] = serde_json::json!(file_id);
                    json["request_index"] = serde_json::json!(i);
                    if let Some(ref meta) = web_meta {
                        json["web_source"] = meta.clone();
                    }
                    json
                }
                Err(_) => {
                    serde_json::json!({
                        "file_id": file_id,
                        "request_index": i,
                        "error": format!(
                            "part not found: file_id={file_id}, part={pn}"
                        ),
                    })
                }
            },
            (None, Some(s), Some(e)) => {
                match neuroncite_store::get_pages_range(&conn, file_id, s, e) {
                    Ok(db_pages) => {
                        let mut parts_json: Vec<serde_json::Value> =
                            Vec::with_capacity((e - s + 1) as usize);
                        for pn in s..=e {
                            if let Some(page) = db_pages.iter().find(|p| p.page_number == pn) {
                                parts_json.push(content_part_to_json(
                                    pn,
                                    &page.content,
                                    &page.backend,
                                ));
                            } else {
                                parts_json.push(serde_json::json!({
                                    "part_number": pn,
                                    "content": null,
                                    "extraction_backend": null,
                                }));
                            }
                        }
                        parts_returned += db_pages.len();
                        let mut json = serde_json::json!({
                            "file_id": file_id,
                            "request_index": i,
                            "start": s,
                            "end": e,
                            "part_count": db_pages.len(),
                            "parts": parts_json,
                        });
                        if let Some(ref meta) = web_meta {
                            json["web_source"] = meta.clone();
                        }
                        json
                    }
                    Err(e_msg) => {
                        serde_json::json!({
                            "file_id": file_id,
                            "request_index": i,
                            "error": format!("part range query failed: {e_msg}"),
                        })
                    }
                }
            }
            _ => unreachable!("validated above"),
        };

        results.push(result);
    }

    Ok(serde_json::json!({
        "request_count": requests.len(),
        "total_parts_returned": parts_returned,
        "results": results,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::common::{MAX_CONTENT_BYTES, content_part_to_json, truncate_content};

    /// T-MCP-BCONTENT-001: Content truncation works correctly.
    #[test]
    fn t_mcp_bcontent_001_truncation() {
        let large = "A".repeat(MAX_CONTENT_BYTES + 100);
        let (result, truncated) = truncate_content(&large);
        assert!(truncated);
        assert!(result.len() <= MAX_CONTENT_BYTES);
    }

    /// T-MCP-BCONTENT-002: content_part_to_json includes truncation metadata.
    #[test]
    fn t_mcp_bcontent_002_json_truncation() {
        let large = "B".repeat(MAX_CONTENT_BYTES + 500);
        let json = content_part_to_json(5, &large, "pdfium");
        assert_eq!(json["truncated"], true);
        assert_eq!(json["part_number"], 5);
    }

    /// T-MCP-BCONTENT-003: content_part_to_json does not include truncation
    /// fields for small content.
    #[test]
    fn t_mcp_bcontent_003_json_no_truncation() {
        let json = content_part_to_json(1, "short", "pdf-extract");
        assert!(json.get("truncated").is_none());
        assert_eq!(json["content"], "short");
    }
}
