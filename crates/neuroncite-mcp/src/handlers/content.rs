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

//! Handler for the `neuroncite_content` MCP tool.
//!
//! Retrieves text content from an indexed document by part number. A "part" is
//! the logical content unit of the source format: a page in PDFs, a heading-based
//! section in HTML, a paragraph block in plain text, or the full OCR output for
//! images.
//!
//! Supports single-part retrieval (via `part`) and contiguous range retrieval
//! (via `start`/`end`, maximum 20 parts per request). Individual part content is
//! truncated at 100 KB to prevent MCP response size overflow.
//!
//! For HTML sources, the response is enriched with web_source metadata (URL,
//! title, domain) from the web_source table.

use std::sync::Arc;

use neuroncite_api::AppState;

use super::common::content_part_to_json;

/// Maximum number of parts allowed in a single range request.
const MAX_PART_RANGE: i64 = 20;

/// Retrieves text content from an indexed document.
///
/// # Parameters (from MCP tool call)
///
/// - `file_id` (required): Database file ID of the document.
/// - `part` (optional): 1-based part number for single-part retrieval.
/// - `start` (optional): First part of a contiguous range (1-based, inclusive).
/// - `end` (optional): Last part of a contiguous range (1-based, inclusive).
///
/// Provide either `part` alone, or both `start` and `end`. For HTML sources,
/// the response includes a `web_source` object with URL, title, and domain.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let file_id = params["file_id"]
        .as_i64()
        .ok_or("missing required parameter: file_id")?;

    let part = params["part"].as_i64();
    let start = params["start"].as_i64();
    let end = params["end"].as_i64();

    // Validate parameter combinations and ranges before any database access.
    // This produces clear validation errors without incurring a DB round-trip
    // for obviously invalid requests.
    match (part, start, end) {
        (Some(pn), None, None) if pn < 1 => {
            return Err(format!("part must be >= 1 (1-indexed), got {pn}"));
        }
        (None, Some(s), Some(e)) => {
            if s < 1 {
                return Err("start must be >= 1".to_string());
            }
            if s > e {
                return Err(format!("start ({s}) must be <= end ({e})"));
            }
            if e - s + 1 > MAX_PART_RANGE {
                return Err(format!(
                    "part range exceeds maximum of {MAX_PART_RANGE} parts \
                     (requested {} parts)",
                    e - s + 1
                ));
            }
        }
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err("'part' is mutually exclusive with 'start'/'end'; \
                 provide either 'part' alone or both 'start' and 'end'"
                .to_string());
        }
        (None, Some(_), None) | (None, None, Some(_)) => {
            return Err("both 'start' and 'end' are required for range retrieval".to_string());
        }
        (None, None, None) => {
            return Err("provide either 'part' or both 'start' and 'end'".to_string());
        }
        _ => {} // Valid combinations: (Some(>=1), None, None) or (None, Some, Some) already validated
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Look up the file record to determine source_type. This enables
    // enriching the response with web_source metadata for HTML files.
    let file_row = neuroncite_store::get_file(&conn, file_id)
        .map_err(|_| format!("file {file_id} not found"))?;

    // For HTML sources, fetch web_source metadata (URL, title, domain, etc.)
    // to include in the response alongside the content.
    let web_meta = if file_row.source_type == "html" {
        neuroncite_store::get_web_source_by_file(&conn, file_id)
            .ok()
            .map(|ws| {
                serde_json::json!({
                    "url": ws.url,
                    "title": ws.title,
                    "domain": ws.domain,
                    "language": ws.language,
                    "author": ws.author,
                })
            })
    } else {
        None
    };

    match (part, start, end) {
        // Single-part retrieval mode.
        (Some(pn), None, None) => {
            // Part numbers map to page_number values in the database.
            let page = neuroncite_store::get_page(&conn, file_id, pn)
                .map_err(|_| format!("part not found: file_id={file_id}, part={pn}"))?;

            let mut response = content_part_to_json(pn, &page.content, &page.backend);
            response["file_id"] = serde_json::json!(file_id);
            response["source_type"] = serde_json::json!(file_row.source_type);
            if let Some(meta) = web_meta {
                response["web_source"] = meta;
            }
            Ok(response)
        }

        // Range retrieval mode: fetches a contiguous range of parts.
        // All range validation (s >= 1, s <= e, range size) is handled by the
        // pre-validation block above.
        (None, Some(s), Some(e)) => {
            // Part numbers map to page_number values in the database.
            let db_pages = neuroncite_store::get_pages_range(&conn, file_id, s, e)
                .map_err(|err| format!("part range query failed: {err}"))?;

            // Build response array with one entry per part in the range. Parts
            // missing from the database (e.g., failed extraction) appear as
            // null entries so the caller sees which parts are absent.
            let mut parts_json: Vec<serde_json::Value> = Vec::with_capacity((e - s + 1) as usize);
            for pn in s..=e {
                if let Some(page) = db_pages.iter().find(|p| p.page_number == pn) {
                    parts_json.push(content_part_to_json(pn, &page.content, &page.backend));
                } else {
                    parts_json.push(serde_json::json!({
                        "part_number": pn,
                        "content": null,
                        "extraction_backend": null,
                    }));
                }
            }

            let mut response = serde_json::json!({
                "file_id": file_id,
                "source_type": file_row.source_type,
                "start": s,
                "end": e,
                "part_count": db_pages.len(),
                "parts": parts_json,
            });
            if let Some(meta) = web_meta {
                response["web_source"] = meta;
            }
            Ok(response)
        }

        // All invalid parameter combinations are caught by the pre-validation
        // block above, so this branch is unreachable.
        _ => unreachable!("parameter validation above covers all invalid combinations"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::common::MAX_CONTENT_BYTES;
    use super::*;

    /// T-MCP-CONTENT-001: content_part_to_json preserves metadata fields.
    #[test]
    fn t_mcp_content_001_json_preserves_metadata() {
        let json = content_part_to_json(42, "test content", "pdfium");
        assert_eq!(json["part_number"], 42);
        assert_eq!(json["extraction_backend"], "pdfium");
        assert_eq!(json["content"], "test content");
    }

    /// T-MCP-CONTENT-002: Truncation includes metadata for large content.
    #[test]
    fn t_mcp_content_002_truncation_metadata() {
        let large = "B".repeat(MAX_CONTENT_BYTES + 1000);
        let json = content_part_to_json(1, &large, "pdf-extract");
        assert_eq!(json["truncated"], true);
        assert_eq!(json["original_bytes"], large.len());
    }

    /// T-MCP-CONTENT-003: part=0 returns a validation error.
    #[tokio::test]
    async fn t_mcp_content_003_part_zero_returns_validation_error() {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let pool = {
            let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
                conn.execute_batch("PRAGMA foreign_keys = ON;")?;
                Ok(())
            });
            r2d2::Pool::builder()
                .max_size(2)
                .build(manager)
                .expect("pool build")
        };
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let state = neuroncite_api::AppState::new(
            pool,
            worker_handle,
            AppConfig::default(),
            true,
            None,
            384,
        )
        .expect("test AppState construction must succeed");

        let params = serde_json::json!({ "file_id": 1, "part": 0 });
        let result = handle(&state, &params).await;
        assert!(result.is_err(), "part=0 must return an error");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("part") && err_msg.contains(">= 1"),
            "error must mention part >= 1 constraint, got: {err_msg}"
        );
    }
}
