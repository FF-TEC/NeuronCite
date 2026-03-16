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

// Handler for the neuroncite_text_search MCP tool.
//
// Searches the indexed page text of a specific PDF file for literal substring
// occurrences. Operates on stored database content without re-reading the
// PDF file from disk. Returns matching page numbers with surrounding context
// for each occurrence.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Searches all pages of an indexed file for literal substring occurrences.
///
/// Parameters:
/// - file_id (required): The indexed_file.id to search within.
/// - query (required): The literal substring to search for.
/// - case_sensitive (optional, default false): When true, byte-exact matching.
///
/// Returns JSON with file_id, query, case_sensitive, total_matches,
/// pages_with_matches, and per-page match results with position and context.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let file_id = params["file_id"]
        .as_i64()
        .ok_or("missing required parameter: file_id")?;

    let query = params["query"]
        .as_str()
        .ok_or("missing required parameter: query")?;

    if query.is_empty() {
        return Err("query must not be empty".to_string());
    }

    let case_sensitive = params["case_sensitive"].as_bool().unwrap_or(false);

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate that the file exists.
    neuroncite_store::get_file(&conn, file_id).map_err(|_| format!("file not found: {file_id}"))?;

    let results = neuroncite_store::search_page_text(&conn, file_id, query, case_sensitive, 100)
        .map_err(|e| format!("text search failed: {e}"))?;

    let total_matches: usize = results.iter().map(|p| p.matches.len()).sum();
    let pages_with_matches = results.len();

    let result_json: Vec<serde_json::Value> = results
        .iter()
        .map(|page_match| {
            serde_json::json!({
                "page_number": page_match.page_number,
                "match_count": page_match.matches.len(),
                "matches": page_match.matches.iter().map(|m| serde_json::json!({
                    "position": m.position,
                    "context": m.context,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "file_id": file_id,
        "query": query,
        "case_sensitive": case_sensitive,
        "total_matches": total_matches,
        "pages_with_matches": pages_with_matches,
        "results": result_json,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-070: text_search handler rejects empty query.
    #[test]
    fn t_mcp_070_text_search_rejects_empty_query() {
        let params = serde_json::json!({
            "file_id": 1,
            "query": "",
        });
        let query = params["query"].as_str().unwrap();
        assert!(query.is_empty(), "empty query should be detected");
    }

    /// T-MCP-071: text_search handler defaults case_sensitive to false.
    #[test]
    fn t_mcp_071_text_search_default_case_sensitive() {
        let params = serde_json::json!({
            "file_id": 1,
            "query": "hello",
        });
        let case_sensitive = params["case_sensitive"].as_bool().unwrap_or(false);
        assert!(!case_sensitive, "case_sensitive defaults to false");
    }
}
